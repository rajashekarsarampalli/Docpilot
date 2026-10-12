"""
DocPilot Telegram Bot
=====================
Commands:
  /start      – welcome message
  /login      – authenticate with Django credentials
  /logout     – disconnect your account
  /register   – create a new DocPilot account
  /status     – show your linked account + usage stats
  /myfiles    – quick list of your files
  /myfolders  – quick list of your folders
  /help       – list all commands
  /cancel     – cancel current login/register flow

Any other text is forwarded to the DocPilot AI agent.
Documents and photos sent to the bot are ingested into the user's workspace.

Per-user isolation
------------------
Every chat_id is stored as a TelegramUser row in the DB.
After /login, that row is linked to a real Django User.
Every agent call receives that Django User so all file
operations are strictly scoped to that user's data.
"""

import logging
import os
import re
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Django bootstrap (guard against double-setup) ────────────────────────────
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
try:
    import django
    django.setup()
except RuntimeError:
    pass  # already set up (e.g. called from manage.py run_bot)

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TimedOut, NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

from telegram_bot.models import TelegramUser
from agent.graph import graph, FILE_GENERATING_ACTIONS
from files.models import UploadedFile, Folder
from files.services.file_tools import (
    list_files,
    list_folders,
    media_root,
    relative_media_path,
    workspace_stats,
)

logger = logging.getLogger(__name__)

# ── ConversationHandler states ───────────────────────────────────────────────
WAITING_USERNAME, WAITING_PASSWORD = range(2)
REG_USERNAME, REG_PASSWORD, REG_CONFIRM = range(10, 13)

# ── Blocked file extensions (security) ───────────────────────────────────────
BLOCKED_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".sh", ".msi", ".com", ".scr", ".pif",
    ".vbs", ".js", ".wsf", ".ps1", ".reg", ".dll", ".sys",
}

# Max conversation memory per user (last N exchanges)
MAX_MEMORY = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_or_create_session(update: Update) -> TelegramUser:
    """Return (or create) the TelegramUser for this chat, storing Telegram metadata."""
    chat_id = update.effective_chat.id
    tg_user = update.effective_user

    obj, created = TelegramUser.objects.get_or_create(chat_id=chat_id)

    # Keep Telegram profile info fresh
    changed = False
    if tg_user:
        if obj.telegram_username != tg_user.username:
            obj.telegram_username = tg_user.username
            changed = True
        if obj.first_name != tg_user.first_name:
            obj.first_name = tg_user.first_name
            changed = True
    if changed or created:
        obj.save()

    return obj


def format_telegram_markdown(text: str) -> str:
    """
    Convert standard markdown (headers, double asterisks, code blocks)
    into clean, valid Telegram MarkdownV1 that will never crash or show broken asterisks.
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("### "):
            header_text = trimmed[4:].strip().replace("*", "")
            lines.append(f"📌 *{header_text}*")
        elif trimmed.startswith("## "):
            header_text = trimmed[3:].strip().replace("*", "")
            lines.append(f"📌 *{header_text}*")
        elif trimmed.startswith("# "):
            header_text = trimmed[2:].strip().replace("*", "")
            lines.append(f"📌 *{header_text}*")
        else:
            lines.append(line)
    text = "\n".join(lines)

    # Convert standard markdown **bold** to Telegram *bold*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)

    # Protect code blocks and escape underscores inside words/filenames outside code
    parts = []
    code_pattern = re.compile(r'(```[\s\S]*?```|`[^`\n]+`)')
    last_idx = 0
    for match in code_pattern.finditer(text):
        non_code = text[last_idx:match.start()]
        non_code = re.sub(r'(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])', r'\_', non_code)
        parts.append(non_code)
        parts.append(match.group(0))
        last_idx = match.end()
    non_code = text[last_idx:]
    non_code = re.sub(r'(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])', r'\_', non_code)
    parts.append(non_code)
    text = "".join(parts)

    # Clean up any leftover double asterisks
    text = text.replace("**", "*")

    # Balance unescaped asterisks
    stars = len(re.findall(r'(?<!\\)\*', text))
    if stars % 2 != 0:
        last = text.rfind('*')
        if last != -1:
            text = text[:last] + text[last+1:]

    # Balance backticks
    if text.count('`') % 2 != 0:
        last_tick = text.rfind('`')
        if last_tick != -1:
            text = text[:last_tick] + text[last_tick+1:]

    return text


def strip_markdown(text: str) -> str:
    """Cleanly remove markdown symbols for pristine plain-text fallback without asterisks."""
    if not text:
        return ""
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    return text.replace('**', '').replace('*', '').replace('`', '').strip()


_safe_text = format_telegram_markdown


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


async def run_agent_async(user, user_request: str, memory: list = None) -> dict:
    """Run the synchronous LangGraph agent in a thread pool so we don't block the event loop.

    Returns the full agent result dict (not just the response string) so the
    caller can inspect download_path and history.
    """
    # Prepend conversation memory for multi-turn context
    if memory:
        context_lines = []
        for m in memory[-MAX_MEMORY:]:
            context_lines.append(f"User: {m['user']}")
            context_lines.append(f"DocPilot: {m['bot']}")
        context_str = "\n".join(context_lines)
        enriched_request = (
            f"Previous conversation:\n{context_str}\n\n"
            f"Current request: {user_request}"
        )
    else:
        enriched_request = user_request

    def _invoke():
        from django.db import close_old_connections
        close_old_connections()
        try:
            return graph.invoke({"user_request": enriched_request, "user": user})
        finally:
            close_old_connections()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _invoke)
        return result
    except Exception as exc:
        logger.error("Agent error for user %s: %s", getattr(user, "username", "?"), exc)
        return {"final_response": f"❌ Something went wrong: {exc}"}


async def safe_send(update: Update, text: str, parse_mode: str = ParseMode.MARKDOWN, reply_markup=None) -> None:
    """Send message to user; handles markdown escaping, deleted messages, callback queries, and falls back to clean plain text."""
    # 1. Try formatted message via reply_text or send_message
    try:
        formatted = format_telegram_markdown(text) if parse_mode == ParseMode.MARKDOWN else text
        kwargs = {"parse_mode": parse_mode, "read_timeout": 30.0, "write_timeout": 30.0}
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup

        if update.message:
            await update.message.reply_text(formatted, **kwargs)
            return
        elif update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(formatted, **kwargs)
            return
        elif update.effective_chat:
            await update.effective_chat.send_message(formatted, **kwargs)
            return
    except Exception as e:
        logger.debug("Formatted send failed (%s), falling back to clean plain text", e)

    # 2. Fallback to clean plain text (strip markdown symbols so no asterisks are displayed)
    try:
        clean_text = strip_markdown(text)
        kwargs = {"read_timeout": 30.0, "write_timeout": 30.0}
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup

        if update.message:
            await update.message.reply_text(clean_text, **kwargs)
        elif update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(clean_text, **kwargs)
        elif update.effective_chat:
            await update.effective_chat.send_message(clean_text, **kwargs)
    except Exception as e:
        logger.error("safe_send failed entirely: %s", e)


# Backward compatibility alias
safe_reply = safe_send


def build_file_action_keyboard(file_id: int, file_name: str) -> InlineKeyboardMarkup:
    """Construct 1-tap interactive inline action buttons for an uploaded or searched document."""
    ext = Path(file_name).suffix.lower()
    keyboard = [
        [
            InlineKeyboardButton("📑 Summarize", callback_data=f"sum:{file_id}"),
            InlineKeyboardButton("📥 Download", callback_data=f"send:{file_id}"),
        ]
    ]

    # Format-specific actions
    if ext == ".pdf":
        keyboard.append([
            InlineKeyboardButton("📝 To Word", callback_data=f"word:{file_id}"),
            InlineKeyboardButton("🗜️ Compress", callback_data=f"comp:{file_id}"),
        ])
    elif ext in (".docx", ".doc"):
        keyboard.append([
            InlineKeyboardButton("📑 To PDF", callback_data=f"pdf:{file_id}"),
            InlineKeyboardButton("📝 Extract Text", callback_data=f"txt:{file_id}"),
        ])
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        keyboard.append([
            InlineKeyboardButton("🔍 OCR Text", callback_data=f"ocr:{file_id}"),
            InlineKeyboardButton("📝 Extract Text", callback_data=f"txt:{file_id}"),
        ])

    elif ext in (".pptx", ".ppt"):
        keyboard.append([
            InlineKeyboardButton("📽️ Summarize Slides", callback_data=f"sum:{file_id}"),
            InlineKeyboardButton("📝 Extract Text", callback_data=f"txt:{file_id}"),
        ])
    elif ext in (".xlsx", ".xls"):
        keyboard.append([
            InlineKeyboardButton("📊 Analyze Sheet", callback_data=f"sum:{file_id}"),
            InlineKeyboardButton("📝 View Data", callback_data=f"txt:{file_id}"),
        ])

    # Powerhouse study & testing actions (applicable to all documents)
    keyboard.append([
        InlineKeyboardButton("❓ Quiz Me", callback_data=f"quiz:{file_id}"),
        InlineKeyboardButton("📇 Flashcards", callback_data=f"fc:{file_id}"),
    ])

    return InlineKeyboardMarkup(keyboard)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks from inline keyboards (1-tap actions)."""
    query = update.callback_query
    await query.answer()

    session = get_or_create_session(update)
    if not session.is_authenticated or not session.user_id:
        await safe_send(update, "⚠️ You must be logged in to use actions. Use /login first.")
        return

    session.refresh_from_db()
    user = session.user
    data = query.data or ""

    if ":" not in data:
        return

    action_code, file_id_str = data.split(":", 1)
    try:
        file_id = int(file_id_str)
    except ValueError:
        return

    action_prompts = {
        "sum": f"Summarize file {file_id}",
        "send": f"Send me file {file_id}",
        "word": f"Convert file {file_id} to Word",
        "comp": f"Compress file {file_id}",
        "pdf": f"Convert file {file_id} to PDF",
        "txt": f"Extract text from file {file_id}",
        "ocr": f"Run OCR on file {file_id}",
        "quiz": f"Generate a quiz from file {file_id}",
        "fc": f"Generate flashcards from file {file_id}",
    }

    prompt = action_prompts.get(action_code)
    if not prompt:
        return

    # Indicate activity
    thinking = None
    if query.message:
        try:
            await query.message.reply_chat_action("typing")
            thinking = await query.message.reply_text(f"⏳ Working on _{prompt}_...", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

    memory = context.user_data.get("memory", [])
    result = await run_agent_async(user, prompt, memory=memory)
    response_text = result.get("final_response") or "✅ Operation completed."

    if thinking:
        try:
            await thinking.delete()
        except Exception:
            pass

    await safe_send(update, response_text)

    # Check if a file was produced to send back
    file_to_send = result.get("download_path")
    file_name = None
    history = result.get("history") or []
    if not file_to_send and history:
        for item in reversed(history):
            action = item.get("action", "")
            action_result = item.get("result")
            if action in FILE_GENERATING_ACTIONS and isinstance(action_result, dict):
                p = action_result.get("path") or action_result.get("export_path")
                if p and Path(p).exists():
                    file_to_send = str(p)
                    file_name = action_result.get("name") or action_result.get("export_name")
                    break

    if file_to_send and Path(file_to_send).exists():
        await send_file_to_chat(
            update,
            file_to_send,
            caption=f"📎 {file_name or Path(file_to_send).name}",
        )

    # Update conversation memory
    memory.append({"user": prompt, "bot": response_text[:3500]})
    if len(memory) > MAX_MEMORY:
        memory = memory[-MAX_MEMORY:]
    context.user_data["memory"] = memory


async def send_file_to_chat(update: Update, file_path: str, caption: str = None) -> None:
    """Send a file from disk back into the Telegram chat."""
    path = Path(file_path)
    if not path.exists():
        return
    # Telegram bot API limit: 50MB for sending files
    if path.stat().st_size > 50 * 1024 * 1024:
        await safe_send(update, "⚠️ File is too large to send via Telegram (>50MB). It's saved in your workspace.")
        return
    try:
        with open(path, "rb") as f:
            doc = InputFile(f, filename=path.name)
            caption_str = caption or f"📎 {path.name}"
            kwargs = {"read_timeout": 120.0, "write_timeout": 120.0, "connect_timeout": 30.0}
            if update.message:
                await update.message.reply_document(document=doc, caption=caption_str, **kwargs)
            elif update.callback_query and update.callback_query.message:
                await update.callback_query.message.reply_document(document=doc, caption=caption_str, **kwargs)
            elif update.effective_chat:
                await update.effective_chat.send_document(document=doc, caption=caption_str, **kwargs)
    except Exception as e:
        logger.error("Failed to send file %s: %s", file_path, e)
        await safe_send(update, f"⚠️ Could not send file '{path.name}' directly ({e}). The file is safely saved in your workspace.")


# ── /start ───────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if session.is_authenticated and session.user_id:
        session.refresh_from_db()
        name = session.user.first_name or session.user.username
        await safe_reply(update,
            f"👋 Welcome back to *DocPilot*, {name}!\n\n"
            "Just type your request and I'll handle it:\n"
            "• _\"List my files\"_\n"
            "• _\"Compress invoice.pdf\"_\n"
            "• _\"Summarize document 3\"_\n\n"
            "📤 You can also send me documents to upload them!\n\n"
            "Type /help to see all commands."
        )
    else:
        await safe_reply(update,
            "👋 Welcome to *DocPilot* — your AI-powered document manager!\n\n"
            "To get started:\n"
            "👉 /login — link an existing account\n"
            "👉 /register — create a new account\n\n"
            "Need help? Type /help"
        )


# ── /help ────────────────────────────────────────────────────────────────────

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(
        update,
        "📋 *DocPilot Commands*\n\n"
        "/start — Welcome message\n"
        "/login — Link your DocPilot account\n"
        "/register — Create a new DocPilot account\n"
        "/logout — Unlink your account\n"
        "/status — Account info + usage stats\n"
        "/tools — View all tools \u0026 actions\n"
        "/myfiles — Quick list of your files\n"
        "/myfolders — Quick list of your folders\n"
        "/help — Show this message\n"
        "/cancel — Cancel current action\n\n"
        "📤 *File Upload*\n"
        "Send me any document (PDF, Word, etc.) and I'll save it to your workspace.\n\n"
        "🤖 *Natural Language Examples:*\n"
        "• _\"List my files\"_\n"
        "• _\"Search for fee process\"_\n"
        "• _\"Summarize document 30\"_\n"
        "• _\"Summarize file 30 and export as Word\"_\n"
        "• _\"Compress report.pdf\"_\n"
        "• _\"Merge files 3 and 7 into combined.pdf\"_\n"
        "• _\"Split pages 1-3 from document 5\"_\n"
        "• _\"Create a zip archive of files 1, 2\"_\n"
        "• _\"Send me file 30\"_\n\n"
        "💡 Type /tools to see every tool DocPilot supports!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(
        update,
        "🛠️ *DocPilot Complete Tools Catalog*\n\n"
        "🔍 *1. File Discovery & Reading*\n"
        "• `list_files` — _\"List all my files\"_\n"
        "• `search_files` — _\"Find PDFs about fee or invoice\"_\n"
        "• `get_file_info` — _\"Show info for file 3\"_\n"
        "• `read_file_content` — _\"Read content of notes.txt\"_\n"
        "• `send_file` — _\"Send me file 3\"_\n\n"
        "📁 *2. Folder Management*\n"
        "• `list_folders` — _\"List my folders\"_\n"
        "• `list_files_in_folder` — _\"Show files in exams folder\"_\n"
        "• `create_folder` — _\"Create folder Reports\"_\n"
        "• `delete_folder` — _\"Delete folder OldArchive\"_\n\n"
        "⚙️ *3. File Operations & Sharing*\n"
        "• `move_file` — _\"Move file 3 to Archive folder\"_\n"
        "• `copy_file` — _\"Copy document 5 to Backup folder\"_\n"
        "• `rename_file` — _\"Rename file 2 to final_report.pdf\"_\n"
        "• `delete_file` — _\"Delete file 4\"_\n"
        "• `share_file` — _\"Share file 30 with username Cooked2\"_\n\n"
        "📦 *4. ZIP Archiving*\n"
        "• `create_zip` — _\"Create a zip called bundle of files 2, 3\"_\n"
        "• `extract_zip` — _\"Extract bundle.zip into Unzipped folder\"_\n\n"
        "📑 *5. PDF & Office Engineering (iLovePDF)*\n"
        "• `compress_pdf` — _\"Compress document 3\"_\n"
        "• `split_pdf` — _\"Extract pages 1-3 from file 2\"_\n"
        "• `merge_pdfs` — _\"Merge files 1 and 2 into combined.pdf\"_\n"
        "• `convert_pdf_to_word` — _\"Convert file 3 to Word docx\"_\n"
        "• `convert_word_to_pdf` — _\"Convert report.docx to PDF\"_\n\n"
        "🧠 *6. AI Intelligence & OCR (PDF, Word, PPTX, Excel)*\n"
        "• `ask_document` — _\"What is the fee amount in file 30?\"_\n"
        "• `ocr_document` — _\"Run OCR on photo 12 to extract text\"_\n"
        "• `extract_text` — _\"Extract all text/slides/data from document 2\"_\n"
        "• `summarize_document` — _\"Summarize lecture.pptx or report.xlsx\"_\n"
        "• `create_document` — _\"Create a docx about project timeline\"_\n"
        "• `summarize_and_export` — _\"Summarize file 30 and export as Word\"_\n\n"
        "🎓 *7. AI Study & Revision Tools*\n"
        "• `generate_quiz` — _\"Generate a 5-question quiz from presentation.pptx\"_\n"
        "• `generate_flashcards` — _\"Create revision flashcards from document 2\"_\n\n"
        "🔒 *8. Document Security & Analytics*\n"
        "• `protect_pdf` — _\"Lock file 30 with password Secret123\"_\n"
        "• `unlock_pdf` — _\"Unlock file 32 using password Secret123\"_\n"
        "• `watermark_pdf` — _\"Add CONFIDENTIAL watermark to file 30\"_\n"
        "• `find_duplicates` — _\"Scan my workspace for duplicate files\"_\n"
        "• `translate_document` — _\"Translate file 30 to Spanish\"_\n"
        "• `workspace_stats` — _\"Show workspace storage analytics\"_ (/status)\n\n"
        "💡 *Tip:* Send any PDF, Word, PowerPoint (.pptx), Excel (.xlsx), or Photo to get interactive 1-tap action buttons!",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── /status (enhanced with visual workspace analytics) ──────────────────────

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if session.is_authenticated and session.user_id:
        session.refresh_from_db()
        u = session.user

        stats = workspace_stats(user=u)
        file_count = stats.get("file_count", 0)
        folder_count = stats.get("folder_count", 0)
        total_size_fmt = stats.get("total_size_formatted", "0 B")
        pct = stats.get("storage_usage_percent", 0.0)

        # Build visual progress bar for storage (10 blocks)
        filled = int(round(pct / 10.0))
        filled = min(max(filled, 0), 10)
        bar = "█" * filled + "░" * (10 - filled)

        breakdown_lines = []
        fmt_breakdown = stats.get("format_breakdown", {})
        format_icons = {
            "PDF": "📑",
            "Word Document": "📝",
            "PowerPoint": "📽️",
            "Excel Spreadsheet": "📊",
            "Plain Text / Markdown": "📄",
            "Image": "🖼️",
            "ZIP Archive": "📦",
            "Other": "📎",
        }
        for fmt_name, count in fmt_breakdown.items():
            icon = format_icons.get(fmt_name, "📄")
            breakdown_lines.append(f"  • {icon} *{fmt_name}*: {count}")

        breakdown_str = "\n".join(breakdown_lines) if breakdown_lines else "  • _No files yet_"

        recent_lines = []
        for rf in stats.get("recent_files", [])[:3]:
            r_name = rf['name'].replace('_', ' ')
            recent_lines.append(f"  • `{rf['id']}`: *{r_name}* ({rf['size']})")
        recent_str = "\n".join(recent_lines) if recent_lines else "  • _None_"

        msg = (
            f"📊 *DocPilot Workspace Analytics*\n\n"
            f"👤 *Account:* `{u.username}`\n"
            f"📧 *Email:* {u.email or '—'}\n"
            f"🏷️ *Name:* {u.get_full_name() or '—'}\n\n"
            f"📁 *Files:* {file_count}  |  📂 *Folders:* {folder_count}\n"
            f"💾 *Total Storage:* {total_size_fmt}\n"
            f"📈 *Storage Quota (500 MB):*\n"
            f"   `[{bar}]` {pct}%\n\n"
            f"📑 *File Types:*\n{breakdown_str}\n\n"
            f"🕐 *Recent Files:*\n{recent_str}\n\n"
            f"🔒 _Your files are isolated and securely encrypted._"
        )
        await safe_send(update, msg, parse_mode=ParseMode.MARKDOWN)
    else:
        await safe_send(
            update,
            "❌ *Not logged in*\n\nUse /login to link your DocPilot account.\n"
            "Don't have an account? Use /register to create one.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ── /myfiles ─────────────────────────────────────────────────────────────────

async def myfiles_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)
    if not session.is_authenticated or not session.user_id:
        await safe_send(update, "⚠️ You must be logged in. Use /login first.")
        return

    session.refresh_from_db()
    files = list_files(user=session.user)

    if not files:
        await safe_send(update, "📁 You don't have any files yet.\n\nSend me a document to upload it!")
        return

    lines = []
    for f in files[:20]:  # limit to 20
        size = _format_size(f.get("size", 0))
        folder = (f.get("folder") or "—").replace('_', ' ')
        name = f['name'].replace('_', ' ')  # underscores break Markdown
        lines.append(f"📄 *{name}* (ID: {f['id']})\n     📂 {folder} | 💾 {size}")

    extra = f"\n\n_...and {len(files) - 20} more_" if len(files) > 20 else ""
    await safe_send(update, f"📁 *Your Files ({len(files)})*\n\n" + "\n\n".join(lines) + extra)


# ── /myfolders ───────────────────────────────────────────────────────────────

async def myfolders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)
    if not session.is_authenticated or not session.user_id:
        await safe_send(update, "⚠️ You must be logged in. Use /login first.")
        return

    session.refresh_from_db()
    folders = list_folders(user=session.user)

    if not folders:
        await safe_send(update, "📂 You don't have any folders yet.")
        return

    lines = []
    for f in folders:
        folder_name = f['name'].replace('_', ' ')
        lines.append(f"📂 *{folder_name}* — {f['file_count']} file(s), {_format_size(f.get('total_size', 0))}")

    await safe_send(update, f"📂 *Your Folders ({len(folders)})*\n\n" + "\n".join(lines))


# ── /login conversation ──────────────────────────────────────────────────────

async def login_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if session.is_authenticated and session.user_id:
        session.refresh_from_db()
        await safe_send(
            update,
            f"✅ You are already logged in as *{session.user.username}*.\n"
            "Use /logout first if you want to switch accounts.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    context.user_data.clear()
    await safe_send(
        update,
        "🔐 *DocPilot Login*\n\n"
        "Step 1 of 2 — Enter your *username*:\n\n"
        "_(Type /cancel to abort)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return WAITING_USERNAME


async def receive_username(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.message.text.strip()
    if not username:
        await safe_send(update, "⚠️ Username cannot be empty. Please enter your username:")
        return WAITING_USERNAME

    context.user_data["tg_username"] = username
    await safe_send(
        update,
        "🔑 Step 2 of 2 — Enter your *password*:\n\n"
        "_(Your message will be deleted immediately for security)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return WAITING_PASSWORD


async def receive_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    username = context.user_data.get("tg_username", "")
    password = update.message.text.strip()

    # Delete the password message immediately for security
    try:
        await update.message.delete()
    except Exception:
        pass  # bot might not have delete permission

    if not password:
        await safe_send(
            update,
            "⚠️ Password cannot be empty. Please try /login again."
        )
        return ConversationHandler.END

    # Authenticate against Django user database
    django_user = authenticate(username=username, password=password)

    if django_user is None:
        await safe_send(
            update,
            "❌ *Login failed*\n\n"
            "Invalid username or password.\n"
            "Please try /login again or check your credentials.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    if not django_user.is_active:
        await safe_send(
            update,
            "❌ *Account disabled*\n\nThis account has been deactivated. "
            "Please contact your administrator.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    # Save the authenticated session
    session, _ = TelegramUser.objects.get_or_create(chat_id=chat_id)
    session.user = django_user
    session.is_authenticated = True
    session.save()

    # Ensure user media directory exists
    user_media = media_root(user=django_user)
    user_media.mkdir(parents=True, exist_ok=True)

    name = django_user.first_name or django_user.username
    await safe_send(
        update,
        f"✅ *Welcome to DocPilot, {name}!* 🎉\n\n"
        "Your account is now linked. Here's what you can do:\n\n"
        "📁 *File Management*\n"
        "• List, search, move, copy, rename, delete files\n\n"
        "📄 *Documents*\n"
        "• Read, summarize, extract text from files\n"
        "• Create Word or PDF documents\n\n"
        "🔧 *PDF Tools*\n"
        "• Merge, split, compress PDFs\n"
        "• Convert PDF → Word\n\n"
        "📤 *Upload*\n"
        "• Send me any document to save it!\n\n"
        "💡 Just type your request naturally!\n"
        "Try: _\"List my files\"_ or _\"Show me my folders\"_",
        parse_mode=ParseMode.MARKDOWN,
    )
    context.user_data.clear()
    return ConversationHandler.END


async def login_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await safe_send(
        update,
        "🚫 Cancelled. Type /login or /register whenever you're ready."
    )
    return ConversationHandler.END


# ── /register conversation ────────────────────────────────────────────────────

async def register_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if session.is_authenticated and session.user_id:
        session.refresh_from_db()
        await safe_send(
            update,
            f"✅ You are already logged in as *{session.user.username}*.\n"
            "Use /logout first if you want to create a different account.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    context.user_data.clear()
    await safe_send(
        update,
        "📝 *DocPilot Registration*\n\n"
        "Step 1 of 3 — Choose a *username*:\n\n"
        "_(Type /cancel to abort)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return REG_USERNAME


async def receive_reg_username(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = update.message.text.strip()
    if not username:
        await safe_send(update, "⚠️ Username cannot be empty. Try again:")
        return REG_USERNAME

    if len(username) < 3:
        await safe_send(update, "⚠️ Username must be at least 3 characters. Try again:")
        return REG_USERNAME

    # Check if username is already taken
    if User.objects.filter(username=username).exists():
        await safe_send(
            update,
            f"⚠️ Username \"{username}\" is already taken. Please choose another:"
        )
        return REG_USERNAME

    context.user_data["reg_username"] = username
    await safe_send(
        update,
        "🔑 Step 2 of 3 — Choose a *password*:\n\n"
        "_(Your message will be deleted for security)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return REG_PASSWORD


async def receive_reg_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    password = update.message.text.strip()

    # Delete the password message for security
    try:
        await update.message.delete()
    except Exception:
        pass

    if not password:
        await safe_send(update, "⚠️ Password cannot be empty. Try again:")
        return REG_PASSWORD

    if len(password) < 6:
        await safe_send(
            update,
            "⚠️ Password must be at least 6 characters. Try again:"
        )
        return REG_PASSWORD

    context.user_data["reg_password"] = password
    await safe_send(
        update,
        "🔑 Step 3 of 3 — *Confirm your password*:\n\n"
        "_(Your message will be deleted for security)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return REG_CONFIRM


async def receive_reg_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    confirm = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Delete the confirmation message for security
    try:
        await update.message.delete()
    except Exception:
        pass

    password = context.user_data.get("reg_password", "")
    username = context.user_data.get("reg_username", "")

    if confirm != password:
        await safe_send(
            update,
            "❌ Passwords don't match. Please try /register again."
        )
        context.user_data.clear()
        return ConversationHandler.END

    # Create the Django user
    try:
        django_user = User.objects.create_user(username=username, password=password)
    except Exception as e:
        await safe_send(
            update,
            f"❌ Failed to create account: {e}\nPlease try /register again."
        )
        context.user_data.clear()
        return ConversationHandler.END

    # Link TelegramUser
    session, _ = TelegramUser.objects.get_or_create(chat_id=chat_id)
    session.user = django_user
    session.is_authenticated = True
    session.save()

    # Ensure user media directory exists
    user_media = media_root(user=django_user)
    user_media.mkdir(parents=True, exist_ok=True)

    await safe_send(
        update,
        f"✅ *Account created successfully!* 🎉\n\n"
        f"👤 Username: `{username}`\n\n"
        "You're now logged in. Here's what you can do:\n"
        "• 📤 Send me any document to save it\n"
        "• 💬 Ask me to summarize or search your documents\n"
        "• 🛠️ Compress, split, merge, or convert PDFs\n\n"
        "Type /help to see all commands.",
        parse_mode=ParseMode.MARKDOWN,
    )
    context.user_data.clear()
    return ConversationHandler.END


# ── /logout ───────────────────────────────────────────────────────────────────

async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if not session.is_authenticated or not session.user_id:
        await safe_send(
            update,
            "⚠️ You are not logged in. Use /login to link your account."
        )
        return

    session.refresh_from_db()
    username = session.user.username
    session.user = None
    session.is_authenticated = False
    session.save()

    # Clear conversation memory
    context.user_data.pop("memory", None)

    await safe_send(
        update,
        f"👋 *{username}* has been logged out successfully.\n\n"
        "Your Telegram chat is now unlinked from DocPilot.\n"
        "Use /login to reconnect.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Document ingestion handler ───────────────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads from users — save to their workspace and register in DB."""
    session = get_or_create_session(update)

    if not session.is_authenticated or not session.user_id:
        await safe_send(
            update,
            "⚠️ *You must be logged in to upload files.*\n\n"
            "Use /login to link your account first!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    session.refresh_from_db()
    user = session.user
    document = update.message.document

    if not document:
        return

    file_name = document.file_name or "untitled"
    file_size = document.file_size or 0

    # Check file extension
    ext = Path(file_name).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        await safe_send(
            update,
            f"🚫 File type `{ext}` is not allowed for security reasons.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Check file size (20MB Telegram limit)
    if file_size > 20 * 1024 * 1024:
        await safe_send(update, "⚠️ File is too large. Maximum size is 20MB.")
        return

    # Show progress
    progress_msg = None
    if update.message:
        try:
            progress_msg = await update.message.reply_text(f"📥 Downloading *{file_name}*...", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

    try:
        # Get or create the uploads folder for this user safely
        uploads_folder = Folder.objects.filter(name="uploads", parent=None, user=user).first()
        if not uploads_folder:
            uploads_folder = Folder.objects.create(name="uploads", parent=None, user=user)

        # Prepare user directory
        user_root = media_root(user=user)
        uploads_dir = user_root / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate filename
        dest_path = uploads_dir / file_name
        counter = 1
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        while dest_path.exists():
            file_name = f"{stem}_{counter}{suffix}"
            dest_path = uploads_dir / file_name
            counter += 1

        # Download from Telegram (with generous timeouts for larger files)
        tg_file = await document.get_file(read_timeout=120.0, write_timeout=120.0, connect_timeout=30.0)
        await tg_file.download_to_drive(str(dest_path), read_timeout=120.0, write_timeout=120.0, connect_timeout=30.0)

        # Register in database
        rel_path = relative_media_path(dest_path)
        db_file = UploadedFile.objects.create(
            user=user,
            original_name=file_name,
            folder=uploads_folder,
            file=rel_path,
        )

        actual_size = dest_path.stat().st_size

        # Delete progress message
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass

        safe_display_name = file_name.replace("_", " ")
        markup = build_file_action_keyboard(db_file.id, file_name)
        await safe_send(
            update,
            f"✅ *File uploaded successfully!*\n\n"
            f"📄 Name: *{safe_display_name}* (`{file_name}`)\n"
            f"🆔 ID: {db_file.id}\n"
            f"📂 Folder: uploads\n"
            f"💾 Size: {_format_size(actual_size)}\n\n"
            f"👇 *Tap an action below to start immediately:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=markup,
        )

    except Exception as e:
        logger.error("Document upload failed for %s: %s", user.username, e)
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        await safe_send(update, f"❌ Upload failed: {e}")


# ── Photo upload handler ─────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads — save the highest resolution version."""
    session = get_or_create_session(update)

    if not session.is_authenticated or not session.user_id:
        await safe_send(
            update,
            "⚠️ *You must be logged in to upload photos.*\n\nUse /login first!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    session.refresh_from_db()
    user = session.user

    # Get highest resolution photo
    photo = update.message.photo[-1]  # last = highest res

    progress_msg = None
    if update.message:
        try:
            progress_msg = await update.message.reply_text("📥 Saving photo...")
        except Exception:
            pass

    try:
        uploads_folder = Folder.objects.filter(name="uploads", parent=None, user=user).first()
        if not uploads_folder:
            uploads_folder = Folder.objects.create(name="uploads", parent=None, user=user)

        user_root = media_root(user=user)
        uploads_dir = user_root / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        file_name = f"photo_{photo.file_unique_id}.jpg"
        dest_path = uploads_dir / file_name

        counter = 1
        stem = f"photo_{photo.file_unique_id}"
        while dest_path.exists():
            file_name = f"{stem}_{counter}.jpg"
            dest_path = uploads_dir / file_name
            counter += 1

        # Download (with generous timeouts)
        tg_file = await photo.get_file(read_timeout=120.0, write_timeout=120.0, connect_timeout=30.0)
        await tg_file.download_to_drive(str(dest_path), read_timeout=120.0, write_timeout=120.0, connect_timeout=30.0)

        # Register in DB
        rel_path = relative_media_path(dest_path)
        db_file = UploadedFile.objects.create(
            user=user,
            original_name=file_name,
            folder=uploads_folder,
            file=rel_path,
        )

        actual_size = dest_path.stat().st_size

        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass

        # Attempt automatic OCR on uploaded photo
        ocr_text = ""
        try:
            from files.services.file_tools import ocr_document
            ocr_res = ocr_document(db_file.id, user=user)
            ocr_text = ocr_res.get("text", "").strip()
        except Exception as ocr_err:
            logger.debug("Automatic photo OCR skipped: %s", ocr_err)

        ocr_snippet = ""
        if ocr_text:
            preview = ocr_text[:250].replace("_", " ")
            ocr_snippet = f"\n\n🔍 *Detected Text (OCR):*\n_{preview}..._"

        markup = build_file_action_keyboard(db_file.id, file_name)
        await safe_send(
            update,
            f"✅ *Photo saved!*\n\n"
            f"📷 Name: `{file_name}`\n"
            f"🆔 ID: {db_file.id}\n"
            f"📂 Folder: uploads\n"
            f"💾 Size: {_format_size(actual_size)}"
            f"{ocr_snippet}\n\n"
            f"👇 *Tap an action below to start immediately:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=markup,
        )

    except Exception as e:
        logger.error("Photo upload failed for %s: %s", user.username, e)
        if progress_msg:
            try:
                await progress_msg.delete()
            except Exception:
                pass
        await safe_send(update, f"❌ Photo upload failed: {e}")


# ── Message handler (agent + file send-back + memory) ─────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_or_create_session(update)

    if not session.is_authenticated or not session.user_id:
        await safe_send(
            update,
            "⚠️ *You must be logged in to use DocPilot.*\n\n"
            "Use /login to link your account — it only takes a few seconds!",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    session.refresh_from_db()
    user_text = update.message.text.strip() if update.message and update.message.text else ""

    if not user_text:
        return

    # Get conversation memory
    memory = context.user_data.get("memory", [])

    # Show a typing indicator + processing message (safely guarded against network timeouts)
    try:
        await update.effective_chat.send_action("typing")
    except Exception:
        pass

    thinking_msg = None
    if update.message:
        try:
            thinking_msg = await update.message.reply_text("⏳ Working on it...")
        except Exception:
            pass

    result = await run_agent_async(session.user, user_text, memory=memory)
    response_text = result.get("final_response") or "✅ Operation completed."

    # Delete the "working on it" message
    if thinking_msg:
        try:
            await thinking_msg.delete()
        except Exception:
            pass

    await safe_send(update, response_text)

    # Check if a file was generated or requested to be sent back
    file_to_send = result.get("download_path")
    file_name = None

    history = result.get("history") or []
    if not file_to_send and history:
        for item in reversed(history):
            action = item.get("action", "")
            action_result = item.get("result")
            if action in FILE_GENERATING_ACTIONS and isinstance(action_result, dict):
                p = action_result.get("path") or action_result.get("export_path")
                if p and Path(p).exists():
                    file_to_send = str(p)
                    file_name = action_result.get("name") or action_result.get("export_name")
                    break

    if file_to_send and Path(file_to_send).exists():
        await send_file_to_chat(
            update,
            file_to_send,
            caption=f"📎 {file_name or Path(file_to_send).name}",
        )

    # Update conversation memory
    memory.append({"user": user_text, "bot": response_text[:3500]})
    if len(memory) > MAX_MEMORY:
        memory = memory[-MAX_MEMORY:]
    context.user_data["memory"] = memory


# ── Cancel command handler ───────────────────────────────────────────────────

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await safe_send(
        update,
        "🚫 No active operation was running, or current action was cancelled.\n"
        "Type /help to see all available commands."
    )


# ── Unknown command handler ───────────────────────────────────────────────────

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_send(
        update,
        "❓ Unknown command. Type /help to see what I can do."
    )


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates and prevent bot unhandled crashes."""
    err = context.error
    if isinstance(err, (TimedOut, NetworkError)):
        logger.warning("Telegram network timeout or connection hiccup: %s", err)
    else:
        logger.error("Unhandled error processing update: %s", err, exc_info=err)

    if isinstance(update, Update) and update.effective_chat:
        try:
            if isinstance(err, (TimedOut, NetworkError)):
                await update.effective_chat.send_message(
                    "⚠️ Telegram connection timed out momentarily. Please try your request again."
                )
        except Exception:
            pass


# ── Bot factory ───────────────────────────────────────────────────────────────

def build_app() -> Application:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set or is empty.\n"
            "Please add your Telegram bot token to your .env file:\n"
            "TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz\n"
            "You can obtain a token from @BotFather on Telegram."
        )

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .pool_timeout(30.0)
        .get_updates_connect_timeout(30.0)
        .get_updates_read_timeout(60.0)
        .get_updates_write_timeout(60.0)
        .get_updates_pool_timeout(30.0)
        .build()
    )

    app.add_error_handler(global_error_handler)

    # Login conversation handler
    login_conv = ConversationHandler(
        entry_points=[CommandHandler("login", login_start)],
        states={
            WAITING_USERNAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_username)
            ],
            WAITING_PASSWORD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_password)
            ],
        },
        fallbacks=[CommandHandler("cancel", login_cancel)],
        allow_reentry=True,
        conversation_timeout=300,  # 5 min — auto-clear stale sessions
    )

    # Register conversation handler
    register_conv = ConversationHandler(
        entry_points=[CommandHandler("register", register_start)],
        states={
            REG_USERNAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_reg_username)
            ],
            REG_PASSWORD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_reg_password)
            ],
            REG_CONFIRM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_reg_confirm)
            ],
        },
        fallbacks=[CommandHandler("cancel", login_cancel)],
        allow_reentry=True,
        conversation_timeout=300,  # 5 min — auto-clear passwords from context
    )

    # Register handlers (order matters!)
    # ConversationHandlers MUST be registered before standalone /cancel
    # so their internal fallbacks take priority during active conversations.
    app.add_handler(login_conv)
    app.add_handler(register_conv)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tools", tools_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("myfiles", myfiles_command))
    app.add_handler(CommandHandler("myfolders", myfolders_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    return app


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    print("🤖 Starting DocPilot Telegram bot...")
    try:
        application = build_app()
        print("✅ Bot is running. Press Ctrl+C to stop.")
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped gracefully.")
    except Exception as exc:
        print(f"\n❌ Error starting bot: {exc}")
