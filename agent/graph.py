from pathlib import Path
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .llm import ask_agent
from .tools import execute_tool


MAX_STEPS = 5


def agent_node(state: AgentState):
    steps = state.get("steps", 0) + 1
    user_request = state.get("user_request", "")

    try:
        decision = ask_agent(
            user_request,
            state.get("tool_result"),
            state.get("history", []),
        )
    except Exception as error:
        print("Planner error:", error)
        return {
            "action": "finish",
            "parameters": {},
            "steps": steps,
            "tool_result": {
                "error": str(error),
            },
        }

    print("\n--- AGENT DECISION ---")
    print("Step:", steps)
    print("Action:", decision["action"])
    print("Parameters:", decision["parameters"])

    return {
        "action": decision["action"],
        "parameters": decision["parameters"],
        "steps": steps,
    }


def tool_node(state: AgentState):
    action = state.get("action", "finish")
    parameters = state.get("parameters", {})

    print("\n--- TOOL EXECUTION ---")
    print("Action:", action)
    print("Parameters:", parameters)

    # finish is not an executable tool
    if action == "finish":
        return {
            "tool_result": state.get("tool_result"),
        }

    # Protect against invalid planner output
    if action in ("none", "null", ""):
        error_result = {
            "error": "The planner returned an invalid action."
        }
        history = list(state.get("history", []))
        history.append({
            "action": action,
            "parameters": parameters,
            "result": error_result,
        })
        return {
            "tool_result": error_result,
            "history": history,
        }

    history = list(state.get("history", []))

    user = state.get("user")

    try:
        result = execute_tool(action, parameters, user=user)
        print("Tool result:", result)

        history.append({
            "action": action,
            "parameters": parameters,
            "result": result,
        })

        return {
            "tool_result": result,
            "history": history,
        }

    except Exception as error:
        error_result = {
            "error": str(error)
        }
        print("Tool error:", error_result)

        history.append({
            "action": action,
            "parameters": parameters,
            "result": error_result,
        })

        return {
            "tool_result": error_result,
            "history": history,
        }


def should_continue(state: AgentState) -> str:
    if state.get("action") == "finish":
        return "response"

    if state.get("steps", 0) >= MAX_STEPS:
        return "response"

    return "agent"


FILE_GENERATING_ACTIONS = {
    "compress_pdf", "split_pdf", "merge_pdfs", "convert_pdf_to_word",
    "convert_word_to_pdf", "create_document", "summarize_and_export",
    "send_file", "create_zip", "protect_pdf", "unlock_pdf", "watermark_pdf",
}


def response_node(state: AgentState):
    """
    Generate the final user-facing response directly
    from the trusted tool result.
    """
    result = state.get("tool_result")
    history = state.get("history") or []

    # Determine if a file was created/processed that should be sent back
    download_path = None
    if history:
        for item in reversed(history):
            action = item.get("action", "")
            action_result = item.get("result")
            if action in FILE_GENERATING_ACTIONS and isinstance(action_result, dict):
                path = action_result.get("path") or action_result.get("export_path")
                if path and Path(path).exists():
                    download_path = str(path)
                    break

    res = _build_response(state, result, history)
    if download_path:
        res["download_path"] = download_path
    return res


def _build_response(state: AgentState, result, history) -> dict:
    from pathlib import Path  # keep for local use in nested helper functions

    # Handle case where no tool was executed or initial empty result
    if result is None or result == "":
        if not history:
            user = state.get("user")
            req = (state.get("user_request") or "").lower()
            uname = getattr(user, "first_name", None) or getattr(user, "username", "there")

            if any(w in req for w in ["who am i", "my name", "my account", "whoami", "logged in as"]):
                if user and getattr(user, "username", None):
                    email_str = f" ({user.email})" if getattr(user, "email", None) else ""
                    return {
                        "final_response": f"👤 You are logged in as **{uname}** (Username: `{user.username}`){email_str}."
                    }
                return {
                    "final_response": "You are currently not logged in. Use /login or /register to link your account."
                }

            if any(w in req for w in ["thank", "thx", "appreciate", "awesome", "great job", "nice"]):
                return {
                    "final_response": f"You're very welcome, {uname}! 😊 Let me know if there's anything else you'd like to do with your files."
                }

            if any(w in req for w in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"]):
                return {
                    "final_response": (
                        f"👋 Hi {uname}! I'm DocPilot, your AI document assistant.\n\n"
                        "How can I help you today? You can send me files, ask questions, summarize PDFs, convert formats, and more!\n\n"
                        "💡 Type `/tools` anytime to see all available capabilities."
                    )
                }

            if any(w in req for w in ["bye", "goodbye", "see you", "cya"]):
                return {
                    "final_response": f"Goodbye {uname}! 👋 Your files and folders are safely stored in your workspace. Have a wonderful day!"
                }

            if any(w in req for w in ["what can you do", "help", "features", "tools", "capabilities"]):
                return {
                    "final_response": (
                        "🤖 **DocPilot Capabilities:**\n\n"
                        "• **Files**: Search, list, inspect metadata, read text, send/download, rename, move, copy, delete\n"
                        "• **Folders**: Create, list, delete folders, view files inside folders\n"
                        "• **Archives**: Create ZIP archives, extract ZIPs into folders\n"
                        "• **PDF & Office**: Compress, split pages, merge multiple PDFs, convert PDF to Word, convert Word to PDF\n"
                        "• **AI Document Intelligence**: Ask pinpoint questions, extract text, native OCR on photos, summarize documents, create Word/PDF files\n"
                        "• **Sharing**: Share files directly with other DocPilot users\n\n"
                        "💡 Tip: Type `/tools` to see all available tools and commands!"
                    )
                }
            return {
                "final_response": (
                    f"Hi {uname}! I'm DocPilot. How can I help you manage or analyze your files today?"
                )
            }
        result = history[-1].get("result")

    # Helper icon and size formatters for human-friendly output
    def _get_icon(fname):
        ext = Path(fname).suffix.lower()
        if ext == ".pdf": return "📑"
        if ext in (".docx", ".doc"): return "📝"
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"): return "🖼️"
        if ext in (".zip", ".rar", ".7z", ".tar"): return "📦"
        if ext in (".txt", ".md", ".csv", ".json", ".py", ".sql"): return "📄"
        return "📄"

    def _fmt_size(s):
        if not s: return "0 B"
        if s < 1024: return f"{s} B"
        if s < 1024 * 1024: return f"{s/1024:.1f} KB"
        return f"{s/(1024*1024):.2f} MB"

    # List of items returned (files or folders)
    if isinstance(result, list):
        last_action = history[-1]["action"] if history else state.get("action")

        if not result:
            if last_action == "list_files":
                return {"final_response": "📂 Your workspace is currently empty. Send me any document or photo to get started!"}
            elif last_action == "list_folders":
                return {"final_response": "📁 You don't have any custom folders yet. Say \"Create folder Projects\" to create one!"}
            elif last_action == "list_files_in_folder":
                return {"final_response": "📂 This folder is currently empty."}
            return {"final_response": "🔍 I couldn't find any matching files or folders. Try searching with a broader keyword!"}

        # Check if list of folders
        if result and "file_count" in result[0]:
            lines = [f"📂 **Your Folders ({len(result)} total):**\n"]
            for f in result:
                lines.append(f"• 📁 **{f['name']}** — {f['file_count']} file(s)")
            lines.append("\n💡 _Tip: Say \"Show files in {folder}\" to view contents._")
            return {"final_response": "\n".join(lines)}

        # Single file
        if len(result) == 1:
            f = result[0]
            fid = f.get("id", "?")
            fname = f.get("name", "a file")
            clean_name = fname.replace("_", " ")
            folder = f.get("folder")
            size_str = f" • {_fmt_size(f.get('size', 0))}" if f.get("size") else ""
            folder_str = f" in folder *{folder}*" if folder else ""
            icon = _get_icon(fname)
            return {
                "final_response": (
                    f"{icon} Found *{clean_name}* (ID: `{fid}`){folder_str}{size_str}.\n\n"
                    f"💡 *What would you like to do with it?*\n"
                    f"• _\"Summarize {clean_name}\"_\n"
                    f"• _\"Quiz me on this document\"_\n"
                    f"• _\"Send me {clean_name}\"_"
                )
            }

        # Multiple files
        lines = [f"📁 *Here are your files ({len(result)} total):*\n"]
        for f in result[:10]:
            fid = f.get("id", "?")
            fname = f.get("name", "unnamed")
            clean_name = fname.replace("_", " ")
            icon = _get_icon(fname)
            folder = f.get("folder")
            folder_info = f" • in `{folder}`" if folder else ""
            size_info = f" • {_fmt_size(f.get('size', 0))}" if f.get("size") else ""
            lines.append(f"• {icon} *{clean_name}* (ID: `{fid}`){folder_info}{size_info}")

        if len(result) > 10:
            lines.append(f"\n_...and {len(result) - 10} more files._")

        first_sample = result[0].get("name", "").replace("_", " ") if result else "document"
        lines.append(f"\n💡 _Tip: You can refer to files by name (e.g. \"Summarize {first_sample}\"), by ID, or just say \"Summarize it\"!_")
        return {"final_response": "\n".join(lines)}

    if not isinstance(result, dict):
        return {
            "final_response": "✅ The requested operation was completed successfully."
        }

    # Error handling
    if "error" in result:
        return {
            "final_response": (
                f"⚠️ **I couldn't complete that operation:**\n{result['error']}\n\n"
                f"💡 *Tip:* Try rephrasing your request, or type `/tools` to see example commands."
            )
        }

    # File info
    if "size_formatted" in result and "extension" in result:
        folder_info = f' in "{result["folder"]}"' if result.get("folder") else ""
        return {
            "final_response": (
                f'File "{result["name"]}"{folder_info}: '
                f'Size: {result["size_formatted"]}, Type: {result["extension"]}, '
                f'Uploaded: {result.get("uploaded_at") or "Unknown"}.'
            )
        }

    # Read file content
    if "content" in result and result.get("message") == "File content read successfully":
        trunc_note = " (truncated)" if result.get("truncated") else ""
        return {
            "final_response": (
                f'Content of "{result["name"]}"{trunc_note}:\n\n{result["content"]}'
            )
        }

    # Move file
    if result.get("message") == "File moved successfully":
        return {
            "final_response": (
                f'"{result.get("name")}" was successfully moved '
                f'to the "{result.get("folder")}" folder.'
            )
        }

    # File already in destination
    if result.get("message") == "File is already in the destination folder.":
        return {
            "final_response": (
                f'"{result.get("name")}" is already in '
                f'the "{result.get("folder")}" folder.'
            )
        }

    # Copy file
    if result.get("message") == "File copied successfully":
        return {
            "final_response": (
                f'"{result.get("name")}" was successfully copied '
                f'to the "{result.get("folder")}" folder.'
            )
        }

    # Rename file
    if result.get("message") == "File renamed successfully":
        return {
            "final_response": f'File renamed successfully to "{result.get("name")}".'
        }

    # Delete file
    if result.get("message") == "File deleted successfully":
        return {
            "final_response": f'File "{result.get("name")}" was deleted successfully.'
        }

    # Create folder
    if result.get("message") == "Folder created successfully":
        return {
            "final_response": (
                f'The folder "{result.get("folder_name")}" was created successfully.'
            )
        }

    # Delete folder
    if result.get("message") == "Folder deleted successfully":
        return {
            "final_response": (
                f'The folder "{result.get("folder_name")}" was deleted successfully.'
            )
        }

    # Create ZIP
    if result.get("message") == "ZIP created successfully":
        return {
            "final_response": (
                f'ZIP file "{result.get("zip_name")}" was created successfully.'
            )
        }

    # Extract ZIP
    if result.get("message") == "ZIP extracted successfully":
        return {
            "final_response": (
                f'Extracted {result.get("count")} file(s) from "{result.get("zip_name")}" '
                f'into folder "{result.get("destination_folder")}".'
            )
        }

    # Document summary
    if "summary" in result and result.get("message") == "Document summarized successfully":
        clean_title = (result.get("name") or "document").replace("_", " ")
        return {
            "final_response": (
                f'📑 *Summary of "{clean_title}"*\n\n'
                f'{result["summary"]}'
            )
        }

    # Document summarize and export
    if "export_name" in result and "exported" in result.get("message", "").lower():
        clean_src = (result.get("source_file") or "document").replace("_", " ")
        clean_export = (result.get("export_name") or "document").replace("_", " ")
        return {
            "final_response": (
                f'📑 *Summary of "{clean_src}"*\n\n'
                f'{result["summary"]}\n\n'
                f'📎 *Exported as:* `{clean_export}` ({result["export_format"].upper()}) '
                f'in folder *{result["export_folder"]}*.'
            )
        }

    # Document text extraction
    if "text" in result and result.get("message") == "Text extracted successfully":
        trunc = " (preview truncated)" if result.get("truncated") else ""
        return {
            "final_response": (
                f'Extracted text from "{result["name"]}" ({result.get("pages_read", 1)} pages, {result["total_characters"]} chars){trunc}:\n\n'
                f'{result["text"]}'
            )
        }

    # Document creation
    if "format" in result and "created successfully" in result.get("message", "").lower():
        return {
            "final_response": (
                f'Created {result["format"].upper()} document "{result["name"]}" '
                f'in folder "{result["folder"]}" (ID: {result["id"]}).'
            )
        }

    # Merge PDFs
    if result.get("message") == "PDF files merged successfully":
        return {
            "final_response": (
                f'Successfully merged {result.get("merged_count", 0)} PDFs '
                f'into "{result.get("name")}" (ID: {result.get("id")}) in folder "{result.get("folder")}".'
            )
        }

    # Split PDF
    if result.get("message") == "PDF split successfully":
        pages = result.get("extracted_pages", [])
        return {
            "final_response": (
                f'Successfully split PDF into "{result.get("name")}" (ID: {result.get("id")}) '
                f'containing page(s) {pages} in folder "{result.get("folder")}".'
            )
        }

    # Compress PDF
    if result.get("message") == "PDF compressed successfully":
        engine_str = f" via {result.get('engine')}" if result.get("engine") else ""
        return {
            "final_response": (
                f'Successfully compressed "{result.get("name")}"{engine_str}. '
                f'Size reduced from {result.get("original_size")} to {result.get("compressed_size")} '
                f'({result.get("reduction_percent")} saved).'
            )
        }

    # Convert PDF to Word
    if result.get("message") == "PDF converted to Word successfully":
        engine_str = f" via {result.get('engine')}" if result.get("engine") else ""
        return {
            "final_response": (
                f'Successfully converted PDF to Word document "{result.get("name")}" '
                f'(ID: {result.get("id")}){engine_str} in folder "{result.get("folder")}".'
            )
        }

    # Convert Word to PDF
    if result.get("message") == "Word document converted to PDF successfully":
        engine_str = f" via {result.get('engine')}" if result.get("engine") else ""
        return {
            "final_response": (
                f'Successfully converted Word document to PDF "{result.get("name")}" '
                f'(ID: {result.get("id")}){engine_str} in folder "{result.get("folder")}".'
            )
        }

    # Ask Document (Document Q&A)
    if result.get("message") == "Question answered successfully":
        return {
            "final_response": (
                f'### Answer for "{result.get("name")}"\n\n'
                f'**Question:** {result.get("question")}\n\n'
                f'**Answer:**\n{result.get("answer")}'
            )
        }

    # OCR Document
    if result.get("message") == "OCR text extracted successfully":
        return {
            "final_response": (
                f'### OCR Text from "{result.get("name")}" ({result.get("total_characters")} chars)\n\n'
                f'{result.get("text")}'
            )
        }

    # Share File
    if "shared successfully with" in result.get("message", "").lower():
        return {
            "final_response": (
                f'Successfully shared "{result.get("file_name")}" with {result.get("recipient")} '
                f'(saved in their "{result.get("folder")}" folder as "{result.get("shared_as")}").'
            )
        }

    # Generate Quiz
    if "quiz" in result and result.get("message", "").startswith("Generated"):
        clean_name = (result.get("name") or "document").replace("_", " ")
        return {
            "final_response": (
                f'📝 *Practice Quiz for "{clean_name}"* ({result.get("num_questions")} Questions)\n\n'
                f'{result.get("quiz")}'
            )
        }

    # Generate Flashcards
    if "flashcards" in result and "revision flashcards" in result.get("message", "").lower():
        clean_name = (result.get("name") or "document").replace("_", " ")
        return {
            "final_response": (
                f'📇 *Revision Flashcards for "{clean_name}"*\n\n'
                f'{result.get("flashcards")}'
            )
        }

    # Protect PDF
    if result.get("message") == "PDF password protected successfully":
        clean_name = (result.get("name") or "PDF").replace("_", " ")
        return {
            "final_response": (
                f'🔒 *PDF successfully encrypted with password protection!*\n\n'
                f'📄 File: *{clean_name}* (ID: `{result.get("id")}`)\n'
                f'📂 Folder: {result.get("folder")}\n\n'
                f'⚠️ Remember your password — you will need it to open or unlock this PDF.'
            )
        }

    # Unlock PDF
    if result.get("message") == "PDF unlocked successfully":
        clean_name = (result.get("name") or "PDF").replace("_", " ")
        return {
            "final_response": (
                f'🔓 *PDF successfully decrypted and unlocked!*\n\n'
                f'📄 Unlocked File: *{clean_name}* (ID: `{result.get("id")}`)\n'
                f'📂 Folder: {result.get("folder")}'
            )
        }

    # Watermark PDF
    if "watermarked successfully" in result.get("message", "").lower():
        clean_name = (result.get("name") or "PDF").replace("_", " ")
        return {
            "final_response": (
                f'🖋️ *Watermark applied successfully!*\n\n'
                f'📄 File: *{clean_name}* (ID: `{result.get("id")}`)\n'
                f'🏷️ Stamp: *"{result.get("watermark")}"*\n'
                f'📂 Folder: {result.get("folder")}'
            )
        }

    # Find Duplicates
    if "duplicate scan completed" in result.get("message", "").lower():
        groups = result.get("duplicate_groups", [])
        if not groups:
            return {
                "final_response": "✨ *Duplicate Scan Complete!*\n\nNo duplicate files were found in your workspace. Everything is clean and organized!"
            }
        lines = [f"🔍 *Found {len(groups)} Duplicate File Group(s)* (Wasting {result.get('wasted_space')}):\n"]
        for idx, g in enumerate(groups, 1):
            orig = g["original"]
            c_name = orig['name'].replace('_', ' ')
            lines.append(f"*Group {idx}:* `{c_name}` (ID: `{orig['id']}`)")
            for c in g.get("copies", []):
                dup_name = c['name'].replace('_', ' ')
                lines.append(f"  • Duplicate: `{dup_name}` (ID: `{c['id']}`) in `{c['folder']}`")
        lines.append("\n💡 _Tip: Say 'Delete file [ID]' to remove unnecessary copies._")
        return {"final_response": "\n".join(lines)}

    # Translate Document
    if "translated successfully" in result.get("message", "").lower():
        clean_name = (result.get("name") or "document").replace("_", " ")
        return {
            "final_response": (
                f'🌐 *Translation ({result.get("language")}) — "{clean_name}":*\n\n'
                f'{result.get("translation")}'
            )
        }

    # Workspace Stats
    if result.get("message") == "Workspace stats computed successfully":
        types = result.get("type_breakdown", {})
        return {
            "final_response": (
                f"📊 *Workspace Analytics for {result.get('username')}*\n\n"
                f"• *Total Files:* {result.get('total_files')} files\n"
                f"• *Total Folders:* {result.get('total_folders')} folders\n"
                f"• *Storage Used:* {result.get('total_size')}\n\n"
                f"📈 *File Type Breakdown:*\n"
                f"• 📑 PDFs ({types.get('PDF', 0)}): {result.get('pdf_bar')}\n"
                f"• 📝 Word ({types.get('Word', 0)}): {result.get('word_bar')}\n"
                f"• 📽️ PowerPoint ({types.get('PowerPoint', 0)}): {result.get('pptx_bar')}\n"
                f"• 📊 Excel ({types.get('Excel', 0)}): {result.get('excel_bar')}\n"
                f"• 🖼️ Photos/OCR ({types.get('Image', 0)}): {result.get('image_bar')}\n"
                f"• 📦 Archives ({types.get('Archive', 0)}): {result.get('archive_bar')}\n"
            )
        }

    # Send file
    if result.get("message") == "File ready to send":
        file_name = result.get("name") or "the file"
        return {
            "final_response": f'Sending "{file_name}" to you...'
        }

    msg = result.get("message")
    if msg:
        return {"final_response": msg}

    return {
        "final_response": "The requested operation was completed successfully."
    }


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tool", tool_node)
graph_builder.add_node("response", response_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", "tool")

graph_builder.add_conditional_edges(
    "tool",
    should_continue,
    {
        "agent": "agent",
        "response": "response",
    },
)

graph_builder.add_edge("response", END)

graph = graph_builder.compile()