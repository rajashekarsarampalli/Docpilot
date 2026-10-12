import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# Environment / Client
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL = "qwen/qwen3.8-27b"


def _safe_text(value) -> str:
    """Safely encode text for Windows terminal output."""
    text = "" if value is None else str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


# --------------------------------------------------
# Clean model response
# --------------------------------------------------

def clean_response(content: str) -> str:
    """
    Remove reasoning blocks (<think>...</think>) and markdown code fences.
    """
    if not content:
        return ""

    # Remove <think>...</think>
    if "<think>" in content:
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        else:
            after_think = content.split("<think>", 1)[1]
            brace = after_think.find("{")
            content = after_think[brace:] if brace != -1 else ""

    content = content.strip()

    # Extract content inside markdown code block if present
    if "```" in content:
        start_fence = content.find("```")
        end_fence = content.rfind("```")
        if start_fence != -1 and end_fence > start_fence:
            inner = content[start_fence + 3:end_fence].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            content = inner

    return content.strip()


# --------------------------------------------------
# Planner
# --------------------------------------------------

def ask_agent(
    user_request: str,
    tool_result=None,
    history=None,
) -> dict:
    """
    Ask Qwen to select exactly one next action for DocPilot.
    """
    if not client:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )

    history = history or []

    history_text = "\n".join(
        [
            (
                f"Action: {item.get('action')}\n"
                f"Parameters: {item.get('parameters')}\n"
                f"Result: {item.get('result')}\n"
            )
            for item in history
        ]
    )

    prompt = f"""Select the next action for DocPilot.

USER:
{user_request}

HISTORY:
{history_text if history_text else "None"}

LATEST RESULT:
{tool_result if tool_result is not None else "None"}

AVAILABLE ACTIONS AND PARAMETERS:
list_files: {{}}
search_files: {{"query": "text"}}
list_folders: {{}}
list_files_in_folder: {{"folder_name": "FolderName"}}
get_file_info: {{"file_id": 1}}
send_file: {{"file_id": 1}}
read_file_content: {{"file_id": 1}}
rename_file: {{"file_id": 1, "new_name": "new.pdf"}}
create_zip: {{"file_ids": [1], "zip_name": "archive"}}
extract_zip: {{"zip_name": "archive.zip", "destination_folder": "FolderName"}}
move_file: {{"file_id": 1, "destination_folder": "FolderName"}}
copy_file: {{"file_id": 1, "destination_folder": "FolderName"}}
create_folder: {{"folder_name": "FolderName"}}
delete_file: {{"file_id": 1}}
delete_folder: {{"folder_name": "FolderName", "recursive": false}}
extract_text: {{"file_id": 1}}
summarize_document: {{"file_id": 1, "focus": "topic"}}
create_document: {{"title": "DocName", "content": "text", "format": "docx"}}
summarize_and_export: {{"file_id": 1, "export_format": "docx"}}
merge_pdfs: {{"file_ids": [1, 2], "output_name": "merged.pdf"}}
split_pdf: {{"file_id": 1, "page_ranges": "1-3"}}
compress_pdf: {{"file_id": 1}}
convert_pdf_to_word: {{"file_id": 1}}
ask_document: {{"file_id": 1, "question": "What is the fee?"}}
ocr_document: {{"file_id": 1}}
convert_word_to_pdf: {{"file_id": 1}}
share_file: {{"file_id": 1, "target_username": "username"}}
generate_quiz: {{"file_id": 1, "num_questions": 5}}
generate_flashcards: {{"file_id": 1, "count": 6}}
protect_pdf: {{"file_id": 1, "password": "password"}}
unlock_pdf: {{"file_id": 1, "password": "password"}}
watermark_pdf: {{"file_id": 1, "watermark_text": "CONFIDENTIAL"}}
find_duplicates: {{}}
translate_document: {{"file_id": 1, "target_language": "Telugu"}}
workspace_stats: {{}}
finish: {{}}

RULES:
- For any action with "file_id", you can pass:
  * A numeric ID (e.g. 30)
  * A file name or keyword (e.g. "MRCET", "invoice.pdf", "fee process")
  * "last" if the user refers to "it", "this", "my document", or their most recent file
  You do NOT need to search first if a name, keyword, or "last" is mentioned.
- Document intelligence tools (ask_document, summarize_document, generate_quiz, generate_flashcards, extract_text, translate_document) support PDF, Word (.docx), PowerPoint slides (.pptx), Excel sheets (.xlsx), and text files.
- To ask specific questions about a document or find specific facts/answers, use ask_document.
- To summarize a document, use summarize_document or summarize_and_export.
- To generate a quiz or MCQs from a document, use generate_quiz.
- To generate revision study flashcards from a document, use generate_flashcards.
- To password-protect or encrypt a PDF, use protect_pdf.
- To unlock or decrypt a protected PDF, use unlock_pdf.
- To watermark a PDF with custom text, use watermark_pdf.
- To find duplicate files or check storage waste, use find_duplicates.
- To translate a document into another language, use translate_document.
- To inspect storage analytics, usage breakdown, or workspace stats, use workspace_stats.
- To generate a Word or PDF file from text or summary, use create_document.
- To extract text from an image or scanned photo, use ocr_document.
- To merge multiple PDF files, use merge_pdfs.
- To extract pages from a PDF, use split_pdf.
- To reduce PDF size, use compress_pdf.
- To convert a PDF to Word (.docx), use convert_pdf_to_word.
- To convert a Word (.docx) document to PDF, use convert_word_to_pdf.
- To share or transfer a file to another user, use share_file.
- When the user asks to send, download, or get a file, use send_file.
- Never repeat a successfully completed action.
- Review the latest result before choosing an action.
- If the file is already in the requested destination folder, use finish.
- If the latest result confirms the user's request is complete, use finish.
- If a tool failed, choose the next appropriate action.
- When creating documents or calling tools, never output meta-commentary, notes, apologies, or remarks like '(Note: ...)' inside parameter values. Parameter values must contain ONLY the actual document text.
- If the user asks to save, export, or convert previous content (such as a quiz, summary, or text) into a file and the content is referenced from the conversation history, extract the full text from the history without truncating or adding commentary notes.
- Choose exactly one action.
- Never invent an action.
- Do not explain your decision.

Return exactly ONE JSON object.

Example:
{{"action":"search_files","parameters":{{"query":"Computer Vision"}}}}
"""

    last_error = None

    for max_tokens in (2000, 3000):
        create_kwargs = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are DocPilot's action selector. "
                        "Do not explain your decision. "
                        "Return only one complete JSON object "
                        "containing action and parameters."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if "qwen" in MODEL.lower():
            create_kwargs["reasoning_effort"] = "none"

        response = client.chat.completions.create(**create_kwargs)

        content = response.choices[0].message.content

        print("\n--- RAW PLANNER RESPONSE ---")
        print(_safe_text(content))

        try:
            decision = parse_planner_decision(content)
        except ValueError as error:
            last_error = error
            continue

        print("\n--- PARSED DECISION ---")
        print(_safe_text(decision))

        return decision

    raise last_error or ValueError("Planner returned no JSON object.")


def parse_planner_decision(content: str) -> dict:
    """Parse and validate JSON decision returned by the model."""
    if not content or not content.strip():
        raise ValueError("Planner returned an empty response.")

    content = clean_response(content.strip())

    decoder = json.JSONDecoder()
    start = content.find("{")

    if start == -1:
        raise ValueError(f"Planner returned no JSON object:\n{content}")

    try:
        decision, _ = decoder.raw_decode(content[start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"Planner returned invalid JSON:\n{content}") from error

    if not isinstance(decision, dict):
        raise ValueError("Planner output must be a JSON object.")

    if "action" not in decision:
        raise ValueError("Planner output missing 'action'.")

    if "parameters" not in decision:
        raise ValueError("Planner output missing 'parameters'.")

    if not isinstance(decision["action"], str):
        raise ValueError("Planner 'action' must be a string.")

    if not isinstance(decision["parameters"], dict):
        raise ValueError("Planner 'parameters' must be a JSON object.")

    allowed_actions = {
        "list_files",
        "search_files",
        "list_folders",
        "list_files_in_folder",
        "get_file_info",
        "send_file",
        "read_file_content",
        "rename_file",
        "create_zip",
        "extract_zip",
        "move_file",
        "copy_file",
        "create_folder",
        "delete_file",
        "delete_folder",
        "extract_text",
        "summarize_document",
        "create_document",
        "summarize_and_export",
        "merge_pdfs",
        "split_pdf",
        "compress_pdf",
        "convert_pdf_to_word",
        "ask_document",
        "ocr_document",
        "convert_word_to_pdf",
        "share_file",
        "generate_quiz",
        "generate_flashcards",
        "protect_pdf",
        "unlock_pdf",
        "watermark_pdf",
        "find_duplicates",
        "translate_document",
        "workspace_stats",
        "finish",
    }

    if decision["action"] not in allowed_actions:
        raise ValueError(f"Unknown planner action: {decision['action']}")

    return decision