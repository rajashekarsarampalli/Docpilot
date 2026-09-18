import inspect

from files.services.file_tools import (
    list_files,
    search_files,
    list_folders,
    list_files_in_folder,
    get_file_info,
    send_file,
    rename_file,
    create_zip,
    extract_zip,
    move_file,
    copy_file,
    create_folder,
    delete_file,
    delete_folder,
    read_file_content,
    extract_text,
    summarize_document,
    create_document,
    summarize_and_export,
    merge_pdfs,
    split_pdf,
    compress_pdf,
    convert_pdf_to_word,
    ask_document,
    ocr_document,
    convert_word_to_pdf,
    share_file,
    generate_quiz,
    generate_flashcards,
    protect_pdf,
    unlock_pdf,
    watermark_pdf,
    find_duplicates,
    translate_document,
    workspace_stats,
)


TOOLS = {
    "list_files": list_files,
    "search_files": search_files,
    "list_folders": list_folders,
    "list_files_in_folder": list_files_in_folder,
    "get_file_info": get_file_info,
    "send_file": send_file,
    "rename_file": rename_file,
    "create_zip": create_zip,
    "extract_zip": extract_zip,
    "move_file": move_file,
    "copy_file": copy_file,
    "create_folder": create_folder,
    "delete_file": delete_file,
    "delete_folder": delete_folder,
    "read_file_content": read_file_content,
    "extract_text": extract_text,
    "summarize_document": summarize_document,
    "create_document": create_document,
    "summarize_and_export": summarize_and_export,
    "merge_pdfs": merge_pdfs,
    "split_pdf": split_pdf,
    "compress_pdf": compress_pdf,
    "convert_pdf_to_word": convert_pdf_to_word,
    "ask_document": ask_document,
    "ocr_document": ocr_document,
    "convert_word_to_pdf": convert_word_to_pdf,
    "share_file": share_file,
    "generate_quiz": generate_quiz,
    "generate_flashcards": generate_flashcards,
    "protect_pdf": protect_pdf,
    "unlock_pdf": unlock_pdf,
    "watermark_pdf": watermark_pdf,
    "find_duplicates": find_duplicates,
    "translate_document": translate_document,
    "workspace_stats": workspace_stats,
}

PARAMETER_ALIASES = {
    "create_folder": {
        "name": "folder_name",
        "folder": "folder_name",
    },
    "delete_folder": {
        "name": "folder_name",
        "folder": "folder_name",
    },
    "list_files_in_folder": {
        "name": "folder_name",
        "folder": "folder_name",
    },
    "create_zip": {
        "name": "zip_name",
        "archive_name": "zip_name",
        "archive": "zip_name",
        "files": "file_ids",
        "ids": "file_ids",
    },
    "extract_zip": {
        "name": "zip_name",
        "archive_name": "zip_name",
        "archive": "zip_name",
        "folder": "destination_folder",
        "destination": "destination_folder",
    },
    "move_file": {
        "id": "file_id",
        "file": "file_id",
        "folder": "destination_folder",
        "destination": "destination_folder",
        "target_folder": "destination_folder",
        "to": "destination_folder",
    },
    "copy_file": {
        "id": "file_id",
        "file": "file_id",
        "folder": "destination_folder",
        "destination": "destination_folder",
        "target_folder": "destination_folder",
        "to": "destination_folder",
    },
    "rename_file": {
        "id": "file_id",
        "file": "file_id",
        "name": "new_name",
        "filename": "new_name",
    },
    "delete_file": {
        "id": "file_id",
        "file": "file_id",
    },
    "get_file_info": {
        "id": "file_id",
        "file": "file_id",
    },
    "send_file": {
        "id": "file_id",
        "file": "file_id",
    },
    "read_file_content": {
        "id": "file_id",
        "file": "file_id",
        "limit": "max_chars",
        "length": "max_chars",
    },
    "search_files": {
        "q": "query",
        "search": "query",
        "text": "query",
        "keyword": "query",
    },
    "extract_text": {
        "id": "file_id",
        "file": "file_id",
        "pages": "max_pages",
    },
    "summarize_document": {
        "id": "file_id",
        "file": "file_id",
        "topic": "focus",
        "about": "focus",
    },
    "create_document": {
        "name": "title",
        "text": "content",
        "body": "content",
        "type": "format",
        "folder": "folder_name",
    },
    "summarize_and_export": {
        "id": "file_id",
        "file": "file_id",
        "format": "export_format",
        "type": "export_format",
        "folder": "folder_name",
        "topic": "focus",
    },
    "merge_pdfs": {
        "files": "file_ids",
        "ids": "file_ids",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "split_pdf": {
        "id": "file_id",
        "file": "file_id",
        "pages": "page_ranges",
        "range": "page_ranges",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "compress_pdf": {
        "id": "file_id",
        "file": "file_id",
        "level": "compression_level",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "convert_pdf_to_word": {
        "id": "file_id",
        "file": "file_id",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "ask_document": {
        "id": "file_id",
        "file": "file_id",
        "q": "question",
        "query": "question",
        "prompt": "question",
        "ask": "question",
    },
    "ocr_document": {
        "id": "file_id",
        "file": "file_id",
        "image": "file_id",
    },
    "convert_word_to_pdf": {
        "id": "file_id",
        "file": "file_id",
        "word": "file_id",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "share_file": {
        "id": "file_id",
        "file": "file_id",
        "recipient": "target_username",
        "to": "target_username",
        "user": "target_username",
        "username": "target_username",
    },
    "generate_quiz": {
        "id": "file_id",
        "file": "file_id",
        "questions": "num_questions",
        "count": "num_questions",
        "num": "num_questions",
    },
    "generate_flashcards": {
        "id": "file_id",
        "file": "file_id",
        "cards": "count",
        "num": "count",
    },
    "protect_pdf": {
        "id": "file_id",
        "file": "file_id",
        "pass": "password",
        "pwd": "password",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "unlock_pdf": {
        "id": "file_id",
        "file": "file_id",
        "pass": "password",
        "pwd": "password",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "watermark_pdf": {
        "id": "file_id",
        "file": "file_id",
        "text": "watermark_text",
        "watermark": "watermark_text",
        "stamp": "watermark_text",
        "name": "output_name",
        "output": "output_name",
        "folder": "folder_name",
    },
    "translate_document": {
        "id": "file_id",
        "file": "file_id",
        "lang": "target_language",
        "language": "target_language",
        "to": "target_language",
    },
}


def normalize_parameters(action: str, parameters: dict) -> dict:
    """Map common parameter aliases to canonical tool parameter names."""
    parameters = dict(parameters or {})
    aliases = PARAMETER_ALIASES.get(action, {})

    for alias, canonical in aliases.items():
        if alias in parameters and canonical not in parameters:
            parameters[canonical] = parameters.pop(alias)
        elif alias in parameters:
            parameters.pop(alias)

    # Universal fallback: if tool expects file_id but LLM supplied filename, document, doc, or name
    if "file_id" not in parameters:
        for candidate in ("filename", "document", "doc", "name", "file", "target", "file_name"):
            if candidate in parameters:
                parameters["file_id"] = parameters.pop(candidate)
                break

    # If create_zip was given a single file_id instead of file_ids list
    if action == "create_zip":
        if "file_id" in parameters and "file_ids" not in parameters:
            parameters["file_ids"] = [parameters.pop("file_id")]
        elif "file_id" in parameters:
            parameters.pop("file_id")

    return parameters


def execute_tool(action: str, parameters: dict, user=None):
    """
    Safely execute a tool with parameter normalization and signature validation
    to prevent unexpected keyword argument errors from LLM hallucinations.
    The ``user`` argument is injected into any tool that declares a ``user``
    parameter so that every DB read/write is scoped to the authenticated user.
    """
    if action not in TOOLS:
        raise ValueError(f"Unknown tool: {action}")

    tool_func = TOOLS[action]
    normalized = normalize_parameters(action, parameters)

    # Inject user into normalized params if the tool accepts it
    sig = inspect.signature(tool_func)
    if "user" in sig.parameters and user is not None:
        normalized["user"] = user

    # Filter parameters to only those accepted by the tool signature
    valid_params = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            if param_name in normalized:
                valid_params[param_name] = normalized[param_name]

    return tool_func(**valid_params)