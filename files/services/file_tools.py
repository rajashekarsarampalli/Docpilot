import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from dotenv import load_dotenv
from groq import Groq

from django.conf import settings

from ..models import UploadedFile, Folder

# --------------------------------------------------
# Shared Groq LLM client (lazy singleton)
# --------------------------------------------------

load_dotenv()

GROQ_MODEL = "qwen/qwen3.8-27b"
_groq_client = None


def _get_groq_client() -> Groq:
    """Return a shared Groq client instance, raising a clear error if the key is missing."""
    global _groq_client
    if _groq_client is None:
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def media_root(user=None) -> Path:
    """Return resolved Path to MEDIA_ROOT, scoped to user subfolder if provided."""
    base = Path(settings.MEDIA_ROOT).resolve()
    if user:
        uname = getattr(user, "username", None) or str(user)
        safe_uname = "".join(c for c in uname if c.isalnum() or c in ("-", "_")).strip() or "default"
        return (base / "users" / safe_uname).resolve()
    return base


def relative_media_path(path) -> str:
    """
    Return a relative path from MEDIA_ROOT using POSIX forward slashes
    for database consistency across Windows and Unix platforms.
    """
    return str(
        Path(path).resolve().relative_to(Path(settings.MEDIA_ROOT).resolve())
    ).replace("\\", "/")


def get_file_object(file_id, user=None) -> UploadedFile:
    """
    Get UploadedFile from the database.
    Intelligently resolves:
    1. Integer or numeric string: 30 or "30"
    2. Context words: "last", "latest", "recent", "it", "this", None, "" -> user's latest upload
    3. Exact filename (case-insensitive)
    4. Substring / partial filename (e.g. "MRCET", "fee process")
    5. Word-by-word matching
    """
    qs = UploadedFile.objects.select_related("folder")
    if user:
        qs = qs.filter(user=user)

    # 1. Contextual pronouns or empty -> grab the most recent file
    if file_id is None or str(file_id).strip().lower() in (
        "last", "latest", "recent", "current", "it", "this", "uploaded", "the file", "that file", "", "none"
    ):
        file_obj = qs.order_by("-uploaded_at").first()
        if file_obj:
            return file_obj
        raise ValueError("No files found in your workspace yet. Please upload a document first.")

    file_str = str(file_id).strip().strip("'\"")

    # 2. Check if numeric ID
    if file_str.isdigit():
        target_id = int(file_str)
        match = qs.filter(id=target_id).first()
        if match:
            return match
        if user and UploadedFile.objects.filter(id=target_id).exists():
            raise PermissionError(f"Access denied to file ID {target_id}.")
        raise ValueError(f"File with ID {target_id} does not exist.")

    # 3. Exact filename match (case-insensitive)
    match = qs.filter(original_name__iexact=file_str).first()
    if match:
        return match

    # 4. Substring match
    matches = list(qs.filter(original_name__icontains=file_str).order_by("-uploaded_at"))
    if matches:
        return matches[0]

    # 5. Word-by-word matching (e.g. "fee process" in "MRCET APP EXAM FEE PROCESS.pdf")
    words = [w.lower() for w in file_str.replace(".", " ").replace("_", " ").replace("-", " ").split() if len(w) >= 3]
    if words:
        for f in qs.order_by("-uploaded_at"):
            f_lower = f.original_name.lower()
            if all(w in f_lower for w in words):
                return f
        # Any single significant word match
        for f in qs.order_by("-uploaded_at"):
            f_lower = f.original_name.lower()
            if any(w in f_lower for w in words):
                return f

    recent_files = [f"'{f.original_name}'" for f in qs.order_by("-uploaded_at")[:3]]
    hint = f" Did you mean one of these? {', '.join(recent_files)}." if recent_files else ""
    raise ValueError(f"Could not find any file matching '{file_str}'.{hint}")


def parse_file_id(file_id, user=None) -> int:
    """Validate and parse a file ID, filename, or contextual keyword into an integer ID."""
    obj = get_file_object(file_id, user=user)
    return obj.id


def parse_file_ids(file_ids, user=None) -> list:
    """Validate and parse a list or JSON/comma string of file IDs, names, or keywords."""
    if file_ids is None:
        raise ValueError("file_ids cannot be empty.")

    if isinstance(file_ids, str):
        cleaned = file_ids.strip()
        try:
            file_ids = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            file_ids = [item.strip() for item in cleaned.split(",") if item.strip()]

    if isinstance(file_ids, (int, float)):
        file_ids = [file_ids]

    if not isinstance(file_ids, (list, tuple, set)):
        raise ValueError("file_ids must be a list of file IDs or names.")

    if not file_ids:
        raise ValueError("file_ids list cannot be empty.")

    return [get_file_object(fid, user=user).id for fid in file_ids]


def serialize_file(file: UploadedFile) -> dict:
    """Return a consistent dictionary representation of an UploadedFile."""
    try:
        file_path = Path(file.file.path).resolve()
        path_str = str(file_path)
        if file_path.exists():
            size = file_path.stat().st_size
        else:
            size = 0
    except Exception:
        path_str = str(file.file)
        size = 0

    return {
        "id": file.id,
        "name": file.original_name,
        "path": path_str,
        "folder": file.folder.name if file.folder else None,
        "size": size,
    }


def get_folder_object(folder_name: str, user=None) -> Folder:
    """
    Get Folder by name, scoped to user if provided.
    Uses .first() to gracefully avoid MultipleObjectsReturned errors.
    """
    if not folder_name or not isinstance(folder_name, str) or not folder_name.strip():
        raise ValueError("Folder name cannot be empty.")

    folder_name = folder_name.strip()
    qs = Folder.objects.filter(name=folder_name, parent=None)
    if user:
        qs = qs.filter(user=user)
    folder = qs.first()

    if not folder:
        raise ValueError(f"Folder '{folder_name}' does not exist.")

    return folder


def get_or_create_default_folder(user=None) -> Folder:
    """Get or create the default 'documents' folder for generated files."""
    folder, _ = Folder.objects.get_or_create(
        name="documents", parent=None, user=user
    )
    mroot = media_root(user=user)
    (mroot / "documents").mkdir(parents=True, exist_ok=True)
    return folder


# --------------------------------------------------
# File & Folder Listing / Inspection Tools
# --------------------------------------------------

def list_files(user=None) -> list:
    """Return all files known to Docpilot, optionally filtered by user."""
    qs = UploadedFile.objects.select_related("folder").order_by("-uploaded_at")
    if user:
        qs = qs.filter(user=user)
    return [serialize_file(file) for file in qs]


def search_files(query, user=None) -> list:
    """Search files by their original filename, optionally filtered by user."""
    if query is None:
        return []

    query = str(query).strip()
    if not query:
        return []

    qs = UploadedFile.objects.filter(
        original_name__icontains=query
    ).select_related("folder").order_by("-uploaded_at")
    if user:
        qs = qs.filter(user=user)

    return [serialize_file(file) for file in qs]


def list_folders(user=None) -> list:
    """Return all folders with their file counts and total sizes, optionally filtered by user."""
    folders = Folder.objects.filter(parent=None).prefetch_related("files")
    if user:
        folders = folders.filter(user=user)
    result = []
    for folder in folders:
        files = folder.files.all()
        if user:
            files = files.filter(user=user)
        total_size = 0
        for f in files:
            try:
                p = Path(f.file.path).resolve()
                if p.exists():
                    total_size += p.stat().st_size
            except Exception:
                pass

        result.append({
            "id": folder.id,
            "name": folder.name,
            "file_count": files.count(),
            "total_size": total_size,
        })
    return result


def list_files_in_folder(folder_name: str, user=None) -> list:
    """List all files located inside a specific folder, scoped to user."""
    folder = get_folder_object(folder_name, user=user)
    files = folder.files.all().order_by("-uploaded_at")
    if user:
        files = files.filter(user=user)
    return [serialize_file(f) for f in files]


def get_file_info(file_id, user=None) -> dict:
    """Return detailed metadata for a file, scoped to user."""
    file = get_file_object(file_id, user=user)
    serialized = serialize_file(file)
    file_path = Path(serialized["path"])
    exists = file_path.exists()
    size_bytes = serialized["size"]

    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"

    return {
        "id": file.id,
        "name": file.original_name,
        "folder": serialized["folder"],
        "extension": file_path.suffix.lower() if file_path.suffix else "none",
        "size_bytes": size_bytes,
        "size_formatted": size_str,
        "path": serialized["path"],
        "uploaded_at": file.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if file.uploaded_at else None,
        "exists_on_disk": exists,
    }


def send_file(file_id, user=None) -> dict:
    """Prepare a file for sending/downloading by validating it exists on disk."""
    file = get_file_object(file_id, user=user)
    file_path = Path(file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"File not found on disk: {file_path}")

    return {
        "message": "File ready to send",
        "id": file.id,
        "name": file.original_name,
        "path": str(file_path),
        "folder": file.folder.name if file.folder else None,
    }


# --------------------------------------------------
# Folder & File Management Tools
# --------------------------------------------------

def create_folder(folder_name: str, user=None) -> dict:
    """Create a new folder in MEDIA_ROOT and record it in the database."""
    if not folder_name or not isinstance(folder_name, str) or not folder_name.strip():
        raise ValueError("Folder name cannot be empty.")

    folder_name = folder_name.strip()

    # Prevent path traversal
    if "/" in folder_name or "\\" in folder_name or ".." in folder_name:
        raise ValueError("Invalid folder name.")

    # Check database first
    qs = Folder.objects.filter(name=folder_name, parent=None)
    if user:
        qs = qs.filter(user=user)
    if qs.exists():
        raise ValueError(f"A folder named '{folder_name}' already exists.")

    # Physical folder location
    mroot = media_root(user=user)
    folder_path = (mroot / folder_name).resolve()

    # Safety check: must reside inside user's media root
    if mroot not in folder_path.parents and folder_path != mroot:
        raise ValueError("Invalid folder destination.")

    # Check filesystem too
    if folder_path.exists():
        if folder_path.is_dir():
            raise ValueError(f"A physical folder named '{folder_name}' already exists.")
        raise ValueError(f"A file with the name '{folder_name}' already exists.")

    # Create physical folder
    folder_path.mkdir(parents=True, exist_ok=False)

    try:
        folder = Folder.objects.create(name=folder_name, parent=None, user=user)
    except Exception:
        if folder_path.exists():
            folder_path.rmdir()
        raise

    return {
        "message": "Folder created successfully",
        "folder_id": folder.id,
        "folder_name": folder.name,
        "path": str(folder_path),
    }


def delete_folder(folder_name: str, recursive: bool = False, user=None) -> dict:
    """Delete an existing folder record and remove the directory from disk."""
    folder = get_folder_object(folder_name, user=user)
    files_count = folder.files.count()

    if files_count > 0 and not recursive:
        raise ValueError(
            f"Folder '{folder_name}' contains {files_count} file(s). Set recursive=True to delete."
        )

    folder_path = (media_root(user=user) / folder.name).resolve()

    if recursive:
        for f in folder.files.all():
            try:
                p = Path(f.file.path).resolve()
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
            f.delete()

    folder.delete()

    if folder_path.exists():
        if recursive:
            shutil.rmtree(folder_path, ignore_errors=True)
        else:
            try:
                folder_path.rmdir()
            except OSError:
                pass

    return {
        "message": "Folder deleted successfully",
        "folder_name": folder_name,
    }


def move_file(file_id, destination_folder: str, user=None) -> dict:
    """Move a file to another folder."""
    uploaded_file = get_file_object(file_id, user=user)

    if not destination_folder or not isinstance(destination_folder, str) or not destination_folder.strip():
        raise ValueError("Destination folder cannot be empty.")

    destination_folder = destination_folder.strip()
    folder = get_folder_object(destination_folder, user=user)

    source_path = Path(uploaded_file.file.path).resolve()

    if not source_path.exists():
        raise ValueError(f"The file does not exist on disk: {source_path}")

    mroot = media_root(user=user)
    destination_path = (mroot / folder.name).resolve()

    # Safety check
    if mroot not in destination_path.parents and destination_path != mroot:
        raise ValueError("Invalid destination folder.")

    destination_path.mkdir(parents=True, exist_ok=True)

    # Check if already there in DB and filesystem
    if uploaded_file.folder_id == folder.id:
        expected_path = (destination_path / uploaded_file.original_name).resolve()
        if source_path == expected_path:
            return {
                "message": "File is already in the destination folder.",
                "id": uploaded_file.id,
                "name": uploaded_file.original_name,
                "folder": folder.name,
                "destination": str(expected_path),
                "already_there": True,
            }

    new_path = (destination_path / uploaded_file.original_name).resolve()

    if new_path.exists() and new_path != source_path:
        raise ValueError(
            f"A file named '{uploaded_file.original_name}' already exists in '{folder.name}'."
        )

    # Move physical file (shutil.move handles cross-drive moves on Windows)
    shutil.move(str(source_path), str(new_path))

    try:
        uploaded_file.file.name = relative_media_path(new_path)
        uploaded_file.folder = folder
        uploaded_file.save(update_fields=["file", "folder"])
    except Exception:
        if new_path.exists():
            new_path.rename(source_path)
        raise

    return {
        "message": "File moved successfully",
        "id": uploaded_file.id,
        "name": uploaded_file.original_name,
        "folder": folder.name,
        "destination": str(new_path),
        "already_there": False,
    }


def copy_file(file_id, destination_folder: str, user=None) -> dict:
    """Copy a file to another folder and register a new DB record."""
    uploaded_file = get_file_object(file_id, user=user)

    if not destination_folder or not isinstance(destination_folder, str) or not destination_folder.strip():
        raise ValueError("Destination folder cannot be empty.")

    destination_folder = destination_folder.strip()
    folder = get_folder_object(destination_folder, user=user)

    source_path = Path(uploaded_file.file.path).resolve()

    if not source_path.exists():
        raise ValueError(f"The file does not exist on disk: {source_path}")

    mroot = media_root(user=user)
    destination_path = (mroot / folder.name).resolve()

    # Safety check
    if mroot not in destination_path.parents and destination_path != mroot:
        raise ValueError("Invalid destination folder.")

    destination_path.mkdir(parents=True, exist_ok=True)

    new_path = (destination_path / uploaded_file.original_name).resolve()

    if new_path.exists():
        raise ValueError(
            f"A file named '{uploaded_file.original_name}' already exists in '{folder.name}'."
        )

    shutil.copy2(source_path, new_path)

    relative_name = relative_media_path(new_path)

    try:
        copied_file = UploadedFile(
            user=user,
            original_name=uploaded_file.original_name,
            folder=folder,
        )
        copied_file.file.name = relative_name
        copied_file.save()
    except Exception:
        if new_path.exists():
            new_path.unlink()
        raise

    return {
        "message": "File copied successfully",
        "id": copied_file.id,
        "name": copied_file.original_name,
        "folder": folder.name,
        "source": str(source_path),
        "destination": str(new_path),
    }


def rename_file(file_id, new_name: str, user=None) -> dict:
    """Rename a stored file."""
    uploaded_file = get_file_object(file_id, user=user)

    if not new_name or not isinstance(new_name, str) or not new_name.strip():
        raise ValueError("New file name cannot be empty.")

    new_name = Path(new_name.strip()).name
    if not new_name:
        raise ValueError("Invalid file name.")

    source_path = Path(uploaded_file.file.path).resolve()

    if not source_path.exists():
        raise ValueError(f"The file does not exist on disk: {source_path}")

    new_path = (source_path.parent / new_name).resolve()

    if new_path.exists() and new_path != source_path:
        raise ValueError(f"A file named '{new_name}' already exists.")

    source_path.rename(new_path)

    try:
        uploaded_file.file.name = relative_media_path(new_path)
        uploaded_file.original_name = new_name
        uploaded_file.save(update_fields=["file", "original_name"])
    except Exception:
        if new_path.exists():
            new_path.rename(source_path)
        raise

    return {
        "message": "File renamed successfully",
        "id": uploaded_file.id,
        "name": uploaded_file.original_name,
    }


def delete_file(file_id, user=None) -> dict:
    """Delete a file from the database and remove it from disk."""
    file = get_file_object(file_id, user=user)
    name = file.original_name
    folder_name = file.folder.name if file.folder else None
    file_path = Path(file.file.path).resolve()

    file.delete()

    disk_deleted = False
    if file_path.exists():
        file_path.unlink(missing_ok=True)
        disk_deleted = True

    return {
        "message": "File deleted successfully",
        "id": file_id,
        "name": name,
        "folder": folder_name,
        "deleted_from_disk": disk_deleted,
    }


def read_file_content(file_id, max_chars: int = 2000, user=None) -> dict:
    """Read and return text preview from a text-based document."""
    file = get_file_object(file_id, user=user)
    file_path = Path(file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"File not found on disk: {file_path}")

    suffix = file_path.suffix.lower()
    text_extensions = {
        ".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".css",
        ".js", ".yaml", ".yml", ".xml", ".ini", ".env", ".rst", ".sql"
    }

    if suffix not in text_extensions:
        raise ValueError(
            f"Cannot read preview of '{file.original_name}' ({suffix}). Only text-based files (.txt, .md, .csv, .json, .py, etc.) are supported."
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(int(max_chars))
        truncated = len(content) >= int(max_chars)
    except Exception as error:
        raise ValueError(f"Failed to read file: {error}")

    return {
        "message": "File content read successfully",
        "id": file.id,
        "name": file.original_name,
        "content": content,
        "truncated": truncated,
        "character_count": len(content),
    }


# --------------------------------------------------
# ZIP Archive Tools
# --------------------------------------------------

def create_zip(file_ids, zip_name: str, user=None) -> dict:
    """Create a ZIP archive containing selected files."""
    if not zip_name or not isinstance(zip_name, str) or not zip_name.strip():
        raise ValueError("ZIP name cannot be empty.")

    zip_name = Path(zip_name.strip()).name
    if not zip_name.endswith(".zip"):
        zip_name += ".zip"

    parsed_file_ids = parse_file_ids(file_ids)

    mroot = media_root(user=user)
    zip_directory = (mroot / "archives").resolve()
    zip_directory.mkdir(parents=True, exist_ok=True)

    zip_path = (zip_directory / zip_name).resolve()

    if mroot not in zip_path.parents:
        raise ValueError("Invalid ZIP destination path.")

    if zip_path.exists():
        raise ValueError(f"ZIP file '{zip_name}' already exists.")

    files = [get_file_object(fid, user=user) for fid in parsed_file_ids]

    file_entries = []
    for uploaded_file in files:
        fpath = Path(uploaded_file.file.path).resolve()
        if not fpath.exists():
            raise ValueError(f"File does not exist: {fpath}")
        file_entries.append((fpath, uploaded_file.original_name))

    try:
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zip_file:
            seen_names = set()
            for fpath, orig_name in file_entries:
                arcname = orig_name
                counter = 1
                stem = Path(orig_name).stem
                suffix = Path(orig_name).suffix
                while arcname in seen_names:
                    arcname = f"{stem} ({counter}){suffix}"
                    counter += 1
                seen_names.add(arcname)
                zip_file.write(fpath, arcname=arcname)
    except Exception:
        if zip_path.exists():
            zip_path.unlink()
        raise

    return {
        "message": "ZIP created successfully",
        "zip_name": zip_name,
        "path": str(zip_path),
        "files": [f.original_name for f in files],
    }


def extract_zip(zip_name: str, destination_folder: str = None, user=None) -> dict:
    """Extract a ZIP archive into a destination folder and register extracted files."""
    if not zip_name or not isinstance(zip_name, str) or not zip_name.strip():
        raise ValueError("zip_name is required.")

    zip_name = Path(zip_name.strip()).name
    if not zip_name.endswith(".zip"):
        zip_name += ".zip"

    mroot = media_root(user=user)
    archive_path = (mroot / "archives" / zip_name).resolve()
    if not archive_path.exists():
        archive_path = (mroot / zip_name).resolve()

    if not archive_path.exists():
        raise ValueError(f"ZIP file '{zip_name}' was not found.")

    if not destination_folder:
        destination_folder = Path(zip_name).stem
    destination_folder = Path(destination_folder.strip()).name

    dest_path = (mroot / destination_folder).resolve()
    if mroot not in dest_path.parents and dest_path != mroot:
        raise ValueError("Invalid destination folder.")

    dest_path.mkdir(parents=True, exist_ok=True)

    folder, _ = Folder.objects.get_or_create(name=destination_folder, parent=None, user=user)

    extracted_files = []
    with ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            if not filename:
                continue
            target_file_path = dest_path / filename
            with zf.open(member) as src, open(target_file_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            rel_name = relative_media_path(target_file_path)
            db_file, _ = UploadedFile.objects.get_or_create(
                original_name=filename,
                folder=folder,
                user=user,
                defaults={"file": rel_name}
            )
            if db_file.file.name != rel_name:
                db_file.file.name = rel_name
                db_file.save(update_fields=["file"])

            extracted_files.append(filename)

    return {
        "message": "ZIP extracted successfully",
        "zip_name": zip_name,
        "destination_folder": folder.name,
        "extracted_files": extracted_files,
        "count": len(extracted_files),
    }


# --------------------------------------------------
# Document Intelligence: Extraction, Summarization & Generation
# --------------------------------------------------

def extract_text(file_id, max_pages: int = 30, max_chars: int = 50000, user=None) -> dict:
    """
    Extract readable text from a document (PDF, Word .docx, or text files).
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")

    suffix = file_path.suffix.lower()
    full_text = ""
    pages_read = 1

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            pages_read = min(total_pages, int(max_pages))
            pages_text = []
            for idx in range(pages_read):
                page_content = reader.pages[idx].extract_text() or ""
                if page_content.strip():
                    pages_text.append(f"--- Page {idx + 1} ---\n{page_content.strip()}")
            full_text = "\n\n".join(pages_text)
        except Exception as error:
            raise ValueError(f"Failed to extract PDF text: {error}")

    elif suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(file_path))
            full_text = "\n\n".join(
                [p.text for p in doc.paragraphs if p.text.strip()]
            )
        except Exception as error:
            raise ValueError(f"Failed to read Word document: {error}")

    elif suffix == ".pptx":
        try:
            from pptx import Presentation
            prs = Presentation(str(file_path))
            slides_text = []
            total_slides = len(prs.slides)
            pages_read = min(total_slides, int(max_pages))
            for idx in range(pages_read):
                slide = prs.slides[idx]
                slide_lines = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            t = paragraph.text.strip()
                            if t:
                                slide_lines.append(t)
                    elif shape.has_table:
                        for row in shape.table.rows:
                            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                            if row_text:
                                slide_lines.append(" | ".join(row_text))
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_lines.append(f"[Presenter Notes: {notes}]")
                if slide_lines:
                    slides_text.append(f"--- Slide {idx + 1} ---\n" + "\n".join(slide_lines))
            full_text = "\n\n".join(slides_text)
        except Exception as error:
            raise ValueError(f"Failed to extract PowerPoint text: {error}")

    elif suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
            sheets_text = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheet_lines = [f"--- Sheet: {sheet_name} ---"]
                row_count = 0
                for row in ws.iter_rows(values_only=True):
                    vals = [str(v).strip() if v is not None else "" for v in row]
                    if any(vals):
                        sheet_lines.append(" | ".join(vals))
                        row_count += 1
                        if row_count >= 100:
                            sheet_lines.append("... [Additional rows truncated]")
                            break
                if len(sheet_lines) > 1:
                    sheets_text.append("\n".join(sheet_lines))
            wb.close()
            full_text = "\n\n".join(sheets_text)
        except Exception as error:
            raise ValueError(f"Failed to extract Excel data: {error}")

    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        try:
            ocr_res = ocr_document(uploaded_file.id, user=user)
            full_text = ocr_res.get("text", "").strip()
        except Exception as error:
            raise ValueError(f"Failed to extract text from image via OCR: {error}")

    else:
        text_extensions = {
            ".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".css",
            ".js", ".yaml", ".yml", ".xml", ".ini", ".env", ".rst", ".sql"
        }
        if suffix in text_extensions:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    full_text = f.read(int(max_chars))
            except Exception as error:
                raise ValueError(f"Failed to read text file: {error}")
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}' for text extraction. "
                "Supported formats: .pdf, .docx, .pptx, .xlsx, images (.png, .jpg), .txt, .md, .csv, .json, .py, etc."
            )

    if not full_text.strip():
        raise ValueError(f"No extractable text found in '{uploaded_file.original_name}'.")

    truncated = len(full_text) > int(max_chars)
    text_sample = full_text[:int(max_chars)]

    return {
        "message": "Text extracted successfully",
        "id": uploaded_file.id,
        "name": uploaded_file.original_name,
        "text": text_sample,
        "total_characters": len(full_text),
        "pages_read": pages_read,
        "truncated": truncated,
    }


def summarize_document(file_id, focus: str = None, user=None) -> dict:
    """
    Extract text from a document and generate an AI-powered structured summary.
    """
    extracted = extract_text(file_id, max_pages=20, max_chars=12000, user=user)
    text_content = extracted["text"]

    client = _get_groq_client()
    model = GROQ_MODEL

    focus_instruction = (
        f"Specific user focus: {focus}. Highlight aspects related to this focus."
        if focus
        else "Provide a comprehensive, structured summary highlighting key concepts, main topics, and essential takeaways."
    )

    prompt = f"""You are DocPilot's document analysis expert. Summarize the following document content clearly and concisely.

DOCUMENT NAME: {extracted['name']}
{focus_instruction}

DOCUMENT CONTENT:
{text_content}

INSTRUCTIONS:
1. Provide a concise Overview (2-3 sentences).
2. List Key Points / Highlights using clean markdown bullet points.
3. List any Important Questions, Definitions, or Action Items if present.
4. Do not output reasoning or <think> tags. Format strictly as clean Markdown."""

    summary_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocPilot's document intelligence engine. "
                    "Provide a clean, structured summary in Markdown. "
                    "Never output <think> tags or internal thoughts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    if "qwen" in model.lower():
        summary_kwargs["reasoning_effort"] = "none"

    response = client.chat.completions.create(**summary_kwargs)

    summary_text = response.choices[0].message.content.strip()
    if "<think>" in summary_text:
        if "</think>" in summary_text:
            summary_text = summary_text.split("</think>", 1)[1]
        else:
            # No closing tag — discard everything from <think> onward
            summary_text = summary_text.split("<think>", 1)[0]
    summary_text = summary_text.strip()

    return {
        "message": "Document summarized successfully",
        "id": extracted["id"],
        "name": extracted["name"],
        "summary": summary_text,
        "focus": focus or "General overview",
        "original_characters": extracted["total_characters"],
    }


def create_document(title: str, content: str, format: str = "docx", folder_name: str = None, user=None) -> dict:
    """
    Create a new Word (.docx), PDF (.pdf), Markdown (.md), or Text (.txt) document
    and register it in Docpilot.
    """
    if not title or not isinstance(title, str) or not title.strip():
        raise ValueError("Document title cannot be empty.")

    title = title.strip()
    format = (format or "docx").lower().strip().replace(".", "")

    if format not in ("docx", "pdf", "md", "txt"):
        raise ValueError(
            f"Unsupported format '{format}'. Supported formats: 'docx', 'pdf', 'md', 'txt'."
        )

    # Determine destination folder and path
    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
        destination_dir = (mroot / target_folder.name).resolve()
    else:
        target_folder = None
        destination_dir = mroot
    destination_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{Path(title).name}.{format}"
    file_path = (destination_dir / filename).resolve()

    # Avoid overwriting existing files
    counter = 1
    stem = Path(title).name
    while file_path.exists():
        filename = f"{stem}_{counter}.{format}"
        file_path = (destination_dir / filename).resolve()
        counter += 1

    if format == "docx":
        try:
            import docx
            from docx.shared import Pt, Inches, RGBColor
            doc = docx.Document()
            doc.add_heading(title, level=0)

            for block in content.split("\n\n"):
                b = block.strip()
                if not b:
                    continue
                if b.startswith("### "):
                    doc.add_heading(b[4:], level=3)
                elif b.startswith("## "):
                    doc.add_heading(b[3:], level=2)
                elif b.startswith("# "):
                    doc.add_heading(b[2:], level=1)
                elif b.startswith("- ") or b.startswith("* "):
                    for line in b.splitlines():
                        clean_line = line.lstrip("- *").strip()
                        if clean_line:
                            doc.add_paragraph(clean_line, style="List Bullet")
                else:
                    doc.add_paragraph(b)

            doc.save(str(file_path))
        except Exception as error:
            raise ValueError(f"Failed to generate Word document: {error}")

    elif format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(title, styles["Title"]))
            story.append(Spacer(1, 14))

            for block in content.split("\n\n"):
                b = block.strip()
                if not b:
                    continue
                clean_html = b.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                if b.startswith("### "):
                    story.append(Paragraph(b[4:], styles["Heading3"]))
                elif b.startswith("## "):
                    story.append(Paragraph(b[3:], styles["Heading2"]))
                elif b.startswith("# "):
                    story.append(Paragraph(b[2:], styles["Heading1"]))
                elif b.startswith("- ") or b.startswith("* "):
                    for line in b.splitlines():
                        l_clean = line.lstrip("- *").strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        if l_clean:
                            story.append(Paragraph(f"• {l_clean}", styles["Normal"]))
                else:
                    story.append(Paragraph(clean_html, styles["Normal"]))
                story.append(Spacer(1, 8))

            doc.build(story)
        except Exception as error:
            raise ValueError(f"Failed to generate PDF document: {error}")

    else:
        # Plain text or Markdown
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{content}\n")
        except Exception as error:
            raise ValueError(f"Failed to write text document: {error}")

    # Register in database
    relative_name = relative_media_path(file_path)
    uploaded_file = UploadedFile.objects.create(
        user=user,
        original_name=filename,
        folder=target_folder,
        file=relative_name,
    )

    return {
        "message": f"{format.upper()} document created successfully",
        "id": uploaded_file.id,
        "name": uploaded_file.original_name,
        "format": format,
        "folder": target_folder.name if target_folder else "root",
        "path": str(file_path),
    }


def summarize_and_export(file_id, export_format: str = "docx", title: str = None, focus: str = None, folder_name: str = None, user=None) -> dict:
    """
    End-to-end pipeline: summarizes a file and immediately exports the summary
    as a new Word (.docx) or PDF (.pdf) document.
    """
    source_file = get_file_object(file_id, user=user)
    summary_result = summarize_document(source_file.id, focus=focus, user=user)

    if not title:
        base_name = Path(summary_result["name"]).stem
        title = f"{base_name}_Summary"

    target_folder_name = folder_name or (source_file.folder.name if source_file.folder else None)
    created = create_document(
        title=title,
        content=summary_result["summary"],
        format=export_format or "docx",
        folder_name=target_folder_name,
        user=user,
    )

    return {
        "message": f"Document summarized and exported to {export_format.upper()} successfully",
        "source_file": summary_result["name"],
        "name": created["name"],
        "summary": summary_result["summary"],
        "export_id": created["id"],
        "export_name": created["name"],
        "export_format": created["format"],
        "export_folder": created["folder"],
        "export_path": created["path"],
        "path": created["path"],
    }


# --------------------------------------------------
# Advanced PDF Tools (iLovePDF Cloud API + Local Engine)
# --------------------------------------------------

def _get_ilovepdf_keys():
    """Retrieve iLovePDF credentials from environment."""
    load_dotenv()
    public_key = (os.getenv("ILOVEPDF_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("ILOVEPDF_SECRET_KEY") or "").strip()
    if public_key and secret_key:
        return public_key, secret_key
    return None, None


def _ilovepdf_process_file(tool_name: str, file_path: Path, extra_params: dict = None) -> bytes:
    """Execute a task via iLovePDF REST API and return processed file bytes."""
    import requests

    public_key, secret_key = _get_ilovepdf_keys()
    if not public_key:
        raise ValueError("iLovePDF public key is not configured in .env.")

    # 1. Auth
    auth_resp = requests.post(
        "https://api.ilovepdf.com/v1/auth",
        json={"public_key": public_key},
        timeout=15,
    )
    if auth_resp.status_code != 200:
        raise ValueError(f"iLovePDF authentication failed ({auth_resp.status_code}): {auth_resp.text}")
    token = auth_resp.json().get("token")
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Start
    start_resp = requests.get(
        f"https://api.ilovepdf.com/v1/start/{tool_name}",
        headers=headers,
        timeout=15,
    )
    if start_resp.status_code != 200:
        raise ValueError(f"iLovePDF start task failed: {start_resp.text}")
    start_data = start_resp.json()
    server = start_data["server"]
    task_id = start_data["task"]

    # 3. Upload
    with open(file_path, "rb") as f:
        upload_resp = requests.post(
            f"https://{server}/v1/upload",
            headers=headers,
            data={"task": task_id},
            files={"file": (file_path.name, f)},
            timeout=60,
        )
    if upload_resp.status_code != 200:
        raise ValueError(f"iLovePDF upload failed: {upload_resp.text}")
    server_filename = upload_resp.json()["server_filename"]

    # 4. Process
    payload = {
        "task": task_id,
        "tool": tool_name,
        "files": [{"server_filename": server_filename, "filename": file_path.name}],
    }
    if extra_params:
        payload.update(extra_params)

    proc_resp = requests.post(
        f"https://{server}/v1/process",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if proc_resp.status_code != 200:
        raise ValueError(f"iLovePDF processing failed: {proc_resp.text}")

    # 5. Download
    down_resp = requests.get(
        f"https://{server}/v1/download/{task_id}",
        headers=headers,
        timeout=120,
    )
    if down_resp.status_code != 200:
        raise ValueError(f"iLovePDF download failed: {down_resp.text}")

    return down_resp.content


def merge_pdfs(file_ids, output_name: str = "merged.pdf", folder_name: str = None, user=None) -> dict:
    """
    Merge multiple PDF files into one single PDF.
    Operates locally with pypdf for maximum speed and reliability.
    """
    parsed_ids = parse_file_ids(file_ids)
    if len(parsed_ids) < 2:
        raise ValueError("At least 2 PDF files are required to merge.")

    files = [get_file_object(fid, user=user) for fid in parsed_ids]

    from pypdf import PdfWriter, PdfReader
    writer = PdfWriter()

    for uploaded_file in files:
        fpath = Path(uploaded_file.file.path).resolve()
        if not fpath.exists():
            raise ValueError(f"File not found on disk: {fpath}")
        if fpath.suffix.lower() != ".pdf":
            raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")
        reader = PdfReader(str(fpath))
        for page in reader.pages:
            writer.add_page(page)

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = files[0].folder if files else None
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name or not output_name.strip():
        output_name = "merged.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "message": "PDF files merged successfully",
        "id": new_record.id,
        "name": new_record.original_name,
        "folder": target_folder.name if target_folder else "root",
        "merged_count": len(files),
        "path": str(output_path),
    }


def split_pdf(file_id, page_ranges: str = "1", output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Extract specific pages or page ranges from a PDF (e.g. '1', '1-3', '1, 3, 5').
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)

    # Parse ranges: e.g. "1-2" or "1, 3"
    target_pages = set()
    for part in str(page_ranges).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start_p = int(start_str.strip())
            end_p = int(end_str.strip()) if end_str.strip().lower() != "end" else total_pages
            for p in range(start_p, end_p + 1):
                target_pages.add(p)
        else:
            target_pages.add(int(part))

    sorted_pages = sorted(p for p in target_pages if 1 <= p <= total_pages)
    if not sorted_pages:
        raise ValueError(f"No valid pages selected. Document has {total_pages} page(s).")

    writer = PdfWriter()
    for p_num in sorted_pages:
        writer.add_page(reader.pages[p_num - 1])

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        base_stem = file_path.stem
        output_name = f"{base_stem}_pages_{min(sorted_pages)}-{max(sorted_pages)}.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "message": "PDF split successfully",
        "id": new_record.id,
        "name": new_record.original_name,
        "folder": target_folder.name if target_folder else "root",
        "extracted_pages": sorted_pages,
        "page_count": len(sorted_pages),
        "path": str(output_path),
    }


def compress_pdf(file_id, compression_level: str = "recommended", output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Compress a PDF file to reduce size. Uses iLovePDF API if keys are set,
    otherwise uses local pypdf stream compression.
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    original_size = file_path.stat().st_size

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}_compressed.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    used_engine = "local"
    public_key, _ = _get_ilovepdf_keys()

    if public_key:
        try:
            compressed_bytes = _ilovepdf_process_file(
                "compress",
                file_path,
                extra_params={"compression_level": compression_level or "recommended"},
            )
            with open(output_path, "wb") as f_out:
                f_out.write(compressed_bytes)
            used_engine = "ilovepdf"
        except Exception:
            used_engine = "local"

    if used_engine == "local":
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(file_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        for page in writer.pages:
            page.compress_content_streams()
        writer.compress_identical_objects()
        with open(output_path, "wb") as f_out:
            writer.write(f_out)

    new_size = output_path.stat().st_size
    saved_bytes = max(0, original_size - new_size)
    percent_saved = (saved_bytes / original_size * 100) if original_size > 0 else 0

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "message": "PDF compressed successfully",
        "id": new_record.id,
        "name": new_record.original_name,
        "engine": used_engine,
        "original_size": f"{original_size / 1024:.1f} KB",
        "compressed_size": f"{new_size / 1024:.1f} KB",
        "reduction_percent": f"{percent_saved:.1f}%",
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def convert_pdf_to_word(file_id, output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Convert a PDF into an editable Word document (.docx).
    Uses iLovePDF API if configured, or local text-to-docx pipeline.
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}.docx"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".docx"):
        output_name += ".docx"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    ext = ".docx"
    while output_path.exists():
        output_name = f"{stem}_{counter}{ext}"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    used_engine = "local"
    public_key, _ = _get_ilovepdf_keys()

    if public_key:
        try:
            docx_bytes = _ilovepdf_process_file("pdf_to_office", file_path)
            with open(output_path, "wb") as f_out:
                f_out.write(docx_bytes)
            used_engine = "ilovepdf"
        except Exception:
            used_engine = "local"

    if used_engine == "local":
        ext = extract_text(file_id, max_pages=50, user=user)
        import docx
        doc = docx.Document()
        doc.add_heading(uploaded_file.original_name, level=0)
        for page_chunk in ext["text"].split("--- Page "):
            p_chunk = page_chunk.strip()
            if not p_chunk:
                continue
            lines = p_chunk.splitlines()
            if lines:
                doc.add_heading(f"Page {lines[0]}", level=2)
                for line in lines[1:]:
                    if line.strip():
                        doc.add_paragraph(line.strip())
        doc.save(str(output_path))

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "message": "PDF converted to Word successfully",
        "id": new_record.id,
        "name": new_record.original_name,
        "engine": used_engine,
        "format": "docx",
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def ask_document(file_id, question: str, user=None) -> dict:
    """
    Answer specific user questions about a document using AI over extracted text.
    Provides precise, grounded, factual answers.
    """
    if not question or not str(question).strip():
        raise ValueError("A question must be provided to ask about the document.")

    question = str(question).strip()
    extracted = extract_text(file_id, max_pages=30, max_chars=20000, user=user)
    text_content = extracted["text"]

    client = _get_groq_client()
    model = GROQ_MODEL

    prompt = f"""You are DocPilot's document question-answering assistant. Answer the user's question accurately and strictly based on the document content provided below.

DOCUMENT NAME: {extracted['name']}

DOCUMENT CONTENT:
{text_content}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question directly and concisely based ONLY on the document.
2. If specific details (amounts, steps, dates, links, names) are mentioned in the text, include them clearly.
3. If the answer cannot be found in the document, explicitly state: "This information is not mentioned in the document."
4. Format your answer using clean Markdown. Never output <think> tags."""

    qa_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are DocPilot's document Q&A engine. Provide direct, factual answers grounded in the provided document text. Never output internal thinking tags.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    if "qwen" in model.lower():
        qa_kwargs["reasoning_effort"] = "none"

    response = client.chat.completions.create(**qa_kwargs)
    answer = response.choices[0].message.content.strip()

    if "<think>" in answer:
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1]
        else:
            # No closing tag — discard everything from <think> onward
            answer = answer.split("<think>", 1)[0]
    answer = answer.strip()

    return {
        "message": "Question answered successfully",
        "id": extracted["id"],
        "name": extracted["name"],
        "question": question,
        "answer": answer,
    }


def ocr_document(file_id, lang: str = "en", user=None) -> dict:
    """
    Perform native Optical Character Recognition (OCR) on an image file
    or scanned image-only PDF to extract readable text.
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")

    suffix = file_path.suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

    import asyncio
    try:
        import winocr
    except ImportError:
        winocr = None

    def _run_ocr_sync(pil_image, ocr_lang):
        """Run winocr in a fresh event loop — safe even if an outer loop exists."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(winocr.recognize_pil(pil_image, lang=ocr_lang))
        finally:
            loop.close()

    extracted_lines = []

    if suffix in image_exts:
        from PIL import Image
        img = Image.open(str(file_path))
        if winocr:
            try:
                res = _run_ocr_sync(img, lang)
                text = getattr(res, "text", "") or str(res)
                if text.strip():
                    extracted_lines.append(text.strip())
            except Exception as e:
                raise ValueError(f"OCR processing failed on image: {e}")
        else:
            raise ValueError("OCR engine is not available on this system.")

    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        from PIL import Image
        import io

        for idx, page in enumerate(reader.pages[:10]):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_lines.append(f"--- Page {idx + 1} (Digital) ---\n{page_text.strip()}")
            elif page.images and winocr:
                img_texts = []
                for img_obj in page.images:
                    try:
                        pil_img = Image.open(io.BytesIO(img_obj.data))
                        res = _run_ocr_sync(pil_img, lang)
                        t = getattr(res, "text", "")
                        if t.strip():
                            img_texts.append(t.strip())
                    except Exception:
                        pass
                if img_texts:
                    extracted_lines.append(f"--- Page {idx + 1} (OCR) ---\n" + "\n".join(img_texts))

    else:
        raise ValueError(
            f"Unsupported format '{suffix}' for OCR. Supported formats: .png, .jpg, .jpeg, .bmp, .webp, .pdf"
        )

    full_text = "\n\n".join(extracted_lines).strip()
    if not full_text:
        raise ValueError(f"No text could be recognized from '{uploaded_file.original_name}'.")

    return {
        "message": "OCR text extracted successfully",
        "id": uploaded_file.id,
        "name": uploaded_file.original_name,
        "text": full_text,
        "total_characters": len(full_text),
    }


def convert_word_to_pdf(file_id, output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Convert a Word (.docx) document to PDF (.pdf).
    Uses iLovePDF 'officepdf' cloud API for pixel-perfect conversion,
    with local ReportLab fallback.
    """
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"The file does not exist on disk: {file_path}")
    if file_path.suffix.lower() not in (".docx", ".doc"):
        raise ValueError(f"'{uploaded_file.original_name}' is not a Word document.")

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    used_engine = "local"
    public_key, _ = _get_ilovepdf_keys()

    if public_key:
        try:
            pdf_bytes = _ilovepdf_process_file("officepdf", file_path)
            with open(output_path, "wb") as f_out:
                f_out.write(pdf_bytes)
            used_engine = "ilovepdf"
        except Exception:
            used_engine = "local"

    if used_engine == "local":
        import docx
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        doc = docx.Document(str(file_path))
        styles = getSampleStyleSheet()
        story = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                clean = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(clean, styles["Normal"]))
                story.append(Spacer(1, 6))
        pdf_doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        pdf_doc.build(story)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "message": "Word document converted to PDF successfully",
        "id": new_record.id,
        "name": new_record.original_name,
        "engine": used_engine,
        "format": "pdf",
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def share_file(file_id, target_username: str, folder_name: str = "Shared", user=None) -> dict:
    """
    Share/copy a document from the current user to another registered DocPilot user's workspace.
    """
    if not target_username or not str(target_username).strip():
        raise ValueError("Target username is required to share a file.")

    target_username = str(target_username).strip()
    from django.contrib.auth.models import User
    try:
        recipient = User.objects.get(username__iexact=target_username)
    except User.DoesNotExist:
        raise ValueError(f"User '{target_username}' was not found in DocPilot.")

    if user and recipient.id == user.id:
        raise ValueError("You cannot share a file with yourself.")

    source_file = get_file_object(file_id, user=user)
    source_path = Path(source_file.file.path).resolve()
    if not source_path.exists():
        raise ValueError(f"Source file not found on disk: {source_path}")

    recip_mroot = media_root(user=recipient)
    shared_folder, _ = Folder.objects.get_or_create(
        name=folder_name or "Shared",
        parent=None,
        user=recipient,
    )
    dest_dir = (recip_mroot / shared_folder.name).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = (dest_dir / source_file.original_name).resolve()
    counter = 1
    stem = Path(source_file.original_name).stem
    suffix = Path(source_file.original_name).suffix
    sender_uname = getattr(user, "username", "Someone") if user else "user"
    while dest_path.exists():
        dest_path = (dest_dir / f"{stem}_from_{sender_uname}_{counter}{suffix}").resolve()
        counter += 1

    shutil.copy2(source_path, dest_path)

    rel_name = relative_media_path(dest_path)
    new_record = UploadedFile.objects.create(
        user=recipient,
        original_name=dest_path.name,
        folder=shared_folder,
        file=rel_name,
    )

    return {
        "message": f"File shared successfully with {recipient.username}",
        "file_name": source_file.original_name,
        "shared_as": new_record.original_name,
        "sender": sender_uname,
        "recipient": recipient.username,
        "folder": shared_folder.name,
        "new_file_id": new_record.id,
    }


def generate_quiz(file_id, num_questions: int = 5, difficulty: str = "medium", user=None) -> dict:
    """
    Generate multiple-choice questions (MCQs) with options A-D, answers, and explanations
    based on the document content.
    """
    extracted = extract_text(file_id, max_pages=20, max_chars=15000, user=user)
    text_content = extracted["text"]

    client = _get_groq_client()
    model = GROQ_MODEL

    prompt = f"""You are DocPilot's academic quiz creator. Based strictly on the following document, generate {num_questions} high-quality Multiple Choice Questions (MCQs) at a {difficulty} difficulty level.

DOCUMENT NAME: {extracted['name']}
DOCUMENT CONTENT:
{text_content}

FORMAT EACH QUESTION EXACTLY AS FOLLOWS:
### Question [N]: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
**Correct Answer:** [Option Letter and text]
**Explanation:** [1-2 sentences explaining why based on the text]

Output strictly clean Markdown. Never output <think> tags."""

    quiz_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are DocPilot's quiz engine. Create challenging, accurate MCQs directly from the text in clean Markdown without internal thoughts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    if "qwen" in model.lower():
        quiz_kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**quiz_kwargs)
    quiz_content = resp.choices[0].message.content.strip()

    return {
        "status": "success",
        "message": f"Generated {num_questions}-question quiz successfully",
        "id": extracted["id"],
        "name": extracted["name"],
        "num_questions": num_questions,
        "quiz": quiz_content,
    }


def generate_flashcards(file_id, count: int = 6, num_cards: int = None, user=None) -> dict:
    """
    Generate revision study flashcards (Question/Concept on front, Answer on back)
    from the document.
    """
    if num_cards is not None:
        count = num_cards
    extracted = extract_text(file_id, max_pages=20, max_chars=15000, user=user)
    text_content = extracted["text"]

    client = _get_groq_client()
    model = GROQ_MODEL

    prompt = f"""You are DocPilot's revision expert. Extract the top {count} most important key concepts, definitions, or exam takeaways from this document into structured study flashcards.

DOCUMENT NAME: {extracted['name']}
DOCUMENT CONTENT:
{text_content}

FORMAT EACH FLASHCARD AS:
### 📇 Card [N]: [Key Concept / Question]
👉 **Explanation / Answer:** [Concise, direct answer]

Output strictly clean Markdown. Never output <think> tags."""

    fc_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are DocPilot's flashcard engine. Create crisp, high-yield study cards in clean Markdown without internal thoughts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    if "qwen" in model.lower():
        fc_kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**fc_kwargs)
    cards_content = resp.choices[0].message.content.strip()

    return {
        "status": "success",
        "message": f"Generated {count} revision flashcards successfully",
        "id": extracted["id"],
        "name": extracted["name"],
        "flashcards": cards_content,
    }


def protect_pdf(file_id, password: str, output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Encrypt a PDF file with 128-bit AES password protection.
    """
    if not password or not str(password).strip():
        raise ValueError("A password must be provided to protect the PDF.")

    password = str(password).strip()
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"File not found on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(file_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(user_password=password, owner_password=password, use_128bit=True)

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}_protected.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "status": "success",
        "message": "PDF password protected successfully",
        "id": new_record.id,
        "protected_file_id": new_record.id,
        "name": new_record.original_name,
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def unlock_pdf(file_id, password: str, output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Decrypt a password-protected PDF and save an unlocked copy.
    """
    if not password:
        raise ValueError("Password is required to unlock the PDF.")

    password = str(password).strip()
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"File not found on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(file_path))

    if not reader.is_encrypted:
        return {
            "message": "PDF is not encrypted. No unlock needed.",
            "id": uploaded_file.id,
            "name": uploaded_file.original_name,
            "path": str(file_path),
        }

    status = reader.decrypt(password)
    if status == 0:
        raise ValueError("Incorrect password. Could not decrypt PDF.")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}_unlocked.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "status": "success",
        "message": "PDF unlocked successfully",
        "id": new_record.id,
        "unlocked_file_id": new_record.id,
        "name": new_record.original_name,
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def watermark_pdf(file_id, watermark_text: str = "CONFIDENTIAL", output_name: str = None, folder_name: str = None, user=None) -> dict:
    """
    Stamp a custom semi-transparent text watermark diagonally across all pages of a PDF.
    """
    watermark_text = (watermark_text or "CONFIDENTIAL").strip()
    uploaded_file = get_file_object(file_id, user=user)
    file_path = Path(uploaded_file.file.path).resolve()

    if not file_path.exists():
        raise ValueError(f"File not found on disk: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{uploaded_file.original_name}' is not a PDF file.")

    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(file_path))
    writer = PdfWriter()

    for page in reader.pages:
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(w, h))
        can.saveState()
        can.setFont("Helvetica-Bold", 36)
        can.setFillColor(colors.Color(0.5, 0.5, 0.5, alpha=0.25))
        can.translate(w / 2, h / 2)
        can.rotate(45)
        can.drawCentredString(0, 0, watermark_text)
        can.restoreState()
        can.save()
        packet.seek(0)

        wm_reader = PdfReader(packet)
        page.merge_page(wm_reader.pages[0])
        writer.add_page(page)

    mroot = media_root(user=user)
    if folder_name and folder_name.strip():
        target_folder = get_folder_object(folder_name.strip(), user=user)
    else:
        target_folder = uploaded_file.folder
    dest_dir = (mroot / target_folder.name).resolve() if target_folder else mroot
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"{file_path.stem}_watermarked.pdf"
    output_name = Path(output_name.strip()).name
    if not output_name.endswith(".pdf"):
        output_name += ".pdf"

    output_path = (dest_dir / output_name).resolve()
    counter = 1
    stem = Path(output_name).stem
    while output_path.exists():
        output_name = f"{stem}_{counter}.pdf"
        output_path = (dest_dir / output_name).resolve()
        counter += 1

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    rel_name = relative_media_path(output_path)
    new_record = UploadedFile.objects.create(
        user=user,
        original_name=output_name,
        folder=target_folder,
        file=rel_name,
    )

    return {
        "status": "success",
        "message": f"PDF watermarked successfully with '{watermark_text}'",
        "id": new_record.id,
        "watermarked_file_id": new_record.id,
        "name": new_record.original_name,
        "watermark": watermark_text,
        "folder": target_folder.name if target_folder else "root",
        "path": str(output_path),
    }


def find_duplicates(user=None) -> dict:
    """
    Identify duplicate files in the user's workspace using cryptographic SHA-256 hashing.
    """
    import hashlib
    files = list(UploadedFile.objects.filter(user=user).select_related("folder"))
    hashes = {}
    duplicates = []
    wasted_bytes = 0

    for f in files:
        try:
            p = Path(f.file.path).resolve()
            if not p.exists():
                continue
            hasher = hashlib.sha256()
            with open(p, "rb") as fp:
                while chunk := fp.read(65536):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()
            size = p.stat().st_size
            item = {
                "id": f.id,
                "name": f.original_name,
                "folder": f.folder.name if f.folder else "root",
                "size": size,
            }
            if file_hash in hashes:
                hashes[file_hash].append(item)
                wasted_bytes += size
            else:
                hashes[file_hash] = [item]
        except Exception:
            continue

    for h, group in hashes.items():
        if len(group) > 1:
            duplicates.append({
                "original": group[0],
                "copies": group[1:],
                "count": len(group) - 1,
            })

    def _fmt(b):
        if b < 1024: return f"{b} B"
        if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
        return f"{b / (1024 * 1024):.2f} MB"

    return {
        "status": "success",
        "message": f"Duplicate scan completed: found {len(duplicates)} duplicate group(s)",
        "duplicate_groups": duplicates,
        "duplicate_groups_count": len(duplicates),
        "group_count": len(duplicates),
        "wasted_space": _fmt(wasted_bytes),
        "wasted_bytes": wasted_bytes,
    }


def translate_document(file_id, target_language: str = "Telugu", user=None) -> dict:
    """
    Translate document content or summary into a specified language using Groq LLM.
    """
    target_language = (target_language or "Telugu").strip()
    extracted = extract_text(file_id, max_pages=10, max_chars=8000, user=user)

    client = _get_groq_client()
    model = GROQ_MODEL

    prompt = f"""You are a professional multilingual document translator. Translate the following document content accurately into {target_language}. Maintain formatting, lists, and headings.

DOCUMENT: {extracted['name']}
TARGET LANGUAGE: {target_language}

CONTENT TO TRANSLATE:
{extracted['text']}

Translate accurately into {target_language}. Format strictly in clean Markdown. Never output <think> tags."""

    trans_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a professional document translator into {target_language}. Output strictly the translation in Markdown without internal thoughts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    if "qwen" in model.lower():
        trans_kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**trans_kwargs)
    translation = resp.choices[0].message.content.strip()

    return {
        "status": "success",
        "message": f"Document translated successfully into {target_language}",
        "id": extracted["id"],
        "exported_file_id": extracted["id"],
        "name": extracted["name"],
        "language": target_language,
        "translation": translation,
    }


def workspace_stats(user=None) -> dict:
    """
    Compute comprehensive workspace metrics: storage used, breakdown by type,
    and visual progress bars.
    """
    files = list(UploadedFile.objects.filter(user=user).select_related("folder"))
    folders = list(Folder.objects.filter(user=user))

    type_counts = {"PDF": 0, "Word": 0, "PowerPoint": 0, "Excel": 0, "Image": 0, "Archive": 0, "Other": 0}
    total_size = 0

    for f in files:
        try:
            p = Path(f.file.path).resolve()
            if p.exists():
                sz = p.stat().st_size
                total_size += sz
                ext = p.suffix.lower()
                if ext == ".pdf": type_counts["PDF"] += 1
                elif ext in (".docx", ".doc"): type_counts["Word"] += 1
                elif ext in (".pptx", ".ppt"): type_counts["PowerPoint"] += 1
                elif ext in (".xlsx", ".xls"): type_counts["Excel"] += 1
                elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"): type_counts["Image"] += 1
                elif ext in (".zip", ".rar", ".7z", ".tar"): type_counts["Archive"] += 1
                else: type_counts["Other"] += 1
        except Exception:
            pass

    def _fmt(b):
        if b < 1024: return f"{b} B"
        if b < 1024 * 1024: return f"{b / 1024:.1f} KB"
        return f"{b / (1024 * 1024):.2f} MB"

    def _bar(cnt, total):
        if not total: return "[░░░░░░░░░░] 0%"
        pct = min(10, max(0, int(cnt / total * 10)))
        return f"[{'█' * pct}{'░' * (10 - pct)}] {int(cnt / total * 100)}%"

    total_files = len(files)

    recent_list = []
    for rf in sorted(files, key=lambda x: x.uploaded_at, reverse=True)[:5]:
        sz_str = "0 B"
        try:
            sz_str = _fmt(Path(rf.file.path).stat().st_size)
        except Exception:
            pass
        recent_list.append({
            "id": rf.id,
            "name": rf.original_name,
            "size": sz_str,
            "date": rf.uploaded_at.strftime("%Y-%m-%d %H:%M"),
        })

    return {
        "status": "success",
        "message": "Workspace stats computed successfully",
        "username": getattr(user, "username", "Guest"),
        "total_files": total_files,
        "file_count": total_files,
        "total_folders": len(folders),
        "folder_count": len(folders),
        "total_size": _fmt(total_size),
        "total_size_formatted": _fmt(total_size),
        "total_bytes": total_size,
        "storage_usage_percent": round(total_size / (500 * 1024 * 1024) * 100, 1),
        "type_breakdown": type_counts,
        "format_breakdown": {
            "PDF": type_counts["PDF"],
            "Word Document": type_counts["Word"],
            "PowerPoint": type_counts["PowerPoint"],
            "Excel Spreadsheet": type_counts["Excel"],
            "Image": type_counts["Image"],
            "ZIP Archive": type_counts["Archive"],
            "Other": type_counts["Other"],
        },
        "recent_files": recent_list,
        "pdf_bar": _bar(type_counts["PDF"], total_files),
        "word_bar": _bar(type_counts["Word"], total_files),
        "pptx_bar": _bar(type_counts["PowerPoint"], total_files),
        "excel_bar": _bar(type_counts["Excel"], total_files),
        "image_bar": _bar(type_counts["Image"], total_files),
        "archive_bar": _bar(type_counts["Archive"], total_files),
    }