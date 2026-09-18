from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import UploadedFile
from .serializers import UploadedFileSerializer
from .services.file_tools import (
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


class FileUploadView(generics.CreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UploadedFileSerializer
    parser_classes = [
        MultiPartParser,
        FormParser,
    ]

    def get_queryset(self):
        return UploadedFile.objects.filter(user=self.request.user)


class FileListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UploadedFileSerializer

    def get_queryset(self):
        return (
            UploadedFile.objects.filter(user=self.request.user)
            .select_related("folder")
            .order_by("-uploaded_at")
        )


class FileToolsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        List files or folders. Use ?type=folders to list folders.
        """
        user = request.user
        view_type = request.query_params.get("type", "files")
        if view_type in ("folders", "folder"):
            return Response(list_folders(user=user))
        return Response(list_files(user=user))

    def post(self, request):
        """
        Execute a file or folder operation.
        Supports both flat and nested 'parameters' payloads.
        All operations are scoped to the authenticated user.
        """
        user = request.user
        data = request.data or {}
        action = data.get("action")

        params = data.get("parameters")
        if isinstance(params, dict):
            merged_data = {**params, **data}
        else:
            merged_data = data

        try:
            if action in ("list", "list_files"):
                result = list_files(user=user)

            elif action in ("list_folders", "folders"):
                result = list_folders(user=user)

            elif action in ("list_files_in_folder", "folder_files"):
                folder_name = (
                    merged_data.get("folder_name")
                    or merged_data.get("folder")
                    or merged_data.get("name")
                )
                result = list_files_in_folder(folder_name, user=user)

            elif action in ("get_file_info", "file_info", "info"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                result = get_file_info(file_id, user=user)

            elif action in ("search", "search_files"):
                query = (
                    merged_data.get("query")
                    or merged_data.get("q")
                    or merged_data.get("search")
                )
                result = search_files(query, user=user)

            elif action in ("send_file", "send", "download"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                result = send_file(file_id, user=user)

            elif action in ("read_file_content", "read_file", "read"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                max_chars = merged_data.get("max_chars", 2000)
                result = read_file_content(file_id, max_chars, user=user)

            elif action in ("rename", "rename_file"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                new_name = merged_data.get("new_name") or merged_data.get("name")
                result = rename_file(file_id, new_name, user=user)

            elif action in ("delete_file", "delete"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                result = delete_file(file_id, user=user)

            elif action in ("delete_folder",):
                folder_name = (
                    merged_data.get("folder_name")
                    or merged_data.get("name")
                    or merged_data.get("folder")
                )
                recursive = bool(merged_data.get("recursive", False))
                result = delete_folder(folder_name, recursive=recursive, user=user)

            elif action in ("zip", "create_zip"):
                file_ids = (
                    merged_data.get("file_ids")
                    or merged_data.get("files")
                    or merged_data.get("ids")
                )
                zip_name = merged_data.get("zip_name") or merged_data.get("name")
                result = create_zip(file_ids, zip_name, user=user)

            elif action in ("unzip", "extract_zip"):
                zip_name = (
                    merged_data.get("zip_name")
                    or merged_data.get("name")
                    or merged_data.get("archive")
                )
                destination_folder = (
                    merged_data.get("destination_folder")
                    or merged_data.get("folder")
                    or merged_data.get("destination")
                )
                result = extract_zip(zip_name, destination_folder, user=user)

            elif action in ("move", "move_file"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                destination_folder = (
                    merged_data.get("destination_folder")
                    or merged_data.get("folder")
                    or merged_data.get("destination")
                )
                result = move_file(file_id, destination_folder, user=user)

            elif action in ("copy", "copy_file"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                destination_folder = (
                    merged_data.get("destination_folder")
                    or merged_data.get("folder")
                    or merged_data.get("destination")
                )
                result = copy_file(file_id, destination_folder, user=user)

            elif action in ("create_folder", "folder"):
                folder_name = (
                    merged_data.get("folder_name")
                    or merged_data.get("name")
                    or merged_data.get("folder")
                )
                result = create_folder(folder_name, user=user)

            elif action in ("extract_text", "read_text"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                max_pages = merged_data.get("max_pages", 30)
                max_chars = merged_data.get("max_chars", 50000)
                result = extract_text(file_id, max_pages=max_pages, max_chars=max_chars, user=user)

            elif action in ("summarize", "summarize_document", "summarize_file"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                focus = merged_data.get("focus")
                result = summarize_document(file_id, focus=focus, user=user)

            elif action in ("create_document", "new_document"):
                title = merged_data.get("title") or merged_data.get("name")
                content = merged_data.get("content") or merged_data.get("text", "")
                doc_format = merged_data.get("format", "docx")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = create_document(title, content, format=doc_format, folder_name=folder_name, user=user)

            elif action in ("summarize_and_export", "summarize_to_doc"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                export_format = merged_data.get("export_format") or merged_data.get("format", "docx")
                title = merged_data.get("title")
                focus = merged_data.get("focus")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = summarize_and_export(
                    file_id,
                    export_format=export_format,
                    title=title,
                    focus=focus,
                    folder_name=folder_name,
                    user=user,
                )

            elif action in ("merge_pdfs", "merge_pdf", "merge"):
                file_ids = (
                    merged_data.get("file_ids")
                    or merged_data.get("files")
                    or merged_data.get("ids")
                )
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = merge_pdfs(file_ids, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("split_pdf", "split"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                page_ranges = merged_data.get("page_ranges") or merged_data.get("pages") or merged_data.get("range", "1")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = split_pdf(file_id, page_ranges=page_ranges, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("compress_pdf", "compress"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                compression_level = merged_data.get("compression_level") or merged_data.get("level", "recommended")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = compress_pdf(file_id, compression_level=compression_level, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("convert_pdf_to_word", "pdf_to_word", "pdf_to_docx"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = convert_pdf_to_word(file_id, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("convert_word_to_pdf", "word_to_pdf", "docx_to_pdf"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = convert_word_to_pdf(file_id, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("ask_document", "ask", "question"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                question = merged_data.get("question") or merged_data.get("q") or merged_data.get("query")
                result = ask_document(file_id, question=question, user=user)

            elif action in ("ocr_document", "ocr"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                result = ocr_document(file_id, user=user)

            elif action in ("share_file", "share"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                target_username = merged_data.get("target_username") or merged_data.get("recipient") or merged_data.get("to")
                result = share_file(file_id, target_username=target_username, user=user)

            elif action in ("generate_quiz", "quiz"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                num_questions = int(merged_data.get("num_questions", 5))
                result = generate_quiz(file_id, num_questions=num_questions, user=user)

            elif action in ("generate_flashcards", "flashcards"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                count = int(merged_data.get("count", 6))
                result = generate_flashcards(file_id, count=count, user=user)

            elif action in ("protect_pdf", "lock_pdf", "encrypt_pdf"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                password = merged_data.get("password")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = protect_pdf(file_id, password=password, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("unlock_pdf", "decrypt_pdf"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                password = merged_data.get("password")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = unlock_pdf(file_id, password=password, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("watermark_pdf", "watermark"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                watermark_text = merged_data.get("watermark_text") or merged_data.get("text", "CONFIDENTIAL")
                output_name = merged_data.get("output_name") or merged_data.get("name")
                folder_name = merged_data.get("folder_name") or merged_data.get("folder")
                result = watermark_pdf(file_id, watermark_text=watermark_text, output_name=output_name, folder_name=folder_name, user=user)

            elif action in ("find_duplicates", "duplicates"):
                result = find_duplicates(user=user)

            elif action in ("translate_document", "translate"):
                file_id = merged_data.get("file_id") or merged_data.get("id")
                target_language = merged_data.get("target_language") or merged_data.get("language", "Telugu")
                result = translate_document(file_id, target_language=target_language, user=user)

            elif action in ("workspace_stats", "stats"):
                result = workspace_stats(user=user)

            else:
                return Response(
                    {"error": f"Unknown action: '{action}'"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            return Response(result)

        except PermissionError as error:
            return Response(
                {"error": str(error)},
                status=status.HTTP_403_FORBIDDEN,
            )
        except (ValueError, TypeError, FileNotFoundError) as error:
            return Response(
                {"error": str(error)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as error:
            return Response(
                {"error": f"Internal server error: {str(error)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )