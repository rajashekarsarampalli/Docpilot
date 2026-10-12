import shutil
import tempfile
from pathlib import Path

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status

from files.models import Folder, UploadedFile
from files.services.file_tools import (
    create_folder,
    list_folders,
    delete_folder,
    create_document,
    list_files,
    search_files,
    get_file_info,
    read_file_content,
    rename_file,
    copy_file,
    move_file,
    delete_file,
    create_zip,
    extract_zip,
    merge_pdfs,
    split_pdf,
    watermark_pdf,
    protect_pdf,
    unlock_pdf,
    find_duplicates,
    workspace_stats,
)


class FileToolsTestCase(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.settings_override = override_settings(MEDIA_ROOT=Path(self.temp_dir))
        self.settings_override.enable()

        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_list_folders(self):
        res = create_folder("Projects", user=self.user)
        self.assertEqual(res["folder_name"], "Projects")
        self.assertTrue(Folder.objects.filter(name="Projects", user=self.user).exists())

        folders = list_folders(user=self.user)
        self.assertEqual(len(folders), 1)
        self.assertEqual(folders[0]["name"], "Projects")

        del_folder = delete_folder("Projects", user=self.user)
        self.assertEqual(del_folder["folder_name"], "Projects")

    def test_create_and_read_documents(self):
        doc_res = create_document(
            title="MeetingNotes",
            content="# Agenda\n\n1. Launch\n2. Next steps",
            format="txt",
            user=self.user,
        )
        self.assertTrue(doc_res["id"])
        self.assertEqual(doc_res["format"], "txt")

        files = list_files(user=self.user)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["id"], doc_res["id"])

        content_res = read_file_content(doc_res["id"], user=self.user)
        self.assertIn("Agenda", content_res["content"])

        info_res = get_file_info(doc_res["id"], user=self.user)
        self.assertEqual(info_res["extension"], ".txt")
        self.assertTrue(info_res["exists_on_disk"])

        searched = search_files("Meeting", user=self.user)
        self.assertEqual(len(searched), 1)

    def test_rename_copy_move_delete_file(self):
        doc = create_document(title="Draft", content="Initial text", format="txt", user=self.user)
        create_folder("Archive", user=self.user)

        # Rename
        renamed = rename_file(doc["id"], "FinalDraft.txt", user=self.user)
        self.assertEqual(renamed["name"], "FinalDraft.txt")

        # Copy
        copied = copy_file(doc["id"], "Archive", user=self.user)
        self.assertEqual(copied["folder"], "Archive")
        self.assertEqual(UploadedFile.objects.filter(user=self.user).count(), 2)

        # Move
        create_folder("Trash", user=self.user)
        moved = move_file(doc["id"], "Trash", user=self.user)
        self.assertEqual(moved["folder"], "Trash")

        # Delete
        del_res = delete_file(doc["id"], user=self.user)
        self.assertEqual(del_res["id"], doc["id"])
        self.assertEqual(UploadedFile.objects.filter(user=self.user).count(), 1)

    def test_create_and_extract_zip(self):
        doc1 = create_document(title="FileOne", content="Content one", format="txt", user=self.user)
        doc2 = create_document(title="FileTwo", content="Content two", format="txt", user=self.user)

        zip_res = create_zip([doc1["id"], doc2["id"]], "archive", user=self.user)
        self.assertEqual(zip_res["zip_name"], "archive.zip")
        self.assertEqual(zip_res["name"], "archive.zip")
        self.assertTrue(Path(zip_res["path"]).exists())

        extract_res = extract_zip("archive.zip", "Extracted", user=self.user)
        self.assertEqual(extract_res["count"], 2)
        self.assertEqual(extract_res["destination_folder"], "Extracted")

    def test_pdf_operations(self):
        # Create two PDF documents
        pdf1 = create_document(title="PDFOne", content="# PDF One\nBody 1", format="pdf", user=self.user)
        pdf2 = create_document(title="PDFTwo", content="# PDF Two\nBody 2", format="pdf", user=self.user)

        # Merge
        merged = merge_pdfs([pdf1["id"], pdf2["id"]], "Combined.pdf", user=self.user)
        self.assertEqual(merged["merged_count"], 2)
        self.assertTrue(Path(merged["path"]).exists())

        # Split
        split_res = split_pdf(merged["id"], page_ranges="1", user=self.user)
        self.assertEqual(split_res["page_count"], 1)

        # Watermark
        wm_res = watermark_pdf(pdf1["id"], watermark_text="TEST", user=self.user)
        self.assertEqual(wm_res["status"], "success")

        # Protect & Unlock
        prot_res = protect_pdf(pdf1["id"], password="pass", user=self.user)
        self.assertEqual(prot_res["status"], "success")

        unlock_res = unlock_pdf(prot_res["id"], password="pass", user=self.user)
        self.assertEqual(unlock_res["status"], "success")

    def test_duplicates_and_stats(self):
        doc1 = create_document(title="Original", content="Some content", format="txt", user=self.user)
        create_folder("Backup", user=self.user)
        copy_file(doc1["id"], "Backup", user=self.user)

        dups = find_duplicates(user=self.user)
        self.assertEqual(dups["status"], "success")
        self.assertGreaterEqual(dups["group_count"], 1)

        stats = workspace_stats(user=self.user)
        self.assertEqual(stats["status"], "success")
        self.assertGreaterEqual(stats["total_files"], 2)

    def test_file_tools_api_view(self):
        # Test GET
        res_get = self.client.get("/api/files/tools/")
        self.assertEqual(res_get.status_code, status.HTTP_200_OK)

        # Test POST create folder
        res_post = self.client.post("/api/files/tools/", {"action": "create_folder", "folder_name": "APIFolder"})
        self.assertEqual(res_post.status_code, status.HTTP_200_OK)
        self.assertEqual(res_post.data.get("folder_name"), "APIFolder")

        # Test POST list folders
        res_folders = self.client.post("/api/files/tools/", {"action": "list_folders"})
        self.assertEqual(res_folders.status_code, status.HTTP_200_OK)
        self.assertTrue(any(f["name"] == "APIFolder" for f in res_folders.data))
