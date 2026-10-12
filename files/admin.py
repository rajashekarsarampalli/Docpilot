from django.contrib import admin
from .models import Folder, UploadedFile


@admin.register(Folder)
class FolderAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "parent", "created_at")
    list_filter = ("user",)
    search_fields = ("name", "user__username")


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ("original_name", "user", "folder", "uploaded_at")
    list_filter = ("user", "folder")
    search_fields = ("original_name", "user__username")

