from django.contrib import admin
from .models import TelegramUser


@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = ("chat_id", "telegram_username", "user", "is_authenticated", "updated_at")
    list_filter = ("is_authenticated",)
    search_fields = ("chat_id", "telegram_username", "user__username")
