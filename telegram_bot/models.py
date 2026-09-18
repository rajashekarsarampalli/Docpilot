from django.db import models
from django.contrib.auth.models import User


class TelegramUser(models.Model):
    chat_id = models.BigIntegerField(unique=True, db_index=True)
    telegram_username = models.CharField(max_length=255, null=True, blank=True)
    first_name = models.CharField(max_length=255, null=True, blank=True)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="telegram_sessions",
    )
    is_authenticated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        auth_status = f"Authed: {self.user.username}" if self.is_authenticated and self.user else "Unauthed"
        return f"Telegram [{self.chat_id}] ({auth_status})"
