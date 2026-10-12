from django.test import TestCase
from django.contrib.auth.models import User
from telegram_bot.models import TelegramUser
from telegram_bot.bot import (
    format_telegram_markdown,
    strip_markdown,
    build_file_action_keyboard,
    _format_size,
)


class TelegramBotTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tguser", password="password123")

    def test_telegram_user_model(self):
        tg_user = TelegramUser.objects.create(
            chat_id=12345678,
            telegram_username="tg_username",
            first_name="Test",
            user=self.user,
            is_authenticated=True,
        )
        self.assertEqual(tg_user.chat_id, 12345678)
        self.assertIn("Authed: tguser", str(tg_user))

    def test_markdown_formatting(self):
        raw = "### Heading\n**Bold text** with `code_snippet` and file_name_test."
        formatted = format_telegram_markdown(raw)
        self.assertIn("📌 *Heading*", formatted)
        self.assertIn("*Bold text*", formatted)

        stripped = strip_markdown("### Title\n**Bold** `code`")
        self.assertEqual(stripped, "Title\nBold code")

    def test_format_size(self):
        self.assertEqual(_format_size(500), "500 B")
        self.assertEqual(_format_size(2048), "2.0 KB")
        self.assertEqual(_format_size(1048576 * 3), "3.00 MB")

    def test_build_keyboard(self):
        kb_pdf = build_file_action_keyboard(1, "document.pdf")
        self.assertTrue(any("sum:1" in btn.callback_data for row in kb_pdf.inline_keyboard for btn in row))
        self.assertTrue(any("word:1" in btn.callback_data for row in kb_pdf.inline_keyboard for btn in row))

        kb_docx = build_file_action_keyboard(2, "notes.docx")
        self.assertTrue(any("pdf:2" in btn.callback_data for row in kb_docx.inline_keyboard for btn in row))
