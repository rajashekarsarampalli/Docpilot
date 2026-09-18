"""
Management command to run the DocPilot Telegram bot.

Usage:
    python manage.py run_bot

Make sure TELEGRAM_BOT_TOKEN is set in your .env file.
"""

import os
import sys
from dotenv import load_dotenv
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Start the DocPilot Telegram bot (long-polling)"

    def handle(self, *args, **options):
        # Allow synchronous ORM calls inside async telegram handlers
        os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

        load_dotenv()

        from telegram_bot.bot import build_app

        self.stdout.write(self.style.SUCCESS("[*] Initializing DocPilot Telegram bot..."))
        try:
            app = build_app()
        except RuntimeError as err:
            raise CommandError(str(err))

        self.stdout.write(self.style.SUCCESS("[+] Bot is running (polling updates). Press Ctrl+C to stop."))
        try:
            app.run_polling(drop_pending_updates=True)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n[-] DocPilot Telegram bot stopped."))
