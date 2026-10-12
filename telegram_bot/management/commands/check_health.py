"""
Management command to run a comprehensive health check on external APIs and services:
- Telegram Bot API
- Groq LLM API
- Django Database
- ILovePDF configuration

Usage:
    python manage.py check_health
"""

import os
import sys
import requests
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from django.db import connection

load_dotenv()


class Command(BaseCommand):
    help = "Check health and connectivity of Telegram Bot, Groq LLM API, and Database"

    def handle(self, *args, **options):
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

        self.stdout.write(self.style.MIGRATE_HEADING("\n=== DOCPILOT SYSTEM HEALTH CHECK ==="))
        all_passed = True

        # 1. Database Check
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            self.stdout.write(self.style.SUCCESS("[✓] Database: Operational and connected."))
        except Exception as e:
            all_passed = False
            self.stdout.write(self.style.ERROR(f"[✗] Database: Error - {e}"))

        # 2. Telegram Bot Token Check
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not tg_token:
            all_passed = False
            self.stdout.write(self.style.ERROR("[✗] Telegram Bot: TELEGRAM_BOT_TOKEN missing in .env"))
        else:
            try:
                resp = requests.get(f"https://api.telegram.org/bot{tg_token}/getMe", timeout=10)
                data = resp.json()
                if data.get("ok"):
                    bot_user = data.get("result", {}).get("username")
                    bot_name = data.get("result", {}).get("first_name")
                    self.stdout.write(
                        self.style.SUCCESS(f"[✓] Telegram Bot: Connected as @{bot_user} ({bot_name})")
                    )
                else:
                    all_passed = False
                    self.stdout.write(self.style.ERROR(f"[✗] Telegram Bot: {data.get('description')}"))
            except Exception as e:
                all_passed = False
                self.stdout.write(self.style.ERROR(f"[✗] Telegram Bot: Connection failed - {e}"))

        # 3. Groq LLM API Check
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            all_passed = False
            self.stdout.write(self.style.ERROR("[✗] Groq LLM: GROQ_API_KEY missing in .env"))
        else:
            try:
                from groq import Groq
                from agent.llm import MODEL

                client = Groq(api_key=groq_key)
                models = [m.id for m in client.models.list().data]
                if MODEL in models:
                    model_status = f"Model '{MODEL}' is available."
                else:
                    model_status = f"Warning: Configured model '{MODEL}' not in models list."

                # Send a lightweight ping
                resp = client.chat.completions.create(
                    model=MODEL if MODEL in models else "llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "Respond 'OK'"}],
                    max_tokens=10,
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"[✓] Groq LLM: Operational ({model_status}) -> Response: {resp.choices[0].message.content.strip()}"
                    )
                )
            except Exception as e:
                all_passed = False
                self.stdout.write(self.style.ERROR(f"[✗] Groq LLM: Connection failed - {e}"))

        # 4. ILovePDF Configuration Check
        pub_key = os.getenv("ILOVEPDF_PUBLIC_KEY")
        sec_key = os.getenv("ILOVEPDF_SECRET_KEY")
        if pub_key and sec_key:
            self.stdout.write(self.style.SUCCESS("[✓] ILovePDF: Credentials configured in environment."))
        else:
            self.stdout.write(self.style.WARNING("[-] ILovePDF: Not fully configured (using local PDF engine)."))

        self.stdout.write(self.style.MIGRATE_HEADING("=====================================\n"))

        if all_passed:
            self.stdout.write(self.style.SUCCESS("All primary services are operational!"))
        else:
            self.stdout.write(self.style.WARNING("One or more services reported warnings or errors."))
