"""Notification services for alerts"""
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class Notifier:
    """Base notifier class"""

    def send(self, subject: str, message: str) -> bool:
        """Send notification. Returns True if successful."""
        raise NotImplementedError


class EmailNotifier(Notifier):
    """Send alerts via Gmail SMTP"""

    def __init__(self):
        self.username = os.environ.get("EMAIL_USERNAME")
        self.password = os.environ.get("EMAIL_PASSWORD")
        self.to_email = os.environ.get("EMAIL_TO")

    def is_configured(self) -> bool:
        return all([self.username, self.password, self.to_email])

    def send(self, subject: str, message: str) -> bool:
        if not self.is_configured():
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = f"Stock Alert Bot <{self.username}>"
            msg["To"] = self.to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Email send failed: {e}")
            return False


class TelegramNotifier(Notifier):
    """Send alerts via Telegram Bot"""

    def __init__(self):
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return all([self.bot_token, self.chat_id])

    def send(self, subject: str, message: str) -> bool:
        if not self.is_configured():
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            text = f"*{subject}*\n\n{message}"

            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=30)

            return resp.status_code == 200
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False


class MultiNotifier(Notifier):
    """Send to all configured notification channels"""

    def __init__(self):
        self.notifiers = []

        email = EmailNotifier()
        if email.is_configured():
            self.notifiers.append(("Email", email))

        telegram = TelegramNotifier()
        if telegram.is_configured():
            self.notifiers.append(("Telegram", telegram))

    def send(self, subject: str, message: str) -> bool:
        if not self.notifiers:
            print("No notifiers configured")
            return False

        success = False
        for name, notifier in self.notifiers:
            if notifier.send(subject, message):
                print(f"Sent via {name}")
                success = True
            else:
                print(f"Failed to send via {name}")

        return success
