"""
attendance/alerts.py
---------------------
Sends email (SMTP) or SMS (Twilio) notifications for absent students.

Configure via environment variables:

  Email:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_FROM, ALERT_TO

  SMS (Twilio):
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertService:
    def notify_absences(self, class_id: str, absent_names: list[str], date: str):
        if not absent_names:
            return

        subject = f"Attendance Alert — {class_id} ({date})"
        body = (
            f"The following students were absent from {class_id} on {date}:\n\n"
            + "\n".join(f"  • {name}" for name in absent_names)
            + "\n\nPlease follow up as required."
        )

        self._send_email(subject, body)
        self._send_sms(f"{subject}\n{body}")

    # ── Email ────────────────────────────────────────────────────────────
    def _send_email(self, subject: str, body: str):
        host  = os.environ.get("SMTP_HOST")
        port  = int(os.environ.get("SMTP_PORT", 587))
        user  = os.environ.get("SMTP_USER")
        pwd   = os.environ.get("SMTP_PASS")
        from_ = os.environ.get("ALERT_FROM", user)
        to_   = os.environ.get("ALERT_TO")

        if not all([host, user, pwd, to_]):
            print("[Alert] Email not configured — skipping.")
            return

        try:
            msg                    = MIMEMultipart()
            msg["Subject"]         = subject
            msg["From"]            = from_
            msg["To"]              = to_
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, pwd)
                server.sendmail(from_, to_, msg.as_string())
            print(f"[Alert] Email sent to {to_}")
        except Exception as e:
            print(f"[Alert] Email failed: {e}")

    # ── SMS (Twilio) ──────────────────────────────────────────────────────
    def _send_sms(self, message: str):
        sid   = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_ = os.environ.get("TWILIO_FROM")
        to_   = os.environ.get("TWILIO_TO")

        if not all([sid, token, from_, to_]):
            print("[Alert] Twilio not configured — skipping SMS.")
            return

        try:
            from twilio.rest import Client
            client = Client(sid, token)
            client.messages.create(body=message[:1600], from_=from_, to=to_)
            print(f"[Alert] SMS sent to {to_}")
        except Exception as e:
            print(f"[Alert] SMS failed: {e}")
