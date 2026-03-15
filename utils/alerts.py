"""
utils/alerts.py — Email + SMS repeat-violator alert system (Sprint 3).

Tracks per-plate violation counts and fires SMTP email + Twilio SMS
when a plate reaches the configured threshold in a session.

Design:
  • Threshold logic is pure Python — fully testable without credentials
  • Alert fires exactly once per plate per session (no duplicate spam)
  • SMTP and Twilio are independent — one can fail without affecting the other
  • All credentials loaded from Settings (which reads env vars)
"""
from __future__ import annotations
import logging
import smtplib
import threading
from collections import defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("AlertSystem")


class AlertSystem:
    """
    Tracks violations per plate and fires alerts at threshold.

    Parameters
    ----------
    threshold : int         — violations before alert fires
    settings  : Settings | None — runtime config with SMTP/Twilio credentials.
                                  Pass None for test/dry-run mode.
    """

    def __init__(self, threshold: int = 3, settings=None):
        self.threshold = threshold
        self.cfg       = settings
        self._counts:  dict[str, int] = defaultdict(int)
        self._alerted: set[str]       = set()
        self._lock     = threading.Lock()

    # ── Public API ────────────────────────────────────────────────

    def record(self, plate: str, violation: str) -> bool:
        """
        Record one violation event for *plate*.

        Returns True the FIRST time threshold is reached (alert dispatched).
        Returns False for all subsequent calls after the alert has fired.
        """
        key = f"{plate}:{violation}"
        with self._lock:
            if key in self._alerted:
                return False
            self._counts[key] += 1
            if self._counts[key] >= self.threshold:
                self._alerted.add(key)
                self._dispatch(plate, violation, self._counts[key])
                return True
        return False

    def reset_plate(self, plate: str):
        """Clear all violation history for *plate* (e.g. vehicle left scene)."""
        with self._lock:
            for k in [k for k in self._counts if k.startswith(f"{plate}:")]:
                del self._counts[k]
                self._alerted.discard(k)

    def get_stats(self) -> dict:
        """Return current counts for API / reporting."""
        with self._lock:
            return {
                "tracked_plates"  : len({k.split(":")[0] for k in self._counts}),
                "alerted_plates"  : len(self._alerted),
                "violation_counts": dict(self._counts),
            }

    # ── Internal dispatch ─────────────────────────────────────────

    def _dispatch(self, plate: str, violation: str, count: int):
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"[ANPR ALERT] Repeat Violator — {plate}"
        body    = (
            f"Repeat Violation Alert\n\n"
            f"  Plate     : {plate}\n"
            f"  Violation : {violation}\n"
            f"  Count     : {count} times this session\n"
            f"  Timestamp : {ts}\n\n"
            f"Plate flagged {count} times. Please take action.\n\n"
            f"— Smart City ANPR System (SRM IST)"
        )
        log.warning(f"ALERT: {plate} | {violation} × {count}")

        if self.cfg is None:
            return   # dry-run / test mode — no sends

        if self.cfg.alert_email_to and self.cfg.alert_smtp_user:
            threading.Thread(target=self._send_email,
                             args=(subject, body), daemon=True).start()

        if self.cfg.alert_twilio_sid and self.cfg.alert_sms_to:
            sms = (f"[ANPR ALERT] Repeat violator {plate}: "
                   f"{violation} x{count} at {ts}")
            threading.Thread(target=self._send_sms,
                             args=(sms,), daemon=True).start()

    def _send_email(self, subject: str, body: str):
        cfg = self.cfg
        try:
            msg             = MIMEMultipart()
            msg["From"]     = cfg.alert_email_from or cfg.alert_smtp_user
            msg["To"]       = cfg.alert_email_to
            msg["Subject"]  = subject
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(cfg.alert_smtp_host, cfg.alert_smtp_port) as s:
                s.ehlo(); s.starttls()
                s.login(cfg.alert_smtp_user, cfg.alert_smtp_pass)
                s.sendmail(msg["From"], [msg["To"]], msg.as_string())
            log.info(f"  ✓ Email sent → {cfg.alert_email_to}")
        except Exception as e:
            log.error(f"Email failed: {e}")

    def _send_sms(self, body: str):
        cfg = self.cfg
        try:
            from twilio.rest import Client
            Client(cfg.alert_twilio_sid, cfg.alert_twilio_token).messages.create(
                body=body, from_=cfg.alert_sms_from, to=cfg.alert_sms_to)
            log.info("  ✓ SMS sent")
        except ImportError:
            log.warning("twilio not installed — SMS skipped")
        except Exception as e:
            log.error(f"SMS failed: {e}")
