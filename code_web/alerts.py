"""One SMS plus one call per fire. Twilio runs only when it is configured."""

from __future__ import annotations

import logging
import os
import threading

from config import TWILIO_ENV_KEYS

logger = logging.getLogger("firevision.alerts")

CALL_TWIML = (
    "<Response><Say>Warning! Fire or Smoke detected. "
    "Please take immediate action.</Say></Response>"
)


def twilio_configured(env: os._Environ[str] | dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return all(source.get(key) for key in TWILIO_ENV_KEYS)


def format_alert_body(fire_detected: bool, smoke_detected: bool) -> str:
    message_body = "🔥 FIRE/SMOKE ALERT! 🔥\n"
    if fire_detected:
        message_body += "🚨 Fire detected and spreading!\n"
    if smoke_detected:
        message_body += "⚠️ Smoke detected, moving in a dangerous direction!\n"
    return message_body


class AlertSender:
    """Send the SMS/call pair, or log the same body when Twilio is unset."""

    def __init__(self, env=None, client_factory=None):
        self.env = os.environ if env is None else env
        self.client_factory = client_factory or _default_client_factory
        self.events: list[dict] = []

    def configured(self) -> bool:
        return twilio_configured(self.env)

    def send(self, fire_detected: bool, smoke_detected: bool) -> dict:
        body = format_alert_body(fire_detected, smoke_detected)
        if not self.configured():
            record = {
                "mode": "demo",
                "sms": body,
                "call": CALL_TWIML,
                "fire": fire_detected,
                "smoke": smoke_detected,
            }
            self.events.append(record)
            logger.info("demo alert (Twilio not configured): %s", body.replace("\n", " | "))
            return record

        try:
            client = self.client_factory(
                self.env["TWILIO_ACCOUNT_SID"],
                self.env["TWILIO_AUTH_TOKEN"],
            )
            message = client.messages.create(
                body=body,
                from_=self.env["TWILIO_PHONE"],
                to=self.env["ALERT_PHONE"],
            )
            call = client.calls.create(
                twiml=CALL_TWIML,
                from_=self.env["TWILIO_PHONE"],
                to=self.env["ALERT_PHONE"],
            )
            record = {
                "mode": "live",
                "sms": body,
                "sms_sid": getattr(message, "sid", ""),
                "call_sid": getattr(call, "sid", ""),
                "fire": fire_detected,
                "smoke": smoke_detected,
            }
            logger.info(
                "twilio alert sent once (sms=%s call=%s)", record["sms_sid"], record["call_sid"]
            )
        except Exception as exc:
            logger.warning("twilio alert failed once and will not retry: %s", exc)
            record = {
                "mode": "error",
                "sms": body,
                "error": str(exc)[:200],
                "fire": fire_detected,
                "smoke": smoke_detected,
            }
        self.events.append(record)
        return record


class AlertGate:
    """Allow a single alert for a fire, then ignore the rest of that video."""

    def __init__(self, sender: AlertSender):
        self.sender = sender
        self.alerted = False
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.alerted = False

    def consider(self, fire_detected: bool, smoke_detected: bool = False):
        with self._lock:
            if self.alerted or not fire_detected:
                return None
            self.alerted = True
        return self.sender.send(True, bool(smoke_detected))


def _default_client_factory(account_sid: str, auth_token: str):
    from twilio.rest import Client

    return Client(account_sid, auth_token)
