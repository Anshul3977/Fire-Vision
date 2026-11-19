"""Twilio SMS / call alerts that degrade to a no-op without credentials.

Previously `Client(ACCOUNT_SID, AUTH_TOKEN)` was built outside the try block in
app.py, so a missing TWILIO_* env var (or no `twilio` package) crashed the
video-processing thread the first time fire was seen. Now every failure path
returns False and logs, and alerts fire at most once per session.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

REQUIRED_VARS = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE", "ALERT_PHONE")


class AlertManager:
    def __init__(self, env: dict | None = None, client_factory=None):
        env = os.environ if env is None else env
        self.sid = env.get("TWILIO_ACCOUNT_SID")
        self.token = env.get("TWILIO_AUTH_TOKEN")
        self.from_phone = env.get("TWILIO_PHONE")
        self.to_phone = env.get("ALERT_PHONE")
        self.enabled = env.get("ALERTS_ENABLED", "1") != "0" and all(env.get(v) for v in REQUIRED_VARS)
        self._client_factory = client_factory
        self._client = None
        self.triggered = False
        if not self.enabled:
            log.info("Twilio alerts disabled (missing credentials or ALERTS_ENABLED=0); alerts are a no-op.")

    def reset(self) -> None:
        self.triggered = False

    def _get_client(self):
        if self._client is None:
            factory = self._client_factory
            if factory is None:
                from twilio.rest import Client as factory  # imported lazily: optional dependency
            self._client = factory(self.sid, self.token)
        return self._client

    @staticmethod
    def message(fire_detected: bool, smoke_detected: bool) -> str:
        body = "FIRE/SMOKE ALERT!\n"
        if fire_detected:
            body += "Fire detected and spreading!\n"
        if smoke_detected:
            body += "Smoke detected, moving in a dangerous direction!\n"
        return body

    def send_sms_alert(self, fire_detected: bool, smoke_detected: bool) -> bool:
        if not self.enabled or self.triggered:
            return False
        try:
            msg = self._get_client().messages.create(
                body=self.message(fire_detected, smoke_detected), from_=self.from_phone, to=self.to_phone
            )
            log.info("SMS alert sent: %s", getattr(msg, "sid", "?"))
            return True
        except Exception as exc:  # network, auth, missing package...
            log.warning("Failed to send SMS alert: %s", exc)
            return False

    def make_call_alert(self) -> bool:
        if not self.enabled or self.triggered:
            return False
        try:
            call = self._get_client().calls.create(
                twiml="<Response><Say>Warning! Fire or Smoke detected. Please take immediate action.</Say></Response>",
                from_=self.from_phone,
                to=self.to_phone,
            )
            log.info("Call alert triggered: %s", getattr(call, "sid", "?"))
            return True
        except Exception as exc:
            log.warning("Failed to make call alert: %s", exc)
            return False

    def alert(self, fire_detected: bool, smoke_detected: bool) -> bool:
        """Send SMS + call once per session. Returns True if anything was delivered."""
        if self.triggered or not (fire_detected or smoke_detected):
            return False
        sent = self.send_sms_alert(fire_detected, smoke_detected)
        called = self.make_call_alert()
        if sent or called:
            self.triggered = True
        return sent or called
