"""The Twilio alert path must never crash the video pipeline."""
from alerts import AlertManager

CREDS = {"TWILIO_ACCOUNT_SID": "AC123", "TWILIO_AUTH_TOKEN": "tok", "TWILIO_PHONE": "+1000", "ALERT_PHONE": "+1999"}


class FakeClient:
    def __init__(self, fail=False):
        self.sent, self.calls_made, self.fail = [], [], fail
        outer = self

        class _Messages:
            def create(self, **kw):
                if outer.fail:
                    raise RuntimeError("twilio down")
                outer.sent.append(kw)
                return type("M", (), {"sid": "SM1"})()

        class _Calls:
            def create(self, **kw):
                if outer.fail:
                    raise RuntimeError("twilio down")
                outer.calls_made.append(kw)
                return type("C", (), {"sid": "CA1"})()

        self.messages, self.calls = _Messages(), _Calls()


def test_missing_credentials_is_a_noop():
    def boom(*a):
        raise AssertionError("client must not be created without credentials")

    m = AlertManager(env={}, client_factory=boom)
    assert m.enabled is False
    assert m.alert(fire_detected=True, smoke_detected=True) is False
    assert m.send_sms_alert(True, False) is False and m.make_call_alert() is False


def test_partial_credentials_is_a_noop():
    env = dict(CREDS)
    del env["ALERT_PHONE"]
    assert AlertManager(env=env).enabled is False


def test_sends_sms_and_call_once_per_session():
    fake = FakeClient()
    m = AlertManager(env=CREDS, client_factory=lambda sid, tok: fake)
    assert m.alert(True, False) is True
    assert m.alert(True, True) is False  # debounced
    assert len(fake.sent) == 1 and len(fake.calls_made) == 1
    assert "Fire detected" in fake.sent[0]["body"] and fake.sent[0]["to"] == "+1999"
    m.reset()
    assert m.alert(False, True) is True and "Smoke detected" in fake.sent[1]["body"]


def test_twilio_errors_are_swallowed_and_retried_later():
    fake = FakeClient(fail=True)
    m = AlertManager(env=CREDS, client_factory=lambda sid, tok: fake)
    assert m.alert(True, False) is False
    assert m.triggered is False  # not marked as sent, so the next detection retries
    fake.fail = False
    assert m.alert(True, False) is True


def test_client_construction_error_is_swallowed():
    def broken(*a):
        raise ImportError("twilio not installed")

    m = AlertManager(env=CREDS, client_factory=broken)
    assert m.alert(True, True) is False


def test_kill_switch():
    assert AlertManager(env={**CREDS, "ALERTS_ENABLED": "0"}).enabled is False


def test_no_detection_no_alert():
    fake = FakeClient()
    m = AlertManager(env=CREDS, client_factory=lambda sid, tok: fake)
    assert m.alert(False, False) is False and not fake.sent
