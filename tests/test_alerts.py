from alerts import CALL_TWIML, AlertGate, AlertSender, format_alert_body


class Boom:
    def __init__(self):
        self.calls = 0

    def messages(self):
        raise AssertionError("demo mode must not build a Twilio client")

    def __getattr__(self, name):
        raise AssertionError(name)


class FakeMessages:
    def __init__(self):
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return type("Msg", (), {"sid": "SM_TEST"})()


class FakeCalls:
    def __init__(self):
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return type("Call", (), {"sid": "CA_TEST"})()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()
        self.calls = FakeCalls()


def test_one_alert_per_fire_and_reset_allows_the_next():
    sender = AlertSender(env={})
    gate = AlertGate(sender)
    assert gate.consider(False, True) is None
    for _ in range(5):
        gate.consider(True, False)
    assert len(sender.events) == 1
    assert sender.events[0]["mode"] == "demo"
    assert "Fire detected" in sender.events[0]["sms"]
    assert sender.events[0]["call"] == CALL_TWIML
    gate.reset()
    gate.consider(True, True)
    assert len(sender.events) == 2
    assert "Smoke detected" in sender.events[1]["sms"]


def test_demo_mode_does_not_construct_twilio(monkeypatch):
    def forbid(_sid, _token):
        raise AssertionError("Twilio client constructed")

    monkeypatch.setattr("alerts._default_client_factory", forbid)
    sender = AlertSender(env={})
    sender.send(True, False)
    assert sender.events[0]["mode"] == "demo"


def test_live_mode_sends_sms_and_call_once():
    env = {
        "TWILIO_ACCOUNT_SID": "AC_TEST",
        "TWILIO_AUTH_TOKEN": "token",
        "TWILIO_PHONE": "+10000000000",
        "ALERT_PHONE": "+10000000001",
    }
    client = FakeClient()
    sender = AlertSender(env=env, client_factory=lambda _sid, _token: client)
    gate = AlertGate(sender)
    gate.consider(True, False)
    gate.consider(True, False)
    assert len(client.messages.created) == 1
    assert len(client.calls.created) == 1
    assert client.messages.created[0]["from_"] == "+10000000000"
    assert client.calls.created[0]["to"] == "+10000000001"
    assert sender.events[0]["mode"] == "live"
    assert sender.events[0]["sms_sid"] == "SM_TEST"


def test_a_twilio_failure_is_not_retried_every_frame():
    env = {
        "TWILIO_ACCOUNT_SID": "AC_TEST",
        "TWILIO_AUTH_TOKEN": "token",
        "TWILIO_PHONE": "+10000000000",
        "ALERT_PHONE": "+10000000001",
    }
    attempts = {"n": 0}

    def factory(_sid, _token):
        attempts["n"] += 1
        raise RuntimeError("network down")

    sender = AlertSender(env=env, client_factory=factory)
    gate = AlertGate(sender)
    for _ in range(4):
        gate.consider(True, False)
    assert attempts["n"] == 1
    assert sender.events[0]["mode"] == "error"


def test_alert_copy_matches_the_original_wording():
    body = format_alert_body(True, True)
    assert body.startswith("🔥 FIRE/SMOKE ALERT! 🔥")
    assert "🚨 Fire detected and spreading!" in body
    assert "⚠️ Smoke detected, moving in a dangerous direction!" in body


def test_pipeline_alerts_once_for_a_clip(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from alerts import AlertGate, AlertSender
    from jobs import Runner
    from processing import process_video

    monkeypatch.setenv("FRAME_STRIDE", "1")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    assert writer.isOpened()
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for _ in range(6):
        writer.write(frame)
    writer.release()

    def fake_yolo(image, session=None, conf=None, iou=None):
        return image, [[1, 1, 20, 20]], [0]

    sender = AlertSender(env={})
    gate = AlertGate(sender)
    events = []
    stats = process_video(
        path,
        run_motion=True,
        run_yolo_detection=True,
        size_factor="1.5",
        m_frames=2,
        n_frames=2,
        emit=lambda event, data: events.append(event),
        runner=Runner(),
        gate=gate,
        delete_after=True,
        yolo_runner=fake_yolo,
    )
    assert stats["fired_alert"] is True
    assert len(sender.events) == 1
    assert not path.exists()
    assert "stable_update" in events
    assert "yolo_update" in events
    # Same fire, still one alert.
    path.write_bytes(b"")  # replaced below
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    for _ in range(4):
        writer.write(frame)
    writer.release()
    process_video(
        path,
        run_motion=False,
        run_yolo_detection=True,
        size_factor=1,
        m_frames=1,
        n_frames=1,
        emit=lambda event, data: None,
        runner=Runner(),
        gate=gate,
        yolo_runner=fake_yolo,
    )
    assert len(sender.events) == 1
    gate.reset()
    process_video(
        path,
        run_motion=False,
        run_yolo_detection=True,
        size_factor=1,
        m_frames=1,
        n_frames=1,
        emit=lambda event, data: None,
        runner=Runner(),
        gate=gate,
        delete_after=True,
        yolo_runner=fake_yolo,
    )
    assert len(sender.events) == 2
    assert not path.exists()
