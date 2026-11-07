"""Progress display checks. These use no generated audio or flight data."""
import os
from pathlib import Path
import threading
import time

import pytest

from dopplergeist import automatic, jobs


def _job(tmp_path):
    directory = tmp_path / "0123456789abcdef0123456789abcdef"
    directory.mkdir()
    jobs.write_json(directory / "request.json", {})
    jobs.write_json(directory / "operator.json", {})
    return directory


def _age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_dead_worker_turns_stale_and_elapsed_time_is_reported(tmp_path):
    directory = _job(tmp_path)
    jobs.result(directory, "running", stage="discovery", progress=0.2, message="Discovery: tracking.")
    _age(directory / "request.json", 300)
    fresh = jobs.status_for_display(directory)
    assert fresh["progress_stale"] is False and fresh["progress"] == 0.2
    assert fresh["elapsed_seconds"] >= 299
    _age(directory / "result.json", 120)
    stale = jobs.status_for_display(directory)
    assert stale["progress_stale"] is True and stale["progress"] is None


def test_heartbeat_rewrites_only_the_latest_status(tmp_path):
    directory = _job(tmp_path)
    lock, latest = threading.Lock(), {}
    value = jobs.result(directory, "running", stage="discovery", progress=0.25, message="Discovery: tracking.")
    latest.update(value=value, time=time.monotonic())
    heartbeat = jobs._Heartbeat(directory, lock, latest, 30.0)
    assert heartbeat.beat() is False  # a recent update needs no heartbeat
    latest["time"] -= 60
    _age(directory / "result.json", 120)
    assert heartbeat.beat() is True
    shown = jobs.status_for_display(directory)
    assert shown["progress_stale"] is False
    assert {key: shown[key] for key in value} == value
    latest["time"] -= 60
    (directory / "cancel").touch()
    assert heartbeat.beat() is False
    (directory / "cancel").unlink()
    heartbeat.stop()
    assert heartbeat.beat() is False


def test_worker_heartbeat_keeps_silent_step_fresh_and_never_overwrites_final(tmp_path, monkeypatch):
    directory = _job(tmp_path)
    monkeypatch.setattr(jobs, "HEARTBEAT_INTERVAL_S", 0.2)
    observed = {}

    def silent_step(request, output, config, progress, resume=False):
        progress("discovery", 0.25, message="Discovery: tracking Doppler ridges across the recording.")
        _age(output / "result.json", 120)
        time.sleep(0.6)  # a long call that reports nothing itself
        observed["shown"] = jobs.status_for_display(output, stale_after_s=0.5)
        return {"status": "inconclusive", "reason_code": "no_supported_events", "message": "none"}

    monkeypatch.setattr(automatic, "analyze", silent_step)
    jobs.execute(directory)
    assert observed["shown"]["progress_stale"] is False
    assert observed["shown"]["stage"] == "discovery" and observed["shown"]["progress"] == 0.25
    time.sleep(0.5)
    assert jobs.read_json(directory / "result.json")["status"] == "inconclusive"


def test_acquisition_moves_within_its_band():
    assert automatic._acquisition_fraction({"phase": "downloading", "bytes_done": 0, "bytes_total": 100}) == 0.35
    assert automatic._acquisition_fraction({"phase": "downloading", "bytes_done": 50, "bytes_total": 100}) == pytest.approx(0.40)
    assert automatic._acquisition_fraction({"phase": "downloading", "bytes_done": 7}) == 0.35
    ordered = ["decoding_archive", "creating_flight_index", "staging_samples", "creating_sample_lookup",
               "building_segments", "preparing_tracks", "compacting_flight_index", "flight_index_ready"]
    fractions = [automatic._acquisition_fraction({"phase": phase}) for phase in ordered]
    assert fractions == sorted(fractions) and 0.45 <= fractions[0] and fractions[-1] < 0.55


def test_decode_progress_and_throttle():
    fraction, message = automatic._decode_progress(0.05, 0.14, "Decoding audio", decoded_s=90, duration_s=180)
    assert fraction == pytest.approx(0.095) and message == "Decoding audio: 1:30 of 3:00 decoded."
    assert automatic._decode_progress(0.05, 0.14, "Decoding audio", decoded_s=4000)[0] == 0.05
    calls = []
    forward = automatic._throttled(lambda **details: calls.append(details), interval_s=60)
    forward(phase="decoding_high_rate", decoded_s=1)
    forward(phase="decoding_high_rate", decoded_s=2)
    forward(phase="tracking_ridges")
    assert [call.get("phase") for call in calls] == ["decoding_high_rate", "tracking_ridges"]


def test_intake_progress_matches_the_job_progress_signature(tmp_path, monkeypatch):
    from dopplergeist import intake

    def report_then_stop(path, output, *, progress_callback=None, **_):
        progress_callback(decoded_s=0.0, duration_s=180.0)
        raise intake.AudioIntakeError("stop_after_progress", "stopped after the first report")

    monkeypatch.setattr(intake, "intake_recording", report_then_stop)
    reports = []

    def job_progress(stage, fraction, **fields):  # same shape as jobs.execute's progress
        reports.append((stage, fraction, fields.get("message")))

    placeholder = tmp_path / "input.m4a"
    placeholder.touch()  # only hashed; intake is replaced above, so nothing is decoded
    final = automatic.analyze({"schema_version": 1, "audio_path": str(placeholder)}, tmp_path / "job", {}, job_progress)
    assert final["reason_code"] == "stop_after_progress"
    assert reports[-1] == ("intake", 0.05, "Decoding audio: 0:00 of 3:00 decoded.")


REAL_RECORDING = os.environ.get("DOPPLERGEIST_REAL_RECORDING")


@pytest.mark.skipif(not REAL_RECORDING, reason="set DOPPLERGEIST_REAL_RECORDING to a real recording")
def test_intake_reports_decoding_progress_on_a_real_recording(tmp_path):
    from dopplergeist.intake import intake_recording

    reports = []
    recording = intake_recording(Path(REAL_RECORDING), tmp_path / "intake",
                                 progress_callback=lambda **details: reports.append(details))
    assert reports[0]["decoded_s"] == 0
    decoded = [report["decoded_s"] for report in reports]
    assert decoded == sorted(decoded) and decoded[-1] <= recording.metadata["duration_s"]
    assert reports[0]["duration_s"] == pytest.approx(recording.metadata["duration_s"], rel=0.02)


def test_elapsed_time_starts_when_worker_starts(tmp_path, monkeypatch):
    directory = _job(tmp_path)
    _age(directory / "request.json", 300)
    observed = {}

    def silent_step(request, output, config, progress, resume=False):
        progress("discovery", 0.1)
        observed["elapsed"] = jobs.status_for_display(output)["elapsed_seconds"]
        return {"status": "inconclusive", "reason_code": "no_supported_events", "message": "none"}

    monkeypatch.setattr(automatic, "analyze", silent_step)
    jobs.execute(directory)
    assert observed["elapsed"] < 5
