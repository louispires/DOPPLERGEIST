"""Opt-in regression using an existing real discovery checkpoint, not audio synthesis."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dopplergeist import automatic
from dopplergeist.matching import UnreconciledHarmonicGroup, prepare_events


def test_unreconciled_real_ridges_stay_unmatched_without_stopping_search(tmp_path, monkeypatch):
    checkpoint = os.environ.get("DOPPLERGEIST_UNRECONCILED_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set DOPPLERGEIST_UNRECONCILED_CHECKPOINT to a real saved job")
    checkpoint = Path(checkpoint)
    request = json.loads((checkpoint / "request.json").read_text())
    metadata = json.loads((checkpoint / "intake/recording.json").read_text())
    discovery = (checkpoint / "discovery/discovery.json").read_bytes()
    events = json.loads(discovery)["events"]
    with pytest.raises(UnreconciledHarmonicGroup):
        prepare_events(events)

    unmatched = []
    prepared, measurements = prepare_events(events, unmatched=unmatched)
    assert prepared and len(prepared) == len(measurements)
    used = {event_id for event in prepared for event_id in event["event_ids"]}
    dropped = {event_id for item in unmatched for event_id in item["event_ids"]}
    assert not used & dropped
    assert used | dropped == {event["event_id"] for event in events}

    # Exercise the orchestration path using the saved measured ridges. The
    # bypassed clock gate is a software control; no flight data are configured,
    # so the job must stop at the data gate rather than the grouping gate.
    recording = SimpleNamespace(metadata=metadata, time_status="ready")
    monkeypatch.setattr(automatic, "_load_audio_checkpoint", lambda *args: (recording, events))
    (tmp_path / "discovery").mkdir()
    (tmp_path / "discovery/discovery.json").write_bytes(discovery)
    result = automatic.analyze(request, tmp_path, resume=True)
    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "data_unavailable"
    assert json.loads((tmp_path / result["artifacts"]["unmatched_proposals"]).read_text()) == unmatched
    assert (tmp_path / result["artifacts"]["discovery"]).read_bytes() == discovery
