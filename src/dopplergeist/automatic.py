"""Automatic stage orchestration; scientific producers own their algorithms."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

from .contracts import utc_timestamp, validate_candidate, validate_event, validate_request
from .geography import EmptySearchArea
from .jobs import write_json


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _audio_identity(request, config):
    modules = ("intake.py", "discovery.py", "advanced_discovery.py", "audio.py", "contracts.py")
    identity = {"request": request, "source_sha256": _digest(request["audio_path"]),
            "discovery_settings": config.get("discovery", {}),
            "modules": {name: _digest(Path(__file__).parent / name) for name in modules},
            "packages": {name: version(name) for name in ("numpy", "scipy", "numba")}}
    backend = os.environ.get("DOPPLERGEIST_MEDIAN_BACKEND", "cpu")
    if backend != "cpu":
        identity["median_backend"] = backend
        if backend == "cuda":
            from cupy import __version__ as cupy_version
            identity["cupy_version"] = cupy_version
    return identity


def _save_audio_checkpoint(output, request, config):
    files = ("intake/recording.f32", "intake/recording.json", "intake/intake.json", "discovery/discovery.json")
    identity = _audio_identity(request, config)
    metadata = json.loads((output / "intake/recording.json").read_text())
    discovery = json.loads((output / "discovery/discovery.json").read_text())
    if metadata["source_sha256"] != identity["source_sha256"] or discovery["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Audio changed while generating the checkpoint")
    write_json(output / "resume.json", {"version": 1, "stage": "audio_complete",
               "identity": identity,
               "files": {name: _digest(output / name) for name in files}})


def _load_audio_checkpoint(output, request, config):
    from .intake import IntakeRecording
    from .contracts import validate_recording
    try:
        checkpoint = json.loads((output / "resume.json").read_text())
        files = ("intake/recording.f32", "intake/recording.json", "intake/intake.json", "discovery/discovery.json")
        if checkpoint["version"] != 1 or checkpoint["stage"] != "audio_complete":
            raise ValueError("Unsupported audio checkpoint")
        if checkpoint["identity"] != _audio_identity(request, config):
            raise ValueError("Audio, request, discovery settings, or detector changed")
        if checkpoint["files"] != {name: _digest(output / name) for name in files}:
            raise ValueError("Saved audio checkpoint was changed")
        metadata = validate_recording(json.loads((output / "intake/recording.json").read_text()))
        diagnostics = json.loads((output / "intake/intake.json").read_text())
        discovery = json.loads((output / "discovery/discovery.json").read_text())
        if discovery["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("Saved detection does not match the recording")
        events = [validate_event(event) for event in discovery["events"]]
        return IntakeRecording(output / "intake/recording.f32", metadata, diagnostics), events
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("A complete, readable audio checkpoint is required to resume") from exc


def analyze(request, output_dir, config=None, progress=lambda *args, **kwargs: None, *, resume=False):
    request = validate_request(request)
    config = config or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "provenance.json").exists() and not resume:
        raise FileExistsError("Analysis output exists; use a new directory")
    began = time.monotonic()
    from .provenance import implementation_files
    sources = {p.relative_to(Path(__file__).parent).as_posix(): _digest(p)
               for p in implementation_files(Path(__file__).parent, recursive=False)}
    provenance = {"request": request, "operator_configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "source_sha256": _digest(request["audio_path"]), "implementation_files": sources,
                  "ground_truth_used_by_search": False, "event_selection": "automatic_whole_recording"}
    if resume:
        recording, events = _load_audio_checkpoint(output, request, config)
        provenance["resumed_from"] = "audio_complete"
    write_json(output / "request.json", request)
    write_json(output / "provenance.json", provenance)
    state = {"coverage": {"search_complete": False, "data_status": "unavailable", "limitations": [], "requested_area": request["search"]}}

    def finish(status, reason_code=None, message=None, **fields):
        provenance["elapsed_s"] = time.monotonic() - began
        provenance["status"] = status
        provenance["coverage"] = fields.get("coverage", state["coverage"])
        write_json(output / "provenance.json", provenance)
        artifacts = {"provenance": "provenance.json", "request": "request.json"}
        for name, path in (("intake", "intake/intake.json"), ("recording", "intake/recording.json"),
                           ("discovery", "discovery/discovery.json"), ("discovery_plot", "discovery/discovery.png"),
                           ("search_report", "search/search-report.json"), ("refinement", "refined-candidates.json"),
                           ("spectral_comparison", "spectral-comparison.json"),
                           ("data_provenance", "data-provenance.json"), ("decision", "decision.json")):
            if (output / path).is_file():
                artifacts[name] = path
        for path in sorted((output / "discovery").glob("event-*.png")):
            artifacts[path.stem] = path.relative_to(output).as_posix()
        value = dict(state, status=status, stage="complete", progress=1, artifacts=artifacts, **fields)
        if reason_code:
            value.update(reason_code=reason_code, message=message)
        return value

    try:
        from .intake import AudioIntakeError, intake_recording
        from .discovery import DiscoverySettings, discover_events
    except ImportError as exc:
        return finish("failed", "component_unavailable", f"Automatic audio components are not installed: {exc}")
    discovery_options = config.get("discovery", {})
    search_workers = config.get("search_workers")
    if search_workers is not None and (type(search_workers) is not int or search_workers < 1):
        return finish("failed", "invalid_configuration", "search_workers must be a positive integer.")
    if not isinstance(discovery_options, dict) or discovery_options.keys() - {"chunk_s", "overlap_s"}:
        return finish("failed", "invalid_configuration", "Only default discovery settings are supported.")
    try:
        discovery_settings = DiscoverySettings(**discovery_options)
    except (TypeError, ValueError) as exc:
        return finish("failed", "invalid_configuration", f"Invalid operator discovery settings: {exc}")
    if discovery_settings != DiscoverySettings():
        return finish("failed", "invalid_configuration", "Custom discovery settings are not supported.")
    spectral_options = config.get("spectral_comparison")
    spectral_keys = {"time_sweep_s", "time_step_s", "frequency_step_hz", "expansion_s",
                     "window_s", "hop_s", "max_window_s", "max_modes", "max_event_clock_trials",
                     "max_curve_samples", "max_spectrogram_bins"}
    if spectral_options is not None and (not isinstance(spectral_options, dict)
                                         or spectral_options.keys() - spectral_keys):
        return finish("failed", "invalid_configuration", "Spectral comparison accepts only work limits and spectral grid settings.")
    if not resume:
        progress("intake", 0.05, message="Decoding audio.")
        def report_intake_progress(**details):
            fraction, message = _decode_progress(0.05, 0.14, "Decoding audio", **details)
            progress("intake", fraction, message=message)

        try:
            recording = intake_recording(request["audio_path"], output / "intake", time_window=request.get("time_window"),
                                         progress_callback=_throttled(report_intake_progress))
        except AudioIntakeError as exc:
            return finish("failed", exc.code, str(exc))
        state["recording"] = recording.metadata
        progress("discovery", 0.15, message="Discovering acoustic events.", recording=recording.metadata)

        def report_discovery_progress(phase, **details):
            if phase == "decoding_high_rate":
                fraction, message = _decode_progress(0.15, 0.24, "Discovery: decoding high-rate copy", **details)
            else:
                fraction, message = _DISCOVERY_PHASES.get(phase, (0.25, f"Discovery: {phase.replace('_', ' ')}."))
            progress("discovery", fraction, message=message, recording=recording.metadata)

        events = [validate_event(event) for event in discover_events(
            recording, output_dir=output / "discovery", settings=discovery_settings,
            progress_callback=_throttled(report_discovery_progress))]
        _save_audio_checkpoint(output, request, config)
    state["recording"] = recording.metadata
    state["events"] = events
    provenance["discovered_events"] = len({event.get("harmonic_group_id", event["event_id"]) for event in events})
    provenance["discovered_ridges"] = len(events)
    if recording.time_status == "needs_input":
        return finish("needs_input", "recording_time_required", "Supply a defensible UTC window for the start of this recording. File creation and upload times are not recording clocks.", required_fields=["time_window"])
    if not events:
        return finish("inconclusive", "no_supported_events", "No supported moving-tone Doppler events were discovered in the recording.")
    # Coverage remains unassessed until the search report is available.
    # Keep the initial state for explicit acquisition failure results.
    running_state = {key: value for key, value in state.items() if key != "coverage"}
    progress("acquisition", 0.35, **running_state)
    last_acquisition_report = 0.0
    last_acquisition_phase = None
    acquisition_fraction = 0.35

    def report_acquisition_progress(**details):
        nonlocal last_acquisition_report, last_acquisition_phase, acquisition_fraction
        now = time.monotonic()
        phase = str(details.get("phase", "working")).replace("_", " ")
        if phase == last_acquisition_phase and now - last_acquisition_report < 2.0:
            return
        label = details.get("label")
        message = f"{label}: {phase}" if label else phase[:1].upper() + phase[1:]
        done, total = details.get("bytes_done"), details.get("bytes_total")
        if isinstance(done, (int, float)) and math.isfinite(done):
            if isinstance(total, (int, float)) and math.isfinite(total) and total > 0:
                message += f" ({done / (1024 * 1024):.0f}/{total / (1024 * 1024):.0f} MiB)"
            else:
                message += f" ({done / (1024 * 1024):.0f} MiB)"
        metrics = []
        for key, metric_label in (("rows", "sample rows"), ("batches", "archive batches"),
                                  ("segments", "flight segments"), ("flights", "flights"),
                                  ("tracks", "prepared tracks")):
            value = details.get(key)
            if type(value) is int:
                metrics.append(f"{value:,} {metric_label}")
        if metrics:
            message += f" ({', '.join(metrics)})"
        acquisition_fraction = max(acquisition_fraction, _acquisition_fraction(details))
        progress("acquisition", acquisition_fraction, message=message, **running_state)
        last_acquisition_report, last_acquisition_phase = now, phase

    def report_search_progress(fraction=0.0):
        fraction = min(1.0, max(0.0, float(fraction)))
        if fraction <= 0.8:
            message = f"Coarse search area: {fraction / 0.8:.1%} processed; coverage not assessed."
        elif fraction < 1.0:
            message = f"Finalist refinement: {(fraction - 0.8) / 0.2:.1%} complete; coverage not assessed."
        else:
            message = "Search scoring complete; coverage not assessed."
        progress("search", 0.55 + 0.29 * fraction,
                 message=message, **running_state)

    try:
        from .data_access import DataUnavailable
        search, prepared_events, measurements, indexes, terrain = _search(
            request, recording, events, output, config, report_search_progress,
            report_acquisition_progress)
    except ImportError as exc:
        return finish("failed", "component_unavailable", f"Automatic matching/data components are not installed: {exc}")
    except EmptySearchArea as exc:
        state["coverage"]["limitations"].append(str(exc))
        return finish("needs_input", exc.reason_code, str(exc), required_fields=["search.bounds_wsen"])
    except MissingData as exc:
        state["coverage"]["limitations"].append(str(exc))
        return finish("inconclusive", "data_unavailable", str(exc))
    except UnreconciledDiscovery as exc:
        state["coverage"]["limitations"].append(str(exc))
        return finish("inconclusive", "unreconciled_discovery_group", str(exc))
    except DataUnavailable as exc:
        state["coverage"]["limitations"].append(str(exc))
        return finish("failed", "data_unavailable", str(exc))
    state["coverage"] = _coverage(request, search, config)
    provenance["search_run_hash"] = search.get("run_hash")
    progress("refinement", 0.85, message="Preparing finalist refinements.", **state)

    last_decision_report = 0.0
    last_decision_fraction = -1.0

    def report_decision_progress(fraction, message):
        nonlocal last_decision_report, last_decision_fraction
        fraction = min(1.0, max(0.0, float(fraction)))
        now = time.monotonic()
        if (fraction < 1.0 and last_decision_report
                and now - last_decision_report < 2.0
                and fraction - last_decision_fraction < 0.002):
            return
        if fraction >= 1.0:
            progress("decision", 0.95, message=message, **state)
        else:
            progress("refinement", 0.85 + 0.099 * fraction, message=message, **state)
        last_decision_report = now
        last_decision_fraction = fraction

    return _decide(search, events, prepared_events, measurements, indexes, terrain, state, output, config, finish,
                   report_decision_progress, recording=recording)


class MissingData(ValueError):
    pass


class UnreconciledDiscovery(ValueError):
    pass


# Display-only progress bands. Fractions never affect scoring or rankings.
_DISCOVERY_PHASES = {
    "tracking_ridges": (0.25, "Discovery: tracking Doppler ridges across the recording."),
    "grouping_proposals": (0.32, "Discovery: grouping harmonics into event proposals."),
    "writing_plots": (0.33, "Discovery: writing diagnostic plots."),
}
# Acquisition spans 0.35-0.55: downloads with known byte totals move within
# 0.35-0.45; later index-building phases have no row totals and use fixed steps.
_ACQUISITION_PHASES = {
    "decoding_archive": 0.45, "cached_flight_index": 0.45, "creating_flight_index": 0.46,
    "staging_samples": 0.47, "creating_sample_lookup": 0.49, "building_segments": 0.50,
    "preparing_tracks": 0.52, "compacting_flight_index": 0.53, "flight_index_ready": 0.54,
}


def _clock(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def _decode_progress(low, high, label, decoded_s=0.0, duration_s=None, **_):
    """Map decoded audio seconds into a display band; unknown durations stay at ``low``."""
    if isinstance(duration_s, (int, float)) and math.isfinite(duration_s) and duration_s > 0:
        share = min(1.0, max(0.0, decoded_s / duration_s))
        return low + (high - low) * share, f"{label}: {_clock(decoded_s)} of {_clock(duration_s)} decoded."
    return low, f"{label}: {_clock(decoded_s)} decoded."


def _acquisition_fraction(details):
    phase = details.get("phase")
    if phase in _ACQUISITION_PHASES:
        return _ACQUISITION_PHASES[phase]
    done, total = details.get("bytes_done"), details.get("bytes_total")
    if (isinstance(done, (int, float)) and isinstance(total, (int, float))
            and math.isfinite(done) and math.isfinite(total) and total > 0):
        return 0.35 + 0.10 * min(1.0, max(0.0, done / total))
    return 0.35


def _throttled(callback, interval_s=2.0):
    """Forward phase changes immediately and repeated phases at most every ``interval_s``."""
    last = {"time": None, "phase": object()}

    def forward(**details):
        now = time.monotonic()
        phase = details.get("phase")
        if phase == last["phase"] and last["time"] is not None and now - last["time"] < interval_s:
            return
        last["time"], last["phase"] = now, phase
        callback(**details)
    return forward


def _earliest_emission(lo, events, max_distance, c, filters):
    """Retain the complete CPA and refinement envelope of all proposals."""
    original = lo - max_distance / 300 - 30
    try:
        # Even when the optional CPA/silence settings cannot narrow the search,
        # refinement loads tracks 120 s before the first observed event.
        earliest_refinement = lo + min(e["start_s"] for e in events) - max_distance / 300 - 121
        fallback = min(original, earliest_refinement) if math.isfinite(earliest_refinement) else original
    except (KeyError, TypeError, ValueError):
        return original
    try:
        offset = float(filters.get("offset_tolerance_s", 5.0))
        if not math.isfinite(offset) or offset < 0 or not math.isfinite(c) or c <= 0:
            return fallback
        silence = filters.get("silence")
        if silence:
            if not isinstance(silence, dict):
                return fallback
            margin = float(silence.get("edge_margin_s", 60))
            if not math.isfinite(margin) or margin < 0:
                return fallback
        # Candidate association allows half the offset tolerance beyond each
        # event's CPA interval, then the wide-clock scorer allows a 5 s sweep.
        earliest_cpa = min(e["cpa_receive_s"] - e["quality"]["cpa_tolerance_s"] for e in events)
        # Round outward relative to the differently grouped Unix-time
        # calculation in _decide, including floating-point error.
        candidate = earliest_cpa - offset / 2 - 5 - max_distance / c - 30
        if not all(math.isfinite(x) for x in (earliest_cpa, earliest_refinement, candidate)):
            return fallback
        return min(lo + candidate, earliest_refinement)
    except (KeyError, TypeError, ValueError):
        return fallback


def _latest_emission(hi, duration, events, filters):
    """Retain whole-recording, CPA and refinement upper time envelopes."""
    original = hi + duration + 30
    refinement = hi + max(e["end_s"] for e in events) + 121
    try:
        offset = float(filters.get("offset_tolerance_s", 5.0))
    except (TypeError, ValueError):
        raise MissingData("Cannot establish coverage with an invalid CPA offset tolerance.") from None
    if not math.isfinite(offset) or offset < 0:
        raise MissingData("Cannot establish coverage with an invalid CPA offset tolerance.")
    latest_cpa = max(e["cpa_receive_s"] + e["quality"]["cpa_tolerance_s"] for e in events)
    candidate = hi + latest_cpa + offset / 2 + 5 + 31
    if not all(math.isfinite(x) for x in (original, refinement, candidate)):
        raise MissingData("Cannot establish finite event and refinement time coverage.")
    return max(original, refinement, candidate)


def _search(request, recording, events, output, config, search_progress, acquisition_progress=None):
    from .data_access import FlightIndex, acquire_inputs
    from .geography import profile_for_search, receiver_mask_for_search
    from .matching import prepare_events, UnreconciledHarmonicGroup
    from .terrain import Terrain
    from .tiled_search import SearchBudget, search_national
    c = float(config.get("sound_speed_mps", 343.0))
    max_distance = float(config.get("max_distance_m", 20000.0))
    try:
        prepared, measurements = prepare_events(events, c=c, max_distance_m=max_distance)
    except UnreconciledHarmonicGroup as exc:
        raise UnreconciledDiscovery(
            f"Acoustic proposals could not be reconciled for matching: {exc}. "
            "The saved discovery diagnostics retain all detected ridges; no location search was performed."
        ) from exc
    window = recording.metadata["time_window"]
    lo, hi = (utc_timestamp(window[name]).timestamp() for name in ("start_utc", "end_utc"))
    filters = {"association": "subsets", **config.get("filters", {})}
    filters.setdefault("max_events", len(prepared))
    silence = filters.get("silence")
    if silence:
        if not isinstance(silence, dict):
            raise MissingData("Cannot establish coverage with an invalid silence configuration.")
        try:
            margin = float(silence.get("edge_margin_s", 60))
        except (TypeError, ValueError):
            raise MissingData("Cannot establish coverage with an invalid silence margin.") from None
        if not math.isfinite(margin) or margin < 0:
            raise MissingData("Cannot establish coverage with a negative or nonfinite silence margin.")
    earliest = _earliest_emission(lo, prepared, max_distance, c, filters)
    latest = _latest_emission(hi, recording.metadata["duration_s"], prepared, filters)
    if not config.get("flight_indexes") and not config.get("cache_root"):
        raise MissingData("Historical flight/terrain data are not configured. The operator must provide cache_root or prebuilt flight indexes with a terrain model.")
    first_day = datetime.fromtimestamp(earliest, timezone.utc).date()
    last_day = datetime.fromtimestamp(latest, timezone.utc).date()
    acquisition_days = (last_day-first_day).days+1
    if not config.get("flight_indexes") and acquisition_days > config.get("max_acquisition_days", 3):
        raise MissingData("Recording time window exceeds the configured historical-data acquisition budget. Supply a narrower defensible window.")
    ground = config.get("terrain", {})
    hashes = {"terrain_configuration": ground}
    profile = profile_for_search(request["search"])
    receiver_mask = receiver_mask_for_search(request["search"],
        cache_root=config.get("cache_root", output / "cache"), offline=config.get("offline", True))
    budget = None
    if receiver_mask is not None:
        budget = SearchBudget(**config.get("search_budget", {}))
        if receiver_mask.preflight_grid(request["search"].get("bounds_wsen", profile.bounds_wsen),
                                        budget.spacing_m, profile.crs) is not True:
            raise MissingData("The bounded receiver-grid check could not establish a German receiver "
                "point in this selection. No flight data were acquired. Choose a smaller area, "
                "or ask the operator to inspect the search area and coarse-grid spacing; "
                "this does not establish that the receiver area is empty.")
    if ground.get("rasters") or "constant_m" in ground:
        # Explicit offline/test terrain override; normal provider acquisition stays
        # in the data owner's adapter. Never infer flight coverage from this model.
        if not config.get("flight_indexes"):
            raise MissingData("A declared local terrain model requires prebuilt flight indexes.")
        indexes = [FlightIndex(path) for path in config["flight_indexes"]]
        terrain = Terrain(ground.get("rasters", []), ground.get("constant_m"), ground.get("geoid_raster"))
        time_range = (earliest, latest)
        bounds = request["search"].get("bounds_wsen", profile.bounds_wsen)
        hashes["terrain_rasters"] = {str(path): _digest(path) for path in ground.get("rasters", [])}
    else:
        bundle = acquire_inputs(request, recording.metadata, cache_root=config.get("cache_root", output / "cache"),
                                index_paths=config.get("flight_indexes"), offline=config.get("offline", True),
                                terrain_options=ground, max_sound_range_m=max_distance,
                                earliest_emission_s=earliest, latest_emission_s=latest,
                                progress_callback=acquisition_progress)
        indexes, terrain, time_range, bounds = bundle.indexes, bundle.terrain, bundle.time_range_s, bundle.bounds
    if ground.get("geoid_raster"):
        hashes["geoid_sha256"] = _digest(ground["geoid_raster"])
    search_progress()
    # A supplied interval is never collapsed to its midpoint. Every event/tile
    # uses the same recording-start clock and globally comparable score units.
    scoring = {"mode": "robust_ridge", "c": c, "time_origin": lo if lo == hi else None,
               "time_sweep_s": 0 if lo == hi else 5, "time_step_s": 0.5,
               "recording_start_window_s": [lo, hi]}
    if config.get("resume_legacy_first_batch"):
        scoring["legacy_first_batch"] = True
    geographic_options = {"receiver_mask": receiver_mask} if receiver_mask is not None else {}
    report = search_national(indexes, terrain, prepared, measurements, time_range_s=time_range, output_dir=output / "search",
                             bounds=bounds, budget=budget or SearchBudget(**config.get("search_budget", {})),
                             scoring=scoring, filters=filters, input_hashes=hashes, crs=profile.crs,
                             workers=config.get("search_workers"),
                             search_progress=search_progress, **geographic_options)
    provenance = {"flight_indexes": [index.report for index in indexes],
                  "terrain": terrain.provenance() if hasattr(terrain, "provenance") else hashes}
    if receiver_mask is not None:
        provenance["search_geography"] = {"country": profile.country, "coverage": profile.coverage,
            "bounds_wsen": list(bounds), "crs": profile.crs, "receiver_mask": receiver_mask.identity}
    write_json(output / "data-provenance.json", provenance)
    return report, prepared, measurements, indexes, terrain


def _bounded_model_assessment(search, support):
    """Validate accounting of missing fits without claiming full model support.

    The confidence decision still compares the certified upper score bound with
    the winner. This only distinguishes bounded finite trials from omitted work.
    """
    if (search.get("model_support_complete") is not False
            or search.get("model_assessment_complete") is not True
            or search.get("status") != "complete" or not isinstance(support, dict)):
        return False
    names = ("unsupported_cpa_track_evaluations", "unclassified_score_candidates", "unknown_trials",
             "unreported_trial_envelopes", "bounded_score_candidates", "unbounded_score_candidates")
    if not all(type(support.get(name)) is int and support[name] >= 0 for name in names):
        return False
    bounded = support["bounded_score_candidates"]
    if (bounded == 0 or bounded != support["unclassified_score_candidates"]
            or support["unknown_trials"] < bounded or support["unsupported_cpa_track_evaluations"]
            or support["unreported_trial_envelopes"] or support["unbounded_score_candidates"]):
        return False
    finite = search.get("finite_search_evidence")
    if (not isinstance(finite, dict) or type(finite.get("schema_version")) is not int
            or finite["schema_version"] != 1 or finite.get("scope") != "evaluated_finite_grid"
            or finite.get("score_kind") != "matched_group_support_minus_normalized_residual"
            or finite.get("complete") is not True or finite.get("pruning_accounted_for") is not True):
        return False
    counts = ("bounded_unknown_envelopes", "bounded_unknown_rejections", "discarded_count",
              "max_event_groups", "unclassified_score_candidates", "unsupported_cpa_track_evaluations")
    if not all(type(finite.get(name)) is int and finite[name] >= 0 for name in counts):
        return False
    if (finite["bounded_unknown_envelopes"] != bounded
            or finite["unclassified_score_candidates"] != bounded
            or finite["bounded_unknown_rejections"] > bounded
            or finite["unsupported_cpa_track_evaluations"] != 0 or finite["max_event_groups"] < 1):
        return False
    possible = bounded - finite["bounded_unknown_rejections"]
    upper = finite.get("best_unknown_score_upper_bound")
    discarded = finite["discarded_count"]
    overall = finite.get("best_discarded_score")
    number = lambda value: type(value) in (int, float) and math.isfinite(value)
    if discarded < possible or (discarded and not number(overall)) or (not discarded and overall is not None):
        return False
    if possible:
        return (number(upper) and 0 <= upper <= finite["max_event_groups"]
                and overall >= upper)
    return upper is None


def _coverage(request, search, config):
    limitations = ["Historical receiver archives may omit aircraft or trajectory samples."]
    if search.get("aircraft_scope") == "confirmed_nonjet":
        limitations.append("Only aircraft classified as piston or turbine were searched; jets and unknown types were excluded.")
    search_complete = (search.get("tiles_total", 0) > 0 and search.get("tiles_completed") == search.get("tiles_total")
                       and not search.get("tile_failures") and not search.get("refinement_failure")
                       and search.get("all_event_groups_considered") is not False)
    if search.get("aircraft_classification_incomplete"):
        search_complete = False
        limitations.append("No confirmed non-jet tracks were selected; aircraft-type metadata may be missing.")
    if any(name in search for name in ("model_support_complete", "model_support", "model_assessment_complete")):
        support = search.get("model_support")
        count_names = ("unsupported_cpa_track_evaluations", "unclassified_score_candidates",
                       "evaluated_trials", "rejected_trials", "unknown_trials", "unreported_trial_envelopes")
        valid_counts = (isinstance(support, dict) and
                        all(type(support.get(name)) is int and support[name] >= 0 for name in count_names))
        model_complete = (search.get("model_support_complete") is True and valid_counts and
                          all(support[name] == 0 for name in (count_names[0], count_names[1], "unknown_trials", "unreported_trial_envelopes")))
        if "model_assessment_complete" in search and search["model_assessment_complete"] is not True:
            model_complete = False
        bounded_assessment = valid_counts and _bounded_model_assessment(search, support)
        if not model_complete and not bounded_assessment:
            search_complete = False
            limitations.append("Trajectory model assessment is incomplete; unsupported competitors remain unassessed. See the search support diagnostics.")
        elif bounded_assessment:
            limitations.append("Some finite acoustic trials lack full trajectory support. Their certified score bounds or rejections are accounted for; the decision still checks whether any bound could compete with the leading location.")
        if not model_complete:
            if valid_counts:
                limitations.append(f"Unsupported CPA track evaluations: {support[count_names[0]]}; unclassified score candidates: {support[count_names[1]]}; unknown clock trials: {support['unknown_trials']}; unreported trial envelopes: {support['unreported_trial_envelopes']}. Counts describe work evaluations, not distinct aircraft.")
            else:
                limitations.append("Trajectory model support counts are unavailable or malformed; missing work is not treated as zero.")
    if not search_complete:
        limitations.append("The search did not assess the complete requested scope within its work or physical model limits.")
    if search.get("all_event_groups_considered") is False:
        limitations.append(search.get("event_limit_reason") or "Not every proposed flyover group was considered within the configured budget.")
    if "constant_m" in config.get("terrain", {}):
        limitations.append("A constant terrain model was supplied; real terrain and altitude-datum accuracy are not established.")
    finite = search.get("finite_search_evidence")
    if finite is not None:
        descriptions = finite.get("limitations") if isinstance(finite, dict) else None
        if isinstance(descriptions, list) and descriptions and all(isinstance(item, str) for item in descriptions):
            limitations.extend(descriptions)
        else:
            limitations.append("Only evaluated grid points and bounded refinements are covered; continuous-space uniqueness is not established.")
    elif search.get("alternative_modes_exhaustively_assessed") is not True:
        limitations.append("Candidate retention is bounded; exhaustive alternative-location assessment is not established.")
    missing = [str(item) for item in search.get("tile_failures", [])]
    if not search.get("data_scope_complete"):
        missing.append("historical_flight_scope_or_audible_border_region")
        limitations.append("Flight inputs do not establish complete coverage of the required emission-time interval and audible border region.")
    return {"search_complete": search_complete, "data_status": "available" if search.get("data_scope_complete") else "partial",
            "limitations": limitations, "requested_area": request["search"], "missing_data": missing}


def _comparable_refinement(candidate, original, prepared_events):
    """Preserve newer normalized results; adapt only older raw-loss producers."""
    if (original.get("score_kind") == "matched_group_support_minus_normalized_residual"
            and candidate.get("score_kind") != "matched_group_support_minus_normalized_residual"):
        from .matching import association_score
        chosen = [prepared_events[i] for i in original["event_indices"]]
        candidate = association_score(candidate, chosen)
    if not math.isfinite(candidate["score"]):
        raise ValueError("Refinement no longer supports the matched event association")
    if "mode_id" in original:
        candidate = dict(candidate, mode_id=original["mode_id"])
    return candidate


def _compare_retained_spectra(recording, rows, prepared_events, indexes, config, start_window):
    """Run the optional physical comparison without transferring ridge evidence."""
    from .data_access import DataUnavailable
    from .spectral_matching import SCORE_KIND, compare_spectral_modes
    options = dict(config["spectral_comparison"])
    lookup = {"requested_tracks": 0, "loaded_tracks": 0, "loaded_samples": 0,
              "missing_ids": [], "ambiguous_ids": [], "failures": []}
    try:
        if recording is None:
            raise ValueError("The decoded recording is unavailable for spectral comparison")
        tracks, ambiguous = {}, set()
        if len(rows) <= options.get("max_modes", 1000):
            needed = {identifier for row in rows for identifier in row["flight_ids"]}
            lookup["requested_tracks"] = len(needed)
            budget = config.get("search_budget", {})
            max_tracks = budget.get("max_tracks_per_tile", 2000)
            max_samples = budget.get("max_track_samples_per_tile", 500000)
            lookup.update(max_tracks=max_tracks, max_samples=max_samples)
            for index in indexes:
                remaining = max_samples-lookup["loaded_samples"]
                if remaining <= 0:
                    lookup["failures"].append("Flight lookup sample budget exhausted")
                    break
                try:
                    for track in index.lookup(needed, max_tracks=max_tracks, max_samples=remaining):
                        lookup["loaded_samples"] += len(track.time)
                        if track.id in tracks or track.id in ambiguous:
                            ambiguous.add(track.id)
                            tracks.pop(track.id, None)
                        else:
                            tracks[track.id] = track
                except DataUnavailable as exc:
                    lookup["failures"].append(str(exc))
            lookup.update(loaded_tracks=len(tracks), missing_ids=sorted(needed-tracks.keys()),
                          ambiguous_ids=sorted(ambiguous))
        report = compare_spectral_modes(recording, rows, tracks, prepared_events,
                    recording_start_window_s=start_window, c=config.get("sound_speed_mps", 343.0), **options)
        if lookup["failures"]:
            report.update(status="inconclusive", all_modes_compared=False)
    except (ValueError, TypeError, KeyError, OSError, DataUnavailable) as exc:
        report = {"status": "inconclusive", "location_decision": "not_evaluated",
                  "score_kind": SCORE_KIND, "all_modes_compared": False,
                  "ranked": [], "rankings_by_event_set": [], "reason": str(exc)}
    report["track_lookup"] = lookup
    return report


def _refinement_tracks(indexes, needed, bounds, time_range, sample_limit):
    """Load only a finalist's whole segments within the original query envelope."""
    from .data_access import DataUnavailable, FlightIndex

    tracks = {}
    for index in indexes:
        # Preserve custom query adapters and their filtering behavior. The native
        # index can restrict names in SQL before decompressing unrelated flights.
        named = type(index) is FlightIndex
        source = (index.lookup(needed, bounds=bounds, time_range=time_range,
                               max_tracks=max(1, len(needed)), max_samples=sample_limit)
                  if named else index.query(bounds, time_range))
        try:
            for track in source:
                if track.id in needed:
                    tracks[track.id] = track
                    if sum(len(item.time) for item in tracks.values()) > sample_limit:
                        raise ValueError("Refinement trajectory sample budget exceeded")
        except DataUnavailable as error:
            if named and str(error) == "Flight lookup sample budget exceeded":
                raise ValueError("Refinement trajectory sample budget exceeded") from error
            raise
    return tracks


def _decide(search, events, prepared_events, measurements, indexes, terrain, state, output, config, finish, decision_progress, *, recording=None):
    # The confidence producer supplies this adapter; missing decision machinery
    # can never turn the highest raw score into a supported location.
    from .confidence import decide_location
    from .geography import profile_for_search, receiver_mask_for_search
    from .matching import _cache_ridge_fits, profile_candidate, refine_candidate
    # Each finalist and the complete sensitivity profile own an exact-input
    # cache. Profile's internal refinements share its scope; other calls do not.
    refine_candidate = _cache_ridge_fits(refine_candidate)
    profile_candidate = _cache_ridge_fits(profile_candidate)
    raw = list(search.get("ranked", []))
    refined, refinement_failures = [], []
    window = state["recording"]["time_window"]
    start_window = tuple(utc_timestamp(window[key]).timestamp() for key in ("start_utc", "end_utc"))
    search_area = state["coverage"]["requested_area"]
    profile = profile_for_search(search_area)
    search_bounds = search_area.get("bounds_wsen", profile.bounds_wsen)
    receiver_mask = receiver_mask_for_search(search_area,
        cache_root=config.get("cache_root", output / "cache"), offline=config.get("offline", True))
    geographic_options = {"receiver_mask": receiver_mask} if receiver_mask is not None else {}
    finite = search.get("finite_search_evidence")
    assessed_ids = set()
    producer_ids = search.get("core_assessed_mode_ids")
    valid_core_mapping = (isinstance(producer_ids, list)
                          and all(isinstance(identity, str) and identity for identity in producer_ids))
    if isinstance(finite, dict) and valid_core_mapping:
        producer_ids = set(producer_ids)
        for row in raw:
            identity = row.get("mode_id")
            groups = row.get("event_indices", range(len(prepared_events)))
            core = row.get("event_diagnostics", [])
            if (isinstance(identity, str) and identity in producer_ids and row.get("score_kind") == finite.get("score_kind")
                    and math.isfinite(row["score"]) and start_window[0] <= row["recording_start_time_s"] <= start_window[1]
                    and len(groups) == len(row["flight_ids"]) == len(core) and len(core) > 0
                    and all(isinstance(fit, dict) and fit.get("support_fraction") == 1.
                            and not isinstance(fit.get("support_fraction"), bool)
                            and isinstance(fit.get("score"), (int, float)) and not isinstance(fit["score"], bool)
                            and math.isfinite(fit["score"]) for fit in core)):
                assessed_ids.add(identity)
    # Refine global competing approaches with their original whole trajectories.
    # The index iterator is bounded by the same local acoustic support envelope.
    def mode_key(row):
        return (tuple(row.get("event_indices", range(len(prepared_events)))), tuple(row["flight_ids"]),
                tuple(row.get("pass_emission_times_s", [])), row["lat"], row["lon"], row["recording_start_time_s"],
                row.get("elev_m"), row.get("mode_id"))

    retained_keys = {mode_key(row) for row in raw}
    refined_keys = set()
    seen, track_sets = set(), {}
    best_refined_score = -math.inf
    finalist_limit = max(0, int(config.get("refine_finalists", 5)))
    finalist_total = min(finalist_limit, len({mode_key(row) for row in raw}))
    for row in raw:
        key = mode_key(row)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > config.get("refine_finalists", 5):
            break
        finalist_index = len(seen)

        def report_finalist_progress(fraction, **fields):
            if finalist_total <= 0:
                return
            fraction = min(1.0, max(0.0, float(fraction)))
            overall = 0.4 * ((finalist_index - 1 + fraction) / finalist_total)
            iteration = fields.get("optimizer_iteration")
            evaluations = fields.get("optimizer_evaluations")
            activity = f"Finalist {finalist_index}/{finalist_total}"
            altitude_profile = fields.get("altitude_profile")
            altitude_profiles = fields.get("altitude_profiles")
            if altitude_profile is not None and altitude_profiles is not None:
                activity += f", altitude profile {altitude_profile}/{altitude_profiles}"
            if iteration is not None:
                activity += f", optimizer iteration {iteration}/60"
            if evaluations is not None:
                activity += f", {evaluations} score evaluations"
            decision_progress(overall, activity)

        try:
            max_range = float(config.get("max_distance_m", 20000.0))
            margin = max_range / 60000.0 + 0.3
            bounds = (row["lon"]-margin, row["lat"]-margin, row["lon"]+margin, row["lat"]+margin)
            needed = set(row["flight_ids"])
            lo = start_window[0] + min(e["start_s"] for e in prepared_events) - max_range / 300 - 120
            hi = start_window[1] + max(e["end_s"] for e in prepared_events) + 120
            sample_limit = config.get("search_budget", {}).get("max_track_samples_per_tile", 500000)
            tracks = _refinement_tracks(indexes, needed, bounds, (lo, hi), sample_limit)
            if needed - tracks.keys():
                raise ValueError("Candidate flight segments are unavailable for refinement")
            row = dict(row, altitude_basis={flight: sorted(set(getattr(track, "altitude_basis", ["unspecified"]))) for flight, track in tracks.items()})
            candidate = refine_candidate(row, tracks, prepared_events, measurements, terrain,
                                         radius_m=config.get("refine_radius_m", 500.0),
                                         recording_start_window_s=start_window, c=config.get("sound_speed_mps", 343.0),
                                         bounds_wsen=search_bounds, crs=profile.crs,
                                         progress_callback=report_finalist_progress, **geographic_options)
            candidate = _comparable_refinement(candidate, row, prepared_events)
            refined.append(candidate)
            refined_keys.add(key)
            if candidate["score"] > best_refined_score:
                best_refined_score = candidate["score"]
                track_sets = {mode_key(candidate): tracks}
        except (ValueError, KeyError) as exc:
            refinement_failures.append(str(exc))
        report_finalist_progress(1.0)
    raw.extend(refined)
    order = lambda row: (row["score"], bool(row.get("sensitivity", {}).get("complete")), len(row.get("sensitivity", {}).get("sources", [])))
    raw.sort(key=order, reverse=True)
    if raw and mode_key(raw[0]) in track_sets:
        leading = raw[0]
        try:
            def report_sensitivity_progress(fraction, **fields):
                fraction = min(1.0, max(0.0, float(fraction)))
                overall = 0.4 + 0.6 * fraction
                scenario = fields.get("scenario")
                iteration = fields.get("optimizer_iteration")
                evaluations = fields.get("optimizer_evaluations")
                activity = "Sensitivity profile"
                if scenario:
                    activity += f" {scenario}"
                altitude_profile = fields.get("altitude_profile")
                altitude_profiles = fields.get("altitude_profiles")
                if altitude_profile is not None and altitude_profiles is not None:
                    activity += f", altitude profile {altitude_profile}/{altitude_profiles}"
                if iteration is not None:
                    activity += f", optimizer iteration {iteration}/60"
                if evaluations is not None:
                    activity += f", {evaluations} score evaluations"
                decision_progress(overall, activity)

            profiled = profile_candidate(leading, track_sets[mode_key(leading)], prepared_events, measurements, terrain,
                         radius_m=config.get("refine_radius_m", 500.0), recording_start_window_s=start_window,
                         c=config.get("sound_speed_mps", 343.0), bounds_wsen=search_bounds,
                         crs=profile.crs, progress_callback=report_sensitivity_progress,
                         **geographic_options)
            # Full profiling updates this exact hypothesis. Its earlier, limited
            # envelope must not compete with incomplete evidence or score roundoff.
            raw[0] = _comparable_refinement(profiled, leading, prepared_events)
        except (ValueError, KeyError) as exc:
            refinement_failures.append(f"Sensitivity profiling: {exc}")
            raw[0] = dict(leading, sensitivity={"complete": False, "sources": [],
                         "failures": [{"source": "profiling", "reason": str(exc)}]})
        # Preserve other modes; a different leader must satisfy its own full
        # sensitivity gate, without borrowing this hypothesis's profile.
        raw.sort(key=order, reverse=True)
    candidates, rows = [], []
    for row in raw:
        event_ids, flight_ids = [], []
        selected_events = [prepared_events[i] for i in row.get("event_indices", range(len(prepared_events)))]
        for event, flight in zip(selected_events, row["flight_ids"]):
            ids = event.get("event_ids", [event["event_id"]])
            event_ids.extend(ids)
            flight_ids.extend([flight]*len(ids))
        candidate = {"lat": float(row["lat"]), "lon": float(row["lon"]), "score": float(row["score"]),
                     "score_kind": row.get("score_kind", "negative_robust_ridge_loss_hz"), "event_ids": event_ids,
                     "flight_ids": flight_ids, "recording_start_utc": datetime.fromtimestamp(row["recording_start_time_s"], timezone.utc).isoformat().replace("+00:00", "Z")}
        sensitivity = row.get("sensitivity")
        if sensitivity and sensitivity.get("complete") and not sensitivity.get("boundary_limited"):
            candidate["uncertainty"] = {key: sensitivity[key] for key in ("radius_m", "method", "kind")}
        candidates.append(validate_candidate(candidate))
        rows.append(row)
    best = rows[0] if rows else {}
    original_ids = {event["event_id"]: event.get("event_ids", [event["event_id"]]) for event in prepared_events}
    unmatched_groups = best.get("unmatched_event_ids", list(original_ids) if not rows else [])
    unmatched_ids = [event_id for group in unmatched_groups for event_id in original_ids.get(group, [group])]
    diagnostics = {"recording_start_window_s": list(start_window), "unmatched_event_ids": unmatched_ids, "clock_locked": search.get("scoring_time_is_locked", False), "refinement_failures": refinement_failures,
                   "refinement_boundary_limited": best.get("sensitivity", {}).get("boundary_limited", True),
                   "sensitivity": best.get("sensitivity", {}),
                   "altitude_basis": best.get("altitude_basis", {}),
                   "retained_hypotheses": len(retained_keys), "refined_hypotheses": len(refined_keys),
                   "unrefined_hypotheses": len(retained_keys-refined_keys),
                   "alternatives_evaluated": search.get("alternative_modes_exhaustively_assessed") is True and search.get("status") == "complete" and not refinement_failures and not retained_keys-refined_keys}
    matched_events = [prepared_events[i] for i in best.get("event_indices", range(len(prepared_events)))]
    for event, evidence in zip(matched_events, best.get("event_diagnostics", [])):
        for event_id in event.get("event_ids", [event["event_id"]]):
            diagnostics[event_id] = evidence
    if finite is not None:
        diagnostics["finite_search_evidence"] = deepcopy(finite)
        diagnostics["core_assessment_mapping_status"] = "present" if valid_core_mapping else "missing_or_invalid"
        diagnostics["assessed_mode_ids"] = sorted(assessed_ids)
        diagnostics["candidate_mode_ids"] = [row.get("mode_id") for row in rows]
    from .region_policy import ResultRegionPolicy, RegionPolicyUnavailable, redact_location_outputs
    try:
        policy = ResultRegionPolicy()
        for candidate in candidates:
            if policy.restricted_region(candidate["lat"], candidate["lon"]) is not None:
                redact_location_outputs(output)
                return finish("failed", "restricted_location",
                              "The result falls within a restricted region.")
    except RegionPolicyUnavailable:
        redact_location_outputs(output)
        return finish("failed", "region_policy_unavailable",
                      "The result could not be checked against the configured regional boundaries.")
    write_json(output / "refined-candidates.json", {"ranked": rows, "matching_diagnostics": diagnostics})
    decision_progress(1.0, "Refinement complete; applying inconclusive and uncertainty gates.")
    decision = decide_location(events, candidates, state["coverage"], matching_diagnostics=diagnostics)
    decision.setdefault("candidates", candidates)
    if config.get("spectral_comparison") is not None:
        spectral = _compare_retained_spectra(recording, rows, prepared_events, indexes, config, start_window)
        write_json(output / "spectral-comparison.json", spectral)

    write_json(output / "decision.json", decision)
    fields = {key: value for key, value in decision.items() if key not in {"schema_version", "job_id", "status", "stage", "progress", "artifacts", "recording", "events", "diagnostics"}}
    status = decision["status"]
    reason = fields.pop("reason_code", None)
    message = fields.pop("message", None)
    return finish(status, reason, message, **fields)
