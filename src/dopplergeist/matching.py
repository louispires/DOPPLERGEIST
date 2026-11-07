"""Conservative detected-event adapters and robust emission-time ridge matching.

Scores are penalized residuals, never probabilities. Geometry from a straight-pass
audio fit is a proposal diagnostic, not a reliable exclusion bound for real tracks.
"""
from collections import OrderedDict
from bisect import bisect_left, bisect_right
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
from itertools import chain
import os
import time

import numpy as np
from pyproj import Geod, Transformer
from scipy.optimize import minimize

from .geometry import Track, _EMISSION_ITERATION_TOLERANCE_S, ecef


class UnreconciledHarmonicGroup(ValueError):
    """Discovery supplied a group without a usable shared frequency scale."""


_ridge_backend_choice = os.environ.get("DOPPLERGEIST_RIDGE_BACKEND", "auto")
if _ridge_backend_choice not in {"auto", "python"}:
    raise ValueError("DOPPLERGEIST_RIDGE_BACKEND must be 'auto' or 'python'")
_native_ridge_module = None
_native_fit_irls = None
if _ridge_backend_choice == "auto":
    try:
        from . import _native_ridge as _native_ridge_module
        from ._native_ridge import fit_irls as _native_fit_irls
    except ImportError:
        _native_ridge_module = None
        _native_fit_irls = None

if _native_ridge_module is None:
    RIDGE_FIT_BACKEND = "python-irls-v1"
else:
    from pathlib import Path

    _native_binary_hash = hashlib.sha256(Path(_native_ridge_module.__file__).read_bytes()).hexdigest()
    RIDGE_FIT_BACKEND = f"native-c-irls-v2-pairwise:{_native_binary_hash}"

_native_validate_ridge_events = getattr(_native_ridge_module, "valid_ridge_events", None)
_CANONICAL_NATIVE_EVENT_VALIDATOR = _native_validate_ridge_events
_native_fit_irls_batch = getattr(_native_ridge_module, "fit_irls_batch", None)
_CANONICAL_NATIVE_FIT_IRLS = _native_fit_irls

_NATIVE_RIDGE_ERROR_POLICY = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


@dataclass
class RidgeTimes:
    time: np.ndarray


_ridge_fit_cache = ContextVar("ridge_fit_cache", default=None)


def _fit_argument_key(value):
    """Preserve scalar precision and array shape without rounding fit inputs."""
    if type(value) is float:
        return (float, value.hex())
    if type(value) in (int, bool):
        return (type(value), value)
    if type(value) in (list, tuple):
        return (type(value), tuple(_fit_argument_key(item) for item in value))
    if type(value) is np.ndarray or isinstance(value, np.generic):
        array = np.asarray(value)
        if array.dtype.kind in "biuf":
            return (type(value), array.dtype.str, array.shape, array.tobytes())
    raise TypeError("Unsupported fit-cache argument representation")


class _RidgeFitCache:
    """Bounded reuse for an immutable set of tracks within one scoring batch."""

    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.acoustic_entries = OrderedDict()
        self.acoustic_bytes = 0
        self.acoustic_limit = 32 * 1024 * 1024
        self.measurement_entries = OrderedDict()
        self.measurement_bytes = 0
        self.measurement_limit = 4 * 1024 * 1024
        self.batch_fit_entries = OrderedDict()
        self.batch_fit_bytes = 0
        self.batch_fit_limit = 4 * 1024 * 1024
        self.batch_fit_function = None
        self.batch_fit_backends = (None, None)
        self.ceiling_entries = OrderedDict()
        self.screen_entries = OrderedDict()
        self.screen_bytes = 0
        self.screen_limit = 8 * 1024 * 1024
        self.score_inputs = {}
        self.score_input_bytes = 0
        self.score_input_limit = 4 * 1024 * 1024

    def prepared_measurement(self, measurement, sigma_hz, drift_fraction, key):
        """Reuse selected samples only after checking their current content key."""
        if key in self.measurement_entries:
            self.measurement_entries.move_to_end(key)
            return self.measurement_entries[key][0]
        prepared = _prepare_ridge_fit_inputs(measurement, sigma_hz, drift_fraction)
        # Include the lazily normalized weights in the bound before allocation.
        size = prepared.times.nbytes + prepared.ridge.nbytes + 2 * prepared.weight.nbytes + 512
        if size <= self.measurement_limit and self.max_entries > 0:
            while (self.measurement_bytes + size > self.measurement_limit or
                   len(self.measurement_entries) >= self.max_entries):
                _, (_, removed) = self.measurement_entries.popitem(last=False)
                self.measurement_bytes -= removed
            self.measurement_entries[key] = (prepared, size)
            self.measurement_bytes += size
        return prepared

    def acoustic_batch(self, track, observer, starts, times, c):
        """Reuse exact batched propagation inputs within this scoring call."""
        try:
            key = (id(track), _fit_argument_key(observer), _fit_argument_key(starts),
                   _fit_argument_key(times), _fit_argument_key(c))
        except (TypeError, ValueError):
            key = None
        if key is not None and key in self.acoustic_entries:
            self.acoustic_entries.move_to_end(key)
            _, factors, emissions, _ = self.acoustic_entries[key]
            return factors.copy(), emissions.copy()
        receive = starts[:, None] + times[None, :]
        unique, inverse = np.unique(receive, return_inverse=True)
        factors, emissions, _ = track.acoustic(unique, observer, c)
        factors = factors[inverse].reshape(receive.shape)
        emissions = emissions[inverse].reshape(receive.shape)
        size = factors.nbytes + emissions.nbytes + starts.nbytes + times.nbytes + 512
        if key is not None and size <= self.acoustic_limit:
            while self.acoustic_bytes + size > self.acoustic_limit:
                _, (_, _, _, removed) = self.acoustic_entries.popitem(last=False)
                self.acoustic_bytes -= removed
            self.acoustic_entries[key] = (track, factors, emissions, size)
            self.acoustic_bytes += size
            return factors.copy(), emissions.copy()
        return factors, emissions

    def fit(self, track, observer, measurement, start_time, f0_bounds, c, sigma_hz, drift_fraction,
            acoustic_result=None):
        arguments = (track, observer, measurement, start_time, f0_bounds, c, sigma_hz, drift_fraction)
        try:
            parameters = tuple(_fit_argument_key(value) for value in
                               (observer, start_time, f0_bounds, c, sigma_hz, drift_fraction))
            if acoustic_result is None:
                acoustic_key = None
            elif (type(acoustic_result) in (tuple, list) and len(acoustic_result) == 2
                  and all(type(value) is np.ndarray and value.dtype.kind in "biuf" for value in acoustic_result)):
                acoustic_key = (type(acoustic_result), tuple(
                    (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).digest())
                    for value in acoustic_result))
            else:
                # Preserve validation and custom input behavior in the fitter.
                return _fit_ridge_uncached(*arguments, acoustic_result=acoustic_result)
        except (TypeError, ValueError):
            # Let the fitter handle custom representations and their validation.
            return _fit_ridge_uncached(*arguments, acoustic_result=acoustic_result)
        # Measurements may be equivalent copies or modified arrays. Hash the
        # numeric inputs actually used by the fit, without retaining their data.
        private = _prepared_score_input(measurement)
        if private is None:
            spec, ridge, weight, _ = measurement
            observed = hashlib.sha256()
            for values in (spec.time, ridge, weight):
                values = np.asarray(values, dtype=float)
                observed.update(repr(values.shape).encode())
                observed.update(values.tobytes())
            measurement_key = observed.digest()
        else:
            measurement_key = private.measurement_key
        key = (id(track), parameters, measurement_key, acoustic_key)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][1].copy()
        prepared = self.prepared_measurement(measurement, sigma_hz, drift_fraction,
                                             (measurement_key, parameters[-2:]))
        result = _fit_ridge_uncached(*arguments, acoustic_result=acoustic_result, _prepared=prepared)
        # Keep the track alive so a different trajectory cannot reuse its id.
        # Fit diagnostics contain only immutable scalar values, so a shallow copy
        # gives each caller an independent dictionary without recursive copying.
        # The enclosing batch already owns these tracks. Exceptions are not cached.
        self.entries[key] = (track, result.copy())
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return result


def _cache_ridge_fits(function, *, max_entries=4096):
    """Reuse fits only while this batch's tracks stay unchanged.

    Each fit cache has at most 4,096 entries by default. Serialized batch
    snapshots, including validated ceiling vectors, and digest bytes are bounded
    to 4 MiB; Python object/container overhead is additional. Prepared measurements
    use their own 4 MiB bound; private immutable scoring inputs have another
    4 MiB budget, and acoustic arrays use 32 MiB. Acoustic screening
    evidence has an additional 8 MiB budget, including conservatively charged
    keys, arrays and containers. Context-local state prevents
    reuse between concurrent jobs, nested batches or later calls. Cached results
    are copied so candidate diagnostics remain independently mutable.
    """
    if type(max_entries) is not int or max_entries < 1:
        raise ValueError("Fit-cache capacity must be a positive integer")

    @wraps(function)
    def run(*args, **kwargs):
        token = _ridge_fit_cache.set(_RidgeFitCache(max_entries))
        try:
            return function(*args, **kwargs)
        finally:
            _ridge_fit_cache.reset(token)

    return run


def _frequency_scale(event):
    quality = event.get("quality", {})
    return quality.get("frequency_loss_scale_hz", quality.get("frequency_sigma_hz",
                       max(1.0, quality.get("frequency_uncertainty_hz", 1.0))))


def _source_fit(factor, u, y, weight, bounds, cap, scale, basis=None):
    """Exact two-variable box least squares; avoid iterative bounded solves."""
    factor2, u2 = (factor**2, u**2) if basis is None else basis
    weighted_factor2 = weight*factor2
    weighted_factor = weight*factor
    a = np.sum(weighted_factor2)
    r = np.sum(weighted_factor*y)
    if not cap:
        return np.array([np.clip(r/a, *bounds), 0.0])
    b = np.sum(weighted_factor2*u)
    prior = (2*scale/cap)**2
    d = np.sum(weighted_factor2*u2)+prior
    s = np.sum(weighted_factor*u*y)
    determinant = a*d-b*b
    unconstrained = np.array([(r*d-s*b)/determinant, (s*a-r*b)/determinant])
    if bounds[0] <= unconstrained[0] <= bounds[1] and abs(unconstrained[1]) <= cap:
        return unconstrained
    # A convex quadratic's constrained minimum lies on one of these four edges.
    candidates = np.array([[f, np.clip((s-b*f)/d, -cap, cap)] for f in bounds] +
                          [[np.clip((r-b*drift)/a, *bounds), drift] for drift in (-cap, cap)])
    residuals = factor*(candidates[:, :1]+candidates[:, 1:]*u)-y
    costs = np.sum(weight*residuals**2, axis=1)+prior*candidates[:, 1]**2
    return candidates[np.argmin(costs)]


def _altitude_bias(track, bias_m):
    """Shift one geodetic trajectory while preserving its measured derivatives."""
    displacement = ecef(track.lat, track.lon, track.altitude+bias_m)-track.xyz
    velocity = track.speed_vector(track.time)+np.gradient(displacement, track.relative_time, axis=0)
    shifted = Track(track.id, track.time, track.lat, track.lon, track.altitude+bias_m,
                    track.typecode, velocity)
    shifted.altitude_basis = getattr(track, "altitude_basis", ["unspecified"]*len(track.time))
    return shifted


def predict_fitted_ridge(track, observer, receive_times_s, diagnostics, c=343.0):
    """Render the frozen fitted source model; unsupported track times stay NaN."""
    factor, emitted, _ = track.acoustic(receive_times_s, observer, c)
    slope = diagnostics.get("source_drift_hz_s", 0.0)
    if slope and "source_reference_emission_time_s" not in diagnostics:
        raise ValueError("A drifting-source prediction needs its fitted emission-time reference")
    reference = diagnostics.get("source_reference_emission_time_s", 0.0)
    return factor*(diagnostics["f0_hz"]+slope*(emitted-reference))


def association_candidates(table, events, offset_tolerance_s=5.0, time_origin=None,
                           time_tolerance_s=5.0, silence=None, min_events=1,
                           max_events=None, max_candidates=20000, anchor_window=None,
                           pair_screen=None):
    """Bounded subset hypotheses, retaining explicitly unmatched proposals.

    Default to every independent group. Intersect shared-clock intervals before
    extending a branch; do not materialize impossible Cartesian products. Index
    entries, branch visits and stored assignments share an explicit work budget.
    Exceeding it fails visibly so the outer driver reports incomplete coverage.
    An optional half-open anchor window selects the first matched passage's
    implied recording start without clipping the shared-clock fit interval.
    ``pair_screen`` may remove an event/flight/receiver row only when its caller
    supplies supported rejection evidence for the fixed shared recording clock;
    its bounded report is returned with the candidate counts.
    """
    from .search import filter_association_event
    max_events = max(min_events, len(events)) if max_events is None else max_events
    if not 1 <= min_events <= max_events or max_candidates < 1:
        raise ValueError("Invalid association event/candidate budget")
    if anchor_window is not None and (len(anchor_window) != 2 or
            np.isnan(anchor_window[0]) or np.isnan(anchor_window[1]) or
            anchor_window[0] >= anchor_window[1]):
        raise ValueError("Anchor window must be an increasing half-open interval")
    candidates, examined, work = [], 0, 0
    pair_screen_reports = []
    def reserve(units):
        nonlocal work
        work += units
        if work > max_candidates*20:
            raise ValueError("Association work/storage budget exceeded; search is incomplete")
    reserve(len(table))
    table = table.reset_index(drop=True)
    records = table.to_dict("records")
    grouped, survivors = [], []
    for event_index, event in enumerate(events):
        frame = filter_association_event(table, event, offset_tolerance_s, time_origin, time_tolerance_s)
        reserve(len(frame))
        before_screen = len(frame)
        if pair_screen is not None and len(frame):
            frame, report = pair_screen(event_index, event, frame)
            if (not hasattr(frame, "index") or len(frame) > before_screen
                    or not frame.index.isin(table.index).all()):
                raise ValueError("Pair screen must return rows from the supplied CPA table")
            if not isinstance(report, dict):
                raise ValueError("Pair screen must return an evidence dictionary")
            pair_screen_reports.append({"event_index": event_index, "before": before_screen,
                                       "after": len(frame), **report})
            reserve(before_screen - len(frame))
        survivors.append(len(frame))
        groups = {}
        tolerance = event.get("quality", {}).get("cpa_tolerance_s", 0) + offset_tolerance_s/2
        for point, group in frame.groupby("point_id"):
            indices = list(group.index)
            reserve(len(indices) * max(1, len(indices).bit_length()))
            ordered = sorted((records[i]["receive_time"] - event["cpa_receive_s"], i) for i in indices)
            groups[point] = ([start - tolerance for start, _ in ordered],
                             [start + tolerance for start, _ in ordered],
                             [index for _, index in ordered])
        grouped.append(groups)
    silence_index = {}
    if silence:
        for point, frame in table.groupby("point_id"):
            indices = list(frame.index)
            reserve(len(indices) * max(1, len(indices).bit_length()))
            ordered = sorted((records[i]["receive_time"], i) for i in indices)
            silence_index[point] = ([value for value, _ in ordered], [i for _, i in ordered])

    def indexed_options(index, point_id, lower, upper):
        starts, ends, indices = grouped[index][point_id]
        # These endpoints use exactly the arithmetic of the intersection below,
        # including floating-point rounding at inclusive boundaries.
        left = bisect_left(ends, lower)
        right = bisect_right(starts, upper)
        # The returned original-order slice must itself fit the work budget;
        # reserve before allocating and sorting it at every branch.
        selected = max(0, right - left)
        if selected:
            reserve(selected * selected.bit_length())
            return sorted(indices[left:right])
        # Most skipped events have no overlapping row. Avoid allocating and
        # sorting an empty slice while preserving the same branch accounting.
        return ()

    def row_options(indices):
        return chain((records[i] for i in indices), [None])

    for point_id in sorted(set().union(*(g.keys() for g in grouped))):
        available = [i for i, group in enumerate(grouped) if point_id in group]
        if len(available) < min_events:
            continue
        lower, upper = (-np.inf, np.inf) if time_origin is None else (time_origin-time_tolerance_s, time_origin+time_tolerance_s)
        path, used = [], set()
        option_lo, option_hi = anchor_window if anchor_window is not None else (lower, upper)
        frames = [(0, row_options(indexed_options(available[0], point_id, option_lo, option_hi)), lower, upper, 0)]
        while frames:
            cursor, options, lower, upper, depth = frames[-1]
            while len(path) > depth:
                _, old = path.pop()
                used.remove((old["flight_id"], old["emission_time"]))
            try:
                row = next(options)
            except StopIteration:
                frames.pop()
                continue
            reserve(1)
            lo, hi = lower, upper
            index = available[cursor]
            if row is not None:
                identity = (row["flight_id"], row["emission_time"])
                if identity in used:
                    continue
                start = row["receive_time"]-events[index]["cpa_receive_s"]
                if not path and anchor_window is not None and not anchor_window[0] <= start < anchor_window[1]:
                    continue
                tolerance = events[index].get("quality", {}).get("cpa_tolerance_s", 0)+offset_tolerance_s/2
                lo, hi = max(lo, start-tolerance), min(hi, start+tolerance)
                if lo > hi:
                    continue
                path.append((index, row))
                used.add(identity)
            following = cursor+1
            if len(path)+len(available)-following < min_events:
                continue
            while following < len(available) and len(path) < max_events:
                option_lo, option_hi = anchor_window if not path and anchor_window is not None else (lo, hi)
                indices = indexed_options(available[following], point_id, option_lo, option_hi)
                if indices:
                    frame = (following, row_options(indices), lo, hi, len(path))
                    if row is None:
                        # None is the last choice, so its exhausted parent does
                        # not need to remain on the stack while skipping ahead.
                        frames[-1] = frame
                    else:
                        frames.append(frame)
                    break
                # An empty interval has only the unmatched choice. Charge that
                # visit exactly as before, without a frame/iterator round trip.
                reserve(1)
                following += 1
                if len(path)+len(available)-following < min_events:
                    break
            if following < len(available) and len(path) < max_events:
                continue
            if len(path) < min_events:
                continue
            examined += 1
            passes = tuple(row for _, row in path)
            if silence and len(path) > 1:
                margin = silence.get("edge_margin_s", 60)
                distance = max(p["distance_m"] for p in passes)*silence.get("distance_multiplier", 1)
                times, indices = silence_index[point_id]
                left = bisect_right(times, passes[0]["receive_time"] + margin)
                right = bisect_left(times, passes[-1]["receive_time"] - margin)
                reserve(2 * max(1, len(times).bit_length()) + max(0, right - left))
                if any(records[i]["distance_m"] <= distance and
                       (records[i]["flight_id"], records[i]["emission_time"]) not in used
                       for i in indices[left:right]):
                    continue
            if len(candidates) >= max_candidates:
                raise ValueError("Association candidate budget exceeded; search is incomplete")
            reserve(len(path))
            candidates.append({"point_id": point_id, "passes": passes, "event_indices": [i for i, _ in path],
                               "recording_start_bounds_s": [float(lo), float(hi)]})
    return candidates, {"per_event_survivors": survivors, "survivors": len(candidates),
                        "hypotheses_examined": examined, "work_units": work, "max_matched_events": max_events,
                        "event_cap_limited": max_events < len(events),
                        "unmatched_proposals_allowed": True,
                        "event_pair_screen": pair_screen_reports}


def association_score(result, events):
    """Compare different matched subsets using bounded normalized support.

    One well-fitted independent event contributes up to one unit. A poor event
    invalidates that hypothesis; another subset can leave it unmatched. This is
    a ranking heuristic, not a likelihood or a calibrated confidence statistic.
    """
    limits = [3 * _frequency_scale(e) for e in events]
    losses = [-s for s in result["event_scores"]]
    supported = all(np.isfinite(v) and v <= limit for v, limit in zip(losses, limits))
    return dict(result, residual_score=result["score"],
                score=float(sum(1-v/limit for v, limit in zip(losses, limits))) if supported else -np.inf,
                score_kind="matched_group_support_minus_normalized_residual")


def prepare_events(detected_events, c=343.0, max_distance_m=20000.0, *, unmatched=None):
    """Adapt contract ridge points to existing search inputs without truth hints.

    Keep every ridge in a canonical group as evidence for one proposal. Measured
    episode scales (legacy rational ratios) normalize to its lowest observed band; no fundamental
    needs to be present. By default a group with inconsistent ratios raises.
    When ``unmatched`` is a list, a ridge that cannot be reconciled with its
    group's lowest band, and a group left with too few samples, stay unmatched
    instead: each is appended to ``unmatched`` with its reason and never scored.
    Separate paths that reuse measured peaks remain separate hypotheses, with
    an explicit conflict preventing their joint independent-support score.
    Timing uncertainty spans the observed window, because a curved track need
    not have its geometric CPA at a fitted straight-pass center.
    """
    from .discovery import _HARMONIC_RATIOS, _overlap
    from .ridge_overlap import reused_ridge_observations

    groups = {}
    for event in detected_events:
        key = event.get("harmonic_group_id") or event["event_id"]
        groups.setdefault(key, []).append(event)
    settings, measurements, retained_groups = [], [], []
    for group_id, members in groups.items():
        if any(e.get("quality", {}).get("matching_eligible") == 0 for e in members):
            continue
        reference = min(members, key=lambda e: np.median([p["frequency_hz"] for p in e["ridge"]]))
        reference_time = np.array([p["time_s"] for p in reference["ridge"]])
        reference_frequency = np.array([p["frequency_hz"] for p in reference["ridge"]])
        times, frequencies, weights, used = [], [], [], []
        for member in members:
            t = np.array([p["time_s"] for p in member["ridge"]], dtype=float)
            f = np.array([p["frequency_hz"] for p in member["ridge"]], dtype=float)
            w = np.array([p.get("weight", 1.0) for p in member["ridge"]], dtype=float)
            harmonic = 1.0
            modern_scale = member.get("quality", {}).get("episode_frequency_scale")
            reference_scale = reference.get("quality", {}).get("episode_frequency_scale")
            if modern_scale is not None and reference_scale is not None:
                if not np.isfinite(modern_scale) or not np.isfinite(reference_scale) or min(modern_scale, reference_scale) <= 0:
                    raise ValueError("Episode frequency scales must be finite and positive")
                harmonic = modern_scale / reference_scale
                # Discovery established the joint scale model. Requiring every
                # child to re-match one base would discard partial support that
                # was reconciled through other measured members.
            elif member is not reference:
                base = np.column_stack((reference_time, reference_frequency))
                shared = _overlap(base, np.column_stack((t, f)))
                reason = None
                if shared is None or len(shared[0]) * 0.5 < 0.6 * min(np.ptp(reference_time), np.ptp(t)):
                    reason = "ridges lack sufficient shared observations"
                else:
                    ratio = shared[2] / shared[1]
                    harmonic = min([1.0, *_HARMONIC_RATIOS], key=lambda value: abs(value - np.median(ratio)))
                    if np.quantile(np.abs(ratio - harmonic) / harmonic, 0.9) >= 0.008:
                        reason = "inconsistent supported frequency ratio"
                if reason is not None:
                    if unmatched is None:
                        raise UnreconciledHarmonicGroup(f"Harmonic group {group_id}: {reason}")
                    unmatched.append({"group_id": group_id, "event_ids": [member["event_id"]],
                                      "reference_event_id": reference["event_id"], "reason": reason})
                    continue
            used.append(member)
            times.extend(t)
            frequencies.extend(f / harmonic)
            # Equal aggregate weight per ridge prevents duplicated harmonics
            # from manufacturing additional independent events.
            weights.extend(w / max(np.sum(w), 1e-12))
        order = np.argsort(times, kind="stable")
        t, f, w = (np.asarray(a)[order] for a in (times, frequencies, weights))
        if unmatched is not None and len(t) < 10:
            unmatched.append({"group_id": group_id, "event_ids": [e["event_id"] for e in used],
                              "reference_event_id": reference["event_id"],
                              "reason": "fewer than ten ridge samples"})
            continue
        if len(t) < 10 or not np.all(np.isfinite(np.column_stack((t, f, w)))) or np.any(w < 0):
            raise ValueError("Matching requires at least ten finite, nonnegative-weight ridge samples")
        members = used
        start, end = min(e["start_s"] for e in members), max(e["end_s"] for e in members)
        center = float(reference.get("cpa_receive_s", (start + end) / 2))
        quality = dict(reference.get("quality", {}))
        quality.setdefault("frequency_loss_scale_hz", max(1.0, _frequency_scale(reference)))
        quality["cpa_tolerance_s"] = max(center - start, end - center)
        # A subsonic source gives f_source = f_received * (1 + v_radial/c).
        # These broad bounds are independent of an unreliable straight-pass fit.
        # Do not impose a minimum speed or near-sonic maximum on CPA proposals;
        # motion support and the physical fit determine whether they can score.
        event = dict(reference, start_s=start, end_s=end, cpa_receive_s=center,
                     speed_range_mps=[0.0, c], slant_range_m=[0.0, max_distance_m],
                     f0_range_hz=[float(f.min() * 0.1), float(f.max() * 1.9)], quality=quality)
        event["event_ids"] = [e["event_id"] for e in members]
        settings.append(event)
        measurements.append((RidgeTimes(t), f, w, {}))
        retained_groups.append((group_id, members))
    # Match the prepared settings exactly: ineligible episodes have no setting
    # and must not shift the indices used to attach reused-observation conflicts.
    for index, (first_id, first_members) in enumerate(retained_groups):
        for following in range(index + 1, len(retained_groups)):
            second_id, second_members = retained_groups[following]
            conflict = next((found for first in first_members for second in second_members
                             if (found := reused_ridge_observations(first, second)) is not None), None)
            if conflict is not None:
                conflict = dict(conflict, group_ids=[first_id, second_id])
                for setting in (settings[index], settings[following]):
                    setting.setdefault("shared_ridge_observations", []).append(conflict)
    order = np.argsort([e["cpa_receive_s"] for e in settings])
    return [settings[i] for i in order], [measurements[i] for i in order]


def _ridge_fit_inputs(measurement, sigma_hz, drift_fraction):
    """Validate the same measurement/model domain for fits and score ceilings."""
    private = _prepared_score_input(measurement)
    if (private is not None and type(sigma_hz) is float and type(drift_fraction) is float
            and sigma_hz == private.sigma and drift_fraction == private.fraction):
        return private.arrays
    spec, ridge, weight, _ = measurement
    times, ridge, weight = (np.asarray(a, dtype=float) for a in (spec.time, ridge, weight))
    if times.shape != ridge.shape or times.shape != weight.shape or times.ndim != 1:
        raise ValueError("Ridge time, frequency and weight must be matching one-dimensional arrays")
    if not np.isfinite(sigma_hz) or sigma_hz <= 0 or not 0 <= drift_fraction <= 0.1:
        raise ValueError("Need positive frequency uncertainty and drift fraction in [0, 0.1]")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("Ridge weights must be finite and nonnegative")
    return times, ridge, weight


@dataclass
class _PreparedRidge:
    times: np.ndarray
    ridge: np.ndarray
    weight: np.ndarray
    independent_samples: int | None = None
    normalized_weight: np.ndarray | None = None

    def fit_values(self):
        # Preserve the original order: normalization is needed only after the
        # acoustic model has supplied finite values for all measured samples.
        if self.normalized_weight is None:
            count = len(np.unique(self.times))
            normalized = self.weight * count / self.weight.sum()
            self.independent_samples = count
            self.normalized_weight = normalized
        return self.ridge, self.normalized_weight, self.independent_samples


def _prepare_ridge_fit_inputs(measurement, sigma_hz, drift_fraction):
    private = _prepared_score_input(measurement)
    if (private is not None and type(sigma_hz) is float and type(drift_fraction) is float
            and sigma_hz == private.sigma and drift_fraction == private.fraction):
        # Keep normalization lazy and owned by the existing scalar cache.
        return _PreparedRidge(*private.selected)
    times, ridge, weight = _ridge_fit_inputs(measurement, sigma_hz, drift_fraction)
    measured = np.isfinite(times) & np.isfinite(ridge) & (weight > 0)
    return _PreparedRidge(times[measured], ridge[measured], weight[measured])


def _fit_ridge_uncached(track, observer, measurement, start_time, f0_bounds, c=343.0,
                        sigma_hz=1.0, drift_fraction=0.02, acoustic_result=None, _prepared=None):
    """Bounded linear source drift at emission time, with robust residual loss.

    Drift is bounded to a fraction of source frequency over the measured event.
    It cannot supply independent event time shifts or arbitrary curve warping.
    Zero-weight samples are missing evidence; negative/nonfinite weights fail.
    """
    prepared = (_prepare_ridge_fit_inputs(measurement, sigma_hz, drift_fraction)
                if _prepared is None else _prepared)
    failed = {"score": -np.inf, "f0_hz": np.nan, "support_fraction": 0.0}
    if len(prepared.times) < 10:
        return failed
    if acoustic_result is None:
        factor, emission, _ = track.acoustic(start_time + prepared.times, observer, c)
    else:
        factor, emission = acoustic_result
        if factor.shape != prepared.times.shape or emission.shape != factor.shape:
            raise ValueError("Batched acoustic result does not match the measured ridge")
    if not np.all(np.isfinite(factor)) or not np.all(np.isfinite(emission)):
        return failed
    y, w, independent_samples = prepared.fit_values()
    span = max(float(np.ptp(emission)), 1e-9)
    u = (emission - (emission.min() + emission.max()) / 2) / span
    cap = float(np.clip(np.median(y / factor), *f0_bounds)) * drift_fraction
    # NumPy scalar parameters can use different promotion and power rules.
    # Preserve those callers and explicit NumPy error policies on the Python path.
    if (_native_fit_irls is not None and type(sigma_hz) is float and type(cap) is float
            and all(type(bound) is float for bound in f0_bounds)
            and np.geterr() == _NATIVE_RIDGE_ERROR_POLICY):
        try:
            native = _native_fit_irls(factor, u, y, w, float(f0_bounds[0]),
                                      float(f0_bounds[1]), cap, float(sigma_hz),
                                      independent_samples)
        except (BufferError, TypeError, ValueError, OverflowError):
            native = None
        if native is not None and all(np.isfinite(value) for value in native):
            score, f0, drift, rmse, outlier_fraction = native
            return {"score": float(score), "f0_hz": float(f0),
                    "source_drift_hz": float(drift), "source_drift_hz_s": float(drift/span),
                    "source_reference_emission_time_s": float((emission.min()+emission.max())/2),
                    "source_span_s": span, "drift_bound_hz": float(cap),
                    "drift_at_bound": bool(cap and abs(drift) > 0.99*cap),
                    "rmse_hz": float(rmse), "support_fraction": 1.0,
                    "sample_count": int(len(y)), "independent_sample_count": independent_samples,
                    "outlier_fraction": float(outlier_fraction)}
    effective = w.copy()
    basis = (factor**2, u**2)
    for iteration in range(6):
        fit = _source_fit(factor, u, y, effective, f0_bounds, cap, sigma_hz, basis)
        residual = factor*(fit[0]+fit[1]*u)-y
        residual_scale = np.sqrt(1 + (residual / sigma_hz) ** 2)
        if iteration < 5:
            effective = w / residual_scale
    loss = np.average(2 * sigma_hz**2 * (residual_scale - 1), weights=w)
    drift = float(fit[1])
    penalty = (2 * sigma_hz * drift / cap)**2 / independent_samples if cap else 0.0
    return {"score": -float(np.sqrt(loss + penalty)), "f0_hz": float(fit[0]),
            "source_drift_hz": drift, "source_drift_hz_s": drift / span,
            "source_reference_emission_time_s": float((emission.min()+emission.max())/2),
            "source_span_s": span,
            "drift_bound_hz": float(cap), "drift_at_bound": bool(cap and abs(drift) > 0.99 * cap),
            "rmse_hz": float(np.sqrt(np.average(residual**2, weights=w))),
            "support_fraction": 1.0, "sample_count": int(len(y)),
            "independent_sample_count": independent_samples,
            "outlier_fraction": float(np.average(np.abs(residual) > 3 * sigma_hz, weights=w))}


def fit_ridge(track, observer, measurement, start_time, f0_bounds, c=343.0,
              sigma_hz=1.0, drift_fraction=0.02, acoustic_result=None):
    """Fit bounded source drift, optionally reusing an identical batch input."""
    cache = _ridge_fit_cache.get()
    fit = _fit_ridge_uncached if cache is None else cache.fit
    return fit(track, observer, measurement, start_time, f0_bounds, c, sigma_hz, drift_fraction,
               acoustic_result=acoustic_result)


def score_ridges(tracks, events, measurements, observer, recording_start_s, c=343.0, *, acoustic_results=None):
    """Score all events at exactly one recording clock; report nuisance fits."""
    if not events or len(tracks) != len(events) or len(measurements) != len(events):
        raise ValueError("Need exactly one track and measurement for each event")
    diagnostics = []
    if acoustic_results is None:
        acoustic_results = [None] * len(events)
    if len(acoustic_results) != len(events):
        raise ValueError("Need one batched acoustic result per event")
    for track, event, measurement, acoustic_result in zip(tracks, events, measurements, acoustic_results):
        quality = event.get("quality", {})
        diagnostics.append(fit_ridge(track, observer, measurement, recording_start_s,
            event["f0_range_hz"], c, _frequency_scale(event),
            quality.get("source_drift_fraction", 0.02), acoustic_result=acoustic_result))
    return {"score": float(sum(d["score"] for d in diagnostics)),
            "event_scores": [d["score"] for d in diagnostics],
            "f0_hz": [d["f0_hz"] for d in diagnostics], "event_diagnostics": diagnostics,
            "score_kind": "negative_robust_ridge_loss_hz"}


def batch_acoustic_trials(tracks, measurements, observer, starts, c=343.0, max_samples=262144):
    """Evaluate repeated receive times together, falling back on support errors."""
    return list(_iter_acoustic_trials(tracks, measurements, observer, starts, c, max_samples))


def _iter_acoustic_trials(tracks, measurements, observer, starts, c=343.0, max_samples=262144):
    """Defer each group's batch until a surviving trial actually needs it."""
    from .geometry import AcousticSupportError

    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 1 or not np.all(np.isfinite(starts)):
        yield from (None for _ in measurements)
        return
    remaining = max_samples
    for track, measurement in zip(tracks, measurements):
        spec, ridge, weight, _ = measurement
        times, ridge, weight = (np.asarray(a, dtype=float) for a in (spec.time, ridge, weight))
        # Once the solver tolerance is below half an emission-time ULP, a
        # stopped individual trial remains unchanged in further iterations.
        # Negative or lower-precision epochs retain the per-trial path.
        if (track.time[0] < 0 or
                np.spacing(track.time[0]) <= 2 * _EMISSION_ITERATION_TOLERANCE_S):
            yield None
            continue
        if times.shape != ridge.shape or times.shape != weight.shape or times.ndim != 1:
            yield None
            continue
        measured = np.isfinite(times) & np.isfinite(ridge) & (weight > 0)
        size = len(starts) * int(measured.sum())
        if measured.sum() < 10 or not np.all(np.isfinite(weight)) or np.any(weight < 0) or size > remaining:
            yield None
            continue
        try:
            cache = _ridge_fit_cache.get()
            if cache is None or cache.acoustic_limit <= 0:
                receive = starts[:, None] + times[measured][None, :]
                unique, inverse = np.unique(receive, return_inverse=True)
                factor, emission, _ = track.acoustic(unique, observer, c)
                batch = factor[inverse].reshape(receive.shape), emission[inverse].reshape(receive.shape)
            else:
                batch = cache.acoustic_batch(track, observer, starts, times[measured], c)
        except AcousticSupportError:
            yield None
            continue
        remaining -= size
        yield batch


def _geographic_refinement_frame(candidate, radius_m, bounds_wsen):
    west, south, east, north = bounds_wsen
    lon0, lat0 = candidate["lon"], candidate["lat"]
    if (not np.isfinite(bounds_wsen).all() or west >= east or south >= north
            or not west <= lon0 <= east or not south <= lat0 <= north):
        raise ValueError("Refinement seed must lie within finite, ordered geographic bounds")
    geod = Geod(ellps="WGS84")
    lon_scale = geod.inv(lon0, lat0, lon0 + 1, lat0)[2]
    lat_scale = geod.inv(lon0, lat0, lon0, lat0 + 0.1)[2] * 10
    limits = [(max(-radius_m, (west-lon0)*lon_scale), min(radius_m, (east-lon0)*lon_scale)),
              (max(-radius_m, (south-lat0)*lat_scale), min(radius_m, (north-lat0)*lat_scale))]
    return limits, (lon_scale, lat_scale)


def refine_candidate(candidate, tracks, events, measurements, terrain, radius_m=500.0,
                     clock_radius_s=5.0, altitude_offsets_m=(-150.0, 0.0, 150.0),
                     crs="EPSG:5070", c=343.0, recording_start_window_s=None,
                     bounds_wsen=None, receiver_mask=None, progress_callback=None):
    """Jointly refine position/shared clock and profile vertical-datum sensitivity.

    A vertical offset changes the observer-to-track relative height; it is a
    sensitivity scenario, not a new terrain measurement or calibrated interval.
    Returned radius describes the explored alternatives only. A boundary optimum
    warns callers that this local search has not enclosed the possible solution.
    """
    return _refine_candidate(candidate, tracks, events, measurements, terrain,
        radius_m=radius_m, clock_radius_s=clock_radius_s,
        altitude_offsets_m=altitude_offsets_m, crs=crs, c=c,
        recording_start_window_s=recording_start_window_s, bounds_wsen=bounds_wsen,
        receiver_mask=receiver_mask, progress_callback=progress_callback)


def _refine_candidate(candidate, tracks, events, measurements, terrain, radius_m=500.0,
                      clock_radius_s=5.0, altitude_offsets_m=(-150.0, 0.0, 150.0),
                      crs="EPSG:5070", c=343.0, recording_start_window_s=None,
                      bounds_wsen=None, receiver_mask=None, *, _transformers=None,
                      progress_callback=None):
    """Share coordinate operations only within one declared sensitivity profile."""
    if receiver_mask is not None and not receiver_mask.contains(candidate["lon"], candidate["lat"]):
        raise ValueError("Refinement seed lies outside the receiver-country mask")
    if _transformers is None:
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    else:
        forward, inverse = _transformers
    x0, y0 = forward.transform(candidate["lon"], candidate["lat"])
    start = candidate["recording_start_time_s"]
    selected = [tracks[f] for f in candidate["flight_ids"]]
    if "event_indices" in candidate:
        events = [events[i] for i in candidate["event_indices"]]
        measurements = [measurements[i] for i in candidate["event_indices"]]
    if radius_m <= 0 or clock_radius_s < 0 or 0.0 not in altitude_offsets_m:
        raise ValueError("Need positive spatial/nonnegative clock bounds and a nominal zero-altitude scenario")
    lower, upper = -clock_radius_s, clock_radius_s
    if recording_start_window_s is not None:
        lo, hi = recording_start_window_s
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("Recording-start window must be finite and ordered")
        lower, upper = max(lower, lo-start), min(upper, hi-start)
    if lower > upper:
        raise ValueError("Refinement clock bounds do not intersect the recording-start window")
    fixed_clock = lower == upper
    bounds = [(-radius_m, radius_m), (-radius_m, radius_m)]
    if bounds_wsen is not None:
        west, south, east, north = bounds_wsen
        # Use east/north metre-scaled geographic coordinates so Powell's box
        # constraints intersect the geographic rectangle exactly. A projected
        # x/y rectangle does not align with lines of latitude and longitude.
        lon0, lat0 = candidate["lon"], candidate["lat"]
        bounds, (lon_scale, lat_scale) = _geographic_refinement_frame(candidate, radius_m, bounds_wsen)
    initial = np.zeros(2) if fixed_clock else np.array([0.0, 0.0, np.clip(0.0, lower, upper)])
    if not fixed_clock:
        bounds.append((lower, upper))
    profiles = []
    altitude_profile_total = len(altitude_offsets_m)
    max_iterations = 60
    for offset_index, offset in enumerate(altitude_offsets_m):
        optimizer_iterations = 0
        objective_evaluations = 0
        last_objective_report = 0.0

        def report_optimizer_progress(fraction):
            if progress_callback is not None:
                progress_callback(fraction, altitude_profile=offset_index + 1,
                                 altitude_profiles=altitude_profile_total,
                                 optimizer_iteration=optimizer_iterations,
                                 optimizer_evaluations=objective_evaluations)

        report_optimizer_progress(offset_index / altitude_profile_total)

        def evaluate(parameters):
            dt = lower if fixed_clock else parameters[2]
            if bounds_wsen is None:
                lon, lat = inverse.transform(x0 + parameters[0], y0 + parameters[1])
            else:
                # Clip only roundoff at the closed geographic boundary.
                lon = float(np.clip(lon0 + parameters[0]/lon_scale, west, east))
                lat = float(np.clip(lat0 + parameters[1]/lat_scale, south, north))
            # A country boundary restricts receiver locations, including every
            # sensitivity scenario. Never sample or score a foreign receiver.
            if receiver_mask is not None and not receiver_mask.contains(lon, lat):
                return {"score": -np.inf}, lat, lon, np.nan
            elevation = float(np.asarray(terrain.sample(lat, lon)))
            score = score_ridges(selected, events, measurements, ecef(lat, lon, elevation + offset),
                                  start + dt, c)
            if "event_indices" in candidate:
                score = association_score(score, events)
            return score, lat, lon, elevation
        def objective(parameters):
            nonlocal objective_evaluations, last_objective_report
            if progress_callback is not None:
                objective_evaluations += 1
            value = evaluate(parameters)[0]["score"]
            now = time.monotonic()
            if (progress_callback is not None and
                    (objective_evaluations == 1 or objective_evaluations % 8 == 0
                     or now - last_objective_report >= 1.0)):
                local = (offset_index + min(0.99, optimizer_iterations / max_iterations)) / altitude_profile_total
                report_optimizer_progress(local)
                last_objective_report = now
            return -value if np.isfinite(value) else 1e6

        def optimizer_callback(_parameters):
            nonlocal optimizer_iterations
            optimizer_iterations += 1
            local = (offset_index + min(0.99, optimizer_iterations / max_iterations)) / altitude_profile_total
            report_optimizer_progress(local)

        options = {"xtol": 0.01, "ftol": 1e-7, "maxiter": max_iterations}
        result = minimize(objective, initial, method="Powell", bounds=bounds,
                          callback=optimizer_callback if progress_callback is not None else None,
                          options=options)
        if receiver_mask is None:
            parameters = result.x if objective(result.x) < objective(initial) else initial
            score, lat, lon, elevation = evaluate(parameters)
        else:
            # A finite optimization penalty must not make an inadmissible
            # border proposal preferable to a valid, low-scoring incumbent.
            initial_fit, optimized_fit = evaluate(initial), evaluate(result.x)
            improves = (np.isfinite(optimized_fit[0]["score"])
                        and (not np.isfinite(initial_fit[0]["score"])
                             or optimized_fit[0]["score"] > initial_fit[0]["score"]))
            parameters = result.x if improves else initial
            score, lat, lon, elevation = optimized_fit if improves else initial_fit
        if not np.isfinite(score["score"]):
            report_optimizer_progress((offset_index + 1) / altitude_profile_total)
            continue
        dt = lower if fixed_clock else parameters[2]
        spatial_boundary = any(min(value-lo, hi-value) <= radius_m * 0.02
                               for value, (lo, hi) in zip(parameters[:2], bounds[:2]))
        if receiver_mask is not None:
            # Powell explores a square about the seed. Its diagonal reaches
            # sqrt(2) times the per-axis radius; checking only the winning point
            # can miss a country edge that truncated another part of that square.
            spatial_boundary |= bool(receiver_mask.boundary_limited(
                candidate["lon"], candidate["lat"], np.sqrt(2.0) * radius_m))
        clock_boundary = bool(not fixed_clock and min(dt-lower, upper-dt) <= (upper-lower)*0.02)
        profiles.append(dict(score, lat=lat, lon=lon, elev_m=elevation,
            recording_start_time_s=float(start + dt), altitude_offset_m=float(offset),
            boundary_limited=spatial_boundary or clock_boundary,
            position_boundary_limited=spatial_boundary, clock_boundary_limited=clock_boundary,
            optimizer_success=bool(result.success)))
        report_optimizer_progress((offset_index + 1) / altitude_profile_total)
    nominal = next((p for p in profiles if p["altitude_offset_m"] == 0), None)
    if nominal is None:
        raise ValueError("No complete trajectory support for local refinement")
    center = ecef(nominal["lat"], nominal["lon"], 0)
    radius = max(float(np.linalg.norm(ecef(p["lat"], p["lon"], 0) - center)) for p in profiles)
    return dict(candidate, **{k: v for k, v in nominal.items() if k != "altitude_offset_m"},
                time_adjustment_s=float(candidate.get("time_adjustment_s", 0) + nominal["recording_start_time_s"]-start),
                refinement_clock_bounds_s=[float(start+lower), float(start+upper)],
                x=float(forward.transform(nominal["lon"], nominal["lat"])[0]),
                y=float(forward.transform(nominal["lon"], nominal["lat"])[1]),
                sensitivity={"radius_m": radius, "kind": "sensitivity",
                    "method": "profiled_relative_altitude_offsets",
                    "profiles": profiles, "boundary_limited": any(p["boundary_limited"] for p in profiles),
                    "complete": len(profiles) == len(altitude_offsets_m) and all(p["optimizer_success"] for p in profiles)})


_CANONICAL_REFINE_CANDIDATE = refine_candidate


def profile_candidate(candidate, tracks, events, measurements, terrain, *,
                      recording_start_window_s=None, radius_m=500.0, clock_radius_s=5.0,
                      altitude_uncertainty_m=150.0, sound_speed_uncertainty_mps=5.0,
                      c=343.0, crs="EPSG:5070", bounds_wsen=None, receiver_mask=None,
                      progress_callback=None):
    """Run declared perturbations, retaining failures and boundary limitations.

    These finite engineering scenarios are not a calibrated confidence region.
    Frequency offsets and tilts vary each matched ridge separately. Shared-clock
    endpoints, relative height, sound speed and constant-source alternatives are
    reoptimized; no scenario may leave the trusted recording-start window.
    """
    if altitude_uncertainty_m < 0 or not 0 <= sound_speed_uncertainty_mps < c:
        raise ValueError("Sensitivity ranges must be nonnegative and sound speed must remain positive")
    kwargs = dict(radius_m=radius_m, clock_radius_s=clock_radius_s,
                  altitude_offsets_m=(0.0,), crs=crs, c=c,
                  recording_start_window_s=recording_start_window_s, bounds_wsen=bounds_wsen)
    if receiver_mask is not None:
        kwargs["receiver_mask"] = receiver_mask
    transformers = None

    def refine(*args, **options):
        nonlocal transformers
        # Custom refiners keep the original dispatch and receive no private
        # arguments. The standard path owns exactly one pair for this profile.
        progress = options.pop("progress_callback", None)
        current_refiner = refine_candidate
        if current_refiner is not _CANONICAL_REFINE_CANDIDATE:
            return current_refiner(*args, **options)
        if transformers is None:
            transformers = (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
                            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))
        return _refine_candidate(*args, _transformers=transformers,
                                 progress_callback=progress, **options)

    if progress_callback is None:
        nominal = refine(candidate, tracks, events, measurements, terrain, **kwargs)
    else:
        nominal = refine(candidate, tracks, events, measurements, terrain, **kwargs,
                         progress_callback=lambda fraction, **fields: progress_callback(
                             0.15 * fraction, scenario="nominal", **fields))
    scenarios = [("nominal", "nominal", events, measurements, {})]
    for sign in (-1, 1):
        scenarios.append(("altitude", f"height_{sign:+d}", events, measurements,
                          {"altitude_offsets_m": (0.0, sign*altitude_uncertainty_m)}))
        scenarios.append(("sound_speed", f"sound_speed_{sign:+d}", events, measurements,
                          {"c": c+sign*sound_speed_uncertainty_mps}))
    indices = candidate.get("event_indices", list(range(len(events))))
    altitude_provenance, provenance_failures = {}, []
    terrain_ellipsoidal = (getattr(terrain, "vertical_datum", None) == "ellipsoidal" or
                          getattr(terrain, "geoid", None) is not None or getattr(terrain, "geoid_path", None) is not None)
    if not terrain_ellipsoidal:
        provenance_failures.append({"source": "altitude", "scenario": "terrain",
            "reason": "Ellipsoidal terrain datum is not established"})
    for identity in sorted(set(candidate["flight_ids"])):
        track = tracks[identity]
        basis = getattr(track, "altitude_basis", ["unspecified"])
        basis = [basis] if isinstance(basis, str) else list(basis)
        basis = [str(value) for value in basis]
        altitude_provenance[identity] = sorted(set(basis))
        if not basis or not set(basis) <= {"ellipsoidal", "ellipsoidal_interpolated"}:
            provenance_failures.append({"source": "altitude", "scenario": identity,
                "reason": "Common-datum geometric flight altitude is not established"})
        for sign in (-1, 1):
            scenarios.append(("altitude", f"flight_{identity}_{sign:+d}", events, measurements,
                              {"track_altitude_bias": (identity, sign*altitude_uncertainty_m)}))
    for index in indices:
        spec, ridge, weight, fit = measurements[index]
        times = np.asarray(spec.time)
        tilt = 2*(times-(times.min()+times.max())/2)/max(float(np.ptp(times)), 1e-9)
        scale = _frequency_scale(events[index])
        for pattern, shape in (("offset", np.ones_like(times)), ("tilt", tilt)):
            for sign in (-1, 1):
                changed = list(measurements)
                changed[index] = (spec, np.asarray(ridge)+sign*scale*shape, weight, fit)
                scenarios.append(("frequency", f"event_{index}_{pattern}_{sign:+d}", events, changed, {}))
    constant = [dict(e, quality=dict(e.get("quality", {}), source_drift_fraction=0.0)) for e in events]
    scenarios.append(("source_frequency", "constant_source", constant, measurements, {}))
    lo, hi = nominal["refinement_clock_bounds_s"]
    # A trusted point window is tested exactly; it is never perturbed outward.
    for when in sorted(set((lo, hi))):
        scenarios.append(("clock", f"fixed_clock_{when:.6f}", events, measurements,
                          {"recording_start_window_s": (when, when)}))
    profiles, failures = [], list(provenance_failures)
    center = ecef(nominal["lat"], nominal["lon"], 0)
    radius = 0.0
    boundary = False
    scenario_total = len(scenarios) - 1
    for scenario_index, (source, label, scenario_events, scenario_measurements, overrides) in enumerate(scenarios):
        try:
            options = dict(kwargs, **overrides)
            flight_bias = options.pop("track_altitude_bias", None)
            scenario_tracks = tracks
            if flight_bias is not None:
                identity, bias = flight_bias
                scenario_tracks = dict(tracks)
                if bias:
                    scenario_tracks[identity] = _altitude_bias(tracks[identity], bias)
            if source == "nominal":
                row = nominal
            elif progress_callback is None:
                row = refine(candidate, scenario_tracks, scenario_events,
                             scenario_measurements, terrain, **options)
            else:
                scenario_base = 0.15 + 0.85 * (scenario_index - 1) / scenario_total
                scenario_span = 0.85 / scenario_total
                row = refine(candidate, scenario_tracks, scenario_events,
                             scenario_measurements, terrain, **options,
                             progress_callback=lambda fraction, _base=scenario_base,
                             _span=scenario_span, _label=label, **fields: progress_callback(
                                 _base + _span * fraction, scenario=_label, **fields))
            # Height scenarios return a nominal plus the perturbed-height fit;
            # retain the latter rather than silently measuring the nominal twice.
            variants = row["sensitivity"]["profiles"] if source == "altitude" else [row]
            for variant in variants:
                displacement = float(np.linalg.norm(ecef(variant["lat"], variant["lon"], 0)-center))
                radius = max(radius, displacement)
                boundary |= variant["boundary_limited"]
                profiles.append({"source": source, "scenario": label,
                    "lat": variant["lat"], "lon": variant["lon"], "score": variant["score"],
                    "recording_start_time_s": variant["recording_start_time_s"],
                    "displacement_m": displacement, "boundary_limited": variant["boundary_limited"],
                    "flight_altitude_bias": None if flight_bias is None else {"flight_id": flight_bias[0], "bias_m": flight_bias[1]},
                    "observer_altitude_offset_m": variant.get("altitude_offset_m", 0.0),
                    "optimizer_success": variant["optimizer_success"]})
            if not row["sensitivity"]["complete"]:
                failures.append({"source": source, "scenario": label, "reason": "Missing altitude scenario support"})
        except ValueError as exc:
            failures.append({"source": source, "scenario": label, "reason": str(exc)})
        if progress_callback is not None:
            completed = max(0, scenario_index)
            fraction = 0.15 + 0.85 * completed / scenario_total
            progress_callback(fraction, scenario=label,
                              scenarios_completed=completed, scenarios_total=scenario_total)
    sensitivity = {"radius_m": radius, "kind": "sensitivity", "method": "declared_perturbation_profiles",
        "sources": sorted({s[0] for s in scenarios if s[0] != "nominal"}),
        "complete": not failures and all(p["optimizer_success"] for p in profiles),
        "boundary_limited": bool(boundary), "profiles": profiles, "failures": failures,
        "altitude_basis": altitude_provenance,
        "terrain_vertical_datum": "ellipsoidal" if terrain_ellipsoidal else "unestablished",
        "settings": {"altitude_uncertainty_m": altitude_uncertainty_m,
                     "sound_speed_uncertainty_mps": sound_speed_uncertainty_mps,
                     "frequency_scale_hz": [_frequency_scale(events[i]) for i in indices],
                     "clock_bounds_s": [lo, hi]}}
    return dict(nominal, sensitivity=sensitivity)



# Captured when the module initializes, before callers can install custom hooks.
_CANONICAL_SCORE_INPUT_FUNCTIONS = (_frequency_scale, _ridge_fit_inputs,
                                   _prepare_ridge_fit_inputs, _fit_argument_key)
_CANONICAL_FIT_RIDGE = fit_ridge
_CANONICAL_SCALAR_FITTER = _fit_ridge_uncached
_CANONICAL_ACOUSTIC_TRIALS = _iter_acoustic_trials
_CANONICAL_NATIVE_IRLS = _native_fit_irls
_CANONICAL_NATIVE_IRLS_BATCH = _native_fit_irls_batch
_CANONICAL_PREFIX_FUNCTIONS = (
    fit_ridge, _fit_ridge_uncached, _iter_acoustic_trials, _source_fit,
    _RidgeFitCache.fit, _RidgeFitCache.acoustic_batch, _native_fit_irls, _native_fit_irls_batch,
    _native_ridge_module, getattr(_native_ridge_module, "event_fit_reason", None))

from .scoring_inputs import (_prepare_score_inputs, _prepared_score_input,
                             _prepared_score_token, _prepared_score_tokens)
