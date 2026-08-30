"""Background external-load monitor for confirmatory latency measurement.

The monitor samples operating-system process state on a locked cadence. Query
threads only read cached samples; they never fork ``ps`` or ``nvidia-smi``
inside or adjacent to a timed solver call. Recollection therefore depends only
on external process state and never on latency or solver output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Mapping


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ps(output: str, *, threshold: float, excluded: set[int]) -> list[dict[str, Any]]:
    busy: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[0])
            cpu = float(fields[1])
        except ValueError:
            continue
        if pid in excluded or cpu < threshold:
            continue
        busy.append(
            {
                "pid": pid,
                "cpu_percent": cpu,
                "stat": fields[2],
                "args": fields[3],
            }
        )
    return busy


def _parse_gpu_processes(output: str, *, excluded: set[int]) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", maxsplit=2)]
        if len(fields) != 3 or fields[0] in {"", "[Not Found]", "N/A"}:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid in excluded:
            continue
        try:
            memory = (
                None
                if fields[2] in {"", "N/A", "[Not Supported]"}
                else int(fields[2])
            )
        except ValueError:
            memory = None
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_gpu_memory_mib": memory,
            }
        )
    return processes


@dataclass(frozen=True)
class GuardCheckpoint:
    cache_read_count: int
    monitor_sample_count: int
    interval_check_count: int
    wait_event_count: int
    contamination_event_count: int


class QuietHostTechnicalInterruption(RuntimeError):
    """External load or monitor failure prevented unbiased collection."""


Sampler = Callable[[], Mapping[str, Any]]


def _proc_cpu_snapshot() -> dict[str, Any]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("/proc/stat has no aggregate CPU row")
    values = [int(value) for value in fields[1:]]
    total_ticks = sum(values)
    idle_ticks = values[3] + (values[4] if len(values) > 4 else 0)
    processes: dict[tuple[int, int], dict[str, Any]] = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            raw = (directory / "stat").read_text(encoding="utf-8")
            right = raw.rfind(")")
            left = raw.find("(")
            if left < 0 or right <= left:
                continue
            command = raw[left + 1 : right]
            rest = raw[right + 2 :].split()
            process_ticks = int(rest[11]) + int(rest[12])
            start_ticks = int(rest[19])
            state = rest[0]
            try:
                args = (
                    (directory / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                args = ""
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
        processes[(pid, start_ticks)] = {
            "pid": pid,
            "process_ticks": process_ticks,
            "state": state,
            "args": args or f"[{command}]",
        }
    return {
        "captured_monotonic_ns": time.monotonic_ns(),
        "total_ticks": total_ticks,
        "idle_ticks": idle_ticks,
        "processes": processes,
    }


class FormalHostGuard:
    """Cache bounded-cadence external-load samples in a background thread."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        sampler: Sampler | None = None,
    ):
        self.cpu_threshold = float(config["max_unrelated_cpu_percent"])
        self.aggregate_cpu_threshold = float(
            config["max_aggregate_external_cpu_percent"]
        )
        self.stable_checks = int(config["quiet_stable_checks"])
        self.poll_seconds = float(config["quiet_poll_seconds"])
        self.sample_interval_seconds = float(config["monitor_sample_interval_seconds"])
        self.max_sample_age_seconds = float(config["max_monitor_sample_age_seconds"])
        self.command_timeout_seconds = float(config["monitor_command_timeout_seconds"])
        self.max_wait_seconds = float(config["max_quiet_wait_seconds"])
        self.max_contaminated_attempts = int(config["max_contaminated_attempts"])
        self.reject_other_gpu_processes = bool(config["reject_other_gpu_compute_pids"])
        configured_names = config.get("allowed_persistent_gpu_compute_process_names", ())
        if not isinstance(configured_names, list) or any(
            not isinstance(value, str) or not value.startswith("/")
            for value in configured_names
        ):
            raise ValueError(
                "allowed persistent GPU process names must be an explicit list of exact paths"
            )
        if len(configured_names) != len(set(configured_names)):
            raise ValueError("allowed persistent GPU process paths must be unique")
        self.allowed_gpu_process_names = frozenset(configured_names)
        if (
            self.cpu_threshold <= 0.0
            or self.aggregate_cpu_threshold <= 0.0
            or self.stable_checks <= 0
            or self.poll_seconds <= 0.0
            or self.sample_interval_seconds <= 0.0
            or self.max_sample_age_seconds < self.sample_interval_seconds
            or self.command_timeout_seconds <= 0.0
            or self.max_wait_seconds <= 0.0
            or self.max_contaminated_attempts <= 0
        ):
            raise ValueError("quiet-host configuration contains an invalid bound")

        # Only the formal process is internal. Parent/launcher processes remain
        # observable so a separate workload cannot be hidden by ancestry.
        self.formal_pid = os.getpid()
        self.excluded_pids = {self.formal_pid}
        self._sampler = sampler or self._sample_external_state
        self._previous_cpu_snapshot: dict[str, Any] | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._monitor_error: BaseException | None = None
        self._latest: dict[str, Any] | None = None
        self._last_busy: dict[str, Any] | None = None
        self._frozen_allowed_gpu_identities: frozenset[tuple[int, str]] | None = None
        self._frozen_allowed_gpu_processes: list[dict[str, Any]] = []
        self._missing_allowlisted_names_at_freeze: list[str] = []
        self._sample_count = 0
        self._cache_read_count = 0
        self._interval_checks: list[dict[str, Any]] = []
        self._wait_events: list[dict[str, Any]] = []
        self._contamination_events: list[dict[str, Any]] = []
        self._max_sample_gap_seconds = 0.0
        self._allowed_baseline_sighting_count = 0
        self._consecutive_quiet_samples = 0
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="test-v4-external-load-monitor",
            daemon=True,
        )
        self._thread.start()
        self._wait_for_initial_sample()

    def _run_command(self, arguments: list[str]) -> tuple[str, int]:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.command_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise QuietHostTechnicalInterruption(
                f"external-state monitor command timed out: {arguments[0]}"
            )
        if process.returncode:
            raise QuietHostTechnicalInterruption(
                f"external-state monitor command failed: {arguments[0]}: {stderr.strip()}"
            )
        return stdout, int(process.pid)

    def _sample_external_state(self) -> Mapping[str, Any]:
        current_cpu = _proc_cpu_snapshot()
        previous_cpu = self._previous_cpu_snapshot
        self._previous_cpu_snapshot = current_cpu
        cpu: list[dict[str, Any]] = []
        unknown_transitions: list[dict[str, Any]] = []
        external_aggregate_cpu_percent = 0.0
        host_total_cpu_percent = 0.0
        cpu_window_seconds: float | None = None
        cpu_window_available = previous_cpu is not None
        if previous_cpu is not None:
            total_delta = int(current_cpu["total_ticks"]) - int(
                previous_cpu["total_ticks"]
            )
            idle_delta = int(current_cpu["idle_ticks"]) - int(
                previous_cpu["idle_ticks"]
            )
            cpu_window_seconds = (
                int(current_cpu["captured_monotonic_ns"])
                - int(previous_cpu["captured_monotonic_ns"])
            ) / 1e9
            if total_delta <= 0 or cpu_window_seconds <= 0.0:
                raise RuntimeError("nonpositive /proc CPU sampling interval")
            host_total_cpu_percent = 100.0 * max(0, total_delta - idle_delta) / total_delta
            cpu_count = max(1, os.cpu_count() or 1)
            previous_processes = previous_cpu["processes"]
            for identity, item in current_cpu["processes"].items():
                pid = int(item["pid"])
                if pid in self.excluded_pids:
                    continue
                previous_item = previous_processes.get(identity)
                # A process born after the previous snapshot has accumulated
                # all of its lifetime ticks inside this monitor window.  It
                # must not receive a one-window grace period.
                tick_delta = (
                    int(item["process_ticks"])
                    if previous_item is None
                    else int(item["process_ticks"])
                    - int(previous_item["process_ticks"])
                )
                if tick_delta <= 0:
                    continue
                percent = 100.0 * tick_delta * cpu_count / total_delta
                external_aggregate_cpu_percent += percent
                if percent >= self.cpu_threshold:
                    cpu.append(
                        {
                            "pid": pid,
                            "cpu_percent_over_monitor_window": percent,
                            "stat": str(item["state"]),
                            "args": str(item["args"]),
                            "process_started_inside_monitor_window": previous_item
                            is None,
                        }
                    )
            # A process that existed at the pre-window snapshot and exited
            # before the post-window snapshot cannot be assigned an exact
            # tick delta. Conservatively invalidate the covered query rather
            # than silently accepting a potentially CPU-polluted interval.
            for identity, item in previous_processes.items():
                if int(item["pid"]) in self.excluded_pids or identity in current_cpu[
                    "processes"
                ]:
                    continue
                unknown_transitions.append(
                    {
                        "pid": int(item["pid"]),
                        "start_ticks": int(identity[1]),
                        "transition": "external_process_exited_inside_monitor_window",
                        "args": str(item["args"]),
                    }
                )
        gpu: list[dict[str, Any]] = []
        if self.reject_other_gpu_processes:
            gpu_output, _ = self._run_command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            gpu = _parse_gpu_processes(gpu_output, excluded=self.excluded_pids)
        return {
            "busy_cpu_processes": cpu,
            "external_aggregate_cpu_percent": external_aggregate_cpu_percent,
            "host_total_cpu_percent": host_total_cpu_percent,
            "cpu_window_seconds": cpu_window_seconds,
            "cpu_window_available": cpu_window_available,
            "unknown_external_process_transitions": unknown_transitions,
            "gpu_compute_processes": gpu,
        }

    def _freeze_and_classify(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        cpu = [dict(item) for item in raw.get("busy_cpu_processes", ())]
        aggregate_cpu = float(raw.get("external_aggregate_cpu_percent", 0.0))
        unknown_transitions = [
            dict(item)
            for item in raw.get("unknown_external_process_transitions", ())
        ]
        gpu = [dict(item) for item in raw.get("gpu_compute_processes", ())]
        if self._frozen_allowed_gpu_identities is None:
            allowed = [
                item
                for item in gpu
                if str(item.get("process_name")) in self.allowed_gpu_process_names
            ]
            identities = {
                (int(item["pid"]), str(item["process_name"])) for item in allowed
            }
            self._frozen_allowed_gpu_identities = frozenset(identities)
            self._frozen_allowed_gpu_processes = allowed
            observed_names = {name for _, name in identities}
            self._missing_allowlisted_names_at_freeze = sorted(
                self.allowed_gpu_process_names - observed_names
            )
        assert self._frozen_allowed_gpu_identities is not None
        allowed_now: list[dict[str, Any]] = []
        foreign: list[dict[str, Any]] = []
        for item in gpu:
            identity = (int(item["pid"]), str(item["process_name"]))
            if identity in self._frozen_allowed_gpu_identities:
                allowed_now.append(item)
            else:
                foreign.append(item)
        self._allowed_baseline_sighting_count += len(allowed_now)
        return {
            "busy_cpu_processes": cpu,
            "allowed_frozen_gpu_compute_processes": allowed_now,
            "foreign_gpu_compute_processes": foreign,
            "external_aggregate_cpu_percent": aggregate_cpu,
            "host_total_cpu_percent": float(raw.get("host_total_cpu_percent", 0.0)),
            "cpu_window_seconds": raw.get("cpu_window_seconds"),
            "cpu_window_available": bool(raw.get("cpu_window_available", False)),
            "unknown_external_process_transitions": unknown_transitions,
            "aggregate_external_cpu_busy": aggregate_cpu
            >= self.aggregate_cpu_threshold,
            "busy": bool(
                cpu
                or foreign
                or unknown_transitions
                or aggregate_cpu >= self.aggregate_cpu_threshold
            ),
            "decision_source": "external_process_state_only",
        }

    def _monitor_loop(self) -> None:
        next_start = time.monotonic()
        while not self._stop.is_set():
            try:
                raw = self._sampler()
                observed_monotonic_ns = time.monotonic_ns()
                with self._condition:
                    classified = self._freeze_and_classify(raw)
                    self._sample_count += 1
                    if classified["busy"]:
                        self._consecutive_quiet_samples = 0
                    else:
                        self._consecutive_quiet_samples += 1
                    sample = {
                        **classified,
                        "sample_index": self._sample_count,
                        "observed_utc": _utc(),
                        "observed_monotonic_ns": observed_monotonic_ns,
                        "consecutive_quiet_samples": self._consecutive_quiet_samples,
                    }
                    if self._latest is not None:
                        gap = (
                            observed_monotonic_ns
                            - int(self._latest["observed_monotonic_ns"])
                        ) / 1e9
                        self._max_sample_gap_seconds = max(
                            self._max_sample_gap_seconds, float(gap)
                        )
                    self._latest = sample
                    if sample["busy"]:
                        self._last_busy = sample
                    self._condition.notify_all()
            except BaseException as error:
                with self._condition:
                    self._monitor_error = error
                    self._condition.notify_all()
                return
            next_start += self.sample_interval_seconds
            delay = max(0.0, next_start - time.monotonic())
            if self._stop.wait(delay):
                return

    def _raise_monitor_error(self) -> None:
        if self._monitor_error is not None:
            raise QuietHostTechnicalInterruption(
                "background external-state monitor failed"
            ) from self._monitor_error

    def _wait_for_initial_sample(self) -> None:
        deadline = time.monotonic() + min(
            self.max_wait_seconds,
            self.command_timeout_seconds * 3.0,
        )
        with self._condition:
            while self._latest is None:
                self._raise_monitor_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise QuietHostTechnicalInterruption(
                        "background external-state monitor produced no initial sample"
                    )
                self._condition.wait(timeout=min(self.poll_seconds, remaining))

    def _cached_sample(
        self,
        *,
        wait_for_fresh: bool,
        minimum_sample_index_exclusive: int | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_wait_seconds
        with self._condition:
            while True:
                self._raise_monitor_error()
                if self._latest is not None:
                    age = (
                        time.monotonic_ns()
                        - int(self._latest["observed_monotonic_ns"])
                    ) / 1e9
                    index_ready = (
                        minimum_sample_index_exclusive is None
                        or int(self._latest["sample_index"])
                        > int(minimum_sample_index_exclusive)
                    )
                    if (
                        index_ready
                        and (not wait_for_fresh or age <= self.max_sample_age_seconds)
                    ):
                        self._cache_read_count += 1
                        return {**self._latest, "cache_age_seconds": float(age)}
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise QuietHostTechnicalInterruption(
                        "background external-state monitor cache remained stale"
                    )
                self._condition.wait(timeout=min(self.poll_seconds, remaining))

    def wait_until_quiet(self, *, context: str) -> dict[str, Any]:
        started = time.monotonic()
        observed_busy: list[dict[str, Any]] = []
        last_index = -1
        while True:
            sample = self._cached_sample(wait_for_fresh=True)
            index = int(sample["sample_index"])
            if index == last_index:
                with self._condition:
                    self._condition.wait(timeout=self.poll_seconds)
                continue
            last_index = index
            if sample["busy"]:
                observed_busy.append(sample)
            elif int(sample["consecutive_quiet_samples"]) >= self.stable_checks:
                event = {
                    "context": context,
                    "wait_seconds": time.monotonic() - started,
                    "had_busy_process": bool(observed_busy),
                    "busy_observations": observed_busy,
                    "monitor_sample_index": index,
                    "monitor_sample_age_seconds": sample["cache_age_seconds"],
                    "consecutive_quiet_samples": int(
                        sample["consecutive_quiet_samples"]
                    ),
                    "timed_out": False,
                }
                # Preserve nontrivial waits; immediate cached admissions are
                # counted through cache/interval coverage rather than logged
                # as 150k near-empty events.
                if observed_busy or float(event["wait_seconds"]) >= self.poll_seconds:
                    with self._condition:
                        self._wait_events.append(event)
                return event
            elapsed = time.monotonic() - started
            if elapsed > self.max_wait_seconds:
                timed_out_event = {
                    "context": context,
                    "wait_seconds": elapsed,
                    "had_busy_process": bool(observed_busy),
                    "busy_observations": observed_busy,
                    "monitor_sample_index": index,
                    "monitor_sample_age_seconds": sample["cache_age_seconds"],
                    "timed_out": True,
                }
                with self._condition:
                    self._wait_events.append(timed_out_event)
                raise QuietHostTechnicalInterruption(
                    "formal latency test did not regain a quiet host; "
                    f"context={context!r}, waited={elapsed:.3f}s, "
                    f"last_sample={sample}"
                )

    def observe(self, *, context: str, since_sample_index: int) -> dict[str, Any]:
        # At least one bounded-cadence monitor sample must close every query
        # interval. This wait is outside perf_counter_ns timing and performs no
        # synchronous process probe.
        sample = self._cached_sample(
            wait_for_fresh=True,
            minimum_sample_index_exclusive=int(since_sample_index),
        )
        with self._condition:
            last_busy = None if self._last_busy is None else dict(self._last_busy)
        latest_index = int(sample["sample_index"])
        busy_since_start = bool(
            last_busy is not None
            and int(last_busy["sample_index"]) > int(since_sample_index)
        )
        interval_samples = max(0, latest_index - int(since_sample_index))
        observation = {
            **sample,
            "context": context,
            "since_monitor_sample_index": int(since_sample_index),
            "monitor_samples_since_query_start": interval_samples,
            "interval_without_new_monitor_sample": interval_samples == 0,
            "busy_sample_since_query_start": last_busy if busy_since_start else None,
            "busy": bool(sample["busy"] or busy_since_start),
            "monitor_sampling_is_background": True,
            "synchronous_process_probe_per_query": False,
            "query_interval_closed_by_new_monitor_sample": interval_samples > 0,
        }
        coverage = {
            "context": context,
            "since_monitor_sample_index": int(since_sample_index),
            "latest_monitor_sample_index": latest_index,
            "monitor_samples_since_query_start": interval_samples,
            "interval_without_new_monitor_sample": interval_samples == 0,
            "post_read_cache_age_seconds": float(sample["cache_age_seconds"]),
        }
        with self._condition:
            self._interval_checks.append(coverage)
        return observation

    def record_contamination(
        self,
        *,
        context: str,
        observation: Mapping[str, Any],
        attempt_index: int,
        discarded_scope: str,
    ) -> None:
        if not bool(observation.get("busy", False)):
            raise ValueError("a quiet observation cannot invalidate an attempt")
        event = {
            "context": context,
            "detected_utc": _utc(),
            "attempt_index": int(attempt_index),
            "discarded_scope": discarded_scope,
            "all_method_outputs_and_timing_repeats_discarded": True,
            "decision_used_solver_or_latency_result": False,
            "observation": dict(observation),
        }
        with self._condition:
            self._contamination_events.append(event)

    def checkpoint(self) -> GuardCheckpoint:
        with self._condition:
            return GuardCheckpoint(
                cache_read_count=self._cache_read_count,
                monitor_sample_count=self._sample_count,
                interval_check_count=len(self._interval_checks),
                wait_event_count=len(self._wait_events),
                contamination_event_count=len(self._contamination_events),
            )

    def summary_since(self, checkpoint: GuardCheckpoint) -> dict[str, Any]:
        with self._condition:
            intervals = self._interval_checks[checkpoint.interval_check_count :]
            waits = self._wait_events[checkpoint.wait_event_count :]
            contaminated = self._contamination_events[
                checkpoint.contamination_event_count :
            ]
            sample_count = self._sample_count - checkpoint.monitor_sample_count
            cache_reads = self._cache_read_count - checkpoint.cache_read_count
            frozen = list(self._frozen_allowed_gpu_processes)
            missing = list(self._missing_allowlisted_names_at_freeze)
            max_gap = self._max_sample_gap_seconds
            sightings = self._allowed_baseline_sighting_count
        sample_coverages = [
            int(item["monitor_samples_since_query_start"]) for item in intervals
        ]
        return {
            "background_monitor": True,
            "synchronous_ps_or_nvidia_smi_per_query": False,
            "monitor_sample_interval_seconds": self.sample_interval_seconds,
            "per_process_cpu_threshold_percent": self.cpu_threshold,
            "aggregate_external_cpu_threshold_percent": self.aggregate_cpu_threshold,
            "cpu_measurement": "/proc tick delta over monitor window",
            "max_monitor_sample_age_seconds": self.max_sample_age_seconds,
            "external_state_cache_read_count": cache_reads,
            "background_monitor_sample_count": sample_count,
            "query_interval_check_count": len(intervals),
            "query_intervals_without_new_monitor_sample": sum(
                bool(item["interval_without_new_monitor_sample"]) for item in intervals
            ),
            "minimum_monitor_samples_since_query_start": (
                min(sample_coverages) if sample_coverages else None
            ),
            "maximum_post_read_cache_age_seconds": max(
                (float(item["post_read_cache_age_seconds"]) for item in intervals),
                default=0.0,
            ),
            "maximum_observed_monitor_sample_gap_seconds": max_gap,
            "coverage_contract": (
                "bounded-cadence cached external-state samples; zero-new-sample "
                "query intervals are explicitly counted"
            ),
            "formal_runner_pid_excluded": self.formal_pid,
            "configured_allowed_gpu_process_names": sorted(
                self.allowed_gpu_process_names
            ),
            "frozen_allowed_gpu_processes": frozen,
            "allowlisted_names_missing_at_baseline_freeze": missing,
            "allowed_baseline_gpu_process_sighting_count_total": sightings,
            "quiet_wait_event_count": len(waits),
            "quiet_wait_seconds": float(
                sum(float(event["wait_seconds"]) for event in waits)
            ),
            "contaminated_attempt_count": len(contaminated),
            "quiet_wait_events": waits,
            "contaminated_attempt_events": contaminated,
            "contamination_decision_source": "external process state only",
            "latency_or_solver_result_used_for_contamination_decision": False,
        }

    def total_summary(self) -> dict[str, Any]:
        return self.summary_since(GuardCheckpoint(0, 0, 0, 0, 0))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.command_timeout_seconds + 1.0)
            if thread.is_alive():
                raise QuietHostTechnicalInterruption(
                    "background external-state monitor did not stop"
                )
        with self._condition:
            monitor_error = self._monitor_error
        if monitor_error is not None:
            raise QuietHostTechnicalInterruption(
                "background external-state monitor failed before shutdown"
            ) from monitor_error

    def __enter__(self) -> "FormalHostGuard":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "FormalHostGuard",
    "GuardCheckpoint",
    "QuietHostTechnicalInterruption",
    "_parse_gpu_processes",
    "_parse_ps",
]
