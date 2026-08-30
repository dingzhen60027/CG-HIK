from __future__ import annotations

import threading

from confik.test_v4_locked.host_guard import (
    FormalHostGuard,
    _parse_gpu_processes,
    _parse_ps,
)


def _config() -> dict[str, object]:
    return {
        "max_unrelated_cpu_percent": 50.0,
        "max_aggregate_external_cpu_percent": 400.0,
        "reject_other_gpu_compute_pids": True,
        "allowed_persistent_gpu_compute_process_names": [
            "/usr/libexec/gnome-remote-desktop-daemon"
        ],
        "monitor_sample_interval_seconds": 0.005,
        "max_monitor_sample_age_seconds": 0.1,
        "monitor_command_timeout_seconds": 1.0,
        "quiet_stable_checks": 1,
        "quiet_poll_seconds": 0.005,
        "max_quiet_wait_seconds": 1.0,
        "max_contaminated_attempts": 3,
    }


class _SequenceSampler:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples
        self.calls = 0
        self.third_sample = threading.Event()

    def __call__(self) -> dict[str, object]:
        index = min(self.calls, len(self.samples) - 1)
        self.calls += 1
        if self.calls >= 3:
            self.third_sample.set()
        return self.samples[index]


def test_process_parsers_exclude_runner_and_keep_exact_process_path() -> None:
    cpu = _parse_ps(
        "10 99.0 R /formal/runner\n11 51.0 S /external/load\n",
        threshold=50.0,
        excluded={10},
    )
    gpu = _parse_gpu_processes(
        "10, /formal/runner, 100\n1787, /usr/libexec/gnome-remote-desktop-daemon, 7\n",
        excluded={10},
    )
    assert [row["pid"] for row in cpu] == [11]
    assert gpu == [
        {
            "pid": 1787,
            "process_name": "/usr/libexec/gnome-remote-desktop-daemon",
            "used_gpu_memory_mib": 7,
        }
    ]


def test_background_monitor_freezes_persistent_gpu_identity_and_rejects_new_pid() -> None:
    name = "/usr/libexec/gnome-remote-desktop-daemon"
    sampler = _SequenceSampler(
        [
            {
                "busy_cpu_processes": [],
                "gpu_compute_processes": [
                    {"pid": 1787, "process_name": name, "used_gpu_memory_mib": 7}
                ],
            },
            {
                "busy_cpu_processes": [],
                "gpu_compute_processes": [
                    {"pid": 1787, "process_name": name, "used_gpu_memory_mib": 7}
                ],
            },
            {
                "busy_cpu_processes": [],
                "gpu_compute_processes": [
                    {"pid": 1787, "process_name": name, "used_gpu_memory_mib": 7},
                    {"pid": 9999, "process_name": name, "used_gpu_memory_mib": 9},
                ],
            },
        ]
    )
    with FormalHostGuard(_config(), sampler=sampler) as guard:
        token = guard.wait_until_quiet(context="unit/before")
        assert sampler.third_sample.wait(timeout=1.0)
        observation = guard.observe(
            context="unit/after",
            since_sample_index=int(token["monitor_sample_index"]),
        )
        assert observation["busy"]
        assert [
            row["pid"] for row in observation["allowed_frozen_gpu_compute_processes"]
        ] == [1787]
        assert [row["pid"] for row in observation["foreign_gpu_compute_processes"]] == [
            9999
        ]
        guard.record_contamination(
            context="unit",
            observation=observation,
            attempt_index=0,
            discarded_scope="complete_same_query_all_methods_all_repeats",
        )
        summary = guard.total_summary()
        assert summary["background_monitor"]
        assert not summary["synchronous_ps_or_nvidia_smi_per_query"]
        assert summary["frozen_allowed_gpu_processes"][0]["pid"] == 1787
        assert summary["contaminated_attempt_count"] == 1


def test_unlisted_gpu_process_is_busy_at_initial_sample() -> None:
    sampler = _SequenceSampler(
        [
            {
                "busy_cpu_processes": [],
                "gpu_compute_processes": [
                    {
                        "pid": 123,
                        "process_name": "/opt/unrelated/trainer",
                        "used_gpu_memory_mib": 1024,
                    }
                ],
            }
        ]
    )
    guard = FormalHostGuard(_config(), sampler=sampler)
    try:
        sample = guard._cached_sample(wait_for_fresh=True)
        assert sample["busy"]
        assert sample["foreign_gpu_compute_processes"][0]["pid"] == 123
    finally:
        guard.close()
