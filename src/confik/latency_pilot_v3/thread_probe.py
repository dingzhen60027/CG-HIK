from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from ..config import load_config, load_robot
from ..data.datasets import TransitionDataset
from ..models.seed import TorchSeedEnsemble, encode_seed_inputs
from .benchmark import distribution_summary
from .optimized_inference import VectorizedSeedMLP


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only CPU thread probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--robot", required=True, choices=("ur5e", "panda"))
    parser.add_argument("--intra", required=True, type=int)
    parser.add_argument("--inter", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=1200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.intra <= 0 or args.inter <= 0:
        raise ValueError("thread counts must be positive")
    torch.set_num_threads(args.intra)
    torch.set_num_interop_threads(args.inter)
    config = load_config(args.config)
    root = Path(args.source_root).resolve()
    validation_path = root / args.robot / "datasets" / "seed_validation.npz"
    if "test" in validation_path.name.lower():
        raise RuntimeError("thread probe refuses test-named data")
    kinematics = load_robot(config, args.robot)
    ensemble = TorchSeedEnsemble.load(
        root / args.robot / "models" / "seed_ensemble.pt",
        kinematics,
        device="cpu",
    )
    dataset = TransitionDataset.load(validation_path)
    features = encode_seed_inputs(
        kinematics,
        dataset.previous_q[:256],
        dataset.target_position[:256],
        dataset.target_rotation[:256],
        use_history=ensemble.config.use_history,
    ).astype(np.float32)
    eager = VectorizedSeedMLP.from_ensemble(ensemble, device="cpu").eval()
    example = torch.from_numpy(features[:1]).contiguous()
    modules = {"optimized_pytorch": eager}
    values: dict[str, list[float]] = {name: [] for name in modules}
    with torch.inference_mode():
        for warmup_index in range(args.warmup):
            row = torch.from_numpy(features[warmup_index % len(features) : warmup_index % len(features) + 1])
            for module in modules.values():
                module(row)
        for repeat in range(args.repeats):
            row_index = repeat % len(features)
            row = torch.from_numpy(features[row_index : row_index + 1])
            order = tuple(reversed(tuple(modules))) if repeat % 2 else tuple(modules)
            for name in order:
                started = perf_counter_ns()
                modules[name](row)
                values[name].append((perf_counter_ns() - started) / 1e6)
    payload = {
        "robot": args.robot,
        "intra_op_threads": args.intra,
        "inter_op_threads": args.inter,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "backends_ms": {name: distribution_summary(samples) for name, samples in values.items()},
    }
    print(json.dumps(payload, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
