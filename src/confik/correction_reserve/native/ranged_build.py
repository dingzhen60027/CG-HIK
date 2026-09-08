"""Build two explicitly identified bridges without changing the pinned checkout.

Run from anywhere using Python 3.12+. Rust/cargo must already be installed (the
local tmp/crik_dependencies toolchain is preferred). Builds and downloaded cargo
dependencies remain under the task-specific tmp directory, not old environments.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[4]
NATIVE = Path(__file__).resolve().parent
UPSTREAM_SHA = "1c48d2ae408b4e024ee037641aac1e728267984e"
URL = "https://github.com/uwgraphics/relaxed_ik_core.git"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(*args, cwd=ROOT, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def build():
    deps = ROOT / "tmp/crik_dependencies"
    upstream = deps / "ranged_ik"
    if not upstream.exists():
        subprocess.run(["git", "clone", "--single-branch", "--branch", "ranged-ik", URL, str(upstream)], check=True)
        subprocess.run(["git", "checkout", "--detach", UPSTREAM_SHA], cwd=upstream, check=True)
    if checked("git", "rev-parse", "HEAD", cwd=upstream) != UPSTREAM_SHA:
        raise ValueError("existing RangedIK checkout does not match the pinned SHA")
    if checked("git", "status", "--porcelain", cwd=upstream):
        raise ValueError("refusing to build from a dirty RangedIK source checkout")
    adapted = deps / "ranged_ik_positive_range"
    files = checked("git", "ls-files", cwd=upstream).splitlines()
    objective_name = "src/groove/objective.rs"
    original = (upstream / objective_name).read_bytes()
    if original.count(b"if (bound <= 1e-2)") != 2:
        raise ValueError("the pinned source's two cutoff conditions were not found")
    replacement = original.replace(b"if (bound <= 1e-2)", b"if (bound <= 0.0)")
    if not adapted.exists():
        adapted.mkdir(parents=True)
        archive = subprocess.check_output(["git", "archive", UPSTREAM_SHA], cwd=upstream)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(adapted, filter="data")
        # A checked, exactly two-line mechanical patch to a new generated copy.
        (adapted / objective_name).write_bytes(replacement)
    for name in files:
        expected = replacement if name == objective_name else (upstream / name).read_bytes()
        if (adapted / name).read_bytes() != expected:
            raise ValueError(f"unexpected positive-range source modification: {name}")

    work = deps / "ranged_adapter_build"
    work.mkdir(exist_ok=True)
    env = os.environ.copy()
    cargo = deps / "cargo_home/bin/cargo"
    rustc = deps / "cargo_home/bin/rustc"
    if cargo.exists():
        env["CARGO_HOME"] = str(deps / "cargo_home")
        env["RUSTUP_HOME"] = str(deps / "rust_home")
    else:
        cargo, rustc = Path("cargo"), Path("rustc")
    manifest = {
        "upstream_url": URL, "upstream_branch": "ranged-ik", "upstream_sha": UPSTREAM_SHA,
        "license": "MIT", "rustc": checked(str(rustc), "--version", env=env),
        "source_diff": str((NATIVE / "ranged_positive_range.patch").relative_to(ROOT)),
        "source_diff_sha256": sha(NATIVE / "ranged_positive_range.patch"),
        "bridge_sha256": sha(NATIVE / "ranged_bridge/src/lib.rs"),
        "upstream_objective_sha256": hashlib.sha256(original).hexdigest(),
        "positive_range_objective_sha256": hashlib.sha256(replacement).hexdigest(),
        "exact_changed_lines": 2, "unchanged_upstream_files": len(files) - 1,
        "public_tolerances_changed": False, "variants": {},
    }
    dependency_template = (NATIVE / "ranged_bridge/Cargo.toml").read_text()
    dependency_lock = NATIVE / "ranged_bridge/Cargo.lock"
    for mode, core in (("upstream", upstream), ("positive_range", adapted)):
        folder = work / mode
        folder.mkdir(exist_ok=True)
        cargo_text = dependency_template.replace(
            'path = "../../../../../tmp/crik_dependencies/ranged_ik"', f'path = "{core.as_posix()}"')
        cargo_text = cargo_text.replace('crate-type = ["cdylib"]',
            f'crate-type = ["cdylib"]\npath = "{(NATIVE / "ranged_bridge/src/lib.rs").as_posix()}"')
        (folder / "Cargo.toml").write_text(cargo_text)
        if dependency_lock.exists():
            (folder / "Cargo.lock").write_bytes(dependency_lock.read_bytes())
        env["CARGO_TARGET_DIR"] = str(folder / "target")
        command = [str(cargo), "build", "--release", "--locked", "--manifest-path", str(folder / "Cargo.toml")]
        log_path = folder / "build.log"
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"RangedIK {mode} compilation failed; see {log_path}")
        library = folder / "target/release/libcrik_ranged_bridge.so"
        manifest["variants"][mode] = {
            "library": str(library.relative_to(ROOT)), "library_sha256": sha(library),
            "cargo_lock_sha256": sha(folder / "Cargo.lock"),
            "build_log": str(log_path.relative_to(ROOT)),
            "core_path": str(core.relative_to(ROOT)),
        }
        print(f"Built RangedIK {mode}: {library}", flush=True)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
