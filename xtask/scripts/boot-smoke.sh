#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Exercise the EFI that will ship, using the repository's synthetic ARM kernel.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 EFI-IMAGE NEW-OUTPUT-DIRECTORY" >&2
    exit 2
fi
image=$1
output=$2

boot_case() {
    local name=$1 cores=$2 threads=$3 profile=$4 status=0
    shift 4
    local run="$output/$name"
    cargo xtask boot run --image "$image" --output "$run" --timeout 120 \
        --host-cores "$cores" --host-threads "$threads" "$@" || status=$?
    # The authored guest idles with mount code 0x10 on isa-debug-exit.
    if [[ "$status" != 33 ]]; then
        for log in stderr.txt serial.txt; do
            if [[ -f "$run/$log" ]]; then tail -80 "$run/$log" >&2; fi
        done
        echo "$name exited with $status, expected 33" >&2
        return 1
    fi
    python3 - "$run" "$cores" "$threads" "$profile" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
cores, threads = map(int, sys.argv[2:4])
profile = sys.argv[4]
report = json.loads((run / "run.json").read_text())
if report["timed_out"] or report["qemu_returncode"] != 33 or report["scenario"] != {"passed": True}:
    sys.exit(f"{run.name}: the authored kernel did not reach its verified idle state")

lines = (run / "serial.txt").read_text().splitlines()
inventories = [
    dict(field.split("=", 1) for field in line.split()[2:])
    for line in lines
    if line.startswith("NATIVE INVENTORY ")
]
expected = {"physical_cores": str(cores), "logical": str(cores * threads)}
if len(inventories) != 1 or any(inventories[0].get(key) != count for key, count in expected.items()):
    sys.exit(f"{run.name}: expected CPU inventory {expected}, got {inventories}")
if not any(line.startswith("HOST MEMORY ") and f"profile={profile}" in line.split() for line in lines):
    sys.exit(f"{run.name}: expected the {profile} memory profile")
print(f"{run.name}: verified {profile} boot with {cores} cores and {cores * threads} threads")
PY
}

boot_case control 1 1 standard
boot_case intel-smt 32 2 standard --no-la57
boot_case compact 1 1 compact --memory 4G
