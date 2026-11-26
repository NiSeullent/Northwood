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
    local name=$1 sockets=$2 cores=$3 threads=$4 expected_exit=$5 status=0
    shift 5
    local run="$output/$name"
    cargo xtask boot run --image "$image" --output "$run" --timeout 120 \
        --host-sockets "$sockets" --host-cores "$cores" --host-threads "$threads" "$@" || status=$?
    # isa-debug-exit encodes mount idle (0x10) as 33, and fatal (0x11) as 35.
    if [[ "$status" != "$expected_exit" ]]; then
        for log in stderr.txt serial.txt; do
            if [[ -f "$run/$log" ]]; then tail -80 "$run/$log" >&2; fi
        done
        echo "$name exited with $status, expected $expected_exit" >&2
        return 1
    fi
    python3 - "$run" "$sockets" "$cores" "$threads" "$expected_exit" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
sockets, cores, threads, expected_exit = map(int, sys.argv[2:])
cores_total = sockets * cores
report = json.loads((run / "run.json").read_text())
if report["timed_out"] or report["qemu_returncode"] != expected_exit:
    sys.exit(f"{run.name}: QEMU did not return {expected_exit}")
if (report["qemu_host_sockets"], report["qemu_host_cores"], report["qemu_host_threads"]) != (sockets, cores, threads):
    sys.exit(f"{run.name}: QEMU did not use the requested topology")

lines = (run / "serial.txt").read_text().splitlines()
admission = "FAIL" if expected_exit == 35 else "PASS"
if not any(
    line.startswith(f"CHECK installed-memory {admission} ") and "minimum=0x200000000" in line.split()
    for line in lines
):
    sys.exit(f"{run.name}: expected the installed RAM check to report {admission}")

if expected_exit == 35:
    if report["scenario"]["passed"] or any(line.startswith("GUEST|") for line in lines):
        sys.exit(f"{run.name}: guest entered below the RAM minimum")
    print(f"{run.name}: verified early RAM refusal")
    sys.exit(0)

if report["scenario"] != {"passed": True}:
    sys.exit(f"{run.name}: the authored kernel did not reach its verified idle state")
if "FIRMWARE watchdog disabled" not in lines or not any(
    line.startswith("FIRMWARE boot services ended at try ") for line in lines
):
    sys.exit(f"{run.name}: firmware handoff diagnostics are missing")

inventories = [
    dict(field.split("=", 1) for field in line.split()[2:])
    for line in lines
    if line.startswith("NATIVE INVENTORY ")
]
expected = {"physical_cores": str(cores_total), "logical": str(cores_total * threads)}
if len(inventories) != 1 or any(inventories[0].get(key) != count for key, count in expected.items()):
    sys.exit(f"{run.name}: expected CPU inventory {expected}, got {inventories}")
other_threads = cores_total * threads - 1
if other_threads and (
    f"NATIVE PARKED BEFORE FIRMWARE {other_threads} of {other_threads}; wait_for_sipi=0" not in lines
):
    sys.exit(f"{run.name}: not all {other_threads} other threads parked before firmware access")
if not any(line.startswith("HOST MEMORY ") and "profile=standard" in line.split() for line in lines):
    sys.exit(f"{run.name}: expected the standard memory profile")
print(f"{run.name}: verified boot with {sockets} socket(s), {cores_total} cores and {cores_total * threads} threads")
PY
}

boot_case control 1 1 1 33
boot_case intel-smt 1 32 2 33 --no-la57
boot_case dual-intel-smt 2 16 2 33 --no-la57
boot_case low-memory 1 4 1 35 --memory 7G
