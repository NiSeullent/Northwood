<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Following Stela upstream

A fresh Northwood clone uses `origin` for `NiSeullent/Northwood`. Fetch a reviewed Stela source separately when comparing changes. [Stela](https://github.com/tylertms/Stela) · [Northwood](https://github.com/NiSeullent/Northwood)

## Inspecting an update

```sh
git fetch --no-tags https://github.com/garvitsurana271/Stela.git refs/heads/main
upstream="$(git rev-parse FETCH_HEAD)"
git merge-base HEAD "$upstream" &&
  git rev-list --left-right --count "HEAD...$upstream" &&
  git merge-tree --write-tree HEAD "$upstream"
```

`merge-tree` works on committed trees and leaves the current checkout and branch refs untouched. The manual **Stela upstream merge check** workflow reports the two revisions, checks that they share a Git ancestor and tests the merge. Compilation and boot validation run in **build and release**.

The archived `tylertms/Stela` main branch no longer shares Northwood's Git history. To review changes from a surviving public fork, enter its repository and branch in **Stela upstream merge check** after checking the source provenance. The original shared baseline was `502dbdd53d3d4d09b9b8bbf0a05cc98c1f9bc36b`.

If the reviewed branch has changes to integrate, create a dedicated integration branch or worktree from the committed Northwood branch. Merge the captured `$upstream` revision into that integration branch and resolve conflicts with the table below and the current roadmap in view. Preserve the upstream file-specific licenses and both projects' source notices. Review submodule pins before updating them, and keep `third_party/patches/` aligned with the pins selected for the build.

Run the x86-64 host checks, build the deployable EFI, then test Standard and Compact in QEMU with the usual fixtures. Re-run the low-memory and hardware tests relevant to the changed paths. Integrate the branch once those checks pass. For a fix that also benefits Stela, prepare a focused upstream patch with its tests and source provenance.

## Fork changes to revisit after a merge

The list starts with the committed differences at Northwood `afb62c1dc`. Update it when a fork-only change to an upstream file is integrated. Keep new policy and implementation in dedicated modules and keep existing call sites small.

| Paths | Behavior to recheck |
| --- | --- |
| `crates/stela-boot-core/src/host_memory.rs`, `crates/stela-boot-core/src/host_memory/tests.rs`, `crates/stela-boot-core/src/native_resources.rs` | Compact memory threshold and allocation budgets |
| `crates/stela-boot/src/system/memory.rs` | Memory admission and profile output |
| `crates/stela-boot-core/src/gpu_status.rs`, `crates/stela-driver-net/src/wifi/unsupported.rs` | GPU and Wi-Fi diagnostics on unsupported devices |
| `README.md`, `.github/workflows/release.yml`, `hardware/lenovo-ideapad-1-15igl7.md` | Fork documentation and distribution requirements |

On 2026-11-11, Stela `origin/main` was `502dbdd53`, an ancestor of `afb62c1dc`. `git merge-tree --write-tree HEAD origin/main` returned the Northwood HEAD tree `e18feccab` without conflicts. A merge with new Stela commits and its build still need verification. The shared Linux checkout was clean at `347246472`, while the superproject pinned `cc9de5d4b`. Keep this local checkout difference out of unrelated changes.

## Build and tag policy

Scheduled and manual runs upload the build artifacts to GitHub Actions. GitHub enables manual and scheduled triggers after the workflow exists on the default branch. The release publisher runs for a pushed `v*` tag, and the tag must match the `[workspace.package]` version. For a prerelease, set the full prerelease version in `Cargo.toml` and use the matching `v` tag. [GitHub Actions workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
