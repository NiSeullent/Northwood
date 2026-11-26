<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Stela and fork compatibility triage for 1.0.0b2

Northwood and the Stambha `main` snapshot at `085e226f4` share Stela ancestor
`502dbdd53`. GitHub lists the original Stela repository as archived and its
issue API returns HTTP 410. The direct fork API returned 309 visible Stela
forks and 26 Stambha forks. Northwood and Stela-Mods had issue tracking
enabled in that enumeration. The accessible trackers contained 11 Stambha
issues, eight Stela-Mods issues and zero Northwood issues, excluding pull
requests. The review also sampled the default branch heads of 16 direct
Stela forks with push dates newer than the archived upstream head.
Published issue and pull-request pages supplement the archived tracker.
The counts describe the accessible snapshot. [Stela forks](https://github.com/tylertms/Stela/forks) · [Stambha](https://github.com/garvitsurana271/stambha) · [Stela-Mods issues](https://github.com/reece-builds/Stela-Mods/issues) · [Northwood issues](https://github.com/NiSeullent/Northwood/issues)

## Changes integrated

| Failure or report | Northwood change |
| --- | --- |
| A delayed application processor could pass TSC sync despite never exchanging samples with the bootstrap processor. The source report reproduced this on a 28-thread Intel i7-14700HX. | Require samples in both directions, retry non-overlapping rounds independently of clock corrections, and stop stalled rendezvous after a deadline. Regression fixtures cover delayed samples and counter wrap. [Stela #189](https://github.com/tylertms/Stela/issues/189) · [Archived report](https://githubhelp.com/tylertms/Stela/issues/189) |
| Intel PCH xHCI can advertise a zero-length standard PCI vendor capability. A firmware-named absent PCI function can also return vendor ID `0000`. | Keep the vendor entry header so the rest of the capability chain can be walked. Skip absent functions in ECAM enumeration. Regression fixtures cover the capability chain, VMD shadow boundaries and empty PCI functions. [Stela PR #53](https://github.com/tylertms/Stela/pull/53) · [Stambha 80ff754de](https://github.com/garvitsurana271/stambha/commit/80ff754de) · [Stambha d2fa5d25b](https://github.com/garvitsurana271/stambha/commit/d2fa5d25b) |
| The PS/2 observation queue can block both i8042 ports after a Synaptics device fills it. Slow active I²C-HID polling can also reduce pointer responsiveness. | Drain unused PS/2 observations, forward TrackPoint input, and give active I²C-HID devices a larger polling budget. Regression fixtures cover keyboard progress and sustained TrackPoint traffic. [Stambha PR #11](https://github.com/garvitsurana271/stambha/pull/11) · [Stambha PR #20](https://github.com/garvitsurana271/stambha/pull/20) |
| A large translation-pack write can exceed a block device's declared command limit. | Split Northwood's synchronous writes on whole-block boundaries within the device limit. A capped block-device fixture checks the resulting bytes. The fork report also covers a queued bulk reader introduced after Northwood's branch point. [Stambha PR #21](https://github.com/garvitsurana271/stambha/pull/21) |
| NVMe integrity issues during read-back and discard command validation. | Initialize recycled read buffers before publishing each command, and validate the number of discard ranges before allocating an NVMe tag. Simulated controllers cover completions without a transfer and out-of-range command counts. [Stambha 4a56ad302](https://github.com/garvitsurana271/stambha/commit/4a56ad302) · [Stambha 7f67ae7f9](https://github.com/garvitsurana271/stambha/commit/7f67ae7f9) |
| NVMe controllers can change their ready timeout after configuration. | Configure the stopped controller before reading the timeout it advertises. [Stambha f5f8121a2](https://github.com/garvitsurana271/stambha/commit/f5f8121a2) |
| Ethernet discovery and initialization can fail for rebadged Realtek adapters or Intel PCH adapters without the power-cycle completion indication. | Use the compatible Realtek device IDs and link-status handling, and continue the Intel PHY sequence after the bounded wait while verifying register access. [Stambha Realtek changes](https://github.com/garvitsurana271/stambha/commit/9fd57360e) · [Stambha Intel PCH change](https://github.com/garvitsurana271/stambha/commit/da395e895) |
| macOS Setup can open at an unusably small logical resolution on dense 1080p displays. | Adjust the advertised density of narrow panels while preserving existing wide-panel scaling. [Stambha PR #13](https://github.com/garvitsurana271/stambha/pull/13) |
| An installer can drop a host Bluetooth controller outside the address space that maps it. | Teardown now uses the loader's address space for the same mapped hardware access available during Bluetooth service. [Stambha 747e1e784](https://github.com/garvitsurana271/stambha/commit/747e1e784) |

## Remaining hardware reports

- Stela's retry-on-next-boot report depends on the persistent device-start
  history introduced after Northwood's branch point. The parser patch
  applies once the history, boot log persistence and protection for essential
  devices are available together. [Stela PR #35](https://github.com/tylertms/Stela/pull/35) · [Stambha 1b67b2c99](https://github.com/garvitsurana271/stambha/commit/1b67b2c99)
- Root PCI bridges whose Bus Master bit is hard-wired need the donor's full
  containment change and tests for both root and downstream bridges. The
  donor verifies the bit and preserves the rules for DMA-capable endpoints.
  [Stela #48](https://github.com/tylertms/Stela/issues/48) · [Stambha f2cbad820](https://github.com/garvitsurana271/stambha/commit/f2cbad820) · [Stambha 4f8d5b58c](https://github.com/garvitsurana271/stambha/commit/4f8d5b58c)
- Separately reproduce the Stela-Mods RTX 5060 black screen, intermittent USB
  freezes and USB-dependent installed boot. The report reaches
  WindowServer and the Language Chooser, so loader progress by itself
  cannot confirm recovery. [Stela-Mods #8](https://github.com/reece-builds/Stela-Mods/issues/8)
- Stambha reports an Intel HD 620 display freeze and, separately, Ryzen
  5800H/RTX 3050 mobile initialization or guest failures. The attached logs
  report an i915 probe failure, a WindowServer watchdog, and a translated
  worker page fault respectively. These configurations need fresh
  Northwood boot logs. [Stambha #8](https://github.com/garvitsurana271/stambha/issues/8) · [Stambha #7](https://github.com/garvitsurana271/stambha/issues/7) · [Stambha #6](https://github.com/garvitsurana271/stambha/issues/6)
- The AMD/RADV and Intel graphics efforts in Stambha and Stela-Mods
  depend on different pinned Linux and Mesa sources, build glue, and
  hardware qualification. Those drivers remain open compatibility work.
  [Stambha graphics PRs](https://github.com/garvitsurana271/stambha/pulls) · [Stela-Mods graphics branches](https://github.com/reece-builds/Stela-Mods/branches)
- The published 32-core/64-thread Xeon x2APIC hang in Memtest86+ comes from
  a separate program. It documents a useful comparison topology. Northwood
  verification on a physical Xeon still needs the specific model and
  firmware, the saved boot log, and boots with SMT enabled and disabled.
  [Memtest86+ #605](https://github.com/memtest86plus/memtest86plus/issues/605)
