<h1 align="center">
  <img src="assets/stela-icon.png" alt="" width="96"><br>
  Northwood
</h1>

Northwood is a fork of Stela that boots Apple silicon macOS on x86-64 PCs.
It combines a UEFI boot program, an arm64-to-x86-64 translator, a Metal driver
for NVIDIA GPUs and a model of Apple silicon hardware.

> [!WARNING]
Northwood is still in development. Check the [hardware notes](hardware/) for
device-specific support and verification status.

## Translator

Northwood keeps Stela's translated arm64 UEFI boot path. Its `northwood
import-oc` tool reads an existing configuration and exports reviewable
Northwood hints for that path.

An ahead-of-time and just-in-time compiler translates arm64e code to x86-64.
It covers more than 3,000 AArch64 instruction encodings, maps guest registers
to host registers and caches translated code. It runs on any x86-64-v2 CPU,
Intel or AMD.

## NVIDIA driver

The Metal driver supports GTX 16 and RTX 20 series GPUs and newer. It compiles
AIR, Metal's shader IR, directly to NIR, Mesa's IR, and supports the Metal3
and Apple7 GPU families.

## Requirements

- An x86-64-v2 CPU (about 2011 or newer), UEFI with Secure Boot off, 4 GB of
  RAM. 8 GB or more is recommended: with less than about 7 GB free, Stela
  boots in a compact profile that gives macOS 2 GB or a little more and runs
  slower.
- An NVIDIA GTX 16 or RTX 20 series GPU or newer.
- A USB stick of 8 GB or more.
- An empty GPT partition of at least 60 GB.
- Network access during the install, which downloads macOS.

Stela currently models the Mac mini (M2) and runs macOS 27.

## Install

1. Download the archive for your system from
   [Northwood releases](https://github.com/NiSeullent/Northwood/releases).
2. On the PC that will run macOS, run `sudo ./northwood setup` on Linux or
   macOS. On Windows, run `northwood.exe setup` or open `stela.exe` for the
   installer wizard. The `stela setup` command also works.
3. Choose the USB stick, the macOS version and the Mac model, then confirm.
   Setup erases the USB stick.
4. Boot the PC from the USB stick.

The removable-media entry is `EFI/BOOT/BOOTX64.EFI`. With the bundled
Northwood loader, a fresh setup writes `EFI/northwood/nwconfig.plist` and
keeps its data under `northwood/`. Setup labels a newly formatted data
partition `NORTHWOOD`. Updates and repairs preserve the recorded layout,
including `EFI/stela` and `stela/` on existing Stela media.

## Build

You need [Rust](https://rustup.rs) and LLVM with Clang 21 or newer.

```sh
git clone https://github.com/NiSeullent/Northwood
cd Northwood
cargo xtask dist
```

The build writes `target/release/northwood`, `target/release/stela` and a
matching `northwood.efi`. The first build fetches and compiles Linux, Mesa
and LLVM sources.

## Legal

Setup obtains macOS from Apple. You are responsible for following Apple's
software license agreement and terms of use.

The project source defaults to GPL-2.0-or-later. The UEFI program is
distributed under GPL-2.0-only and the CLI under GPL-3.0-or-later. See
[LICENSING.md](LICENSING.md), [LICENSE](LICENSE), [NOTICE](NOTICE) and
[CREDITS.md](CREDITS.md).

## Community

[Discord Server](https://discord.gg/aSkwrJEMYc)
