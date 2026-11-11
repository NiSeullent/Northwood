<h1 align="center">
  <img src="assets/stela-icon.png" alt="" width="96"><br>
  Stela
</h1>

Stela boots unmodified Apple silicon macOS on x86-64 PCs. It is a UEFI boot
program with an arm64-to-x86-64 translator, a Metal driver for NVIDIA GPUs and
a model of Apple silicon hardware. macOS and most apps start, but it is slow and needs
much more testing, as of v0.1.0

> [!WARNING]
Stela is very much still in development. Your chances of getting it booted are low as of v0.1.0, major hardware support/fixes are on the way.

## Translator

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

1. Download `stela` from the
   [releases page](https://github.com/tylertms/Stela/releases).
2. On the PC that will run macOS, run `sudo ./stela setup` (Linux, macOS) or
   double-click `stela.exe` (Windows).
3. Choose the USB stick, the macOS version and the Mac model, then confirm.
   Setup erases the USB stick.
4. Boot the PC from the USB stick.

## Build

You need [Rust](https://rustup.rs) and LLVM with Clang 21 or newer.

```sh
git clone https://github.com/tylertms/Stela
cd Stela
cargo xtask dist
```

This writes `target/release/stela`. The first build fetches and compiles
Linux, Mesa and LLVM sources and takes a long time.

## Legal

Stela contains no Apple software. You are responsible for following Apple's
software license agreement and terms of use.

Stela's code is GPL-2.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Community

[Discord Server](https://discord.gg/aSkwrJEMYc)
