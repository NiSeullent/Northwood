<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Northwood licensing

Northwood is a fork of Stela. Original project code uses `GPL-2.0-or-later` by default. A file's SPDX identifier and copyright notices identify any different terms. See the [default license](LICENSE), [distribution notices and source inventory](NOTICE), [license texts](LICENSES/) and [credits](CREDITS.md).

## Distributed programs

| Program | License | What it contains |
| --- | --- | --- |
| `stela.efi` | `GPL-2.0-only` | Linux code and Rust ports of Linux drivers. |
| `stela` | `GPL-3.0-or-later` | The CLI and its linked dependencies, including Apache-2.0 crates. The EFI application embedded in the installer is a separate program under its own license. |
| `stela-gpu-maintenance`, `Stela.app`, `StelaGPU.kext`, `StelaGPU.bundle` | `GPL-2.0-or-later` | The maintenance tool and guest graphics components. The kernel extension uses XNU headers with their own notices. |

For binary redistribution, follow the corresponding-source terms of the applicable GPL version. The source identified in `NOTICE` includes the matching repository revision, the pinned submodules, the third-party patches and the build scripts. Keep `LICENSE`, `NOTICE` and `LICENSES/` with the distribution. [GPL version 2](LICENSES/GPL-2.0-only.txt) · [GPL version 3](LICENSES/GPL-3.0-or-later.txt)

Setup obtains supported firmware from vendor sources under the applicable vendor terms. Users obtain macOS separately under Apple's terms. The default Northwood distribution excludes both payloads.

## Third-party contributions

| Tier | Material | Project policy |
| --- | --- | --- |
| A | Permissively licensed code and data, such as MIT, BSD or ISC. | Import when the terms fit every destination program. Preserve notices and record the source. |
| B | Copyleft and file-specific licenses, such as GPL, LGPL, MPL and APSL. | Check the destination and the original files before porting or linking. Preserve the required notices, source and distribution terms. |
| C | Source-available software with noncommercial or other use restrictions, such as PolyForm Noncommercial 1.0.0. | Keep it out of the repository and default release. An approved integration may obtain a verified artifact from its original distributor with user consent and run it as an independent program. Keep its notices and terms with it. |
| D | Unlicensed code, leaked material and proprietary implementation details, including extracted Apple binaries. | Use authorized public documentation for independent implementation. User-supplied macOS and vendor firmware used at runtime remain subject to the owners' terms. |

For new code and dependencies linked into the EFI program, choose terms compatible with `GPL-2.0-only`. That rules out new Apache-2.0-only or GPLv3-family code in the EFI program. The CLI can use Apache-2.0 code under GPLv3 and excludes `GPL-2.0-only` code. For shared libraries, meet the requirements of both consumers. Review any addition to a `GPL-2.0-or-later` program that would limit its distribution to a later GPL version. The existing third-party inventory remains recorded in `NOTICE`. [FSF license guidance](https://www.gnu.org/licenses/license-list.html) · [Apache Software Foundation compatibility guidance](https://www.apache.org/licenses/GPL-compatibility.html)

Record the upstream project, exact revision, source path and SPDX expression when importing or porting a file. Preserve its copyright and license notices, add the matching entry under `Ported sources` in `NOTICE` and include the applicable license text in `LICENSES/`. Use `SPDX-License-Identifier: GPL-2.0-or-later` on new original source files. When evaluating Tier C implementation details, record the materials reviewed and keep the reviewer separate from the corresponding GPL implementation work.

## Optional noncommercial components

The terms of each optional component govern its use and distribution. A distribution, installer or service that includes or obtains such a component for the recipient must honor those terms. The GPL grants for the Northwood programs include commercial use under their respective licenses. [GPL version 2, section 6](LICENSES/GPL-2.0-only.txt) · [GPL version 3, section 10](LICENSES/GPL-3.0-or-later.txt)

Before enabling an optional Tier C download, identify each payload and its copyright holder, verify the license and the artifact's pinned hash, and present the applicable restrictions before asking the user to opt in. Keep the download and its execution separate from the GPL programs, including dynamically loaded plugins. Retain a default installation path that works without the component. NullMoth publishes terms for individual source components. Assess any candidate kit and the planned integration separately before enabling acquisition. [NullMoth license](https://github.com/nullmoth/nvidia-macos-driver/blob/v3.0.0-beta.1/LICENSE) · [NullMoth third-party notices](https://github.com/nullmoth/nvidia-macos-driver/blob/v3.0.0-beta.1/NOTICE)
