# Lenovo IdeaPad 1 15IGL7 firmware provisioning

An existing Northwood USB can be provisioned from a setup computer with
internet access. Mount the USB's EFI partition and pass its path to `--usb`.
The profile selects the Intel CNVi `8086:31dc` Wi-Fi firmware candidate and
the signed wireless regulatory database. The current legacy Intel code
checks and plans the firmware sections on the host. PCI transport, radio
initialization and Bluetooth firmware selection still need on-device work.

The first 15IGL7 Diagnostics run on the physical USB enumerated
`0000:00:0c.0 8086:31dc`. It recorded zero of 18 Intel Wi-Fi UEFI variables,
an absent ESP firmware manifest, and zero host firmware images. Its recovered
ESP log ends before the post-EBS checks. The current probe rejects this
controller before Wi-Fi attachment. Diagnostics does not supply the Wi-Fi
settings and native ownership services needed for a carrier test.

The exact physical `dcdd1e13…` EFI and the private ESP candidate
`4d8162be…` both advertise host firmware capability revision 3. The loader
accepts manifest schema 1, up to 64 resources, 64 MiB per resource, and
256 MiB total. Both archived ESPs lack a host firmware directory. The
three-file recipe below fits both EFIs. A previous local CLI test checked
identical online and offline manifests in temporary directories. The
API 46 CNVi image below was independently rehashed from the local oracle
copy. Keep the regulatory database and signature together.
[Loader limits](../crates/stela-loader/src/host_firmware.rs)
[Profile recipe](../crates/stela-loader/src/host_firmware/recipes.rs)

- Physical EFI `mem02-bootkc.efi` SHA-256
  `dcdd1e13f9cae3b13e2fc95f6fd96189884cba530d7e29f55c9100ecebb62ddc`
- Private candidate `install-target.efi` SHA-256
  `4d8162beca53b5bd5c1a523e454a598222fc90c172d8f735d22c4109c860c740`

| 15IGL7 profile resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `iwlwifi-9000-pu-b0-jf-b0-46.ucode` | 1,519,568 | `9a0c5d938646942f9798e6212579c25771e4bb44301cd409b82abe361ca2c3e0` |
| `regulatory.db` | 6,348 | `7e236caecd939c8ec98be4870bf30422f28ffef2565a38aaaa2d9ddabd0c2641` |
| `regulatory.db.p7s` | 1,085 | `50332f0db09b8bcc719235ec4985c27377f2cc2f42abb7b5dcf951866c5a888a` |

The cache also needs the Wi-Fi notice `16d5040c7cf851fc693b7542e20870935b99802533ea1bfd231c377a2305e5c5`
and the regulatory database notice `678b0df753c86198fc496d1f1033429bbd57f101472132ee7eaaf9f5e0a7fae1`.
Stage the offline package on a private copy of an already configured ESP.
Use the profile's offline command below and compare the manifest targets,
lengths and hashes against the recipe and the copied files. On the next boot,
look for `HOST FIRMWARE phase=before-ebs images=3 bytes=1527001` when the
package contains just these three resources.

Bluetooth needs its own live USB and bootloader identity. The Intel RSA
recipe can select `intel/ibt-20-1-3.sfi` (716,180 bytes,
`1c8f0dac48d108c0d36f4297119857465d2830b35f3b4f697995f94a32f36db0`)
and `intel/ibt-20-1-3.ddc` (70 bytes,
`597f26161c81a2ba899d42c1aafdedc427b20b00cb6ce157587814166ff178a0`)
only when the observed controller and complete revision match the recipe,
including loader tuple `37:14:01:03`. Its cache notice is
`5181b0b51efc79d5acb2c9bb92042878fdbad97a92114d4ab5e32e2b5b52fce4`.
The 15IGL7 Wi-Fi profile does not select them.
[Intel Bluetooth recipe](../crates/stela-loader/src/host_firmware/recipes.rs)

```sh
stela setup host-firmware fetch --usb /path/to/esp \
  --profile lenovo-ideapad-1-15igl7 --cache /path/to/firmware-cache --dry-run
stela setup host-firmware fetch --usb /path/to/esp \
  --profile lenovo-ideapad-1-15igl7 --cache /path/to/firmware-cache
```

For offline reuse, copy the complete cache directory, including the files
named by SHA-256 for license notices, and use `--offline` with the same
profile. Setup checks size and SHA-256 again and publishes the package only
after verifying all required objects. If the ESP already holds additional
firmware, copy its cache objects too. Setup preserves the existing bindings.

```sh
stela setup host-firmware fetch --usb /path/to/esp \
  --profile lenovo-ideapad-1-15igl7 --cache /path/to/copied-cache --offline
```

Vendor images are acquired during setup. The cache and the USB package carry
the pinned vendor notices. The independent NVIDIA `--gpu CHIP` option and
observed `REQUIREMENTS` inventory continue to work.
