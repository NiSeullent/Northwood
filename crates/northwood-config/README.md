# Northwood configuration v1

Put the settings file at `EFI/northwood/nwconfig.plist` on the EFI System Partition. [The annotated sample](examples/nwconfig.sample.plist) lists the keys and defaults. [The 15IGL7 sample](examples/nwconfig.15igl7.plist) requests its built-in hardware profile. Setup supplies the boot source and target values for a particular disk.

```sh
northwood config validate path/to/nwconfig.plist
```

The shared parser accepts UTF-8 XML property lists with a dictionary at the root. It supports dictionaries, arrays, strings, decimal 64-bit integers, booleans, and base64 data. The standard Apple property-list declaration may appear in the file. Duplicate keys, invalid field types, and unsupported schema versions produce errors. Unknown keys produce warnings. The loader and host tools use the same parser.

| Section | Configuration |
| --- | --- |
| `Northwood` | Schema version 1, optional hardware profile ID and automatic matching |
| `ACPI` | Host quirks with a matcher, kind, optional value and comment |
| `Boot` | Sources, installation policy, startup timeout and named entries |
| `Drivers` | Auto, Safe or Manual selection, module rules and host network settings |
| `Graphics` | Backend selection and scanout mirroring |
| `Memory` | Auto, Standard, Compact or Paged profile, Paged budgets, and swap settings |
| `Guest` | Storage, NOR, topology and guest kernel arguments |
| `Misc` | Boot log destinations and diagnostics |
| `NVRAM` | Guest variable defaults |
| `PlatformInfo` | Guest Mac model |
| `UEFI` | Firmware connection, memory reclamation and firmware location |

A startup entry is a dictionary with a `Name` and an `Overrides` dictionary. An override can contain any of the same sections. Nested dictionaries merge with the top-level settings and arrays replace the corresponding array. Up to 35 entries are allowed. Names use up to 40 printable ASCII bytes.

In a diagnostics build, setting `Misc/Debug/HostCheck` to true opens the hardware report directly. It defaults to false. Set `Misc/BootLog/Volume` to false to disable the readable data-volume log.

A driver `Block` rule accepts a `Match` and optional `Comment`. A `Force` rule accepts a `Match` and `Module`. Both use `pci:vvvv:dddd` with four hexadecimal digits for each ID. Keep the Wi-Fi passphrase in the separate file named by `Drivers/Network/WiFi/KeyFile`.

The canonical swap keys are `SizeGiB`, `Partition`, and `VolumeMaxGiB`. The parser accepts the memory design's `Bytes`, `PartitionUUID`, `VolumeMaxBytes`, and `Location=Volume` spellings too. When a canonical key and an alias both appear, their values must agree.
