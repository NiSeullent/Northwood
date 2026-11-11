// SPDX-License-Identifier: GPL-2.0-or-later
use northwood_config::{parse, render::options_line};
use stela_boot_core::options::System;

#[test]
fn the_two_published_samples_render_valid_stela_options() {
    for bytes in [
        &include_bytes!("../../northwood-config/examples/nwconfig.sample.plist")[..],
        &include_bytes!("../../northwood-config/examples/nwconfig.15igl7.plist")[..],
    ] {
        let parsed = parse(bytes).unwrap();
        assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
        let configs = core::iter::once(&parsed.config)
            .chain(parsed.config.boot.entries.iter().map(|entry| &entry.config));
        for config in configs {
            let line = options_line(config, false).unwrap();
            let system = System::parse(&line).unwrap();
            assert_eq!(
                system.ssid.as_deref(),
                (!config.drivers.network.wifi.ssid.is_empty())
                    .then_some(config.drivers.network.wifi.ssid.as_str())
            );
            assert_eq!(system.no_volume_log, !config.misc.boot_log.volume);
            assert_eq!(System::parse(&system.to_string()).unwrap(), system);
        }
    }
}
