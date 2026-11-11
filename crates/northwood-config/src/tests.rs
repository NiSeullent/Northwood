// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use alloc::{format, string::String, vec};
use plist::Value;

fn document(body: &str) -> String {
    format!("<plist version=\"1.0\"><dict>{body}</dict></plist>")
}

#[test]
fn examples_parse_and_defaults_match_the_schema() {
    let generic = parse(include_bytes!("../examples/nwconfig.sample.plist")).unwrap();
    assert!(generic.warnings.is_empty());
    assert_eq!(generic.config, NwConfig::default());
    assert_eq!(
        render::options_line(&generic.config, false).unwrap(),
        "--system"
    );
    let lenovo = parse(include_bytes!("../examples/nwconfig.15igl7.plist")).unwrap();
    assert!(lenovo.warnings.is_empty());
    assert_eq!(
        lenovo.config.northwood.hardware_profile,
        "lenovo-ideapad-1-15igl7"
    );
    assert_eq!(lenovo.config.memory.swap.bytes, 32 << 30);
}

#[test]
fn unknown_keys_warn_and_boot_entries_inherit_sparse_values() {
    let text = document(
        r#"
        <key>Boot</key><dict>
          <key>Timeout</key><integer>25</integer>
          <key>Entries</key><array>
            <dict><key>Name</key><string>Install</string>
            <key>Overrides</key><dict>
              <key>Graphics</key><dict><key>ScanoutMirror</key><true/></dict>
              <key>Memory</key><dict><key>Swap</key><dict><key>Enabled</key><false/></dict></dict>
              <key>Future</key><true/>
            </dict></dict>
          </array>
        </dict>
        <key>Memory</key><dict><key>Swap</key><dict><key>SizeGiB</key><integer>4</integer>
          <key>FutureKey</key><integer>6</integer></dict></dict>
    "#,
    );
    let parsed = parse(text.as_bytes()).unwrap();
    assert_eq!(
        parsed.warnings,
        ["Memory/Swap/FutureKey", "Boot/Entries[0]/Overrides/Future",]
    );
    assert_eq!(parsed.config.boot.timeout, BootTimeout::Seconds(25));
    assert!(parsed.config.memory.swap.enabled);
    let choice = &parsed.config.boot.entries[0];
    assert_eq!(choice.name, "Install");
    assert!(choice.config.graphics.scanout_mirror);
    assert!(!choice.config.memory.swap.enabled);
    assert_eq!(choice.config.memory.swap.bytes, 4 << 30);
    assert!(choice.config.boot.entries.is_empty());
}

#[test]
fn errors_produce_defaults_and_explain_the_field() {
    for (body, field) in [
        (
            r#"<key>Northwood</key><dict><key>SchemaVersion</key><integer>2</integer></dict>"#,
            "Northwood/SchemaVersion",
        ),
        (
            r#"<key>Boot</key><dict><key>Timeout</key><integer>61</integer></dict>"#,
            "Boot/Timeout",
        ),
        (
            r#"<key>Boot</key><dict><key>VerifyOnRead</key><string>true</string></dict>"#,
            "Boot/VerifyOnRead",
        ),
        (
            r#"<key>Memory</key><dict><key>Swap</key><dict><key>SizeGiB</key><integer>0</integer></dict></dict>"#,
            "Memory/Swap/Bytes",
        ),
        (
            r#"<key>Memory</key><dict><key>Swap</key><dict><key>SizeGiB</key><integer>4</integer><key>Bytes</key><integer>2147483648</integer></dict></dict>"#,
            "Memory/Swap/Bytes",
        ),
        (
            r#"<key>PlatformInfo</key><dict><key>Ecid</key><integer>1</integer></dict>"#,
            "PlatformInfo/Ecid",
        ),
        (
            r#"<key>Drivers</key><dict><key>Network</key><dict><key>WiFi</key><dict><key>Password</key><string>secret</string></dict></dict></dict>"#,
            "Drivers/Network/WiFi/Password",
        ),
        (
            r#"<key>Drivers</key><dict><key>Block</key><array><dict><key>Match</key><string>pci:8086:31DC</string></dict></array><key>Force</key><array><dict><key>Module</key><string>IntelWiFi</string><key>Match</key><string>pci:8086:31dc</string></dict></array></dict>"#,
            "Drivers/Force[0]/Match",
        ),
    ] {
        let bytes = document(body);
        let (parsed, error) = parse_or_failsafe(bytes.as_bytes());
        assert_eq!(parsed.config, NwConfig::default());
        assert!(matches!(error, Some(Error::Field { path, .. }) if path == field));
    }
}

#[test]
fn rejects_inline_credentials_at_known_paths_and_keeps_future_keys() {
    let secret = "SYNTHETIC-PASS-MARKER";
    for (body, path) in [
        (
            format!(
                "<key>Drivers</key><dict><key>Network</key><dict>\
                <key>WiFi</key><dict/><key>Wifi</key><dict>\
                <key>preSharedKey</key><string>{secret}</string></dict></dict></dict>"
            ),
            "Drivers/Network/Wifi/preSharedKey",
        ),
        (
            "<key>Misc</key><dict><key>Security</key><dict>\
             <key>PasswordSalt</key><data>AAEC</data></dict></dict>"
                .into(),
            "Misc/Security/PasswordSalt",
        ),
        (
            format!(
                "<key>PlatformInfo</key><dict><key>Generic</key><dict>\
                <key>SystemSerialNumber</key><string>{secret}</string></dict></dict>"
            ),
            "PlatformInfo/Generic/SystemSerialNumber",
        ),
        (
            format!(
                "<key>Boot</key><dict><key>Entries</key><array><dict>\
                <key>Name</key><string>Network QA</string><key>Overrides</key><dict>\
                <key>Drivers</key><dict><key>Network</key><dict><key>WiFi</key><dict>\
                <key>password</key><string>{secret}</string></dict></dict></dict>\
                </dict></dict></array></dict>"
            ),
            "Boot/Entries[0]/Overrides/Drivers/Network/WiFi/password",
        ),
    ] {
        let error = parse(document(&body).as_bytes()).unwrap_err();
        assert!(!format!("{error}").contains(secret));
        assert!(matches!(error, Error::Field { path: actual, .. } if actual == path));
    }

    let text = document(
        r"<key>Drivers</key><dict><key>Network</key><dict><key>WiFi</key><dict>
            <key>SSID</key><string>fixture</string>
            <key>KeyFile</key><string>\EFI\northwood\fixture.key</string>
            <key>FutureWiFiKey</key><true/>
        </dict></dict></dict>
        <key>Future</key><dict><key>Password</key><string>opaque value</string></dict>",
    );
    let parsed = parse(text.as_bytes()).unwrap();
    assert_eq!(
        parsed.config.drivers.network.wifi.key_file,
        r"\EFI\northwood\fixture.key"
    );
    assert_eq!(
        parsed.warnings,
        ["Drivers/Network/WiFi/FutureWiFiKey", "Future"]
    );
}

#[test]
fn hid_quirk_values_match_driver_bounds_and_fall_back_on_error() {
    let quirk = |kind: &str, value: &str| {
        document(&format!(
            "<key>ACPI</key><dict><key>Quirks</key><array><dict>\
             <key>Match</key><string>SYNA2BA6</string><key>Kind</key><string>{kind}</string>\
             {value}</dict></array></dict>"
        ))
    };
    for (kind, min, max) in [
        ("I2cHidAddress", 0, 0x3ff),
        ("I2cHidDescriptor", 0, 0xffff),
        ("I2cHidForcePolling", 1_000, 50_000),
    ] {
        for number in [min, max] {
            let text = quirk(
                kind,
                &format!("<key>Value</key><integer>{number}</integer>"),
            );
            let parsed = parse(text.as_bytes()).unwrap();
            assert_eq!(
                parsed.config.acpi.quirks[0].value,
                Some(Value::Integer(number))
            );
        }
        for value in [
            String::new(),
            String::from("<key>Value</key><string>8000</string>"),
            format!("<key>Value</key><integer>{}</integer>", min - 1),
            format!("<key>Value</key><integer>{}</integer>", max + 1),
        ] {
            let (parsed, error) = parse_or_failsafe(quirk(kind, &value).as_bytes());
            assert_eq!(parsed.config, NwConfig::default());
            assert!(parsed.warnings.is_empty());
            assert!(matches!(
                error,
                Some(Error::Field { path, .. }) if path == "ACPI/Quirks[0]/Value"
            ));
        }
    }
    let future = quirk(
        "FutureKind",
        "<key>Value</key><array><string>future</string></array>",
    );
    assert!(matches!(
        parse(future.as_bytes()).unwrap().config.acpi.quirks[0].value,
        Some(Value::Array(_))
    ));
}

#[test]
fn host_check_selects_diagnostics_and_respects_the_volume_log() {
    let text = document(
        r#"
        <key>Misc</key><dict>
          <key>Debug</key><dict><key>HostCheck</key><true/></dict>
          <key>BootLog</key><dict><key>Volume</key><false/></dict>
        </dict>
    "#,
    );
    let config = parse(text.as_bytes()).unwrap().config;
    assert!(render::options_line(&config, false).is_err());
    assert_eq!(
        render::options_line(&config, true).unwrap(),
        "--host-check --no-volume-log"
    );
}

#[test]
fn renderer_preserves_operands_and_diagnostics_gate() {
    let text = document(
        r#"
        <key>Boot</key><dict><key>Bundle</key><string>stela/boot/build</string></dict>
        <key>Drivers</key><dict>
          <key>Input</key><true/>
          <key>Network</key><dict>
            <key>WiFi</key><dict><key>SSID</key><string>집 &amp; 5%</string></dict>
            <key>IPv4</key><string>192.168.1.5/24</string>
            <key>Gateway</key><string>192.168.1.1</string>
            <key>DNS</key><array><string>1.1.1.1</string><string>2606:4700:4700::1111</string></array>
          </dict>
        </dict>
        <key>Guest</key><dict><key>Cores</key><integer>1</integer><key>Parallel</key><false/></dict>
        <key>Misc</key><dict><key>Debug</key><dict><key>Break</key><integer>4096</integer></dict></dict>
    "#,
    );
    let parsed = parse(text.as_bytes()).unwrap();
    assert!(render::options_line(&parsed.config, false).is_err());
    let line = render::options_line(&parsed.config, true).unwrap();
    assert_eq!(
        line,
        "--system --bundle stela/boot/build --input --wifi 집%20&%205%25 --wifi-key \\EFI\\northwood\\wifi.key --ipv4 192.168.1.5/24 --ipv4-gateway 192.168.1.1 --dns 1.1.1.1,2606:4700:4700::1111 --native-cores 1 --serialized --break 0x1000"
    );

    let mut config = parsed.config;
    config.misc.boot_log.volume = false;
    assert!(
        render::options_line(&config, true)
            .unwrap()
            .contains("--no-volume-log")
    );
    config.misc.debug.raw = String::from("--ecid 1");
    assert!(render::options_line(&config, true).is_err());
    config.boot.bundle = String::from("one --single-user");
    assert!(
        matches!(render::options_line(&config, true), Err(Error::Field { path, .. }) if path == "Boot/Bundle")
    );
}

#[test]
fn malformed_xml_is_bounded() {
    for body in [
        "<key>X</key><data>AA=A</data>",
        "<key>X</key><data>AB==</data>",
        "<key>X</key><string>&bogus;</string>",
        "<key>X</key><string>&#0;</string>",
        "<key>X</key><string>&#x+41;</string>",
        "<key>X</key><integer>18446744073709551616</integer>",
        "<key>X</key><true/><key>X</key><false/>",
        "<key>X</key><string><dict/></string>",
        "<!-- a--b -->",
    ] {
        let text = document(body);
        assert!(
            matches!(parse(text.as_bytes()), Err(Error::Xml(_))),
            "{body}"
        );
    }
    let text = document(&format!(
        "<key>X</key>{}{}",
        "<array>".repeat(48),
        "</array>".repeat(48)
    ));
    assert!(parse(text.as_bytes()).is_err());
    assert!(parse(&vec![b' '; plist::MAX_BYTES + 1]).is_err());
    assert!(parse(br#"<?xml version="1.0" encoding="UTF-16"?><plist><dict/></plist>"#).is_err());
    assert!(parse(br#"<?xml version="1.0"?><plist version = '1.0' ><dict/></plist>"#).is_ok());
    let sample = include_bytes!("../examples/nwconfig.sample.plist");
    for offset in 0..sample.len() - 1 {
        assert!(parse(&sample[..offset]).is_err(), "truncated at {offset}");
    }
}

#[test]
fn xml_decodes_escaped_data_and_signed_integers() {
    let text = document(
        r#"
        <key>Text</key><string>&lt;&amp;&quot;&apos;&gt;&#x41;&#65;</string>
        <key>Data</key><data>AAF+ /w==</data>
        <key>Number</key><integer>-9223372036854775808</integer>
        <key>Unsigned</key><integer>18446744073709551615</integer>
    "#,
    );
    let dictionary = plist::parse(text.as_bytes()).unwrap();
    assert_eq!(dictionary["Text"], Value::String(String::from("<&\"'>AA")));
    assert_eq!(dictionary["Data"], Value::Data(vec![0, 1, 126, 255]));
    assert_eq!(dictionary["Number"], Value::Integer(i64::MIN.into()));
    assert_eq!(dictionary["Unsigned"], Value::Integer(u64::MAX.into()));
}

#[test]
fn xml_normalizes_literal_line_endings_before_comparing_keys() {
    for (first, second) in [("A\rB", "A\nB"), ("A\r\nB", "A&#10;B")] {
        let text = document(&format!(
            "<key>{first}</key><true/><key>{second}</key><false/>"
        ));
        assert!(matches!(
            plist::parse(text.as_bytes()),
            Err(plist::XmlError {
                reason: "duplicate dictionary key",
                ..
            })
        ));
    }
    let text = document("<key>A&#13;B</key><string>x\r\ny\rz&#13;</string><key>A\nB</key><true/>");
    let dictionary = plist::parse(text.as_bytes()).unwrap();
    assert_eq!(dictionary["A\rB"], Value::String(String::from("x\ny\nz\r")));
    assert_eq!(dictionary["A\nB"], Value::Boolean(true));
}

#[test]
fn xml_accepts_whitespace_before_the_end_tag_delimiter() {
    let text = "<plist><dict><key>X</key ><array><string>x</string\t>\
                <integer>1</integer\n><data>AA==</data >\
                <true></true ><false></false\r></array ></dict ></plist >";
    let dictionary = plist::parse(text.as_bytes()).unwrap();
    assert_eq!(
        dictionary["X"],
        Value::Array(vec![
            Value::String(String::from("x")),
            Value::Integer(1),
            Value::Data(vec![0]),
            Value::Boolean(true),
            Value::Boolean(false),
        ])
    );
    let wrong = text.replace("</string\t>", "</stringExtra\t>");
    assert!(plist::parse(wrong.as_bytes()).is_err());
}

#[cfg(feature = "std")]
#[test]
fn xml_writer_round_trips_future_keys_and_base64_padding() {
    let text = document(
        r#"
        <key>Future</key><array>
        <string>&lt;&amp;&quot;&apos;&gt;&#13;한글</string>
        <data></data><data>AA==</data><data>AAE=</data><data>AAEC</data>
        <dict/><array/><false/><integer>-2</integer>
        </array>
    "#,
    );
    let dictionary = plist::parse(text.as_bytes()).unwrap();
    let xml = plist::to_xml(&dictionary);
    assert_eq!(plist::parse(xml.as_bytes()).unwrap(), dictionary);
}
