//! Complete configuration validation and class-driver-based selection. USB 2.0
//! 9.6.3/9.6.5/9.6.6: descriptor indices, values and alternate settings are
//! separate namespaces. Unknown class descriptors remain in the raw inventory.
use super::*;

/// Endpoint types (USB 2.0 Table 9-13) and the largest wMaxPacketSize Linux
/// `usb_parse_endpoint` accepts for each at each speed, indexed by type:
/// control, isochronous, bulk, interrupt.
const ISOCHRONOUS: usize = 1;
const BULK: usize = 2;
const INTERRUPT: usize = 3;
const LOW_SPEED_MAXIMUMS: [u16; 4] = [8, 0, 0, 8];
const FULL_SPEED_MAXIMUMS: [u16; 4] = [64, 1023, 64, 64];
const HIGH_SPEED_MAXIMUMS: [u16; 4] = [64, 1024, 512, 1024];
const SUPER_SPEED_MAXIMUMS: [u16; 4] = [512, 1024, 1024, 1024];
/// wMaxPacketSize's additional-transactions field (high-speed periodic).
const MULTIPLE_TRANSACTIONS: u16 = 3 << 11;
const ENDPOINT_NUMBER: u8 = 0x0f;
const ENDPOINT_TYPE: u8 = 0x03;
/// bInterval replacements: 8 ms in microframe or frame exponents, and the
/// low-speed minimum period.
const HIGH_SPEED_DEFAULT_INTERVAL: u8 = 7;
const FULL_SPEED_ISOCHRONOUS_INTERVAL: u8 = 4;
const SLOW_INTERRUPT_INTERVAL: u8 = 10;
/// SuperSpeed companion limits (Linux `usb_parse_ss_endpoint_companion`).
const MAX_BURST: u8 = 15;
const MAX_STREAMS_EXPONENT: u8 = 16;
const COMPANION_LENGTH: u8 = 6;
const ISOCHRONOUS_MULT: u8 = 0x03;

/// A configuration descriptor as Linux `usb_parse_configuration` reads it:
/// the header must be valid, but a bad descriptor ends the stream, nested
/// configurations, duplicate interface settings and invalid or duplicate
/// endpoints are skipped, endpoint fields are corrected as Linux corrects
/// them, a SuperSpeed endpoint without a companion gets the default one,
/// and the counts and total length describe what remains.
pub(crate) fn sanitize(bytes: &[u8], speed: u8) -> Result<Vec<u8>, &'static str> {
    if bytes.len() < 9 || bytes[0] != 9 || bytes[1] != CONFIGURATION_DESCRIPTOR || bytes[5] == 0 {
        return Err("invalid USB configuration header/value");
    }
    let total = usize::from(u16::from_le_bytes([bytes[2], bytes[3]])).min(bytes.len());
    let mut out = bytes[..9].to_vec();
    let mut settings: Vec<(u8, u8)> = Vec::new();
    let mut owners: Vec<(u8, u8)> = Vec::new();
    // The current interface's number and where its bNumEndpoints is.
    let mut interface: Option<(u8, usize)> = None;
    let mut setting_endpoints: Vec<u8> = Vec::new();
    let mut skipping = false;
    // A SuperSpeed endpoint still waiting for its companion: its type and
    // packet size, and the default companion's bytes per interval.
    let mut companion_due: Option<(usize, u16, u16)> = None;
    let mut at = 9;
    while at + 2 <= total {
        let length = usize::from(bytes[at]);
        if length < 2 || at + length > total {
            break;
        }
        let d = &bytes[at..at + length];
        at += length;
        let kind = d[1];
        if (kind != SS_ENDPOINT_COMPANION || length != usize::from(COMPANION_LENGTH))
            && let Some((_, _, bytes_per_interval)) = companion_due.take()
        {
            out.extend_from_slice(&default_companion(bytes_per_interval));
        }
        match kind {
            CONFIGURATION_DESCRIPTOR => {}
            INTERFACE_DESCRIPTOR => {
                skipping = length < 9 || settings.contains(&(d[2], d[3]));
                if skipping {
                    continue;
                }
                settings.push((d[2], d[3]));
                finish_interface(&mut out, interface, setting_endpoints.len());
                interface = Some((d[2], out.len() + 4));
                setting_endpoints.clear();
                out.extend_from_slice(d);
            }
            ENDPOINT_DESCRIPTOR => {
                let Some((number, _)) = interface.filter(|_| !skipping && length >= 7) else {
                    continue;
                };
                let address = d[2];
                let owned_elsewhere = owners
                    .iter()
                    .any(|(other, owned)| *owned == address && *other != number);
                let maximum = u16::from_le_bytes([d[4], d[5]]);
                if address & ENDPOINT_NUMBER == 0
                    || maximum == 0
                    || setting_endpoints.contains(&address)
                    || owned_elsewhere
                {
                    continue;
                }
                let mut endpoint = d.to_vec();
                fix_endpoint(&mut endpoint, speed);
                setting_endpoints.push(address);
                owners.push((number, address));
                out.extend_from_slice(&endpoint);
                if matches!(speed, SUPER_SPEED | SUPER_SPEED_PLUS) {
                    let kind = usize::from(d[3] & ENDPOINT_TYPE);
                    let packet = u16::from_le_bytes([endpoint[4], endpoint[5]]);
                    let periodic = matches!(kind, ISOCHRONOUS | INTERRUPT);
                    companion_due = Some((kind, packet, if periodic { packet } else { 0 }));
                }
            }
            SS_ENDPOINT_COMPANION => {
                // Only a whole companion right after its endpoint is read.
                let Some((kind, packet, _)) = companion_due.take() else {
                    continue;
                };
                let mut companion = d.to_vec();
                companion[2] = companion[2].min(MAX_BURST);
                if kind == BULK {
                    companion[3] = companion[3].min(MAX_STREAMS_EXPONENT);
                }
                if matches!(kind, ISOCHRONOUS | INTERRUPT) {
                    // A periodic endpoint moves at most its packets per burst
                    // and per isochronous Mult each interval.
                    let mult = match kind {
                        ISOCHRONOUS => u32::from(companion[3] & ISOCHRONOUS_MULT) + 1,
                        _ => 1,
                    };
                    let most = u32::from(packet) * (u32::from(companion[2]) + 1) * mult;
                    let claimed = u16::from_le_bytes([companion[4], companion[5]]);
                    let fixed = u32::from(claimed).min(most) as u16;
                    companion[4..6].copy_from_slice(&fixed.to_le_bytes());
                }
                out.extend_from_slice(&companion);
            }
            _ if skipping => {}
            _ => out.extend_from_slice(d),
        }
    }
    if let Some((_, _, bytes_per_interval)) = companion_due {
        out.extend_from_slice(&default_companion(bytes_per_interval));
    }
    finish_interface(&mut out, interface, setting_endpoints.len());
    let mut numbers: Vec<u8> = settings.iter().map(|(number, _)| *number).collect();
    numbers.sort_unstable();
    numbers.dedup();
    out[4] = numbers.len() as u8;
    let length = u16::try_from(out.len()).map_err(|_| "USB configuration too long")?;
    out[2..4].copy_from_slice(&length.to_le_bytes());
    Ok(out)
}

/// Record the endpoints an interface descriptor really has.
fn finish_interface(out: &mut [u8], interface: Option<(u8, usize)>, endpoints: usize) {
    if let Some((_, count)) = interface {
        out[count] = endpoints as u8;
    }
}

/// Linux's companion for a SuperSpeed endpoint without one: no bursts, no
/// streams, and a periodic endpoint's packet per service interval.
fn default_companion(bytes_per_interval: u16) -> [u8; 6] {
    let [low, high] = bytes_per_interval.to_le_bytes();
    [COMPANION_LENGTH, SS_ENDPOINT_COMPANION, 0, 0, low, high]
}

/// Linux `usb_parse_endpoint`'s corrections of bInterval and wMaxPacketSize.
fn fix_endpoint(endpoint: &mut [u8], speed: u8) {
    let kind = usize::from(endpoint[3] & ENDPOINT_TYPE);
    let interval = endpoint[6];
    let high_speed = matches!(speed, HIGH_SPEED | SUPER_SPEED | SUPER_SPEED_PLUS);
    endpoint[6] = match kind {
        INTERRUPT if high_speed && !(1..=16).contains(&interval) => {
            // Many devices give full-speed periods: convert to microframes.
            match (u32::from(interval) * 8).checked_ilog2() {
                Some(bit) => bit as u8 + 1,
                None => HIGH_SPEED_DEFAULT_INTERVAL,
            }
        }
        INTERRUPT if !high_speed && interval == 0 => SLOW_INTERRUPT_INTERVAL,
        ISOCHRONOUS if !(1..=16).contains(&interval) => match speed {
            HIGH_SPEED => HIGH_SPEED_DEFAULT_INTERVAL,
            _ => FULL_SPEED_ISOCHRONOUS_INTERVAL,
        },
        _ => interval,
    };
    let field = u16::from_le_bytes([endpoint[4], endpoint[5]]);
    let multiple = match speed {
        HIGH_SPEED if matches!(kind, ISOCHRONOUS | INTERRUPT) => field & MULTIPLE_TRANSACTIONS,
        _ => 0,
    };
    let maximums = match speed {
        LOW_SPEED => LOW_SPEED_MAXIMUMS,
        FULL_SPEED => FULL_SPEED_MAXIMUMS,
        SUPER_SPEED | SUPER_SPEED_PLUS => SUPER_SPEED_MAXIMUMS,
        _ => HIGH_SPEED_MAXIMUMS,
    };
    let packet = (field & !multiple).min(maximums[kind]);
    endpoint[4..6].copy_from_slice(&(multiple | packet).to_le_bytes());
}

#[cfg(test)]
pub(super) fn plan(
    device: &Device,
    interface: &Interface,
    all: &[Interface],
) -> Result<Plan, &'static str> {
    plan_with_controller(device, interface, all, None)
}

/// Whether a driver binds through this interface. CDC data and audio streaming
/// interfaces belong to the communications or audio control interface that
/// claims them.
pub(super) fn primary(interface: &Interface) -> bool {
    interface.alternate == 0
        && interface.class != cdc::DATA_CLASS
        && !(interface.class == audio::CLASS && interface.subclass == audio::STREAMING_SUBCLASS)
}

pub(super) fn plan_with_controller(
    device: &Device,
    interface: &Interface,
    all: &[Interface],
    controller: Option<&Controller>,
) -> Result<Plan, &'static str> {
    if bluetooth::ath3k::is_candidate(device) && interface.number == 0 {
        return bluetooth::ath3k::Binding::new(device, interface)
            .map(Plan::BluetoothBootloader)
            .and_then(|binding| supported(binding, device.speed));
    }
    let binding = match interface.class {
        audio::CLASS => audio::plan(interface, all, &device.configuration, device.speed)
            .map(Plan::Audio)
            .map_err(|_| "unsupported or malformed UAC2 function descriptors"),
        storage::CLASS => storage::plan(device, interface, all, controller),
        hid::CLASS => hid::plan(interface),
        cdc::COMMUNICATIONS => cdc::plan(interface, all),
        bluetooth::CLASS if cdc::is_rndis(interface) => cdc::plan(interface, all),
        bluetooth::CLASS => bluetooth::Binding::new(device, interface).map(Plan::Bluetooth),
        0xff if bluetooth::broadcom::is_vendor_interface(device, interface) => {
            bluetooth::Binding::new(device, interface).map(Plan::Bluetooth)
        }
        0xff if ethernet::listed(&device.descriptor) => {
            let mut failure = "no supported vendor Ethernet interface setting";
            let mut chosen = None;
            for candidate in all
                .iter()
                .filter(|candidate| candidate.number == interface.number && candidate.class == 0xff)
            {
                match ethernet::Binding::new(device, candidate)
                    .map(Plan::VendorEthernet)
                    .and_then(|binding| supported(binding, device.speed))
                {
                    Ok(binding) => {
                        chosen = Some(binding);
                        break;
                    }
                    Err(error) if candidate.alternate == 0 => failure = error,
                    Err(_) => {}
                }
            }
            chosen.ok_or(failure)
        }
        _ => Err("no class driver for this interface"),
    }?;
    supported(binding, device.speed)
}

fn supported(binding: Plan, speed: u8) -> Result<Plan, &'static str> {
    for endpoint in binding.endpoints() {
        endpoint::Context::new(&endpoint, speed)
            .map_err(|_| "USB function has an unsupported speed-specific endpoint")?;
    }
    Ok(binding)
}

/// All simultaneously active settings, including interfaces left to other
/// drivers, must have distinct endpoint addresses. Class functions must own
/// disjoint interfaces; two CDC masters cannot both claim one data interface.
pub(super) fn active_settings(
    all: &[Interface],
    selected: &[(u8, u8)],
) -> Result<(), &'static str> {
    let mut endpoints = Vec::new();
    for interface in all {
        let alternate = selected
            .iter()
            .find(|(number, _)| *number == interface.number)
            .map_or(0, |(_, alternate)| *alternate);
        if interface.alternate != alternate {
            continue;
        }
        for endpoint in &interface.endpoints {
            if endpoints.contains(&endpoint.address) {
                return Err("USB alternate settings activate overlapping endpoints");
            }
            endpoints.push(endpoint.address);
        }
    }
    Ok(())
}

/// Choose the unique configuration offering the greatest number of supported
/// functions. At equal coverage, an implemented observed vendor Ethernet mode
/// wins over its generic CDC alternative, as r8152's cfgselector requires.
/// Prefer NCM, ECM and RNDIS in that order among tied CDC modes.
/// No supported function leaves the first descriptor selected only
/// for unused-interface diagnostics; bind will issue no configuration request.
#[cfg(test)]
pub(crate) fn select(device: &mut Device) -> Result<(), &'static str> {
    select_with_controller(device, None)
}

pub(super) fn select_with_controller(
    device: &mut Device,
    controller: Option<&Controller>,
) -> Result<(), &'static str> {
    let mut candidates = Vec::new();
    for index in 0..device.configurations.len() {
        let Ok(configuration) = sanitize(&device.configurations[index], device.speed) else {
            continue;
        };
        device.configuration = configuration;
        let all = interfaces(&device.configuration);
        let mut count = 0;
        let mut vendor_ethernet = false;
        let mut uas_count = 0;
        let mut asix = false;
        let mut cdc = 0;
        let mut endpoints = Vec::new();
        let mut claimed = Vec::new();
        for interface in all.iter().filter(|i| primary(i)) {
            if device.descriptor[4] == HUB_CLASS && interface.class == HUB_CLASS {
                if hub::status_endpoint(device, interface).is_some() {
                    count += 1;
                }
            } else if let Ok(binding) = plan_with_controller(device, interface, &all, controller) {
                if let Plan::VendorEthernet(binding) = &binding {
                    vendor_ethernet |= binding.observed_vendor();
                    asix |= binding.asix_candidate();
                }
                if let Plan::Ethernet(binding) = &binding {
                    cdc = cdc.max(binding.preference());
                }
                uas_count +=
                    usize::from(matches!(&binding, Plan::Storage(binding) if binding.uas()));
                for (number, alternate) in binding.settings(interface.number) {
                    if claimed.iter().any(|(previous, _)| *previous == number) {
                        return Err("supported USB functions claim the same interface");
                    }
                    claimed.push((number, alternate));
                }
                for endpoint in binding.endpoints() {
                    if endpoints.contains(&endpoint.address) {
                        return Err("supported USB functions claim the same endpoint");
                    }
                    endpoints.push(endpoint.address);
                }
                count += 1;
            }
        }
        active_settings(&all, &claimed)?;
        candidates.push((index, (count, vendor_ethernet, uas_count), asix, cdc));
    }
    let maximum = candidates
        .iter()
        .map(|candidate| candidate.1)
        .max()
        .unwrap_or((0, false, 0));
    let mut finalists: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.1 == maximum)
        .collect();
    // Unlike Realtek cfgselector, ASIX has no source-backed preconfiguration
    // silicon check. At equal supported coverage keep a usable CDC alternative
    // over an ASIX-only candidate; unrelated configurations still tie normally.
    let cdc = finalists
        .iter()
        .map(|candidate| candidate.3)
        .max()
        .unwrap_or(0);
    if !maximum.1 && cdc > 0 {
        finalists.retain(|candidate| !candidate.2 || candidate.3 > 0);
    }
    finalists.retain(|candidate| candidate.3 == 0 || candidate.3 == cdc);
    // Linux `usb_choose_configuration` takes the first of equal candidates.
    let best = finalists.first().map(|candidate| candidate.0);
    if maximum.0 == 0 && device.descriptor[4] == HUB_CLASS {
        return Err("USB hub has no supported configuration");
    }
    let chosen = device
        .configurations
        .get(best.unwrap_or(0))
        .ok_or("USB device has no configuration descriptors")?;
    device.configuration = sanitize(chosen, device.speed)?;
    Ok(())
}

#[cfg(test)]
pub(super) mod tests;
