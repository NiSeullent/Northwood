//! Ordered cross-field rules over the complete record; no parsing or mutation.
use super::*;

pub(super) fn system(options: &System) -> Result<(), Error> {
    restore(options)?;
    wifi(options)?;
    ipv4(options)?;
    install(options)?;
    boot_selection(options)?;
    dram(options)?;
    if options.audio_own_hpet && options.audio_check.is_none() {
        return Err(Error::Constraint(Constraint::AudioHpetDependency));
    }
    if options.native_pci_own_ioapic
        && (options.audio_check.is_some()
            || options.native_bootstrap_check
            || options.native_shared_check)
    {
        return Err(Error::Constraint(Constraint::NativePciIoapicExclusive));
    }
    if options.network_check_ranges.len() > MAX_NETWORK_CHECK_RANGES {
        return Err(Error::NetworkCheckRangeLimit);
    }
    if !options.network_check_ranges.is_empty()
        && (options.network_check.is_some() || options.network_check_get.is_some())
    {
        return Err(Error::DuplicateNetworkCheck);
    }
    for range in &options.network_check_ranges {
        range.check()?;
    }
    if options.network_check_reopen
        && !matches!(options.network_check_ranges.as_slice(), [first, second]
            if first.url == second.url && first.total == second.total && first.etag == second.etag
                && parse_url(&first.url)?.scheme == stela_runtime::net::http::Scheme::Http)
    {
        return Err(Error::NetworkCheckReopen);
    }
    Ok(())
}

fn restore(options: &System) -> Result<(), Error> {
    let restore_inputs = options.restore_source.is_some()
        || options.restore_manifest
        || options.restore_identity.is_some();
    if options.restore
        && (!options.bundle
            || options.preboot.is_some()
            || options.storage.is_none()
            || options.restore_source.is_none()
            || !options.restore_manifest
            || options.restore_identity.is_none()
            || options.survey)
    {
        return Err(Error::Constraint(Constraint::RestoreConfiguration));
    }
    if !options.restore && restore_inputs {
        return Err(Error::Constraint(Constraint::UnexpectedRestoreInputs));
    }
    Ok(())
}

fn wifi(options: &System) -> Result<(), Error> {
    if options.ssid.is_none() && (options.wifi_key.is_some() || options.wifi_country.is_some()) {
        return Err(Error::Constraint(Constraint::WifiDependency));
    }
    Ok(())
}

fn ipv4(options: &System) -> Result<(), Error> {
    if let Some(gateway) = options.ipv4_gateway
        && !options.ipv4.is_some_and(|ipv4| ipv4.neighbors(gateway))
    {
        return Err(Error::Constraint(Constraint::Ipv4Gateway));
    }
    Ok(())
}

fn install(options: &System) -> Result<(), Error> {
    if options.install_source.is_some() && options.install.is_none() {
        return Err(Error::Constraint(Constraint::InstallSourceDependency));
    }
    if options.install.is_some()
        && (!options.bundle
            || options.preboot.is_some()
            || options.storage.is_none()
            || options.nor.is_none()
            || options.restore
            || options.survey)
    {
        return Err(Error::Constraint(Constraint::InstallConfiguration));
    }
    if options.update_restore && options.install.is_none() {
        return Err(Error::Constraint(Constraint::UpdateDependency));
    }
    if options.development_restore && (options.install.is_none() || options.update_restore) {
        return Err(Error::Constraint(Constraint::DevelopmentRestore));
    }
    if options.verify_on_read && options.install.is_none() && !options.restore {
        return Err(Error::Constraint(Constraint::VerifyDependency));
    }
    Ok(())
}

fn boot_selection(options: &System) -> Result<(), Error> {
    if options.preboot.is_none()
        && (options.boot_volume.is_some()
            || options.boot_command.is_some()
            || options.safe_boot
            || options.maintenance_write
            || options.single_user
            || options.upgrade
            || options.recovery.is_some())
    {
        return Err(Error::Constraint(Constraint::PrebootSelection));
    }
    if options.upgrade && (options.recovery.is_some() || options.safe_boot) {
        return Err(Error::Constraint(Constraint::UpgradeSelection));
    }
    if options.recovery.is_some() && options.safe_boot {
        return Err(Error::Constraint(Constraint::RecoverySafeMode));
    }
    Ok(())
}

fn dram(options: &System) -> Result<(), Error> {
    if !options.dram_base.is_multiple_of(LARGE) {
        return Err(Error::Constraint(Constraint::DramAlignment));
    }
    Ok(())
}
