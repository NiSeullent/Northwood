//! `cargo xtask dist`: the release CLIs with their matching Northwood EFI
//! embedded. The boot image builds first, from the pinned sources `gpu
//! fetch` checks out; `stela-cli`'s build script then embeds the file
//! `STELA_EFI` names. The embedded image carries no debug sections: setup
//! writes it beside the old one on the USB's 200 MiB EFI partition.
use crate::tools::{cargo, exec, root};
use anyhow::Result;

pub fn run() -> Result<()> {
    crate::gpu::fetch::fetch(false)?;
    let built = crate::boot::build(true)?;
    let efi = built.with_file_name("northwood.efi");
    crate::boot::deploy_image(&built, &efi)?;
    exec(
        cargo()
            .args([
                "build",
                "--release",
                "--locked",
                "-p",
                "stela-cli",
                "-p",
                "northwood",
                "--features",
                "northwood/installer",
            ])
            .env("STELA_EFI", &efi),
    )?;
    let release = root().join("target/release");
    for name in ["northwood", "stela"] {
        println!(
            "{}",
            release
                .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
                .display()
        );
    }
    println!("{}", efi.display());
    Ok(())
}
