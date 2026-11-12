// SPDX-License-Identifier: GPL-2.0-or-later

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "pthread-runtime-probe")]
    build_pthread_probe();
}

#[cfg(feature = "pthread-runtime-probe")]
fn build_pthread_probe() {
    use std::path::Path;

    let libc = Path::new("../stela-driver-gpu/src/compat/libc");
    let mut build = stela_c_build::native::configure();
    let resource = build
        .get_compiler()
        .to_command()
        .arg("-print-resource-dir")
        .output()
        .expect("locate clang resource headers");
    assert!(resource.status.success());
    let resource = String::from_utf8(resource.stdout).expect("clang resource path");
    build
        .flag("-nostdinc")
        .flag("-w")
        .flag("-isystem")
        .flag(libc.to_str().expect("libc include directory"))
        .flag("-isystem")
        .flag(format!("{}/include", resource.trim()))
        .file("src/engine/gpu/pthread_probe.c");
    if !cfg!(feature = "nvk") {
        build.file(libc.join("pthread.c"));
        build.define("STELA_PTHREAD_STANDALONE", None);
    }
    println!("cargo:rerun-if-changed={}", libc.display());
    println!("cargo:rerun-if-changed=src/engine/gpu/pthread_probe.c");
    build.compile("stela_pthread_probe");
}
