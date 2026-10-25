//! Backend availability shared by the CLI and CUDA crate build scripts.

/// Emit target-based backend capabilities; Cargo features alone do not imply OS support.
pub fn emit(target_os: &str, cuda_feature: bool) {
    let cuda = target_os == "linux" && cuda_feature;
    let metal = target_os == "macos";
    for (name, enabled) in [
        ("cuda_backend", cuda),
        ("metal_backend", metal),
        ("native_backend", cuda || metal),
    ] {
        println!("cargo:rustc-check-cfg=cfg({name})");
        if enabled {
            println!("cargo:rustc-cfg={name}");
        }
    }
}
