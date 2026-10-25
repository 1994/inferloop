//! Shared build contract: Cargo build script and standalone packaging planner.
//! Never recursively invokes Cargo: archive assembly runs after compilation.
#[path = "tools/build/backend_cfg.rs"]
mod backend_cfg;

use std::result::Result as StdResult;
use std::{env, error::Error, fs, path::PathBuf};

type Result<T> = StdResult<T, Box<dyn Error>>;

struct Plan<'a> {
    target: &'a str,
    zig_target: String,
    platform: &'static str,
    backend: &'static str,
    arch: &'static str,
}

impl<'a> Plan<'a> {
    fn new(requested: &'a str, platform: &str) -> Result<Self> {
        let (target, glibc) = requested
            .split_once('.')
            .map_or((requested, None), |(t, v)| (t, Some(v)));
        let (selected, backend, arch) = match target {
            "x86_64-unknown-linux-gnu" => ("linux-cuda", "cuda", "x86_64"),
            "aarch64-unknown-linux-gnu" => ("linux-cuda", "cuda", "aarch64"),
            "x86_64-apple-darwin" => ("macos-metal", "metal", "x86_64"),
            "aarch64-apple-darwin" => ("macos-metal", "metal", "aarch64"),
            _ => return Err(format!("unsupported production target {target}").into()),
        };
        if platform != "auto" && platform != selected {
            return Err(
                format!("{platform} conflicts with {target}; set TARGET explicitly").into(),
            );
        }
        if let Some(version) = glibc
            && (backend != "cuda"
                || !matches!(
                    version,
                    "2.17" | "2.28" | "2.31" | "2.34" | "2.35" | "2.36" | "2.39" | "2.41"
                ))
        {
            return Err("unsupported glibc baseline; use a documented Linux baseline".into());
        }
        Ok(Self {
            target,
            platform: selected,
            backend,
            arch,
            zig_target: if backend == "cuda" {
                format!("{target}.{}", glibc.unwrap_or("2.28"))
            } else {
                target.to_owned()
            },
        })
    }

    fn json(&self) -> String {
        let features = if self.backend == "cuda" {
            "\"cuda\""
        } else {
            ""
        };
        format!(
            "{{\"schema\":2,\"target\":\"{}\",\"zig_target\":\"{}\",\"platform\":\"{}\",\"backend\":\"{}\",\"arch\":\"{}\",\"features\":[{}]}}",
            self.target, self.zig_target, self.platform, self.backend, self.arch, features
        )
    }

    fn preflight(&self, host: &str) -> Result<()> {
        if self.backend == "metal" && !host.ends_with("apple-darwin") {
            let sdk = PathBuf::from(
                env::var("SDKROOT")
                    .map_err(|_| "cross-building Metal requires SDKROOT pointing to a macOS SDK")?,
            );
            for name in ["Metal.framework", "Foundation.framework"] {
                if !sdk.join("System/Library/Frameworks").join(name).is_dir() {
                    return Err(format!("SDKROOT has no {name}").into());
                }
            }
        }
        Ok(())
    }
}

fn cargo_build() -> Result<()> {
    println!("cargo:rerun-if-changed=../../../build.rs");
    println!("cargo:rerun-if-changed=../../../tools/build/backend_cfg.rs");
    for name in [
        "INFER_PACKAGE_TARGET",
        "INFER_PACKAGE_BUILD",
        "SDKROOT",
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_TEST_BACKENDS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let target = env::var("TARGET")?;
    let requested = env::var("INFER_PACKAGE_TARGET").unwrap_or_else(|_| target.clone());
    let plan = Plan::new(&requested, "auto")?;
    if plan.target != target {
        return Err("packaging target differs from Cargo TARGET".into());
    }
    if env::var_os("INFER_PACKAGE_BUILD").is_some() {
        if env::var_os("CARGO_FEATURE_TEST_BACKENDS").is_some()
            || (env::var_os("CARGO_FEATURE_CUDA").is_some() != (plan.backend == "cuda"))
        {
            return Err("production package backend/features mismatch".into());
        }
        plan.preflight(&env::var("HOST")?)?;
    }
    backend_cfg::emit(
        &env::var("CARGO_CFG_TARGET_OS")?,
        env::var_os("CARGO_FEATURE_CUDA").is_some(),
    );
    let output = PathBuf::from(env::var("OUT_DIR")?).join("infer-build.json");
    fs::write(output, plan.json())?;
    println!("cargo:rustc-env=INFER_BUILD_TARGET={target}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [] => cargo_build(),
        [mode, target, platform] if mode == "--plan" => {
            println!("{}", Plan::new(target, platform)?.json());
            Ok(())
        }
        [mode, target, host] if mode == "--preflight" => Plan::new(target, "auto")?.preflight(host),
        _ => Err("usage: build-contract --plan TARGET PLATFORM | --preflight TARGET HOST".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_drives_backend_not_host() -> Result<()> {
        let arm = Plan::new("aarch64-unknown-linux-gnu.2.34", "auto")?;
        assert_eq!(arm.arch, "aarch64");
        assert_eq!(arm.backend, "cuda");
        assert_eq!(arm.zig_target, "aarch64-unknown-linux-gnu.2.34");
        assert_eq!(
            Plan::new("x86_64-unknown-linux-gnu", "auto")?.zig_target,
            "x86_64-unknown-linux-gnu.2.28"
        );
        assert_eq!(
            Plan::new("aarch64-apple-darwin", "macos-metal")?.backend,
            "metal"
        );
        Ok(())
    }

    #[test]
    fn invalid_platforms_and_abi_fail_closed() {
        for (target, platform) in [
            ("x86_64-unknown-linux-gnu", "macos-metal"),
            ("aarch64-apple-darwin.2.28", "auto"),
            ("x86_64-unknown-linux-gnu.99.99", "auto"),
            ("x86_64-pc-windows-msvc", "auto"),
            ("aarch64-unknown-linux-musl", "auto"),
        ] {
            assert!(Plan::new(target, platform).is_err());
        }
    }
}
