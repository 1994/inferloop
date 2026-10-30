use crate::aot::artifact::Manifest;
use crate::contract::target::{Arch, Profile, Target};
use sha2::{Digest, Sha256};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::process;

const CUBIN: &[u8] = b"fake cubin bytes";

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn manifest(arch: &str, sha256: &str) -> String {
    format!(
        r#"{{"schema":1,"profile":"aot","arch":"{arch}","toolchain":"nvcc 12.9.86","entries":[{{"name":"projection.dense_bf16","contract":"bf16-f32acc","cubin":"sm89/projection.cubin","sha256":"{sha256}"}}]}}"#
    )
}

#[test]
fn parses_and_selects_the_matching_target() {
    let document = manifest("sm_89", "abc");
    let parsed = Manifest::parse(document.as_bytes()).expect("manifest");
    let entries = parsed
        .select(Target {
            arch: Arch::Sm89,
            profile: Profile::Aot,
        })
        .expect("matching target");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        parsed
            .entry("projection.dense_bf16")
            .map(|entry| entry.name.as_str()),
        Some("projection.dense_bf16")
    );
    assert!(
        parsed
            .select(Target {
                arch: Arch::Sm90,
                profile: Profile::Aot
            })
            .is_err()
    );
    assert!(
        parsed
            .select(Target {
                arch: Arch::Sm89,
                profile: Profile::Tile
            })
            .is_err()
    );
}

#[test]
fn rejects_an_unknown_schema_or_profile() {
    let document = manifest("sm_89", "abc");
    assert!(Manifest::parse(document.replace("\"schema\":1", "\"schema\":2").as_bytes()).is_err());
    assert!(
        Manifest::parse(
            document
                .replace("\"profile\":\"aot\"", "\"profile\":\"tile\"")
                .as_bytes()
        )
        .is_err()
    );
}

#[test]
fn verifies_and_rejects_cubin_digests() {
    let digest = hex(&Sha256::digest(CUBIN));
    let directory = env::temp_dir().join(format!("infer-aot-{}", process::id()));
    let cubin_dir = directory.join("sm89");
    fs::create_dir_all(&cubin_dir).expect("cubin dir");
    fs::write(cubin_dir.join("projection.cubin"), CUBIN).expect("cubin");
    let parsed = Manifest::parse(manifest("sm_89", &digest).as_bytes()).expect("manifest");
    parsed.verify(&directory).expect("verified");
    fs::write(cubin_dir.join("projection.cubin"), b"corrupt").expect("corrupt");
    assert!(parsed.verify(&directory).is_err());
    fs::remove_dir_all(&directory).expect("cleanup");
}
