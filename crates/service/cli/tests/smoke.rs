use std::path::Path;
use std::process::Command;

#[test]
fn cuda_remains_primary_and_explicit_unavailable_backend_does_not_fallback() {
    let binary = env!("CARGO_BIN_EXE_infer");
    let doctor = Command::new(binary).arg("doctor").output().unwrap();
    assert!(doctor.status.success());
    let report: serde_json::Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(report["backends"]["primary_target"], "cuda");
    assert_eq!(
        report["backends"]["auto_priority"],
        serde_json::json!(["cuda", "metal"])
    );
    assert_eq!(
        report["backends"]["supported"],
        serde_json::json!(["cuda", "metal"])
    );
    assert!(report["backends"].get("host").is_none());
    // Check the built binary against target/feature facts, independently of its cfg aliases.
    assert_eq!(
        report["backends"]["cuda"]["implemented"],
        cfg!(all(target_os = "linux", feature = "cuda"))
    );
    assert_eq!(
        report["backends"]["metal"]["implemented"],
        cfg!(target_os = "macos")
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let output = Command::new(binary)
        // Make the unavailable-device case deterministic even on GPU hosts.
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .args(["--backend", "cuda", "run", "--package"])
        .arg(root.join("qwen-hybrid-tiny"))
        .arg("--requests")
        .arg(root.join("requests.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    let expected_code = if cfg!(all(target_os = "linux", feature = "cuda")) {
        "Backend"
    } else {
        "Unsupported"
    };
    assert_eq!(error["code"], expected_code, "{error}");
    if cfg!(target_os = "macos") {
        assert_eq!(
            error["message"],
            "CUDA execution requires Linux and a build with --features cuda"
        );
    }
    assert!(
        output.stdout.is_empty(),
        "failed backend must not emit results"
    );
    for backend in ["metal", "cuda"] {
        let output = Command::new(binary)
            .args(["--backend", backend, "verify"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires --package"));
    }
}

#[test]
fn production_binary_rejects_cpu_choices_and_implicit_fixture_execution() {
    let binary = env!("CARGO_BIN_EXE_infer");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    for backend in ["host", "cpu", "test-cpu"] {
        let output = Command::new(binary)
            .args(["--backend", backend, "run", "--package"])
            .arg(root.join("qwen-hybrid-tiny"))
            .arg("--requests")
            .arg(root.join("requests.json"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
    }
    let output = Command::new(binary)
        .args(["run", "--requests"])
        .arg(root.join("requests.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --package"));
    let output = Command::new(binary).arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(!help.contains("test-cpu") && !help.contains("host-memory-mib"));
}
