use super::*;
use std::sync::mpsc::channel;
#[test]
fn cpulists_are_bounded_sorted_and_reject_malformed_ranges() -> Result<()> {
    assert_eq!(parse_cpu_list("3,0-2,2,7")?, [0, 1, 2, 3, 7]);
    for input in ["3-1", "0-1048576", "x", "1-2-3", "+1", "1,,2", "1,", ",1"] {
        assert!(parse_cpu_list(input).is_err());
    }
    Ok(())
}
#[test]
fn unconfigured_scope_and_owner_complete_without_changing_affinity() -> Result<()> {
    let policy = ThreadPlacement::default();
    let before = policy.report()?;
    assert_eq!(policy.scope(|| policy.report())?, before);
    let (tx, rx) = channel();
    let thread = policy.spawn("infer-placement-test".into(), move |report| {
        let _ = tx.send(report);
    })?;
    assert_eq!(
        rx.recv()
            .map_err(|error| Error::invalid(error.to_string()))?,
        before
    );
    thread
        .join()
        .map_err(|_| Error::invariant("owner failed"))?;
    Ok(())
}
#[cfg(target_os = "macos")]
#[test]
fn macos_owner_applies_qos_and_rejects_hard_binding_before_startup() -> Result<()> {
    let policy = ThreadPlacement {
        qos: ThreadQos::UserInitiated,
        ..ThreadPlacement::default()
    };
    let (tx, rx) = channel();
    let thread = policy.spawn("infer-qos-test".into(), move |report| {
        let _ = tx.send(report);
    })?;
    assert_eq!(
        rx.recv()
            .map_err(|error| Error::invalid(error.to_string()))?
            .qos,
        ThreadQos::UserInitiated
    );
    thread
        .join()
        .map_err(|_| Error::invariant("QoS owner failed"))?;
    let invalid = ThreadPlacement {
        cpus: vec![0],
        ..ThreadPlacement::default()
    };
    assert!(invalid.spawn("infer-invalid-test".into(), |_| {}).is_err());
    Ok(())
}
#[cfg(target_os = "linux")]
#[test]
fn linux_scope_restores_affinity_and_owner_observes_single_cpu() -> Result<()> {
    let before = CpuTopology::discover()?;
    let cpu = *before
        .allowed_cpus
        .first()
        .ok_or_else(|| Error::invalid("no CPU allowed"))?;
    let policy = ThreadPlacement {
        cpus: vec![cpu],
        ..ThreadPlacement::default()
    };
    policy.scope(|| {
        assert_eq!(CpuTopology::discover()?.allowed_cpus, [cpu]);
        Ok(())
    })?;
    assert_eq!(CpuTopology::discover()?, before);
    let thread = policy.spawn("infer-affinity-test".into(), move |report| {
        assert_eq!(report.allowed_cpus, [cpu]);
    })?;
    thread
        .join()
        .map_err(|_| Error::invariant("affinity owner failed"))?;
    Ok(())
}
#[cfg(target_os = "linux")]
#[test]
fn linux_rejects_outside_inherited_cpuset_without_running_owner() -> Result<()> {
    let before = CpuTopology::discover()?;
    let cpu = before
        .allowed_cpus
        .last()
        .ok_or_else(|| Error::invalid("no allowed CPU"))?
        + 1;
    let policy = ThreadPlacement {
        cpus: vec![cpu],
        ..ThreadPlacement::default()
    };
    assert!(
        policy
            .spawn("infer-invalid-cpuset".into(), |_| panic!(
                "invalid owner ran"
            ))
            .is_err()
    );
    assert_eq!(CpuTopology::discover()?, before);
    Ok(())
}
#[cfg(target_os = "linux")]
#[test]
fn linux_scope_failure_restores_affinity() -> Result<()> {
    let before = CpuTopology::discover()?;
    let cpu = *before
        .allowed_cpus
        .first()
        .ok_or_else(|| Error::invalid("no allowed CPU"))?;
    let policy = ThreadPlacement {
        cpus: vec![cpu],
        ..ThreadPlacement::default()
    };
    assert!(
        policy
            .scope::<()>(|| Err(Error::invalid("initializer failed")))
            .is_err()
    );
    assert_eq!(CpuTopology::discover()?, before);
    Ok(())
}
