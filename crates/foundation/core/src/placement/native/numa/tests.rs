use super::*;
use crate::placement::CpuTopology;
use crate::placement::ThreadPlacement;
use std::sync::mpsc::channel;
fn identity(policy: &Policy) -> (i32, Vec<usize>) {
    (
        policy.mode,
        (0..policy.mask.bits())
            .filter(|node| policy.mask.contains(*node))
            .collect(),
    )
}
#[test]
#[ignore = "requires a Linux production host with get/set_mempolicy permitted; make check-linux-numa"]
fn binding_preference_scope_failure_and_worker_restore() -> Result<()> {
    let node = *CpuTopology::discover()?
        .allowed_nodes
        .first()
        .ok_or_else(|| Error::invalid("no allowed NUMA node"))?;
    let before = identity(&Policy::current()?);
    for (mode, numa) in [(2, NumaPolicy::Bind(node)), (1, NumaPolicy::Prefer(node))] {
        let placement = ThreadPlacement {
            numa: Some(numa),
            ..ThreadPlacement::default()
        };
        placement.scope(|| {
            assert_eq!(identity(&Policy::current()?), (mode, vec![node]));
            Ok(())
        })?;
        assert_eq!(identity(&Policy::current()?), before);
        assert!(
            placement
                .scope::<()>(|| Err(Error::invalid("initializer failed")))
                .is_err()
        );
        assert_eq!(identity(&Policy::current()?), before);
        let (tx, rx) = channel();
        let thread = placement.spawn("infer-numa-test".into(), move |report| {
            let _ = tx.send((
                report.numa,
                Policy::current().map(|policy| identity(&policy)),
            ));
        })?;
        let (reported, actual) = rx
            .recv()
            .map_err(|error| Error::invalid(error.to_string()))?;
        assert_eq!(reported, Some(numa));
        assert_eq!(actual?, (mode, vec![node]));
        thread
            .join()
            .map_err(|_| Error::invariant("NUMA owner failed"))?;
    }
    assert_eq!(identity(&Policy::current()?), before);
    Ok(())
}
