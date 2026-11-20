use super::*;
use crate::resource::ResourcePhase;
fn pending() -> PendingResource {
    PendingResource {
        ticket: None,
        phase: ResourcePhase::Reset,
        retry_epoch: None,
        started: 0,
    }
}
#[test]
fn polling_is_fair_across_removal_reinsertion_and_empty_slots() -> Result<()> {
    let mut waiters = ResourceWaiters::new(4)?;
    for id in 1..=4 {
        waiters.insert(RequestId::new(id)?, pending())?;
    }
    let mut ids = Vec::with_capacity(4);
    waiters.poll_window(2, &mut ids)?;
    assert_eq!(ids, [RequestId::ONE, RequestId::new(2)?]);
    waiters.remove(RequestId::new(3)?);
    waiters.insert(RequestId::new(5)?, pending())?;
    waiters.poll_window(2, &mut ids)?;
    assert_eq!(ids, [RequestId::new(4)?, RequestId::new(5)?]);
    waiters.poll_window(4, &mut ids)?;
    assert_eq!(
        ids,
        [
            RequestId::ONE,
            RequestId::new(2)?,
            RequestId::new(4)?,
            RequestId::new(5)?
        ]
    );
    assert!(waiters.insert(RequestId::new(5)?, pending()).is_err());
    for id in ids {
        waiters.remove(id);
    }
    assert!(waiters.is_empty());
    waiters.insert(RequestId::new(6)?, pending())?;
    let mut ids = Vec::with_capacity(4);
    waiters.poll_window(4, &mut ids)?;
    assert_eq!(ids, [RequestId::new(6)?]);
    Ok(())
}
