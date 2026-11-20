use super::*;
use std::thread::spawn;

/// Arbitrary prefix token count sent through a pooled reply cell.
const PREFIX_TOKEN_COUNT: usize = 8;
/// Arbitrary reservation size in bytes sent through a pooled reply cell.
const RESERVATION_BYTES: u64 = 24;

#[test]
fn fixed_reply_credit_survives_ticket_abandonment_until_responder_finishes() -> Result<()> {
    let pool = ResourcePool::new(1)?;
    let (ticket, reply) = pool.channel()?;
    drop(ticket);
    assert!(reply.abandoned());
    assert!(pool.channel().is_err());
    assert!(reply.send(Ok(ResourceReply::Reserved)).is_err());
    let (mut ticket, reply) = pool.channel()?;
    reply
        .send(Ok(ResourceReply::Reset))
        .map_err(|_| Error::invariant("pooled receiver disappeared"))?;
    assert!(matches!(ticket.poll()?, Some(ResourceReply::Reset)));
    assert!(ticket.poll().is_err());
    assert!(pool.channel().is_err());
    drop(ticket);
    assert!(pool.channel().is_ok());
    Ok(())
}
#[test]
fn disconnected_or_unconsumed_acknowledgements_recycle_exactly_once() -> Result<()> {
    let pool = ResourcePool::new(1)?;
    for _ in 0..1000 {
        let (mut ticket, reply) = pool.channel()?;
        drop(reply);
        assert_eq!(
            ticket.poll().err().map(|error| error.code),
            Some(ErrorCode::Backend)
        );
        drop(ticket);
        let (ticket, reply) = pool.channel()?;
        reply
            .send(Ok(ResourceReply::Prefix(PREFIX_TOKEN_COUNT)))
            .map_err(|_| Error::invariant("pooled receiver disappeared"))?;
        drop(ticket);
    }
    assert!(pool.channel().is_ok());
    Ok(())
}
#[test]
fn acknowledgement_and_cancel_races_never_reuse_a_live_cell() -> Result<()> {
    let pool = ResourcePool::new(1)?;
    for _ in 0..100 {
        let (ticket, reply) = pool.channel()?;
        let owner = spawn(move || {
            let _ = reply.send(Ok(ResourceReply::Reserved));
        });
        drop(ticket);
        owner
            .join()
            .map_err(|_| Error::invariant("resource publisher panicked"))?;
        let (mut ticket, reply) = pool.channel()?;
        assert!(ticket.poll()?.is_none());
        reply
            .send(Ok(ResourceReply::ReservationBytes(Some(RESERVATION_BYTES))))
            .map_err(|_| Error::invariant("pooled receiver disappeared"))?;
        assert!(matches!(
            ticket.poll()?,
            Some(ResourceReply::ReservationBytes(Some(RESERVATION_BYTES)))
        ));
    }
    Ok(())
}
