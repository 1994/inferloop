use super::*;
#[test]
fn rejection_corrects_draft_bias_and_greedy_mismatch() -> Result<()> {
    let p = [0.2, 0.8];
    let q = [0.8, 0.2];
    assert_eq!(
        verify_draft(&p, &q, 0, 0.1, 0.3)?,
        Verification::Accepted(0)
    );
    assert_eq!(
        verify_draft(&p, &q, 0, 0.9, 0.3)?,
        Verification::Replaced(1)
    );
    assert_eq!(
        verify_draft(&[0.0, 1.0], &[1.0, 0.0], 0, 0.0, 0.5)?,
        Verification::Replaced(1)
    );
    let mut counts = [0u32; 2];
    for i in 0..40_000 {
        let proposal = draw_distribution(&q, crate::sampling_uniform(13, 2, i))?;
        let result = verify_draft(
            &p,
            &q,
            proposal,
            crate::sampling_uniform(13, 3, i),
            crate::sampling_uniform(13, 4, i),
        )?;
        let token = match result {
            Verification::Accepted(t) | Verification::Replaced(t) => t,
        };
        counts[token as usize] += 1;
    }
    assert!((f64::from(counts[0]) / 40_000.0 - p[0]).abs() < 0.01);
    Ok(())
}

#[test]
fn a_draft_budget_never_outruns_depth_lanes_or_the_sequence() {
    assert_eq!(draft_budget(4, 8, 10, 100), 4);
    assert_eq!(draft_budget(4, 3, 10, 100), 2);
    assert_eq!(draft_budget(4, 8, 97, 100), 2);
    assert_eq!(draft_budget(4, 8, 99, 100), 0);
    assert_eq!(draft_budget(4, 8, 200, 100), 0);
    assert_eq!(draft_budget(4, 0, 10, 100), 0);
}

#[test]
fn drafts_only_propose_under_greedy() {
    assert!(draft_may_propose(0.0));
    assert!(!draft_may_propose(0.7));
}

#[test]
fn the_accepted_prefix_stops_at_divergence_or_an_end_token() -> Result<()> {
    let target = [5, 6, 7, 8];
    let mut generated = vec![1];
    let accepted = accepted_prefix(
        &[5, 6, 9, 8],
        &mut generated,
        |offset, _| Ok(target[offset]),
        |_| false,
    )?;
    assert_eq!(accepted, [5, 6]);
    assert_eq!(generated, [1, 5, 6]);
    let mut generated = Vec::new();
    let accepted = accepted_prefix(
        &[5, 6, 7],
        &mut generated,
        |offset, _| Ok(target[offset]),
        |token| token == 6,
    )?;
    assert_eq!(accepted, [5, 6]);
    Ok(())
}

#[test]
fn selection_sees_the_tokens_accepted_so_far() -> Result<()> {
    let mut seen = Vec::new();
    let mut generated = vec![1];
    accepted_prefix(
        &[2, 3],
        &mut generated,
        |offset, history| {
            seen.push(history.len());
            Ok([2, 3][offset])
        },
        |_| false,
    )?;
    assert_eq!(seen, [1, 2]);
    Ok(())
}
