use super::*;

#[test]
fn greedy_preserves_ties_signed_zero_and_rejects_nonfinite_tail() -> Result<()> {
    let sampling = Sampling::default();
    for (logits, expected) in [
        (vec![2.0, 2.0, 1.0], 0),
        (vec![-0.0, 0.0, 0.0], 1),
        (vec![f32::MIN, -5.0, -6.0], 1),
        (vec![f32::MAX, f32::MAX], 0),
    ] {
        assert_eq!(sample(&logits, &sampling, 1, 0)?, expected);
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(sample(&[f32::MAX, invalid], &sampling, 1, 0).is_err());
    }
    assert!(sample(&[], &sampling, 1, 0).is_err());
    Ok(())
}
fn reference(logits: &[f32], sampling: &Sampling, request: u64, position: usize) -> Result<u32> {
    let mut candidates: Vec<_> = logits.iter().copied().enumerate().collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if let Some(k) = sampling.top_k {
        candidates.truncate(k);
    }
    let probabilities = crate::softmax(
        &candidates
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>(),
        sampling.temperature,
    )?;
    let mut cumulative = 0.0;
    let uniform = random(sampling.seed, request, position);
    for ((index, _), probability) in candidates.iter().zip(probabilities) {
        cumulative += f64::from(probability);
        if uniform < cumulative {
            return token(*index);
        }
    }
    token(
        candidates
            .last()
            .ok_or_else(|| Error::invariant("empty reference"))?
            .0,
    )
}
#[test]
fn partial_top_k_matches_full_sort_distribution_and_seeded_trajectory() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    for seed in 0..128_u64 {
        let logits: Vec<f32> = (0..257_u16)
            .map(|index| {
                f32::from(
                    (u64::from(index) * 17 + seed)
                        .rem_euclid(53)
                        .try_into()
                        .unwrap_or(0_u16),
                )
            })
            .collect();
        for top_k in [None, Some(1), Some(7), Some(256), Some(1024)] {
            let sampling = Sampling {
                seed,
                top_k,
                temperature: 0.7,
                ..Sampling::default()
            };
            for position in 0..8 {
                assert_eq!(
                    sample_reusing(&logits, &sampling, seed + 1, position, &mut scratch)?,
                    reference(&logits, &sampling, seed + 1, position)?
                );
            }
        }
    }
    Ok(())
}
#[test]
fn repeated_sampling_preserves_allocations_and_greedy_ties_use_lowest_token() -> Result<()> {
    let logits = vec![1.0; 1024];
    let mut scratch = SamplingWorkspace::default();
    let mut sampling = Sampling {
        temperature: 1.0,
        ..Sampling::default()
    };
    sample_reusing(&logits, &sampling, 1, 0, &mut scratch)?;
    let pointers = (scratch.candidates.as_ptr(), scratch.weights.as_ptr());
    for position in 1..128 {
        sampling.top_k = Some(7);
        sample_reusing(&logits, &sampling, 1, position, &mut scratch)?;
        assert_eq!(
            (scratch.candidates.as_ptr(), scratch.weights.as_ptr()),
            pointers
        );
    }
    assert_eq!(sample(&logits, &Sampling::default(), 1, 0)?, 0);
    assert!(sample(&[f32::NAN], &sampling, 1, 0).is_err());
    Ok(())
}

#[test]
fn nucleus_minimum_probability_and_penalties_change_candidates() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    let mut parameters = Sampling {
        temperature: 1.0,
        top_p: 0.5,
        ..Sampling::default()
    };
    for seed in 0..20 {
        parameters.seed = seed;
        assert_eq!(sample(&[3.0, 1.0, 0.0], &parameters, 1, 0)?, 0);
    }
    parameters.top_p = 1.0;
    parameters.min_p = 0.9;
    assert_eq!(sample(&[3.0, 1.0, 0.0], &parameters, 1, 0)?, 0);
    parameters.temperature = 0.0;
    parameters.presence_penalty = 2.0;
    let history = SamplingHistory {
        prompt: &[1],
        generated: &[0, 0],
    };
    assert_eq!(
        sample_with_history(&[3.0, 2.0], &parameters, 1, 0, history, &mut scratch)?,
        1
    );
    parameters.presence_penalty = 0.0;
    parameters.repetition_penalty = 2.0;
    let history = SamplingHistory {
        prompt: &[0],
        generated: &[],
    };
    assert_eq!(
        sample_with_history(&[3.0, 2.0], &parameters, 1, 0, history, &mut scratch)?,
        1
    );
    Ok(())
}

#[test]
fn token_masks_validate_shape_and_ignore_padding() -> Result<()> {
    assert!(TokenMask::new(&[], 0).is_err());
    assert!(TokenMask::new(&[], 1).is_err());
    assert!(TokenMask::new(&[0, 0], 32).is_err());
    assert!(TokenMask::new(&[0], 33).is_err());
    let mask = TokenMask::new(&[1, u32::MAX], 33)?;
    assert!(mask.allows(0));
    assert!(!mask.allows(1));
    assert!(!mask.allows(31));
    assert!(mask.allows(32));
    assert!(!mask.allows(33));
    assert!(!mask.allows(usize::MAX));
    Ok(())
}

#[test]
fn constrained_greedy_excludes_forbidden_maxima_and_eos() -> Result<()> {
    let sampling = Sampling {
        eos_tokens: vec![0],
        ..Sampling::default()
    };
    let mask = Some(TokenMask::new(&[0b1110], 4)?);
    let mut scratch = SamplingWorkspace::default();
    for (logits, expected) in [
        ([100.0, 2.0, 2.0, 1.0], 1),
        ([100.0, -0.0, 0.0, 0.0], 2),
        ([100.0, f32::MIN, -6.0, -5.0], 3),
    ] {
        assert_eq!(
            sample_constrained(
                &logits,
                &sampling,
                1,
                0,
                SamplingHistory::default(),
                mask,
                &mut scratch
            )?,
            expected
        );
    }
    assert_eq!(scratch.candidates.capacity(), 0);
    assert_eq!(scratch.weights.capacity(), 0);
    Ok(())
}

#[test]
fn masks_cannot_hide_invalid_logits_or_fall_back_on_empty_sets() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    for temperature in [0.0, 1.0] {
        let sampling = Sampling {
            temperature,
            ..Sampling::default()
        };
        for words in [0, 0b100] {
            // Padding alone is not a legal continuation.
            let storage = [words];
            let mask = Some(TokenMask::new(&storage, 2)?);
            assert!(
                sample_constrained(
                    &[100.0, 1.0],
                    &sampling,
                    1,
                    0,
                    SamplingHistory::default(),
                    mask,
                    &mut scratch
                )
                .is_err()
            );
        }
        let mask = Some(TokenMask::new(&[0b10], 2)?);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                sample_constrained(
                    &[invalid, 1.0],
                    &sampling,
                    1,
                    0,
                    SamplingHistory::default(),
                    mask,
                    &mut scratch
                )
                .is_err()
            );
        }
        let mask = Some(TokenMask::new(&[1], 1)?);
        assert!(
            sample_constrained(
                &[0.0, 1.0],
                &sampling,
                1,
                0,
                SamplingHistory::default(),
                mask,
                &mut scratch
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn constraints_precede_top_k_nucleus_and_minimum_probability() -> Result<()> {
    let logits = [100.0, 1.0, 2.0, 99.0];
    let mask = Some(TokenMask::new(&[0b0110], 4)?);
    let mut scratch = SamplingWorkspace::default();
    for (top_k, top_p, min_p) in [(Some(1), 1.0, 0.0), (None, 0.5, 0.0), (None, 1.0, 0.9)] {
        let sampling = Sampling {
            temperature: 1.0,
            top_k,
            top_p,
            min_p,
            ..Sampling::default()
        };
        let probabilities =
            probabilities_constrained(&logits, &sampling, SamplingHistory::default(), mask)?;
        assert_eq!(probabilities, vec![0.0, 0.0, 1.0, 0.0]);
        for position in 0..64 {
            assert_eq!(
                sample_constrained(
                    &logits,
                    &sampling,
                    1,
                    position,
                    SamplingHistory::default(),
                    mask,
                    &mut scratch
                )?,
                2
            );
        }
    }
    Ok(())
}

#[test]
fn constrained_penalties_preserve_original_token_indices() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    let sampling = Sampling {
        repetition_penalty: 2.0,
        presence_penalty: 2.0,
        ..Sampling::default()
    };
    let mask = Some(TokenMask::new(&[0b1010], 4)?);
    let history = SamplingHistory {
        prompt: &[0, 3],
        generated: &[3],
    };
    assert_eq!(
        sample_constrained(
            &[100.0, 3.0, 99.0, 8.0],
            &sampling,
            1,
            0,
            history,
            mask,
            &mut scratch
        )?,
        1
    );
    // A negative presence penalty must not revive a forbidden generated token.
    let sampling = Sampling {
        presence_penalty: -2.0,
        ..Sampling::default()
    };
    assert_eq!(
        sample_constrained(
            &[100.0, 3.0, 99.0, 2.0],
            &sampling,
            1,
            0,
            SamplingHistory {
                prompt: &[],
                generated: &[0, 2]
            },
            mask,
            &mut scratch
        )?,
        1
    );
    assert!(
        sample_constrained(
            &[100.0, 3.0, 99.0, 2.0],
            &sampling,
            1,
            0,
            SamplingHistory {
                prompt: &[4],
                generated: &[]
            },
            mask,
            &mut scratch
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn allow_all_masks_preserve_existing_distributions_and_seeded_trajectories() -> Result<()> {
    let logits = [3.0, -2.0, 1.0, 4.0];
    let history = SamplingHistory {
        prompt: &[0],
        generated: &[3],
    };
    let mut plain = SamplingWorkspace::default();
    let mut constrained = SamplingWorkspace::default();
    for temperature in [0.0, 0.7, 1.0] {
        for presence_penalty in [0.0, 2.0] {
            let sampling = Sampling {
                temperature,
                presence_penalty,
                top_k: Some(3),
                top_p: 0.9,
                min_p: 0.01,
                ..Sampling::default()
            };
            let mask = Some(TokenMask::new(&[u32::MAX], logits.len())?);
            assert_eq!(
                probabilities(&logits, &sampling, history)?,
                probabilities_constrained(&logits, &sampling, history, mask)?
            );
            for position in 0..128 {
                assert_eq!(
                    sample_with_history(&logits, &sampling, 19, position, history, &mut plain)?,
                    sample_constrained(
                        &logits,
                        &sampling,
                        19,
                        position,
                        history,
                        mask,
                        &mut constrained
                    )?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn constrained_distribution_renormalizes_and_matches_sample_frequencies() -> Result<()> {
    let logits = [100.0, 0.0, 99.0, 1.0];
    let sampling = Sampling {
        temperature: 1.0,
        ..Sampling::default()
    };
    let mask = Some(TokenMask::new(&[0b1010], 4)?);
    let distribution =
        probabilities_constrained(&logits, &sampling, SamplingHistory::default(), mask)?;
    let expected = 1.0 / (1.0 + 1.0_f64.exp());
    assert!((distribution[1] - expected).abs() < 1e-12);
    assert!((distribution.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mut counts = [0_u32; 4];
    let mut scratch = SamplingWorkspace::with_capacity(logits.len())?;
    for position in 0..20_000 {
        let token = sample_constrained(
            &logits,
            &sampling,
            13,
            position,
            SamplingHistory::default(),
            mask,
            &mut scratch,
        )?;
        counts[token as usize] += 1;
    }
    assert_eq!(counts[0], 0);
    assert_eq!(counts[2], 0);
    assert!((f64::from(counts[1]) / 20_000.0 - expected).abs() < 0.02);
    Ok(())
}

#[test]
fn changing_prefix_masks_do_not_leak_candidates_or_grow_warmed_storage() -> Result<()> {
    let logits = vec![1.0; 64];
    let sampling = Sampling {
        temperature: 1.0,
        repetition_penalty: 1.1,
        ..Sampling::default()
    };
    let history = SamplingHistory {
        prompt: &[0],
        generated: &[1, 33],
    };
    let mut scratch = SamplingWorkspace::with_capacity(logits.len())?;
    sample_with_history(&logits, &sampling, 1, 0, history, &mut scratch)?;
    let pointers = (
        scratch.candidates.as_ptr(),
        scratch.weights.as_ptr(),
        scratch.seen.as_ptr(),
    );
    for position in 0..128 {
        let words = if position % 2 == 0 {
            [0b10, 0]
        } else {
            [0, 0b10]
        };
        let mask = Some(TokenMask::new(&words, logits.len())?);
        let expected = if position % 2 == 0 { 1 } else { 33 };
        assert_eq!(
            sample_constrained(&logits, &sampling, 1, position, history, mask, &mut scratch)?,
            expected
        );
        assert_eq!(
            (
                scratch.candidates.as_ptr(),
                scratch.weights.as_ptr(),
                scratch.seen.as_ptr()
            ),
            pointers
        );
    }
    Ok(())
}
