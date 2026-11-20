use crate::resident::attention_mma::catalog::*;
#[test]
fn hardware_candidates_cannot_cross_instruction_families() {
    assert_eq!(Variant::candidates("sm_90"), &[Variant::Hopper]);
    assert_eq!(
        Variant::candidates("sm_100"),
        &[Variant::Sm100, Variant::Sm100Pair]
    );
    assert_eq!(
        Variant::candidates("sm_103"),
        &[Variant::Sm103, Variant::Sm103Pair]
    );
    assert_eq!(Variant::candidates("sm_120"), &[Variant::Sm120]);
    for unknown in ["sm_80", "sm_89", "sm_101", "sm_121", ""] {
        assert_eq!(Variant::candidates(unknown), []);
    }
    assert_eq!(Variant::Sm100Pair.cluster(), 2);
    assert_eq!(Variant::Sm103Pair.cluster(), 2);
    assert_eq!(Variant::Hopper.cluster(), 1);
}
#[test]
fn dynamic_width_has_geometry_and_abi_bounds_without_context_ceiling() {
    for lanes in [64, 128, 256, 512, 1024, 4096, 65536] {
        assert!(geometry(24, 4, 256, lanes, 0));
    }
    for (heads, kv, dim, lanes, window) in [
        (24, 4, 256, 0, 0),
        (24, 4, 256, 63, 0),
        (24, 4, 256, 256, 1),
        (32, 4, 256, 256, 0),
        (24, 8, 256, 256, 0),
        (24, 4, 128, 256, 0),
    ] {
        assert!(!geometry(heads, kv, dim, lanes, window));
    }
    assert!(!geometry(24, 4, 256, usize::MAX, 0));
}
#[test]
fn scratch_charge_is_one_pair_with_checked_overflow() -> Result<()> {
    let fixed = workspace_bytes(0, 256)?;
    let one = workspace_bytes(32768, 256)?;
    let two = workspace_bytes(65536, 256)?;
    assert_eq!(one - fixed, 32768 * 4 * 256 * 2 * 2);
    assert_eq!(two - one, one - fixed);
    assert_eq!(
        workspace_bytes(131_072, 128)?,
        131_072 * 4 * 256 * 4 + 128 * 24 * 256 * 4 + 8
    );
    assert!(workspace_bytes(usize::MAX, 256).is_err());
    assert!(workspace_bytes(32768, usize::MAX).is_err());
    Ok(())
}
