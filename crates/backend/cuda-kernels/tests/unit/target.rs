use crate::contract::target::Arch;

#[test]
fn maps_known_and_unknown_compute_capabilities() {
    assert_eq!(Arch::from_sm(8, 9), Arch::Sm89);
    assert_eq!(Arch::from_sm(9, 0), Arch::Sm90);
    assert_eq!(Arch::from_sm(10, 3), Arch::Sm103);
    assert_eq!(Arch::from_sm(12, 0), Arch::Sm120);
    assert_eq!(Arch::from_sm(9, 1), Arch::Other { major: 9, minor: 1 });
}

#[test]
fn names_architectures_for_manifests() {
    assert_eq!(Arch::Sm89.name(), "sm_89");
    assert_eq!(Arch::Sm100.name(), "sm_100");
    assert_eq!(Arch::Other { major: 9, minor: 1 }.name(), "sm_91");
}

#[test]
fn parses_driver_names_and_rejects_anything_else() {
    assert_eq!(Arch::from_sm_name("sm_89"), Some(Arch::Sm89));
    assert_eq!(Arch::from_sm_name("sm_120"), Some(Arch::Sm120));
    assert_eq!(
        Arch::from_sm_name("sm_121"),
        Some(Arch::Other {
            major: 12,
            minor: 1
        })
    );
    assert_eq!(Arch::from_sm_name("sm_"), None);
    assert_eq!(Arch::from_sm_name("sm_90a"), None);
    assert_eq!(Arch::from_sm_name("gpu_89"), None);
}
