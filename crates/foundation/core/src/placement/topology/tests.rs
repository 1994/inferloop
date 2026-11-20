use super::*;
#[test]
fn pci_identity_rejects_bad_separators_and_out_of_range_slot_function() {
    assert!(validate_pci("0000:01:00.0").is_ok());
    for address in [
        "0000.01:00.0",
        "0000:01:20.0",
        "0000:01:00.8",
        "../01:00.0",
        "0000:01:é.0",
    ] {
        assert!(validate_pci(address).is_err());
    }
}
