use super::*;
use crate::constants::GIB_U64;

#[test]
fn sizes_stay_readable() {
    assert_eq!(human_bytes(0), "0 MiB");
    assert_eq!(human_bytes(GIB_U64 + GIB_U64 / 2), "1.5 GiB");
}
