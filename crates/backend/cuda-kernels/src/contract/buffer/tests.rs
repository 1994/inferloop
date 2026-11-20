use super::{disjoint, span};

#[test]
fn a_span_covers_whole_elements() {
    assert_eq!(span(0x1000, 3, 4), Some(0x1000..0x100c));
    assert_eq!(span(0x1000, 0, 4), Some(0x1000..0x1000));
    assert_eq!(span(u64::MAX - 1, 1, 4), None);
    assert_eq!(span(0, usize::MAX, 2), None);
}

#[test]
fn adjacent_spans_are_disjoint_and_overlapping_ones_are_not() {
    let whole = span(0x1000, 8, 4);
    assert!(disjoint(span(0x1000, 4, 4), span(0x1010, 4, 4)));
    assert!(disjoint(span(0x1010, 4, 4), span(0x1000, 4, 4)));
    assert!(!disjoint(whole.clone(), span(0x1004, 1, 4)));
    assert!(!disjoint(span(0x1004, 1, 4), whole));
    // Element widths differ: 16 one-byte codes end where four floats would.
    assert!(disjoint(span(0x1000, 16, 1), span(0x1010, 1, 4)));
    assert!(!disjoint(span(0x1000, 17, 1), span(0x1010, 1, 4)));
}

#[test]
fn a_span_past_the_address_space_is_never_disjoint() {
    assert!(!disjoint(span(u64::MAX - 1, 1, 4), span(0, 1, 4)));
}
