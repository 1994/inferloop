use super::*;
#[test]
fn forked_prefix_uses_copy_on_write_and_preserves_parent() {
    let mut parent = PagedRows::new(2, 2, 8).unwrap();
    parent.push(&[1.0, 2.0]).unwrap();
    let mut child = parent.clone();
    assert_eq!(parent.shared_pages(), 1);
    child.push(&[3.0, 4.0]).unwrap();
    assert_eq!(parent.rows, 1);
    assert_eq!(child.row(1).unwrap(), &[3.0, 4.0]);
    assert_eq!(parent.row(0).unwrap(), &[1.0, 2.0]);
    assert_eq!(parent.shared_pages(), 0);
    child.validate().unwrap();
}
