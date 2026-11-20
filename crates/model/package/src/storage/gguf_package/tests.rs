use super::sibling;
use std::env::temp_dir;
use std::fs::{create_dir_all, remove_dir_all, write};
use std::os::unix::fs::symlink;
use std::process::id;

#[test]
fn the_mmproj_is_found_beside_a_linked_model_not_its_target() {
    let root = temp_dir().join(format!("infer-gguf-sibling-{}", id()));
    let _ = remove_dir_all(&root);
    let (store, deployment) = (root.join("store"), root.join("deployment"));
    create_dir_all(&store).expect("store");
    create_dir_all(&deployment).expect("deployment");
    write(store.join("model.gguf"), b"").expect("model");
    let linked = deployment.join("model.gguf");
    symlink(store.join("model.gguf"), &linked).expect("link");
    write(deployment.join("mmproj-model-f16.gguf"), b"").expect("mmproj");
    assert_eq!(
        sibling(&linked),
        Some(deployment.join("mmproj-model-f16.gguf"))
    );
    assert_eq!(sibling(&store.join("model.gguf")), None);
    write(deployment.join("mmproj-other.gguf"), b"").expect("second mmproj");
    assert_eq!(
        sibling(&linked),
        None,
        "an ambiguous sibling attaches nothing"
    );
    remove_dir_all(root).expect("cleanup");
}
