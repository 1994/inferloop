use super::{STORE_EXTENSION, append, prune, read};
use std::env::temp_dir;
use std::fs::{create_dir_all, read_dir, remove_dir_all, write};
use std::path::PathBuf;
use std::process::id;

fn scratch(name: &str) -> PathBuf {
    let directory = temp_dir().join(format!("infer-identities-{name}-{}", id()));
    let _ = remove_dir_all(&directory);
    create_dir_all(&directory).expect("scratch directory");
    directory
}

fn digest(fill: char) -> String {
    fill.to_string().repeat(64)
}

#[test]
fn appended_identities_read_back() {
    let directory = scratch("round-trip");
    let path = directory.join(format!("store.{STORE_EXTENSION}"));
    append(&path, &digest('a'), &digest('b')).expect("append");
    append(&path, &digest('c'), &digest('d')).expect("append");
    let known = read(&path);
    assert_eq!(known.len(), 2);
    assert_eq!(known.get(&digest('a')), Some(&digest('b')));
    assert_eq!(known.get(&digest('c')), Some(&digest('d')));
    remove_dir_all(directory).expect("cleanup");
}

#[test]
fn torn_and_foreign_lines_are_ignored() {
    let directory = scratch("torn");
    let path = directory.join(format!("store.{STORE_EXTENSION}"));
    let valid = format!("{} {}\n", digest('1'), digest('2'));
    let torn = &format!("{} {}", digest('3'), digest('4'))[..100];
    let foreign = format!("{} not-a-digest\n", digest('5'));
    write(&path, format!("{valid}{foreign}{torn}")).expect("write");
    let known = read(&path);
    assert_eq!(known.len(), 1);
    assert_eq!(known.get(&digest('1')), Some(&digest('2')));
    assert!(read(&directory.join("absent")).is_empty());
    remove_dir_all(directory).expect("cleanup");
}

#[test]
fn pruning_keeps_the_current_store_and_unrelated_files() {
    let directory = scratch("prune");
    let current = directory.join(format!("current.{STORE_EXTENSION}"));
    write(&current, "").expect("current");
    write(directory.join("unrelated.txt"), "").expect("unrelated");
    for index in 0..super::RETAINED_STORES + 4 {
        write(directory.join(format!("{index}.{STORE_EXTENSION}")), "").expect("store");
    }
    prune(&directory, &current);
    let stores = read_dir(&directory)
        .expect("listing")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == STORE_EXTENSION))
        .collect::<Vec<_>>();
    assert_eq!(stores.len(), super::RETAINED_STORES);
    assert!(current.exists());
    assert!(directory.join("unrelated.txt").exists());
    remove_dir_all(directory).expect("cleanup");
}
