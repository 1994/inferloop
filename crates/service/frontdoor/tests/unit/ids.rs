use super::*;
use uuid::Uuid;

fn uuid_part(id: &str, prefix: &str) -> Uuid {
    let raw = id.strip_prefix(prefix).expect("the id keeps its prefix");
    Uuid::parse_str(raw).expect("the body is a version 7 UUID")
}

#[test]
fn every_call_gets_its_own_time_ordered_uuid() {
    let first = tool_call();
    let second = tool_call();
    assert_ne!(first, second);
    assert_eq!(uuid_part(&first, "call_").get_version_num(), 7);
    assert!(first < second, "version 7 ids sort by creation time");
}

#[test]
fn every_response_gets_its_own_uuid() {
    let first = response();
    assert_ne!(first, response());
    assert_eq!(uuid_part(&first, "resp_").get_version_num(), 7);
}
