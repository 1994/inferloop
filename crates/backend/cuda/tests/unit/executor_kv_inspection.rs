use super::logical_kv_blocks;
use crate::constants::KV_BLOCK_TOKENS;

#[test]
fn deferred_pool_supports_future_concurrent_requests_with_a_small_model_context() {
    // Before capture there are zero physical pool pages. Eight 4352-token requests
    // fit the funded future layout even though their sum exceeds the 32k model context.
    let logical = logical_kv_blocks(0, Some((16, 8192)), 32768);
    let request_pages = 4352_usize.div_ceil(KV_BLOCK_TOKENS);
    assert!(logical >= 8 * request_pages);
    // Once captured, the same field follows the actual pool or isolated private ceiling.
    assert_eq!(logical_kv_blocks(5000, None, 32768), 5000);
    assert_eq!(logical_kv_blocks(0, None, 32768), 1024);
}
