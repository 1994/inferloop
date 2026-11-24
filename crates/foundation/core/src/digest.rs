//! Text forms of content digests, as fingerprints and artifact identities record them.

/// Lowercase hexadecimal digits, indexed by nibble.
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
/// Bits of one hexadecimal digit.
const NIBBLE_BITS: u32 = 4;
/// The low nibble of a byte.
const NIBBLE_MASK: u8 = 0x0f;

/// `bytes` as lowercase hexadecimal, two digits per byte, most significant nibble first: the
/// form `{:x}` gave digests, which persisted fingerprints and cache keys keep.
#[must_use]
pub fn lower_hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(HEX_DIGITS[usize::from(byte >> NIBBLE_BITS)]));
        text.push(char::from(HEX_DIGITS[usize::from(byte & NIBBLE_MASK)]));
    }
    text
}

#[cfg(test)]
mod tests;
