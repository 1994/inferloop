use super::*;
use image::GenericImageView;
use std::io::Cursor;

fn frame(marker: u8, width: u16, height: u16) -> Vec<u8> {
    let [h1, h2] = height.to_be_bytes();
    let [w1, w2] = width.to_be_bytes();
    vec![
        0xff, 0xd8, // start of image
        0xff, 0xe1, 0, 4, 0x11, 0x22, // length-prefixed metadata
        0xff, marker, 0, 11, 8, h1, h2, w1, w2, 1, 1, 0x11, 0,
    ]
}

#[test]
fn frame_candidates_cover_jpeg_frame_types_and_marker_fill() {
    for marker in SOF_MARKERS {
        let bytes = frame(marker, 321, 123);
        assert_eq!(frame_dimensions(&bytes, None).unwrap(), (321, 123));
        let mut filled = bytes.clone();
        filled.splice(2..2, [0xff; 7]);
        assert_eq!(frame_dimensions(&filled, None).unwrap(), (321, 123));
    }
}

#[test]
fn candidate_header_does_not_claim_the_image_decodes() {
    let bytes = frame(0xc0, 321, 123);
    assert_eq!(frame_dimensions(&bytes, None).unwrap(), (321, 123));
    assert!(image::load_from_memory(&bytes).is_err());
}

#[test]
fn progressive_fixture_matches_independent_image_decode() {
    let bytes = include_bytes!("../fixtures/progressive-16x16.jpg");
    let image = image::load_from_memory(bytes).unwrap();
    assert_eq!(frame_dimensions(bytes, None).unwrap(), image.dimensions());
    for end in 0..bytes.len() {
        if let Ok(dimensions) = frame_dimensions(&bytes[..end], None) {
            assert_eq!(dimensions, image.dimensions());
        }
    }
}

#[test]
fn rgb_and_gray_jpeg_headers_match_full_decodes() {
    for (width, height) in [(1, 1), (64, 32), (127, 193)] {
        let rgb = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([
                x.to_le_bytes()[0],
                y.to_le_bytes()[0],
                (x ^ y).to_le_bytes()[0],
            ])
        });
        let gray = image::GrayImage::from_fn(width, height, |x, y| {
            image::Luma([(x ^ y).to_le_bytes()[0]])
        });
        for source in [
            image::DynamicImage::ImageRgb8(rgb),
            image::DynamicImage::ImageLuma8(gray),
        ] {
            let mut encoded = Cursor::new(Vec::new());
            source
                .write_to(&mut encoded, image::ImageFormat::Jpeg)
                .unwrap();
            let bytes = encoded.into_inner();
            let decoded = image::load_from_memory(&bytes).unwrap();
            assert_eq!(
                frame_dimensions(&bytes, None).unwrap(),
                decoded.dimensions()
            );
            assert_eq!(decoded.dimensions(), (width, height));
        }
    }
}

#[test]
fn malformed_segments_and_dimensions_are_rejected() {
    let valid = frame(0xc0, 321, 123);
    for end in 0..valid.len() {
        assert!(frame_dimensions(&valid[..end], None).is_err());
    }
    for bytes in [
        vec![0xff, 0xd8, 0xff, 0xe1, 0, 1],
        vec![0xff, 0xd8, 0xff, 0xe1, 0, 5, 1, 2],
        vec![0xff, 0xd8, 0xff, 0x00],
        vec![0xff, 0xd8, 0xff, 0xd9],
        vec![0xff, 0xd8, 0xff, 0xda, 0, 2],
        frame(0xc0, 0, 123),
        frame(0xc0, 321, 0),
    ] {
        assert!(frame_dimensions(&bytes, None).is_err());
    }
    let mut zero_components = valid;
    zero_components[17] = 0;
    assert!(frame_dimensions(&zero_components, None).is_err());
}

#[test]
fn long_marker_fill_is_bounded_and_cancellable() {
    let mut bytes = vec![0xff, 0xd8];
    bytes.extend(std::iter::repeat_n(0xff, FILL_SCAN_CHUNK_BYTES * 2));
    bytes.extend_from_slice(&frame(0xc2, 64, 32)[3..]);
    assert_eq!(frame_dimensions(&bytes, None).unwrap(), (64, 32));
    let context = crate::cpu::cancelled_context();
    assert_eq!(
        frame_dimensions(&bytes, Some(&context)).unwrap_err().code,
        infer_core::ErrorCode::Capacity
    );
}

#[test]
fn arbitrary_bounded_inputs_never_produce_empty_dimensions() {
    let mut state = 1_u64;
    for length in 0..512 {
        let mut bytes = vec![0xff, 0xd8, 0xff];
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            bytes.push(state.to_le_bytes()[4]);
        }
        if let Ok((width, height)) = frame_dimensions(&bytes, None) {
            assert!(width > 0 && height > 0);
        }
    }
}
