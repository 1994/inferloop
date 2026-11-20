use super::*;
use infer_core::ModelId;
use std::env::var;
use std::fs::read;
use std::path::Path;

const IMAGE: u32 = 248_056;

fn shipped() -> Result<ImportedModel> {
    let root = var("INFER_VISION_PACKAGE").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../examples/qwen3.8-27b").into()
    });
    let config = read(Path::new(&root).join("config.json"))
        .map_err(|error| Error::invalid(error.to_string()))?;
    let provider = infer_models::default_registry().resolve(&config)?;
    provider.import(ModelId::ONE, &config)
}

#[test]
fn placeholders_without_encodings_are_rejected() -> Result<()> {
    let imported = shipped()?;
    let tokens = [7u32, IMAGE, IMAGE, 9];
    assert_eq!(media_positions(&imported, &tokens), vec![1, 2]);
    let Err(error) = require_placements(&imported, &tokens, &[]) else {
        panic!("media placeholders without encodings must fail");
    };
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    assert!(error.message.contains("without encodings"));
    // A text-only prompt stays valid.
    require_placements(&imported, &[1, 2, 3], &[])?;
    // Encodings for a prompt without placeholders are equally wrong.
    let stray = [VisualPlacement {
        rotary_position: None,
        position: 0,
        embedding: vec![0.0; 4],
    }];
    let Err(error) = require_placements(&imported, &[1, 2, 3], &stray) else {
        panic!("encodings without placeholders must fail");
    };
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    Ok(())
}

#[test]
fn misaligned_placements_are_rejected() -> Result<()> {
    let imported = shipped()?;
    let tokens = [7u32, IMAGE, IMAGE, 9];
    let wrong_position = [
        VisualPlacement {
            rotary_position: None,
            position: 0,
            embedding: vec![0.0; 4],
        },
        VisualPlacement {
            rotary_position: None,
            position: 1,
            embedding: vec![0.0; 4],
        },
    ];
    assert!(require_placements(&imported, &tokens, &wrong_position).is_err());
    let duplicate = [
        VisualPlacement {
            rotary_position: None,
            position: 1,
            embedding: vec![0.0; 4],
        },
        VisualPlacement {
            rotary_position: None,
            position: 1,
            embedding: vec![0.0; 4],
        },
    ];
    assert!(require_placements(&imported, &tokens, &duplicate).is_err());
    let empty = [
        VisualPlacement {
            rotary_position: None,
            position: 1,
            embedding: Vec::new(),
        },
        VisualPlacement {
            rotary_position: None,
            position: 2,
            embedding: vec![0.0; 4],
        },
    ];
    assert!(require_placements(&imported, &tokens, &empty).is_err());
    let aligned = [
        VisualPlacement {
            rotary_position: None,
            position: 1,
            embedding: vec![0.0; 4],
        },
        VisualPlacement {
            rotary_position: None,
            position: 2,
            embedding: vec![0.0; 4],
        },
    ];
    require_placements(&imported, &tokens, &aligned)?;
    Ok(())
}

#[test]
fn greedy_argmax_is_strict_and_finite() -> Result<()> {
    assert_eq!(argmax(&[0.5, 2.0, 1.0])?, 1);
    assert_eq!(argmax(&[-3.0, -2.0])?, 1);
    assert!(argmax(&[]).is_err());
    assert!(argmax(&[f32::NAN, 1.0]).is_err());
    Ok(())
}
