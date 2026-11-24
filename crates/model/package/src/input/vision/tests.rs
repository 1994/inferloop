use super::*;
use crate::QwenProvider;
use infer_core::ModelId;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env::var;
use std::fs::read;
use std::path::PathBuf;

fn declared() -> ModalityEncoder {
    let imported = QwenProvider
        .import_manifest(
            ModelId::new(1).unwrap(),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../examples/qwen3.8-27b/config.json"
            )),
        )
        .unwrap();
    let plan = imported
        .modalities
        .iter()
        .find(|plan| plan.modality == infer_ir::Modality::Image)
        .expect("image modality");
    plan.encoder.clone().expect("declared vision encoder")
}

#[test]
fn inventory_matches_the_package_index_exactly() -> Result<()> {
    let encoder = declared();
    let index = crate::SafetensorsIndex::parse(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../examples/qwen3.8-27b/model.safetensors.index.json"
    )))?;
    let declared: BTreeSet<_> = slots(&encoder)?
        .into_iter()
        .map(|slot| slot.tensor)
        .collect();
    let packaged: BTreeSet<_> = index
        .weight_map
        .keys()
        .filter(|name| name.starts_with(&encoder.prefix))
        .cloned()
        .collect();
    let missing: Vec<_> = packaged.difference(&declared).collect();
    let extra: Vec<_> = declared.difference(&packaged).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing {missing:?} extra {extra:?}"
    );
    assert_eq!(declared.len(), 333);
    Ok(())
}

/// Shapes and dtypes predicted by the inventory must match the real checkpoint headers.
///
/// Skipped when `INFER_VISION_PACKAGE` is unset, so the portable gate stays host-independent.
#[test]
fn inventory_shapes_match_the_real_checkpoint() -> Result<()> {
    let Ok(root) = var("INFER_VISION_PACKAGE") else {
        return Ok(());
    };
    let root = PathBuf::from(root);
    let encoder = declared();
    let index = crate::SafetensorsIndex::parse(
        &read(root.join("model.safetensors.index.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let shards: BTreeSet<_> = index
        .weight_map
        .iter()
        .filter(|(name, _)| name.starts_with(&encoder.prefix))
        .map(|(_, shard)| shard.clone())
        .collect();
    let mut headers = BTreeMap::new();
    for shard in shards {
        let file = crate::SafetensorsFile::open(root.join(&shard))?;
        for (name, header) in file.tensors {
            if name.starts_with(&encoder.prefix) {
                headers.insert(name, header);
            }
        }
    }
    let slots = slots(&encoder)?;
    assert_eq!(headers.len(), slots.len());
    for slot in slots {
        let header = headers
            .get(&slot.tensor)
            .ok_or_else(|| Error::invalid(format!("missing {}", slot.tensor)))?;
        assert_eq!(header.shape, slot.shape, "{}", slot.tensor);
        assert_eq!(header.dtype, crate::TensorDtype::BF16, "{}", slot.tensor);
    }
    Ok(())
}

#[test]
fn geometry_matches_the_published_configuration() -> Result<()> {
    let encoder = declared();
    assert_eq!(encoder.depth, 27);
    assert_eq!(encoder.hidden_size, 1152);
    assert_eq!(encoder.heads, 16);
    assert_eq!(head_dim(&encoder)?, 72);
    assert_eq!(encoder.out_hidden_size, 5120);
    assert_eq!(position_grid(&encoder)?, 48);
    assert_eq!(merged_width(&encoder)?, 4608);
    assert_eq!(slots(&encoder)?.len(), 12 * 27 + 9);
    Ok(())
}

/// Merge-block ordering and bilinear taps must match the official implementation exactly.
///
/// Skipped when `INFER_VISION_GOLDEN` is unset.
#[test]
fn position_taps_match_the_reference() -> Result<()> {
    let Ok(path) = var("INFER_VISION_GOLDEN") else {
        return Ok(());
    };
    let encoder = declared();
    let mut file = crate::SafetensorsFile::open(&path)?;
    let expected_indices = file.read_bytes("c0/interp_indices", 1 << 20)?;
    let expected_weights = file.read_f32("c0/interp_weights", 1 << 20)?;
    let taps = position_taps((1, 4, 4), &encoder)?;
    let expected: Vec<i64> = expected_indices
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    assert_eq!(taps.indices.len(), 16);
    assert_eq!(expected.len(), 16 * 4);
    for patch in 0..16 {
        let gold = &expected[patch * 4..patch * 4 + 4];
        assert_eq!(
            taps.indices[patch].map(i64::from),
            gold,
            "indices patch {patch}"
        );
    }
    assert!(expected_weights.data.len() == 16 * 4, "golden weight count");
    for patch in 0..16 {
        for slot in 0..4 {
            let expected = expected_weights.data[patch * 4 + slot];
            let actual = taps.weights[patch][slot];
            assert!(
                (expected - actual).abs() < 1e-6,
                "weight {patch}/{slot}: {expected} vs {actual}"
            );
        }
    }
    Ok(())
}

/// The host gather must reproduce the reference position contribution exactly.
///
/// Skipped unless both the package and the golden are provided.
#[test]
fn gathered_positions_match_the_reference() -> Result<()> {
    let (Ok(package), Ok(golden)) = (var("INFER_VISION_PACKAGE"), var("INFER_VISION_GOLDEN"))
    else {
        return Ok(());
    };
    let encoder = declared();
    let package = PathBuf::from(package);
    let index = crate::SafetensorsIndex::parse(
        &read(package.join("model.safetensors.index.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let name = format!("{}pos_embed.weight", encoder.prefix);
    let shard = index
        .weight_map
        .get(&name)
        .ok_or_else(|| Error::invalid("missing position table"))?;
    let mut shard_file = crate::SafetensorsFile::open(package.join(shard))?;
    let table = shard_file.read_f32(&name, 1 << 30)?;
    let table: Vec<f32> = table.data;

    let taps = position_taps((1, 4, 4), &encoder)?;
    let actual = gather_positions(&table, &encoder, &taps)?;

    let mut golden_file = crate::SafetensorsFile::open(&golden)?;
    let patch_embed = golden_file.read_f32("c0/patch_embed", 1 << 30)?;
    let post_pos = golden_file.read_f32("c0/post_pos", 1 << 30)?;
    assert_eq!(actual.len(), post_pos.data.len());
    let mut max_abs = 0f32;
    let scale = post_pos.data.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (value, (after, before)) in actual
        .iter()
        .zip(post_pos.data.iter().zip(patch_embed.data.iter()))
    {
        max_abs = max_abs.max((value - (after - before)).abs());
    }
    assert!(
        max_abs <= 1e-4 * scale.max(1.0),
        "position gather deviates by {max_abs} (scale {scale})"
    );
    Ok(())
}

/// Axial grid coordinates must match the reference ordering exactly.
#[test]
fn position_ids_match_the_reference() -> Result<()> {
    let Ok(golden) = var("INFER_VISION_GOLDEN") else {
        return Ok(());
    };
    let encoder = declared();
    let mut file = crate::SafetensorsFile::open(&golden)?;
    let expected = file.read_bytes("c0/position_ids", 1 << 20)?;
    let ids = position_ids((1, 4, 4), &encoder)?;
    assert_eq!(ids.len(), 16);
    let values: Vec<i64> = expected
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    assert_eq!(values.len(), 16 * 2);
    for (patch, id) in ids.iter().enumerate() {
        assert_eq!(
            [i64::from(id[0]), i64::from(id[1])],
            [values[patch * 2], values[patch * 2 + 1]],
            "position {patch}"
        );
    }
    assert_eq!(rope_frequencies(&encoder)?.len(), 18);
    Ok(())
}

#[test]
fn invalid_geometry_is_rejected() {
    let mut encoder = declared();
    encoder.heads = 7;
    assert!(head_dim(&encoder).is_err());
    let mut encoder = declared();
    encoder.position_embeddings = 2300;
    assert!(position_grid(&encoder).is_err());
    let mut encoder = declared();
    encoder.spatial_merge_size = 0;
    assert!(merge_unit(&encoder).is_err());
    let mut encoder = declared();
    encoder.prefix.clear();
    assert!(validate(&encoder).is_err());
}
