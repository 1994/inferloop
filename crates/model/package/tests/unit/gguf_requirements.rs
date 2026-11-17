use super::encoding_for;
use infer_format::GgufType;
use infer_ir::{ExecutableEncoding, GroupedIntFormat};

#[test]
fn maps_tensor_types_to_weight_encodings() {
    for (ty, format) in [
        (GgufType::Q4_0, GroupedIntFormat::Q4_0),
        (GgufType::Q4_1, GroupedIntFormat::Q4_1),
        (GgufType::Q5_0, GroupedIntFormat::Q5_0),
        (GgufType::Q5_1, GroupedIntFormat::Q5_1),
        (GgufType::Q8_0, GroupedIntFormat::Q8_0),
        (GgufType::Q4K, GroupedIntFormat::Q4K),
        (GgufType::Q5K, GroupedIntFormat::Q5K),
        (GgufType::Q6K, GroupedIntFormat::Q6K),
    ] {
        assert_eq!(
            encoding_for(ty).unwrap(),
            ExecutableEncoding::GroupedInt(format)
        );
    }
    assert_eq!(
        encoding_for(GgufType::F16).unwrap(),
        ExecutableEncoding::Dense
    );
    assert!(encoding_for(GgufType::Q2K).is_err());
    assert!(encoding_for(GgufType::Q8_1).is_err());
}
