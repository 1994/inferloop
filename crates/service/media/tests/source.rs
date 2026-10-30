use axum::{Router, response::Redirect, routing::get};
use infer_core::ErrorCode;
use infer_media_input::{HttpSource, decode_data_uri, decode_rgb8};
use std::io::Cursor;
use tokio::net::TcpListener;

#[tokio::test]
async fn http_source_reuses_a_client_and_checks_status_and_limit() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/media", get(|| async { "123456" }))
        .route("/redirect", get(|| async { Redirect::temporary("/media") }));
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let source = HttpSource::new().unwrap();
    let url = format!("http://{address}/media");
    assert_eq!(source.fetch(&url, None).await.unwrap(), b"123456");
    assert_eq!(
        source
            .fetch(&format!("http://{address}/redirect"), None)
            .await
            .unwrap(),
        b"123456"
    );
    assert_eq!(
        source.fetch(&url, Some(5)).await.unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(
        source
            .fetch(&format!("http://{address}/missing"), None)
            .await
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    server.abort();
}

#[test]
fn decoder_returns_checked_rgb_pixels() {
    let expected = vec![255, 4, 7, 0, 32, 200];
    let image = image::RgbImage::from_raw(2, 1, expected.clone()).unwrap();
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let (width, height, pixels) = decode_rgb8(&bytes.into_inner()).unwrap().into_parts();
    assert_eq!((width, height, pixels), (2, 1, expected));
    assert!(decode_rgb8(b"not an image").is_err());
}

#[test]
fn inline_source_decodes_and_reports_errors_and_optional_limits() {
    assert_eq!(
        decode_data_uri("data:image/png;base64,AQID", None).unwrap(),
        [1, 2, 3]
    );
    assert_eq!(
        decode_data_uri("data:application/octet-stream;base64,AQID", Some(2))
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert!(decode_data_uri("data:image/png;base64,!!!", None).is_err());
    assert!(decode_data_uri("data:image/png,abc", None).is_err());
    assert!(decode_data_uri("other;base64,AQID", None).is_err());
}
