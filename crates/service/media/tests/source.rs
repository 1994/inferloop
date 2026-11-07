use axum::{Router, response::Redirect, routing::get};
use infer_core::ErrorCode;
use infer_media_input::{HttpSource, data_uri_reader, decode_data_uri, decode_rgb8};
use std::io::{Cursor, Read};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::time::timeout;

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

#[tokio::test]
async fn http_stream_yields_data_before_the_response_completes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (continue_tx, continue_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n")
            .await
            .unwrap();
        continue_rx.await.unwrap();
        socket.write_all(b"3\r\ndef\r\n0\r\n\r\n").await.unwrap();
    });
    let source = HttpSource::new().unwrap();
    let mut response = source
        .stream(&format!("http://{address}/media"), None)
        .await
        .unwrap();
    let first = timeout(Duration::from_secs(5), response.next_chunk())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.as_ref(), b"abc");
    continue_tx.send(()).unwrap();
    let second = response.next_chunk().await.unwrap().unwrap();
    assert_eq!(second.as_ref(), b"def");
    assert!(response.next_chunk().await.unwrap().is_none());
    timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn http_stream_enforces_limits_without_a_content_length() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n")
            .await
            .unwrap();
    });
    let source = HttpSource::new().unwrap();
    let mut response = source
        .stream(&format!("http://{address}/media"), Some(5))
        .await
        .unwrap();
    let error = timeout(Duration::from_secs(5), async {
        loop {
            match response.next_chunk().await {
                Ok(Some(_)) => {}
                Ok(None) => panic!("stream completed beyond its limit"),
                Err(error) => break error,
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(error.code, ErrorCode::Capacity);
    assert!(response.next_chunk().await.is_err());
    timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
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

#[test]
fn inline_reader_decodes_without_materializing_the_input() {
    let mut reader = data_uri_reader("data:video/mp4;base64,AQIDBAU=").unwrap();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, [1, 2, 3, 4, 5]);
    assert!(data_uri_reader("data:video/mp4,not-base64").is_err());
    let mut invalid = data_uri_reader("data:video/mp4;base64,!!!!").unwrap();
    assert!(invalid.read_to_end(&mut Vec::new()).is_err());
}
