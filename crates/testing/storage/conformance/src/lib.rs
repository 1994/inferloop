//! Behavioral suites every storage backend runs, whoever wrote it.
//!
//! [`records::suite`] covers the [`infer_spi::RecordStore`] primitive; [`responses::suite`]
//! covers any [`infer_spi::ResponseStore`], whether it is the record adapter over some backend
//! or a direct implementation. Cases use fresh keys, so suites can share one database with
//! other runs. Each suite returns the first failed expectation as an error.
mod expect;
pub mod records;
pub mod responses;
