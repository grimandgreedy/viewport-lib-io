use std::path::PathBuf;

/// Errors returned by `viewport-lib-io`.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// A filesystem operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The file contents could not be parsed.
    #[error("parse error: {0}")]
    Parse(String),
    /// The requested format requires a disabled crate feature.
    #[error("feature `{feature}` is required for {context}")]
    MissingFeature {
        /// Required Cargo feature name.
        feature: &'static str,
        /// Human-readable context for the failure.
        context: &'static str,
    },
    /// The file or extension is not supported by the active registry.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    /// A referenced dependency was not found.
    #[error("missing dependency: {0}")]
    MissingDependency(PathBuf),
    /// Upload to viewport-lib failed.
    #[error("viewport upload failed: {0}")]
    Upload(#[from] viewport_lib::ViewportError),
}
