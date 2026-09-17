//! Fixtures the loader tests build in memory: a temp directory, and a GLB
//! packed from glTF JSON plus a binary buffer.

/// A fresh directory under the system temp dir, named for the test plus the
/// process id and a timestamp so parallel runs cannot collide.
pub(super) fn temp_dir(name: &str) -> std::path::PathBuf {
    let unique = format!(
        "viewport_lib_io_{name}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Pack a glTF JSON + binary buffer into a self-contained GLB blob.
/// Used by tests that want a `scene_from_slice`-friendly fixture with
/// no external `.bin` file on disk.
#[cfg(feature = "gltf")]
pub(super) fn make_glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
    // glTF 2.0 GLB layout: 12-byte header, 8-byte JSON chunk header,
    // padded JSON body, 8-byte BIN chunk header, padded BIN body.
    fn pad4(len: usize) -> usize {
        (4 - (len & 3)) & 3
    }
    let json_pad = pad4(json.len());
    let bin_pad = pad4(bin.len());
    let json_len = json.len() + json_pad;
    let bin_len = bin.len() + bin_pad;
    let total = 12 + 8 + json_len + 8 + bin_len;

    let mut out = Vec::with_capacity(total);
    // Header.
    out.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
    out.extend_from_slice(&2u32.to_le_bytes()); // version
    out.extend_from_slice(&(total as u32).to_le_bytes());
    // JSON chunk.
    out.extend_from_slice(&(json_len as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(json);
    for _ in 0..json_pad {
        out.push(b' ');
    }
    // BIN chunk.
    out.extend_from_slice(&(bin_len as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(bin);
    for _ in 0..bin_pad {
        out.push(0);
    }
    out
}
