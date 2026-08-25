//! Dump per-submesh UV coverage from an FBX, to diagnose a character that
//! imports but textures wrong (e.g. the face texture smeared over the body).
//!
//! For each decoded mesh it prints the material name and the UV bounding box
//! (umin,umax)x(vmin,vmax). A body submesh whose UVs collapse into a small
//! sub-rect, or land in the same rect as the head, means the loader picked the
//! wrong UV channel (lightmap / data UV) instead of the albedo channel.
//!
//! Pair with `VIEWPORT_FBX_LOG_UV=1` to also see every candidate channel and
//! which one the picker chose per model.
//!
//!   VIEWPORT_FBX_LOG_UV=1 cargo run --example probe-fbx-uv -- path/to.fbx

use std::path::PathBuf;

fn bbox(uvs: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let (mut umin, mut umax, mut vmin, mut vmax) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    for uv in uvs {
        umin = umin.min(uv[0]);
        umax = umax.max(uv[0]);
        vmin = vmin.min(uv[1]);
        vmax = vmax.max(uv[1]);
    }
    (umin, umax, vmin, vmax)
}

fn main() {
    let path: PathBuf = match std::env::args().nth(1) {
        Some(s) => s.into(),
        None => {
            eprintln!("usage: cargo run --example probe-fbx-uv -- <path-to.fbx>");
            std::process::exit(2);
        }
    };
    if !path.exists() {
        eprintln!("file not found: {}", path.display());
        std::process::exit(1);
    }

    let scene = match viewport_lib_io::loaders::fbx::scene_from_path(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to load {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    println!("file: {}", path.display());
    println!(
        "summary: {} meshes, {} materials",
        scene.meshes.len(),
        scene.materials.len()
    );
    println!(
        "\n{:<34} {:<28} {:>8}  uv-bbox",
        "mesh", "material", "verts"
    );
    for m in &scene.meshes {
        let mat = m
            .material_index
            .and_then(|i| scene.materials.get(i))
            .map(|mm| mm.name.as_str())
            .unwrap_or("<none>");
        let sm = &m.mesh;
        let uv = match &sm.uvs {
            Some(uvs) if !uvs.is_empty() => {
                let (umin, umax, vmin, vmax) = bbox(uvs);
                format!(
                    "[{:.3},{:.3}] x [{:.3},{:.3}]  area={:.3}",
                    umin,
                    umax,
                    vmin,
                    vmax,
                    (umax - umin).max(0.0) * (vmax - vmin).max(0.0)
                )
            }
            _ => "<no uvs>".to_string(),
        };
        println!(
            "{:<34} {:<28} {:>8}  {}",
            m.name,
            mat,
            sm.positions.len(),
            uv
        );
    }
}
