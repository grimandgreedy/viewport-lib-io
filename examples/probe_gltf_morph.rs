//! Dump the blend shapes (morph targets) and weight animations a glTF carries.
//!
//! ```sh
//! cargo run --example probe-gltf-morph --features gltf -- path/to/model.glb
//! ```

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe-gltf-morph <file.glb>");
    let scene = viewport_lib_io::loaders::gltf::scene_from_path(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("decode {path}: {e:?}"));

    for mesh in &scene.meshes {
        if mesh.mesh.morph_targets.is_empty() {
            continue;
        }
        println!(
            "mesh '{}': {} verts, {} morph targets",
            mesh.name,
            mesh.mesh.positions.len(),
            mesh.mesh.morph_targets.len(),
        );
        for t in &mesh.mesh.morph_targets {
            let moved = t
                .position_deltas
                .iter()
                .filter(|d| d != &&[0.0, 0.0, 0.0])
                .count();
            println!("    {:<28} {moved} verts moved", t.name);
        }
    }

    println!(
        "\n{} morph weight animation(s):",
        scene.morph_animations.len()
    );
    for clip in &scene.morph_animations {
        println!(
            "  '{}': {:.2}s, {} targets, {} keyframes",
            clip.name,
            clip.duration,
            clip.target_count,
            clip.times.len(),
        );
    }
}
