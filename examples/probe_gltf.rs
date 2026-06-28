//! Dump what `viewport_lib_io::loaders::gltf::scene_from_path` extracts from a
//! `.glb` or `.gltf` file. Useful when a character imports but renders wrong,
//! or when you want to check whether a rig is properly skinned vs.
//! bone-parented.
//!
//! Usage:
//!   cargo run --example probe-gltf -- path/to/file.glb
//!
//! Prints per-mesh attribute presence (positions/normals/uvs/tangents/skin
//! weights), per-skeleton joint counts, and per-clip track counts + durations.

use std::path::PathBuf;

fn main() {
    let path: PathBuf = match std::env::args().nth(1) {
        Some(s) => s.into(),
        None => {
            eprintln!("usage: cargo run --example probe-gltf -- <path-to.glb>");
            std::process::exit(2);
        }
    };

    if !path.exists() {
        eprintln!("file not found: {}", path.display());
        std::process::exit(1);
    }

    let scene = match viewport_lib_io::loaders::gltf::scene_from_path(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to load {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    println!("file: {}", path.display());
    println!(
        "summary: {} meshes, {} skeletons, {} animations, {} materials, {} point sets",
        scene.meshes.len(),
        scene.skeletons.len(),
        scene.animations.len(),
        scene.materials.len(),
        scene.point_sets.len(),
    );

    println!("\nmeshes:");
    let mut skinned = 0usize;
    let mut bone_parented_only = 0usize;
    let mut static_meshes = 0usize;
    for (i, m) in scene.meshes.iter().enumerate() {
        let sm = &m.mesh;
        let skin = sm.skin_weights.is_some();
        let skel = m.skeleton_index;
        let class = match (skin, skel) {
            (true, Some(_)) => {
                // Cannot distinguish real skin from synthesised bone-parented
                // skin here: both look the same downstream. The loader
                // synthesises rigid skinning when JOINTS_0/WEIGHTS_0 are
                // absent on the source primitive.
                skinned += 1;
                "skinned"
            }
            (false, Some(_)) => {
                // Should not occur with the current loader.
                "skeleton-tagged but no weights (?)"
            }
            (false, None) => {
                static_meshes += 1;
                "static"
            }
            (true, None) => {
                bone_parented_only += 1;
                "weights without skeleton (?)"
            }
        };
        println!(
            "  [{i:>3}] {:<28} verts={:<6} idx={:<6} uvs={} tan={} skel={:?} -> {class}",
            truncate(&m.name, 28),
            sm.positions.len(),
            sm.indices.len(),
            sm.uvs.is_some(),
            sm.tangents.is_some(),
            skel,
        );
    }
    println!("  totals: skinned={skinned}, static={static_meshes}, anomalous={bone_parented_only}",);

    println!("\nskeletons:");
    for (i, sk) in scene.skeletons.iter().enumerate() {
        let roots = sk.joints.iter().filter(|j| j.parent.is_none()).count();
        println!(
            "  [{i}] name={:?} joints={} roots={}",
            sk.name,
            sk.joints.len(),
            roots,
        );
        // Show the first few joints so the hierarchy is recognisable.
        for (ji, j) in sk.joints.iter().take(8).enumerate() {
            println!(
                "    joint {ji:>3} parent={:>4} name={:?}",
                j.parent
                    .map(|p| format!("{p}"))
                    .unwrap_or_else(|| "-".into()),
                j.name,
            );
        }
        if sk.joints.len() > 8 {
            println!("    ... ({} more)", sk.joints.len() - 8);
        }
    }

    println!("\nanimations:");
    for (i, c) in scene.animations.iter().enumerate() {
        let translation = c
            .tracks
            .iter()
            .filter(|t| matches!(t.channel, viewport_lib_io::AnimationChannel::Translation))
            .count();
        let rotation = c
            .tracks
            .iter()
            .filter(|t| matches!(t.channel, viewport_lib_io::AnimationChannel::Rotation))
            .count();
        let scale = c
            .tracks
            .iter()
            .filter(|t| matches!(t.channel, viewport_lib_io::AnimationChannel::Scale))
            .count();
        let cubic = c
            .tracks
            .iter()
            .filter(|t| {
                matches!(
                    t.sampler.interpolation,
                    viewport_lib_io::AnimationInterpolation::CubicSpline
                )
            })
            .count();
        println!(
            "  [{i}] {:<24} skeleton={} duration={:.3}s tracks={} (T={translation} R={rotation} S={scale}) cubic={cubic}",
            truncate(&c.name, 24),
            c.skeleton_index,
            c.duration,
            c.tracks.len(),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}
