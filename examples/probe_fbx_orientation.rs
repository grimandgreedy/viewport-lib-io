// Post-pipeline orientation check: loads an FBX through
// `viewport_lib_io::loaders::fbx::scene_from_path` (exactly what
// `drake-assets::FbxLoader` consumes), bakes each mesh's transform into
// its vertex positions, and prints the world-space bounding box. The
// expected DRAKE convention is Z-up: for "tall" props (roof, pot, arc,
// tree) the dominant Z extent should be the largest dimension.
//
// cargo run --example probe-fbx-orientation --features fbx -- <file.fbx> ...

use glam::Vec3;
use std::path::Path;

use viewport_lib_io::loaders::fbx::scene_from_path;

fn dominant_axis(ext: [f32; 3]) -> char {
    let mut best = 0usize;
    for i in 1..3 {
        if ext[i] > ext[best] {
            best = i;
        }
    }
    ['X', 'Y', 'Z'][best]
}

fn probe(path: &Path) {
    println!("=== {} ===", path.display());
    let scene = match scene_from_path(path) {
        Ok(s) => s,
        Err(e) => {
            println!("  load failed: {e:?}");
            return;
        }
    };
    for io_mesh in &scene.meshes {
        let t = io_mesh.transform;
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &io_mesh.mesh.positions {
            let w = t.transform_point3(Vec3::from(*p));
            let a = [w.x, w.y, w.z];
            for k in 0..3 {
                if a[k] < min[k] {
                    min[k] = a[k];
                }
                if a[k] > max[k] {
                    max[k] = a[k];
                }
            }
        }
        let ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        println!(
            "  '{}': bbox min=({:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3}) ext=({:.3},{:.3},{:.3}) dominant={}",
            io_mesh.name,
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2],
            ext[0],
            ext[1],
            ext[2],
            dominant_axis(ext),
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe-fbx-orientation <file.fbx> [more.fbx ...]");
        std::process::exit(1);
    }
    for a in args {
        probe(Path::new(&a));
    }
}
