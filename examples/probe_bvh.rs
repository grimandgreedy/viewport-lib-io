//! Decode a `.bvh` file and print its skeleton and clip shape.
//!
//! ```sh
//! cargo run --example probe-bvh --features bvh -- path/to/motion.bvh
//! ```

use viewport_lib_io::loaders::bvh;
use viewport_lib_io::types::{AnimationChannel, AnimationTrackValues};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe-bvh <file.bvh>");
    let scene = bvh::scene_from_path(std::path::Path::new(&path)).expect("decode bvh");

    let skel = &scene.skeletons[0];
    println!("skeleton `{}`: {} joints", skel.name, skel.joints.len());
    for (i, j) in skel.joints.iter().enumerate() {
        println!("  [{i:>2}] {:<16} parent={:?}", j.name, j.parent);
    }

    for clip in &scene.animations {
        let rots = clip
            .tracks
            .iter()
            .filter(|t| t.channel == AnimationChannel::Rotation)
            .count();
        let frames = clip
            .tracks
            .first()
            .map(|t| t.sampler.times.len())
            .unwrap_or(0);
        println!(
            "clip `{}`: {:.3}s, {} frames, {} tracks ({rots} rotation)",
            clip.name,
            clip.duration,
            frames,
            clip.tracks.len(),
        );
        if let Some(t) = clip.tracks.iter().find(|t| t.joint == 0) {
            if let AnimationTrackValues::Quat(q) = &t.sampler.values {
                println!("  root rotation frame 0: {:?}", q[0]);
            }
        }
    }
}
