//! Biovision Hierarchy (`.bvh`) motion-capture decoding.
//!
//! BVH is the canonical open, text-based skeletal-animation interchange format:
//! a `HIERARCHY` block declaring a joint tree with per-joint `OFFSET`s and
//! `CHANNELS`, followed by a `MOTION` block of one Euler-angle sample row per
//! frame. It carries no mesh, material, or skin weights, so the decoded
//! [`IoScene`] populates only `skeletons` (one entry) and `animations` (one
//! clip). The free mocap libraries (CMU, Bandai Namco, SFU, ACCAD) all ship in
//! this format.
//!
//! BVH is authored Y-up; like the glTF loader, samples and bind matrices are
//! reoriented once here into viewport-lib-io's right-handed Z-up convention, so
//! downstream consumers never re-rotate.

use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use crate::error::IoError;
use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, IoScene, Joint, SceneData, Skeleton,
};

/// Decode a BVH file into a scene carrying one skeleton and one clip.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    let text = std::fs::read_to_string(path)?;
    scene_from_str(&text)
}

/// Decode BVH from an in-memory byte buffer (UTF-8).
pub fn scene_from_bytes(bytes: &[u8]) -> Result<IoScene, IoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| IoError::Parse(format!("bvh: not valid UTF-8: {e}")))?;
    scene_from_str(text)
}

/// Which component of a joint's local transform one BVH channel drives, and
/// about which axis. Channels are listed in application order per joint.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Channel {
    PosX,
    PosY,
    PosZ,
    RotX,
    RotY,
    RotZ,
}

impl Channel {
    fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "Xposition" => Channel::PosX,
            "Yposition" => Channel::PosY,
            "Zposition" => Channel::PosZ,
            "Xrotation" => Channel::RotX,
            "Yrotation" => Channel::RotY,
            "Zrotation" => Channel::RotZ,
            _ => return None,
        })
    }

    fn is_rotation(self) -> bool {
        matches!(self, Channel::RotX | Channel::RotY | Channel::RotZ)
    }
}

/// One parsed joint of the hierarchy. End Sites are not stored: they only fix a
/// leaf bone's tip, which animation never drives.
struct BvhJoint {
    name: String,
    parent: Option<usize>,
    offset: Vec3,
    channels: Vec<Channel>,
}

/// Decode BVH from a UTF-8 string.
pub fn scene_from_str(text: &str) -> Result<IoScene, IoError> {
    let mut tokens = text.split_whitespace().peekable();

    expect(&mut tokens, "HIERARCHY")?;
    let mut joints: Vec<BvhJoint> = Vec::new();
    expect(&mut tokens, "ROOT")?;
    let root_name = next(&mut tokens)?.to_string();
    parse_joint(&mut tokens, &mut joints, None, root_name)?;

    if joints.len() > 256 {
        return Err(IoError::Parse(format!(
            "bvh: {} joints exceeds the 256-joint limit",
            joints.len()
        )));
    }

    // MOTION block.
    expect(&mut tokens, "MOTION")?;
    expect(&mut tokens, "Frames:")?;
    let frame_count: usize = next(&mut tokens)?
        .parse()
        .map_err(|_| IoError::Parse("bvh: bad Frames count".into()))?;
    expect(&mut tokens, "Frame")?;
    expect(&mut tokens, "Time:")?;
    let frame_time: f32 = next(&mut tokens)?
        .parse()
        .map_err(|_| IoError::Parse("bvh: bad Frame Time".into()))?;

    let per_frame: usize = joints.iter().map(|j| j.channels.len()).sum();
    let mut samples: Vec<f32> = Vec::with_capacity(frame_count * per_frame);
    for token in tokens {
        let v: f32 = token
            .parse()
            .map_err(|_| IoError::Parse(format!("bvh: non-numeric motion sample `{token}`")))?;
        samples.push(v);
    }
    if per_frame != 0 && samples.len() < frame_count * per_frame {
        return Err(IoError::Parse(format!(
            "bvh: motion has {} samples, expected {} ({frame_count} frames x {per_frame} channels)",
            samples.len(),
            frame_count * per_frame,
        )));
    }

    let (skeleton, clip) = build(&joints, frame_count, frame_time, &samples);

    Ok(SceneData {
        skeletons: vec![skeleton],
        animations: vec![clip],
        ..Default::default()
    })
}

/// Parse one joint and, recursively, its children. `tokens` sits just after the
/// joint's name; on return it sits just after the joint's closing brace.
fn parse_joint<'a, I>(
    tokens: &mut std::iter::Peekable<I>,
    joints: &mut Vec<BvhJoint>,
    parent: Option<usize>,
    name: String,
) -> Result<(), IoError>
where
    I: Iterator<Item = &'a str>,
{
    let index = joints.len();
    joints.push(BvhJoint {
        name,
        parent,
        offset: Vec3::ZERO,
        channels: Vec::new(),
    });

    expect(tokens, "{")?;
    loop {
        match next(tokens)? {
            "OFFSET" => {
                let x = num(tokens)?;
                let y = num(tokens)?;
                let z = num(tokens)?;
                joints[index].offset = Vec3::new(x, y, z);
            }
            "CHANNELS" => {
                let count: usize = next(tokens)?
                    .parse()
                    .map_err(|_| IoError::Parse("bvh: bad CHANNELS count".into()))?;
                let mut channels = Vec::with_capacity(count);
                for _ in 0..count {
                    let tok = next(tokens)?;
                    let ch = Channel::parse(tok)
                        .ok_or_else(|| IoError::Parse(format!("bvh: unknown channel `{tok}`")))?;
                    channels.push(ch);
                }
                joints[index].channels = channels;
            }
            "JOINT" => {
                let child_name = next(tokens)?.to_string();
                parse_joint(tokens, joints, Some(index), child_name)?;
            }
            "End" => {
                // "End Site": a leaf tip, `{ OFFSET x y z }`. Consumed and
                // dropped; it carries no channels and animation never drives it.
                expect(tokens, "Site")?;
                expect(tokens, "{")?;
                expect(tokens, "OFFSET")?;
                num(tokens)?;
                num(tokens)?;
                num(tokens)?;
                expect(tokens, "}")?;
            }
            "}" => return Ok(()),
            other => {
                return Err(IoError::Parse(format!(
                    "bvh: unexpected token `{other}` in joint body"
                )));
            }
        }
    }
}

/// Turn the parsed joints and flat motion samples into a reoriented skeleton and
/// clip. Rotations compose in the channels' listed order; every orientation is
/// carried from BVH Y-up into Z-up.
fn build(
    joints: &[BvhJoint],
    frame_count: usize,
    frame_time: f32,
    samples: &[f32],
) -> (Skeleton, AnimationClip) {
    let per_frame: usize = joints.iter().map(|j| j.channels.len()).sum();

    // Bind-pose world transforms accumulate the OFFSET translations down the
    // tree (bind rotation is identity), giving each joint its inverse-bind.
    let mut world_bind: Vec<Mat4> = Vec::with_capacity(joints.len());
    let mut skel_joints: Vec<Joint> = Vec::with_capacity(joints.len());
    for j in joints.iter() {
        let local = Mat4::from_translation(j.offset);
        let world = match j.parent {
            Some(p) => world_bind[p] * local,
            None => local,
        };
        world_bind.push(world);
        skel_joints.push(Joint {
            name: j.name.clone(),
            parent: j.parent.map(|p| p as u8),
            inverse_bind: reorient_affine(world.inverse()),
        });
    }

    // Per-joint sample columns: translation (Vec3) and rotation (Quat) streams.
    let mut translations: Vec<Vec<Vec3>> = vec![Vec::new(); joints.len()];
    let mut rotations: Vec<Vec<Quat>> = vec![Vec::new(); joints.len()];
    let has_pos: Vec<bool> = joints
        .iter()
        .map(|j| j.channels.iter().any(|c| !c.is_rotation()))
        .collect();
    let has_rot: Vec<bool> = joints
        .iter()
        .map(|j| j.channels.iter().any(|c| c.is_rotation()))
        .collect();

    for f in 0..frame_count {
        let row = &samples[f * per_frame..(f + 1) * per_frame];
        let mut cursor = 0usize;
        for (i, j) in joints.iter().enumerate() {
            let mut pos = j.offset;
            let mut rot = Quat::IDENTITY;
            for &ch in &j.channels {
                let v = row[cursor];
                cursor += 1;
                match ch {
                    Channel::PosX => pos.x = v,
                    Channel::PosY => pos.y = v,
                    Channel::PosZ => pos.z = v,
                    // Compose in listed order: the first-listed rotation is the
                    // outermost (left-most) factor.
                    Channel::RotX => rot *= Quat::from_rotation_x(v.to_radians()),
                    Channel::RotY => rot *= Quat::from_rotation_y(v.to_radians()),
                    Channel::RotZ => rot *= Quat::from_rotation_z(v.to_radians()),
                }
            }
            if has_pos[i] {
                translations[i].push(reorient_vec3(pos));
            }
            if has_rot[i] {
                rotations[i].push(reorient_quat(rot));
            }
        }
    }

    let times: Vec<f32> = (0..frame_count).map(|f| f as f32 * frame_time).collect();
    let duration = times.last().copied().unwrap_or(0.0);

    let mut tracks: Vec<AnimationTrack> = Vec::new();
    for i in 0..joints.len() {
        if has_rot[i] && !rotations[i].is_empty() {
            tracks.push(AnimationTrack {
                joint: i,
                channel: AnimationChannel::Rotation,
                sampler: AnimationSampler {
                    interpolation: AnimationInterpolation::Linear,
                    times: times.clone(),
                    values: AnimationTrackValues::Quat(std::mem::take(&mut rotations[i])),
                },
            });
        }
        if has_pos[i] && !translations[i].is_empty() {
            tracks.push(AnimationTrack {
                joint: i,
                channel: AnimationChannel::Translation,
                sampler: AnimationSampler {
                    interpolation: AnimationInterpolation::Linear,
                    times: times.clone(),
                    values: AnimationTrackValues::Vec3(std::mem::take(&mut translations[i])),
                },
            });
        }
    }

    let skeleton = Skeleton {
        name: joints.first().map(|j| j.name.clone()).unwrap_or_default(),
        joints: skel_joints,
    };
    let clip = AnimationClip {
        name: "bvh".to_string(),
        duration,
        skeleton_index: 0,
        tracks,
    };
    (skeleton, clip)
}

// --- Y-up to Z-up reorientation (matches the glTF loader) ---

/// +90 degrees about X, as a quaternion: Y -> Z, Z -> -Y, X unchanged.
fn y_up_to_z_up() -> Quat {
    Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

/// Rotate a position or direction vector from Y-up into Z-up.
fn reorient_vec3(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}

/// Conjugate a unit quaternion by the reorientation: `R * q * R^-1`.
fn reorient_quat(q: Quat) -> Quat {
    let r = y_up_to_z_up();
    r * q * r.conjugate()
}

/// Conjugate an affine transform by the reorientation: `R * M * R^-1`.
fn reorient_affine(m: Mat4) -> Mat4 {
    let r = Mat4::from_quat(y_up_to_z_up());
    r * m * r.inverse()
}

fn next<'a, I: Iterator<Item = &'a str>>(tokens: &mut I) -> Result<&'a str, IoError> {
    tokens
        .next()
        .ok_or_else(|| IoError::Parse("bvh: unexpected end of file".into()))
}

fn expect<'a, I: Iterator<Item = &'a str>>(tokens: &mut I, want: &str) -> Result<(), IoError> {
    let got = next(tokens)?;
    if got == want {
        Ok(())
    } else {
        Err(IoError::Parse(format!(
            "bvh: expected `{want}`, found `{got}`"
        )))
    }
}

fn num<'a, I: Iterator<Item = &'a str>>(tokens: &mut I) -> Result<f32, IoError> {
    let tok = next(tokens)?;
    tok.parse()
        .map_err(|_| IoError::Parse(format!("bvh: expected a number, found `{tok}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal two-joint rig: a root with a translation+rotation channel set
    // and one child that only rotates. Two motion frames.
    const SAMPLE: &str = "\
HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.0 5.0 0.0
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.0 2.0 3.0 0.0 90.0 0.0 0.0 0.0 0.0
";

    #[test]
    fn parses_hierarchy_and_names() {
        let scene = scene_from_str(SAMPLE).expect("parse");
        assert_eq!(scene.skeletons.len(), 1);
        let skel = &scene.skeletons[0];
        // End Site is dropped, so two joints remain.
        assert_eq!(skel.joints.len(), 2);
        assert_eq!(skel.joints[0].name, "Hips");
        assert_eq!(skel.joints[1].name, "Spine");
        assert_eq!(skel.joints[0].parent, None);
        assert_eq!(skel.joints[1].parent, Some(0));
    }

    #[test]
    fn clip_duration_and_frame_count() {
        let scene = scene_from_str(SAMPLE).expect("parse");
        let clip = &scene.animations[0];
        // Two frames at 0.033333 => last sample time is one frame in.
        assert!((clip.duration - 0.033333).abs() < 1e-4);
        // Root: rotation + translation; child: rotation only => three tracks.
        assert_eq!(clip.tracks.len(), 3);
    }

    #[test]
    fn root_translation_is_reoriented_z_up() {
        let scene = scene_from_str(SAMPLE).expect("parse");
        let clip = &scene.animations[0];
        let track = clip
            .tracks
            .iter()
            .find(|t| t.joint == 0 && t.channel == AnimationChannel::Translation)
            .expect("root translation track");
        let AnimationTrackValues::Vec3(values) = &track.sampler.values else {
            panic!("translation track must hold Vec3 samples");
        };
        // Frame 1 authored (1,2,3) in Y-up becomes (1,-3,2) in Z-up.
        let f1 = values[1];
        assert!((f1.x - 1.0).abs() < 1e-4);
        assert!((f1.y + 3.0).abs() < 1e-4);
        assert!((f1.z - 2.0).abs() < 1e-4);
    }

    #[test]
    fn rotation_track_carries_samples() {
        let scene = scene_from_str(SAMPLE).expect("parse");
        let clip = &scene.animations[0];
        let track = clip
            .tracks
            .iter()
            .find(|t| t.joint == 0 && t.channel == AnimationChannel::Rotation)
            .expect("root rotation track");
        let AnimationTrackValues::Quat(values) = &track.sampler.values else {
            panic!("rotation track must hold Quat samples");
        };
        assert_eq!(values.len(), 2);
        // Frame 0 is all-zero Euler => identity.
        assert!(values[0].abs_diff_eq(Quat::IDENTITY, 1e-4));
        // Frame 1 rotates 90 degrees, so it is not identity.
        assert!(!values[1].abs_diff_eq(Quat::IDENTITY, 1e-3));
    }

    #[test]
    fn rejects_short_motion() {
        let truncated = SAMPLE.replace("1.0 2.0 3.0 0.0 90.0 0.0 0.0 0.0 0.0\n", "");
        assert!(scene_from_str(&truncated).is_err());
    }
}
