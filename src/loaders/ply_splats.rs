//! Trained Gaussian splat sets, as a trainer writes them out.
//!
//! A `.ply` from a 3D Gaussian Splatting trainer is an ordinary point cloud
//! with a great many extra per-vertex properties, so the parse is
//! [`ply_points`](super::ply_points)'s and nothing here re-reads a byte. What
//! this module owns is the other half: the properties are not stored in the
//! units anything draws them in.
//!
//! Three conventions come from the reference implementation and every trainer
//! that followed it:
//!
//! - `scale_0..2` are logarithms. A splat of extent 1 is written as 0.
//! - `opacity` is a logit. A splat of opacity 0.5 is written as 0.
//! - `rot_0..3` are a quaternion in `w, x, y, z` order, unnormalised.
//!
//! None of those is announced in the file. A reader that takes the numbers at
//! face value produces a set that loads, draws, and is wrong in a way that
//! looks like a bad training run: everything the same size, everything
//! half-transparent, everything facing somewhere arbitrary.

use std::path::Path;

use crate::error::IoError;
use crate::types::{GaussianSplatSet, ShDegree};

/// Decode a trained Gaussian splat set from a `.ply` file.
pub fn splats_from_path(path: &Path) -> Result<GaussianSplatSet, IoError> {
    #[cfg(feature = "ply")]
    {
        let bytes = std::fs::read(path)?;
        splats_from_bytes(&bytes)
    }

    #[cfg(not(feature = "ply"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "ply",
            context: "PLY gaussian-splat decoding",
        })
    }
}

/// Decode a trained Gaussian splat set from bytes already read.
///
/// The spherical-harmonic degree is read from how many `f_rest_*` properties
/// the file carries: none is degree zero, three per channel is degree one,
/// fifteen per channel is degree three. A file carrying some other number is
/// refused rather than loaded at a lower degree, because dropping the
/// view-dependent terms is a change to how the set looks and a silent one
/// would be found by someone wondering why their reconstruction went flat.
pub fn splats_from_bytes(bytes: &[u8]) -> Result<GaussianSplatSet, IoError> {
    #[cfg(feature = "ply")]
    {
        let cloud = super::ply_points::point_cloud_from_bytes(bytes)?;
        let count = cloud.positions.len();
        if count == 0 {
            return Err(IoError::Parse("ply: the file holds no splats".into()));
        }

        let column = |name: &str| -> Result<&Vec<f32>, IoError> {
            cloud.scalar_attributes.get(name).ok_or_else(|| {
                IoError::Parse(format!(
                    "ply: no `{name}` property, so this is a point cloud rather than a trained splat set"
                ))
            })
        };

        let scale = [column("scale_0")?, column("scale_1")?, column("scale_2")?];
        let rot = [
            column("rot_0")?,
            column("rot_1")?,
            column("rot_2")?,
            column("rot_3")?,
        ];
        let opacity = column("opacity")?;
        let dc = [column("f_dc_0")?, column("f_dc_1")?, column("f_dc_2")?];

        for values in scale
            .iter()
            .chain(rot.iter())
            .chain(dc.iter())
            .chain(std::iter::once(&opacity))
        {
            if values.len() != count {
                return Err(IoError::Parse(
                    "ply: a splat property has a different length from the positions".into(),
                ));
            }
        }

        let rest = rest_columns(&cloud)?;
        let degree = match rest.len() {
            0 => ShDegree::Zero,
            9 => ShDegree::One,
            45 => ShDegree::Three,
            other => {
                return Err(IoError::Parse(format!(
                    "ply: {other} `f_rest_*` properties is no spherical-harmonic degree this reads \
                     (expected 0, 9 or 45)"
                )));
            }
        };
        // Per channel rather than in total, which is how the file lays them
        // out: all of red's, then all of green's, then all of blue's.
        let per_channel = rest.len() / 3;

        let mut scales = Vec::with_capacity(count);
        let mut rotations = Vec::with_capacity(count);
        let mut opacities = Vec::with_capacity(count);
        let mut sh_coefficients = Vec::with_capacity(count * degree.coeff_count());

        for i in 0..count {
            scales.push([scale[0][i].exp(), scale[1][i].exp(), scale[2][i].exp()]);
            rotations.push(unit([rot[0][i], rot[1][i], rot[2][i], rot[3][i]]));
            opacities.push(sigmoid(opacity[i]));

            sh_coefficients.extend_from_slice(&[dc[0][i], dc[1][i], dc[2][i]]);
            // The file is channel-major and the packed form is
            // coefficient-major, so this is a transpose rather than a copy:
            // coefficient k of channel c sits at `c * per_channel + k` in the
            // file and at `3 * (k + 1) + c` in the result, after the three DC
            // terms.
            for k in 0..per_channel {
                for c in 0..3 {
                    sh_coefficients.push(rest[c * per_channel + k][i]);
                }
            }
        }

        Ok(GaussianSplatSet {
            positions: cloud.positions,
            scales,
            rotations,
            opacities,
            sh_coefficients,
            sh_degree: degree,
        })
    }

    #[cfg(not(feature = "ply"))]
    {
        let _ = bytes;
        Err(IoError::MissingFeature {
            feature: "ply",
            context: "PLY gaussian-splat decoding",
        })
    }
}

/// The `f_rest_*` columns in their own numeric order.
///
/// Sorted by index rather than by name, because `f_rest_10` sorts before
/// `f_rest_2` as text and a set read in that order is a reconstruction whose
/// view-dependent terms are shuffled: it looks right from the angle it was
/// framed at and wrong from every other.
#[cfg(feature = "ply")]
fn rest_columns(cloud: &crate::types::PointSet) -> Result<Vec<&Vec<f32>>, IoError> {
    let mut indexed: Vec<(usize, &Vec<f32>)> = Vec::new();
    for (name, values) in &cloud.scalar_attributes {
        if let Some(suffix) = name.strip_prefix("f_rest_") {
            let index: usize = suffix
                .parse()
                .map_err(|_| IoError::Parse(format!("ply: `{name}` is not a numbered property")))?;
            indexed.push((index, values));
        }
    }
    indexed.sort_by_key(|(index, _)| *index);
    for (place, (index, _)) in indexed.iter().enumerate() {
        if *index != place {
            return Err(IoError::Parse(
                "ply: the `f_rest_*` properties are not numbered from zero without gaps".into(),
            ));
        }
    }
    Ok(indexed.into_iter().map(|(_, values)| values).collect())
}

/// The logistic function, which is what an opacity was stored through.
#[cfg(feature = "ply")]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// A stored quaternion as a unit one in `x, y, z, w` order.
///
/// Two changes at once because they belong together. The file writes `w`
/// first, and it writes whatever the optimiser last held rather than something
/// normalised, so a set taken at face value is both rotated wrongly and scaled
/// by the length of its own rotation.
#[cfg(feature = "ply")]
fn unit([w, x, y, z]: [f32; 4]) -> [f32; 4] {
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n < 1e-12 {
        // No rotation at all rather than a division by nothing. A zero
        // quaternion is not a direction, and the identity is the only answer
        // that leaves the splat's own axes where its scales put them.
        return [0.0, 0.0, 0.0, 1.0];
    }
    [x / n, y / n, z / n, w / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trained splat set as a trainer would write one, in the stored units
    /// rather than the drawn ones.
    ///
    /// `rest` is how many view-dependent coefficients each channel carries, so
    /// zero is a flat-coloured set, three is degree one and fifteen is degree
    /// three.
    fn written(rest: usize) -> String {
        let splats = 2usize;
        let mut out = String::from("ply\nformat ascii 1.0\n");
        out.push_str(&format!("element vertex {splats}\n"));
        for name in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] {
            out.push_str(&format!("property float {name}\n"));
        }
        for i in 0..rest * 3 {
            out.push_str(&format!("property float f_rest_{i}\n"));
        }
        for name in [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ] {
            out.push_str(&format!("property float {name}\n"));
        }
        out.push_str("end_header\n");
        for s in 0..splats {
            let base = s as f32;
            // Position, then the three DC terms.
            out.push_str(&format!("{base} {} {} 1 2 3", base + 1.0, base + 2.0));
            // The rest terms, channel-major: the value encodes which channel
            // and which coefficient it is, so a transpose that went the wrong
            // way is visible rather than plausible.
            for c in 0..3 {
                for k in 0..rest {
                    out.push_str(&format!(" {}", (c * 100 + k) as f32));
                }
            }
            // Opacity as a logit of a half, scales as logarithms of one, and
            // a quaternion written w first and deliberately not normalised.
            out.push_str(" 0 0 0 0 2 0 0 0");
            out.push('\n');
        }
        out
    }

    #[test]
    fn the_stored_units_are_the_ones_a_trainer_writes() {
        let set = splats_from_bytes(written(0).as_bytes()).expect("reads");
        assert_eq!(set.positions.len(), 2);
        // exp(0) is one. A reader taking the number at face value gives every
        // splat an extent of zero, and the whole set disappears.
        assert_eq!(set.scales[0], [1.0, 1.0, 1.0]);
        // sigmoid(0) is a half. At face value every splat is fully opaque,
        // which is the difference between a reconstruction and a wall.
        assert!((set.opacities[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_quaternion_is_reordered_and_normalised() {
        // Written as `w, x, y, z` = `2, 0, 0, 0`: no rotation, at twice unit
        // length. Taken at face value it is `x = 2`, which is a quarter turn
        // about the wrong axis and a scale of two on top of it.
        let set = splats_from_bytes(written(0).as_bytes()).expect("reads");
        assert_eq!(set.rotations[0], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_zero_quaternion_becomes_the_identity_rather_than_a_division() {
        assert_eq!(unit([0.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_flat_set_is_degree_zero_and_carries_only_its_base_colour() {
        let set = splats_from_bytes(written(0).as_bytes()).expect("reads");
        assert_eq!(set.sh_degree, ShDegree::Zero);
        assert_eq!(set.sh_coefficients.len(), 2 * 3);
        assert_eq!(&set.sh_coefficients[0..3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_view_dependent_terms_are_transposed_into_coefficient_order() {
        // The file is channel-major -- all of red's coefficients, then green's,
        // then blue's -- and the packed form is coefficient-major. A copy
        // rather than a transpose gives a set that looks right from one angle
        // and wrong from every other, which reads as a bad training run.
        let set = splats_from_bytes(written(3).as_bytes()).expect("reads");
        assert_eq!(set.sh_degree, ShDegree::One);
        assert_eq!(set.sh_coefficients.len(), 2 * 12);
        assert_eq!(&set.sh_coefficients[0..3], &[1.0, 2.0, 3.0], "the DC terms");
        // Coefficient 0 of each channel: red's is 0, green's is 100, blue's
        // is 200.
        assert_eq!(&set.sh_coefficients[3..6], &[0.0, 100.0, 200.0]);
        // Coefficient 1 of each.
        assert_eq!(&set.sh_coefficients[6..9], &[1.0, 101.0, 201.0]);
        assert_eq!(&set.sh_coefficients[9..12], &[2.0, 102.0, 202.0]);
    }

    #[test]
    fn a_degree_three_set_reads_at_its_own_degree() {
        let set = splats_from_bytes(written(15).as_bytes()).expect("reads");
        assert_eq!(set.sh_degree, ShDegree::Three);
        assert_eq!(set.sh_coefficients.len(), 2 * 48);
    }

    #[test]
    fn the_rest_properties_are_ordered_by_number_rather_than_by_name() {
        // `f_rest_10` sorts before `f_rest_2` as text, so a set read in name
        // order has its coefficients shuffled. Fifteen per channel is where
        // that first bites.
        let set = splats_from_bytes(written(15).as_bytes()).expect("reads");
        // Channel red, coefficient 10, which is `f_rest_10` and holds 10.
        assert_eq!(set.sh_coefficients[3 * (10 + 1)], 10.0);
        assert_eq!(set.sh_coefficients[3 * (10 + 1) + 1], 110.0);
    }

    #[test]
    fn a_degree_this_cannot_pack_is_refused_rather_than_flattened() {
        // Eight per channel is degree two, which the packed form has no shape
        // for. Loading it at degree zero would drop the view-dependent terms
        // and look like a reconstruction that went flat.
        let err = splats_from_bytes(written(8).as_bytes()).expect_err("refused");
        assert!(format!("{err}").contains("24"), "got {err}");
    }

    #[test]
    fn an_ordinary_point_cloud_is_not_read_as_a_splat_set() {
        let plain = "ply\nformat ascii 1.0\nelement vertex 1\n\
                     property float x\nproperty float y\nproperty float z\n\
                     end_header\n0 0 0\n";
        let err = splats_from_bytes(plain.as_bytes()).expect_err("refused");
        assert!(format!("{err}").contains("scale_0"), "got {err}");
    }
}
