//! glTF stores its scenes Y-up and this crate hands out Z-up, so every
//! orientation-bearing value is rotated once, here, during decoding. Positions,
//! normals, tangents, transforms, inverse binds, and animation samples all pass
//! through these helpers, which is what lets a consumer skip re-rotating
//! anything downstream.

use crate::types::IoMesh;

/// +90 degree rotation about X as a Mat4.
pub(super) const Y_UP_TO_Z_UP: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
    glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Inverse of [`Y_UP_TO_Z_UP`]: -90 degrees about X, i.e. its transpose.
pub(super) const Y_UP_TO_Z_UP_INV: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, -1.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Rotate a position or direction vector from Y-up into Z-up:
/// `(x, y, z) -> (x, -z, y)`.
pub(super) fn reorient_vec3(v: [f32; 3]) -> [f32; 3] {
    [v[0], -v[2], v[1]]
}

/// Rotate a tangent vec4: xyz is direction, w is bitangent sign and stays.
pub(super) fn reorient_tangent(t: [f32; 4]) -> [f32; 4] {
    [t[0], -t[2], t[1], t[3]]
}

/// Conjugate an affine transform by the Y-up to Z-up rotation: `R * M * R^-1`.
pub(super) fn reorient_affine_mat4(m: glam::Mat4) -> glam::Mat4 {
    Y_UP_TO_Z_UP * m * Y_UP_TO_Z_UP_INV
}

/// Y-up to Z-up rotation as a quaternion (used for animation rotation tracks).
pub(super) fn y_up_to_z_up_quat() -> glam::Quat {
    glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

/// Conjugate a unit quaternion by the Y-up to Z-up rotation: `R * q * R^-1`.
pub(super) fn reorient_quat(q: glam::Quat) -> glam::Quat {
    let r = y_up_to_z_up_quat();
    r * q * r.conjugate()
}

/// Permute the components of a per-axis scale vector under the X-axis +90
/// rotation. Anisotropic scales remain correct because the rotation is
/// axis-aligned; arbitrary-axis non-uniform scale is not representable as a
/// scale vector in any frame.
pub(super) fn reorient_scale(s: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::new(s.x, s.z, s.y)
}

/// Rotate a single [`IoMesh`]'s vertex attributes and world transform into
/// Z-up. Skin weights are indices and per-vertex scalars, so they need no
/// reorientation.
pub(super) fn reorient_mesh_z_up(mesh: &mut IoMesh) {
    for p in &mut mesh.mesh.positions {
        *p = reorient_vec3(*p);
    }
    for n in &mut mesh.mesh.normals {
        *n = reorient_vec3(*n);
    }
    if let Some(tangents) = mesh.mesh.tangents.as_mut() {
        for t in tangents.iter_mut() {
            *t = reorient_tangent(*t);
        }
    }
    // Morph displacements live in the same space as the base attributes, so
    // they rotate by the same Y-up to Z-up transform. The rotation is linear,
    // so a displacement reorients exactly like a position. Tangent deltas are
    // xyz-only (no bitangent sign), so reuse the vec3 rotation.
    for target in &mut mesh.mesh.morph_targets {
        for d in &mut target.position_deltas {
            *d = reorient_vec3(*d);
        }
        if let Some(normals) = target.normal_deltas.as_mut() {
            for n in normals.iter_mut() {
                *n = reorient_vec3(*n);
            }
        }
        if let Some(tangents) = target.tangent_deltas.as_mut() {
            for t in tangents.iter_mut() {
                *t = reorient_vec3(*t);
            }
        }
    }
    mesh.transform = reorient_affine_mat4(mesh.transform);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_y_axis_maps_to_z_axis() {
        let v = reorient_vec3([0.0, 1.0, 0.0]);
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_z_axis_maps_to_negative_y_axis() {
        let v = reorient_vec3([0.0, 0.0, 1.0]);
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1] + 1.0).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn vec3_x_axis_unchanged() {
        let v = reorient_vec3([1.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn tangent_xyz_rotates_but_w_preserved() {
        let t = reorient_tangent([0.0, 1.0, 0.0, -1.0]);
        assert!((t[2] - 1.0).abs() < 1e-6);
        assert!((t[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn quat_y_axis_rotation_becomes_z_axis_rotation() {
        let q_y_up = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let q_z_up = reorient_quat(q_y_up);
        let expected = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!(q_z_up.abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn quat_x_axis_rotation_unchanged() {
        let q = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let r = reorient_quat(q);
        assert!(r.abs_diff_eq(q, 1e-5));
    }

    #[test]
    fn scale_swaps_y_and_z_components() {
        let s = reorient_scale(glam::Vec3::new(2.0, 3.0, 5.0));
        assert_eq!(s, glam::Vec3::new(2.0, 5.0, 3.0));
    }

    #[test]
    fn affine_translation_along_y_lands_along_z() {
        let m = glam::Mat4::from_translation(glam::Vec3::Y * 2.0);
        let m_z = reorient_affine_mat4(m);
        let p = m_z.transform_point3(glam::Vec3::ZERO);
        assert!((p - glam::Vec3::Z * 2.0).length() < 1e-5);
    }

    #[test]
    fn affine_inverse_pair_is_identity() {
        let product = Y_UP_TO_Z_UP * Y_UP_TO_Z_UP_INV;
        let i = glam::Mat4::IDENTITY;
        for c in 0..4 {
            for r in 0..4 {
                assert!((product.col(c)[r] - i.col(c)[r]).abs() < 1e-6);
            }
        }
    }
}
