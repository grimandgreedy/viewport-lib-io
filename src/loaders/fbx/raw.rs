//! Format plumbing: typed reads out of FBX nodes and properties, and the stable
//! document object order every walk uses.

use fbxcel_dom::fbxcel;
use fbxcel_dom::v7400::Document;
use fbxcel_dom::v7400::object::TypedObjectHandle;

pub(super) fn read_vec3_property(
    props: &fbxcel_dom::v7400::object::property::PropertiesHandle<'_>,
    name: &str,
) -> Option<glam::Vec3> {
    let property = props.get_property(name)?;
    let values = property.value_part();
    if values.len() < 3 {
        return None;
    }
    let x = attr_to_f64(&values[0])?;
    let y = attr_to_f64(&values[1])?;
    let z = attr_to_f64(&values[2])?;
    Some(glam::Vec3::new(x as f32, y as f32, z as f32))
}

pub(super) fn read_int_property(
    props: &fbxcel_dom::v7400::object::property::PropertiesHandle<'_>,
    name: &str,
) -> Option<i32> {
    let property = props.get_property(name)?;
    let values = property.value_part();
    values.first().and_then(|value| match value {
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i),
        fbxcel::low::v7400::AttributeValue::I16(i) => Some(*i as i32),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i as i32),
        _ => None,
    })
}

pub(super) fn attr_to_f64(value: &fbxcel::low::v7400::AttributeValue) -> Option<f64> {
    match value {
        fbxcel::low::v7400::AttributeValue::F64(f) => Some(*f),
        fbxcel::low::v7400::AttributeValue::F32(f) => Some(*f as f64),
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i as f64),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i as f64),
        _ => None,
    }
}

/// The document's objects in a stable order: ascending FBX object id.
///
/// `Document::objects()` yields its keys from a `HashMap`, so the order varies
/// between decodes of the same file, even within one process. Anything that
/// appends to an ordered output walks this instead, so a positional index
/// recorded against one decode still addresses the same thing in the next.
/// Object ids come from the file, so the order is also the same across runs and
/// machines, and it follows the file's own numbering rather than being
/// arbitrary.
///
/// Code that only looks an object up, or fills a map keyed by object id, does
/// not need this.
#[cfg(feature = "fbx")]
pub(super) fn objects_in_stable_order(
    document: &Document,
) -> Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> {
    let mut objects: Vec<_> = document.objects().collect();
    objects.sort_by_key(|object| object.object_id().raw());
    objects
}

#[cfg(feature = "fbx")]
pub(super) fn lookup_object<'a>(
    document: &'a Document,
    object_id: i64,
) -> Option<fbxcel_dom::v7400::object::ObjectHandle<'a>> {
    document
        .objects()
        .find(|o| o.object_id().raw() == object_id)
}

#[cfg(feature = "fbx")]
pub(super) fn parent_model_id(obj: &fbxcel_dom::v7400::object::ObjectHandle<'_>) -> Option<i64> {
    use fbxcel_dom::v7400::object::model::TypedModelHandle as M;
    let model = match obj.get_typed() {
        TypedObjectHandle::Model(M::LimbNode(n)) => n.parent_model()?,
        TypedObjectHandle::Model(M::Null(n)) => n.parent_model()?,
        _ => return None,
    };
    // TypedModelHandle derefs to ModelHandle -> ObjectHandle, so the chained
    // `**` reaches the ObjectHandle and we can read its raw id.
    let object_id_raw = match model {
        M::LimbNode(n) => (**n).object_id().raw(),
        M::Null(n) => (**n).object_id().raw(),
        M::Mesh(n) => (**n).object_id().raw(),
        M::Camera(n) => (**n).object_id().raw(),
        M::Light(n) => (**n).object_id().raw(),
        _ => return None,
    };
    Some(object_id_raw)
}

#[cfg(feature = "fbx")]
pub(super) fn read_i32_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<i32>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrI32(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrI64(v) => {
            Some(v.iter().map(|&i| i as i32).collect())
        }
        _ => None,
    }
}

#[cfg(feature = "fbx")]
pub(super) fn read_f64_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<f64>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrF64(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrF32(v) => {
            Some(v.iter().map(|&f| f as f64).collect())
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Animation extraction
// ---------------------------------------------------------------------------

#[cfg(feature = "fbx")]
pub(super) fn read_i64_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<i64>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrI64(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrI32(v) => {
            Some(v.iter().map(|&i| i as i64).collect())
        }
        _ => None,
    }
}

#[cfg(feature = "fbx")]
pub(super) fn read_f32_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<f32>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrF32(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrF64(v) => {
            Some(v.iter().map(|&f| f as f32).collect())
        }
        _ => None,
    }
}

#[cfg(feature = "fbx")]
pub(super) fn read_mat4(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<glam::Mat4> {
    let values = read_f64_array(node, name)?;
    if values.len() < 16 {
        return None;
    }
    // FBX stores matrices column-major.
    let m: [f32; 16] = std::array::from_fn(|i| values[i] as f32);
    Some(glam::Mat4::from_cols_array(&m))
}
