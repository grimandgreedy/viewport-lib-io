//! Dump the blend shapes (morph targets) an FBX carries, per sub-mesh.
//!
//! ```sh
//! cargo run --example probe-fbx-morph --features fbx -- path/to/character.fbx
//! ```
//!
//! Prints each sub-mesh's vertex count and the name + non-zero delta count of
//! every morph target on it. A character with an ARKit facial rig lists the
//! familiar `jawOpen` / `mouthSmileLeft` / `eyeBlinkLeft` set on its head mesh.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe-fbx-morph <file.fbx>");
    let scene = viewport_lib_io::loaders::fbx::scene_from_path(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("decode {path}: {e:?}"));

    let mut total_targets = 0usize;
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
            total_targets += 1;
            if moved > 0 {
                println!("    {:<28} {moved} verts moved", t.name);
            }
        }
    }
    println!(
        "\n{} meshes, {} morph-target entries across the scene",
        scene.meshes.len(),
        total_targets,
    );

    // Raw census: what object kinds does fbxcel-dom recognise? Reveals whether
    // BlendShape deformers / Shape geometries are present at all.
    census(&path);
}

fn census(path: &str) {
    use fbxcel_dom::v7400::object::{
        TypedObjectHandle, deformer::TypedDeformerHandle, deformer::TypedSubDeformerHandle,
        geometry::TypedGeometryHandle,
    };
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let doc = match fbxcel_dom::any::AnyDocument::from_seekable_reader(reader).unwrap() {
        fbxcel_dom::any::AnyDocument::V7400(_, doc) => doc,
        _ => panic!("not a v7400 FBX"),
    };
    let (mut mesh, mut shape, mut skin, mut cluster, mut blend, mut chan, mut other) =
        (0, 0, 0, 0, 0, 0, 0);
    for obj in doc.objects() {
        match obj.get_typed() {
            TypedObjectHandle::Geometry(TypedGeometryHandle::Mesh(_)) => mesh += 1,
            TypedObjectHandle::Geometry(TypedGeometryHandle::Shape(_)) => shape += 1,
            TypedObjectHandle::Deformer(TypedDeformerHandle::Skin(_)) => skin += 1,
            TypedObjectHandle::Deformer(TypedDeformerHandle::BlendShape(_)) => blend += 1,
            TypedObjectHandle::SubDeformer(TypedSubDeformerHandle::Cluster(_)) => cluster += 1,
            TypedObjectHandle::SubDeformer(TypedSubDeformerHandle::BlendShapeChannel(_)) => {
                chan += 1
            }
            _ => other += 1,
        }
    }
    println!(
        "census: {mesh} mesh geom, {shape} shape geom, {skin} skin, {cluster} cluster, {blend} blendshape, {chan} channel, {other} other",
    );

    // Distinct (class, subclass) pairs with counts: reveals how this file
    // labels its objects, so a mislabelled blend shape shows up here.
    let mut pairs: std::collections::BTreeMap<(String, String), usize> =
        std::collections::BTreeMap::new();
    for obj in doc.objects() {
        *pairs
            .entry((obj.class().to_string(), obj.subclass().to_string()))
            .or_insert(0) += 1;
    }
    println!("--- (class, subclass) census ---");
    for ((class, sub), n) in &pairs {
        println!("  {class:<16} {sub:<20} {n}");
    }
}
