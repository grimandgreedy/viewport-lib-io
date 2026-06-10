// Detailed FBX probe: GlobalSettings, full parent chains for every Mesh
// (including Null / LimbNode ancestors and root), every TRS property,
// plus raw vertex bounding box BEFORE any transform is applied.
//
// cargo run --example probe-fbx-axis --features fbx -- <path1.fbx> ...

use std::io::BufReader;
use std::path::Path;

use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::fbxcel;
use fbxcel_dom::v7400::object::model::{ModelHandle, TypedModelHandle};
use fbxcel_dom::v7400::object::TypedObjectHandle;
use fbxcel_dom::v7400::data::mesh::{PolygonVertexIndex, PolygonVertices};

fn fan_tri(
    pvs: &PolygonVertices<'_>,
    indices: &[PolygonVertexIndex],
    out: &mut Vec<[PolygonVertexIndex; 3]>,
) -> Result<(), anyhow::Error> {
    let _ = pvs;
    if indices.len() < 3 { return Ok(()); }
    for i in 1..indices.len() - 1 {
        out.push([indices[0], indices[i], indices[i + 1]]);
    }
    Ok(())
}

fn attr_f64(v: &fbxcel::low::v7400::AttributeValue) -> Option<f64> {
    match v {
        fbxcel::low::v7400::AttributeValue::F64(f) => Some(*f),
        fbxcel::low::v7400::AttributeValue::F32(f) => Some(*f as f64),
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i as f64),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i as f64),
        _ => None,
    }
}

fn attr_i64(v: &fbxcel::low::v7400::AttributeValue) -> Option<i64> {
    match v {
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i as i64),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i),
        fbxcel::low::v7400::AttributeValue::I16(i) => Some(*i as i64),
        _ => None,
    }
}

fn props_str(model: &ModelHandle<'_>) -> String {
    let Some(props) = model.direct_properties() else {
        return String::new();
    };
    let mut row = String::new();
    let triplets = [
        "Lcl Translation", "Lcl Rotation", "Lcl Scaling",
        "PreRotation", "PostRotation",
        "RotationOffset", "RotationPivot",
        "ScalingOffset", "ScalingPivot",
        "GeometricTranslation", "GeometricRotation", "GeometricScaling",
    ];
    for n in triplets {
        if let Some(p) = props.get_property(n) {
            let vals = p.value_part();
            if vals.len() >= 3 {
                let x = attr_f64(&vals[0]).unwrap_or(0.0);
                let y = attr_f64(&vals[1]).unwrap_or(0.0);
                let z = attr_f64(&vals[2]).unwrap_or(0.0);
                if x != 0.0 || y != 0.0 || z != 0.0
                    || n == "Lcl Scaling"
                    || n == "GeometricScaling"
                {
                    // only print non-identity (interesting) properties; scaling
                    // is shown even if 1,1,1 only when explicitly set
                    if (n == "Lcl Scaling" || n == "GeometricScaling")
                        && (x - 1.0).abs() < 1e-6 && (y - 1.0).abs() < 1e-6 && (z - 1.0).abs() < 1e-6
                    {
                        continue;
                    }
                    row.push_str(&format!(" {n}=({x:.4},{y:.4},{z:.4})"));
                }
            }
        }
    }
    if let Some(p) = props.get_property("RotationOrder") {
        if let Some(v) = p.value_part().first().and_then(attr_i64) {
            if v != 0 {
                row.push_str(&format!(" RotationOrder={v}"));
            }
        }
    }
    row
}

fn print_parent_chain(label: &str, mesh: &ModelHandle<'_>) {
    // Build full chain mesh -> root
    let mut chain: Vec<(String, String, String)> = Vec::new();
    let name = mesh.name().unwrap_or("<unnamed>").to_string();
    chain.push(("Mesh".into(), name, props_str(mesh)));
    let mut current: Option<TypedModelHandle<'_>> = mesh.parent_model();
    let mut guard = 0;
    while let Some(parent) = current {
        guard += 1;
        if guard > 64 { break; }
        let (kind, p) = match &parent {
            TypedModelHandle::Mesh(m) => ("Mesh", (**m).clone()),
            TypedModelHandle::Null(n) => ("Null", (**n).clone()),
            TypedModelHandle::LimbNode(n) => ("LimbNode", (**n).clone()),
            TypedModelHandle::Light(l) => ("Light", (**l).clone()),
            TypedModelHandle::Camera(c) => ("Camera", (**c).clone()),
            _ => break,
        };
        let n = p.name().unwrap_or("<unnamed>").to_string();
        chain.push((kind.into(), n, props_str(&p)));
        current = match parent {
            TypedModelHandle::Mesh(m) => m.parent_model(),
            TypedModelHandle::Null(n) => n.parent_model(),
            TypedModelHandle::LimbNode(n) => n.parent_model(),
            TypedModelHandle::Light(l) => l.parent_model(),
            TypedModelHandle::Camera(c) => c.parent_model(),
            _ => None,
        };
    }
    println!("  {label} chain (leaf -> root, depth {}):", chain.len());
    for (i, (k, n, r)) in chain.iter().enumerate() {
        println!("    [{i}] {k} '{n}'{}", if r.is_empty() { String::new() } else { r.clone() });
    }
}

fn probe(path: &Path) {
    println!("=== {} ===", path.display());
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => { println!("  open failed: {e}"); return; }
    };
    let reader = BufReader::new(file);
    let doc = match AnyDocument::from_seekable_reader(reader) {
        Ok(AnyDocument::V7400(_, d)) => d,
        Ok(_) => { println!("  unsupported FBX version"); return; }
        Err(e) => { println!("  parse failed: {e:?}"); return; }
    };

    // GlobalSettings
    if let Some(settings) = doc.global_settings() {
        let props = settings.raw_properties();
        for n in [
            "UpAxis", "UpAxisSign",
            "FrontAxis", "FrontAxisSign",
            "CoordAxis", "CoordAxisSign",
            "OriginalUpAxis", "OriginalUpAxisSign",
            "UnitScaleFactor", "OriginalUnitScaleFactor",
        ] {
            let v = props.get_property(n).and_then(|p| {
                p.value_part().first().and_then(attr_f64).map(|f| format!("{f}"))
            });
            println!("  GS {n}: {}", v.unwrap_or_else(|| "<absent>".into()));
        }
    }

    // For each Mesh: print parent chain + raw vertex bbox.
    for obj in doc.objects() {
        if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh)) = obj.get_typed() {
            let name = mesh.name().unwrap_or("<unnamed>").to_string();
            println!("\n  Mesh '{name}':");
            print_parent_chain("parent", &mesh);

            // Raw vertex bbox (walk triangulated control points)
            if let Ok(geom) = mesh.geometry() {
                if let Ok(pv) = geom.polygon_vertices() {
                    if let Ok(tri) = pv.triangulate_each(fan_tri) {
                        let mut min = [f64::INFINITY; 3];
                        let mut max = [f64::NEG_INFINITY; 3];
                        let mut count = 0usize;
                        for ti in tri.triangle_vertex_indices() {
                            if let Some(p) = tri.control_point(ti) {
                                for k in 0..3 {
                                    let v = [p.x, p.y, p.z][k];
                                    if v < min[k] { min[k] = v; }
                                    if v > max[k] { max[k] = v; }
                                }
                                count += 1;
                            }
                        }
                        if count > 0 {
                            let ext = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];
                            println!("    raw cp bbox: min=({:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3}) ext=({:.3},{:.3},{:.3}) n={count}",
                                min[0],min[1],min[2], max[0],max[1],max[2], ext[0],ext[1],ext[2]);
                        }
                    }
                }
            }
        }
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe-fbx-axis <file.fbx> [more.fbx ...]");
        std::process::exit(1);
    }
    for a in args { probe(Path::new(&a)); }
}
