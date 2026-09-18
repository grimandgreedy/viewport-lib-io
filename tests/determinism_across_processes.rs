//! Decoding is reproducible across processes, not only within one.
//!
//! This is the check the conformance floor cannot make. Rust seeds each
//! process's `HashMap` hasher differently, so a loader that walks a map to build
//! ordered output agrees with itself all day inside one process and disagrees
//! with the next run. Both bugs of that shape found here hid behind exactly
//! that: the FBX mesh order, and the VTK surface extractor.
//!
//! The test re-runs itself as a child process several times and compares what
//! each one decoded, loader by loader, over whatever fixtures the enabled
//! features can author.

#![cfg(any(
    feature = "obj",
    feature = "ply",
    feature = "stl",
    feature = "msh",
    feature = "vtk",
    feature = "csv",
    feature = "numpy",
    feature = "svg",
    feature = "gltf",
    feature = "fbx"
))]

use std::fmt::Write as _;

/// One line per loader: its name and a projection of what it decoded, chosen so
/// a reordering shows up as a difference.
fn report() -> String {
    #[allow(unused_mut)]
    let mut out = String::new();

    #[cfg(any(
        feature = "obj",
        feature = "ply",
        feature = "stl",
        feature = "msh",
        feature = "vtk",
        feature = "csv",
        feature = "numpy",
        feature = "svg",
        feature = "gltf"
    ))]
    let dir = viewport_lib_io_testkit::synth::temp_dir("determinism_processes");

    #[cfg(any(
        feature = "obj",
        feature = "ply",
        feature = "stl",
        feature = "msh",
        feature = "vtk",
        feature = "csv",
        feature = "numpy",
        feature = "svg"
    ))]
    let write = |name: &str, bytes: &[u8]| {
        let path = dir.join(name);
        viewport_lib_io_testkit::synth::write(&path, bytes);
        path
    };

    #[cfg(feature = "obj")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write("quad.obj", synth::obj_quad(None).as_bytes());
        let scene = viewport_lib_io::loaders::obj::scene_from_path(&path).expect("obj");
        let _ = writeln!(out, "obj {:?}", scene_digest(&scene));
    }

    #[cfg(feature = "ply")]
    {
        use viewport_lib_io_testkit::synth;
        let mesh = write("quad.ply", synth::ply_ascii_quad(true).as_bytes());
        let scene = viewport_lib_io::loaders::ply_scene::scene_from_path(&mesh).expect("ply");
        let _ = writeln!(out, "ply_scene {:?}", scene_digest(&scene));

        let cloud = write("cloud.ply", synth::ply_ascii_points(true).as_bytes());
        let points = viewport_lib_io::loaders::ply_points::point_cloud_from_path(&cloud)
            .expect("ply points");
        let _ = writeln!(out, "ply_points {:?}", points.positions);
    }

    #[cfg(feature = "stl")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write("quad.stl", synth::stl_binary_quad().as_slice());
        let mesh = viewport_lib_io::loaders::stl::mesh_from_path(&path).expect("stl");
        let _ = writeln!(out, "stl {:?} {:?}", mesh.positions, mesh.indices);
    }

    #[cfg(feature = "msh")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write("quad.msh", synth::msh_v4_quad().as_bytes());
        let mesh = viewport_lib_io::loaders::msh::mesh_from_path(&path).expect("msh");
        let _ = writeln!(out, "msh {:?} {:?}", mesh.positions, mesh.indices);
    }

    #[cfg(feature = "vtk")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write("quad.vtk", synth::vtk_legacy_quad("density").as_bytes());
        let sets = viewport_lib_io::loaders::vtk::datasets_from_path(&path).expect("vtk");
        for set in &sets {
            let _ = writeln!(
                out,
                "vtk {} {:?} {:?}",
                set.name,
                set.as_surface_mesh().map(|m| m.indices.clone()),
                set.as_surface_mesh()
                    .map(|m| m.attributes.keys().cloned().collect::<Vec<_>>())
            );
        }
    }

    #[cfg(feature = "csv")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write("points.csv", synth::csv_points(8, "density").as_bytes());
        let points = viewport_lib_io::loaders::csv::point_cloud_from_path(&path).expect("csv");
        let _ = writeln!(out, "csv {:?}", points.positions);
    }

    #[cfg(feature = "numpy")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write(
            "volume.npy",
            synth::npy_f32(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).as_slice(),
        );
        let volume = viewport_lib_io::loaders::numpy::volume_from_path(&path).expect("npy");
        let mut fields: Vec<&String> = volume.point_fields.keys().collect();
        fields.sort();
        let _ = writeln!(out, "numpy {:?} {fields:?}", volume.dims);
    }

    #[cfg(feature = "svg")]
    {
        use viewport_lib_io_testkit::synth;
        let path = write(
            "art.svg",
            synth::svg_filled_path("M 0 0 L 10 0 L 10 10 Z", "#ff0000").as_bytes(),
        );
        let art = viewport_lib_io::loaders::svg::vector_from_path(&path).expect("svg");
        let _ = writeln!(
            out,
            "svg {} {:?}",
            art.shapes.len(),
            art.shapes.iter().map(|s| s.fill).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "gltf")]
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fox.glb");
        if path.is_file() {
            let scene = viewport_lib_io::loaders::gltf::scene_from_path(&path).expect("glb");
            let _ = writeln!(
                out,
                "gltf {:?} {:?} {:?}",
                scene
                    .meshes
                    .iter()
                    .map(|m| m.name.clone())
                    .collect::<Vec<_>>(),
                scene
                    .animations
                    .iter()
                    .map(|a| (a.name.clone(), a.skeleton_index))
                    .collect::<Vec<_>>(),
                scene
                    .skeletons
                    .iter()
                    .map(|s| s.joints.iter().map(|j| j.name.clone()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[cfg(feature = "fbx")]
    {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fox_rigged.fbx");
        if path.is_file() {
            let scene = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("fbx");
            let _ = writeln!(
                out,
                "fbx {:?} {:?}",
                scene
                    .meshes
                    .iter()
                    .map(|m| m.name.clone())
                    .collect::<Vec<_>>(),
                scene
                    .skeletons
                    .iter()
                    .map(|s| s.joints.iter().map(|j| j.name.clone()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
        }
    }

    out
}

#[cfg(any(feature = "obj", feature = "ply"))]
fn scene_digest(scene: &viewport_lib_io::types::SceneData) -> Vec<(String, usize, Vec<u32>)> {
    scene
        .meshes
        .iter()
        .map(|m| {
            (
                m.name.clone(),
                m.mesh.positions.len(),
                m.mesh.indices.clone(),
            )
        })
        .collect()
}

#[test]
fn every_enabled_loader_decodes_the_same_in_a_fresh_process() {
    const MARKER: &str = "VIEWPORT_LIB_IO_DETERMINISM_CHILD";
    const TEST: &str = "every_enabled_loader_decodes_the_same_in_a_fresh_process";

    if std::env::var_os(MARKER).is_some() {
        println!("REPORT_BEGIN\n{}REPORT_END", report());
        return;
    }

    let mine = report();
    assert!(
        !mine.trim().is_empty(),
        "no loader feature is enabled, so this run checked nothing"
    );

    let exe = std::env::current_exe().expect("this test binary");
    for run in 0..4 {
        let output = std::process::Command::new(&exe)
            .args(["--exact", TEST, "--nocapture"])
            .env(MARKER, "1")
            .output()
            .expect("re-run this test as a child process");
        assert!(
            output.status.success(),
            "child run {run} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let theirs = stdout
            .split_once("REPORT_BEGIN\n")
            .and_then(|(_, rest)| rest.split_once("REPORT_END"))
            .map(|(body, _)| body.to_string())
            .unwrap_or_else(|| panic!("child run {run} printed no report:\n{stdout}"));

        assert_eq!(
            theirs, mine,
            "child run {run} decoded the same files differently"
        );
    }
}
