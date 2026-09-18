use std::path::{Path, PathBuf};

use quick_xml::Reader;
use quick_xml::events::Event;

use super::error::ReadError;

/// A single timestep entry from a `.pvd` collection file.
#[derive(Clone, Debug)]
pub struct TimestepEntry {
    /// Physical simulation time for this step.
    pub time: f64,
    /// Absolute path to the VTK data file for this step.
    pub file: PathBuf,
    /// Optional format-specific selector for timestep data within `file`.
    ///
    /// Used by container formats like XDMF where multiple timesteps can live in a
    /// single XML file.
    pub selector: Option<String>,
}

/// A parsed PVD series containing all timestep entries, sorted by time.
#[derive(Clone, Debug)]
pub struct PvdSeries {
    /// Timestep entries in the series, sorted by time.
    pub timesteps: Vec<TimestepEntry>,
}

/// Parse a `.pvd` XML collection file and return the series metadata.
///
/// File paths inside the PVD are resolved relative to the PVD file's parent directory.
/// The returned `timesteps` are sorted ascending by `time`.
pub fn read_pvd(path: &Path) -> Result<PvdSeries, ReadError> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let file = std::fs::File::open(path)?;
    let buf_reader = std::io::BufReader::new(file);

    let mut reader = Reader::from_reader(buf_reader);
    reader.config_mut().trim_text(true);

    let mut timesteps: Vec<TimestepEntry> = Vec::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) | Ok(Event::Start(ref e)) => {
                if e.local_name().as_ref() == b"DataSet" {
                    let mut time_val: Option<f64> = None;
                    let mut file_val: Option<String> = None;

                    for attr_result in e.attributes() {
                        let attr = attr_result
                            .map_err(|err| ReadError::Pvd(format!("XML attribute error: {err}")))?;
                        match attr.key.local_name().as_ref() {
                            b"timestep" => {
                                let v = attr.unescape_value().map_err(|err| {
                                    ReadError::Pvd(format!("timestep unescape error: {err}"))
                                })?;
                                time_val = v.parse::<f64>().ok();
                            }
                            b"file" => {
                                let v = attr.unescape_value().map_err(|err| {
                                    ReadError::Pvd(format!("file unescape error: {err}"))
                                })?;
                                file_val = Some(v.into_owned());
                            }
                            _ => {}
                        }
                    }

                    if let (Some(time), Some(rel_file)) = (time_val, file_val) {
                        let file_path = if Path::new(&rel_file).is_absolute() {
                            PathBuf::from(rel_file)
                        } else {
                            parent.join(&rel_file)
                        };
                        timesteps.push(TimestepEntry {
                            time,
                            file: file_path,
                            selector: None,
                        });
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(ReadError::Pvd(format!("XML parse error: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    if timesteps.is_empty() {
        return Err(ReadError::Pvd(
            "PVD file contains no DataSet entries".to_string(),
        ));
    }

    timesteps.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(PvdSeries { timesteps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_io_testkit::synth;

    fn collection(entries: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\">\n  \
             <Collection>\n{entries}  </Collection>\n</VTKFile>\n"
        )
    }

    /// A writer is free to list its timesteps in any order, and several do.
    /// The series is sorted by time, because every consumer indexes it as a
    /// timeline.
    #[test]
    fn entries_come_back_sorted_by_time() {
        let path = synth::temp_path("pvd_sorted", "series.pvd");
        synth::write(
            &path,
            collection(
                "    <DataSet timestep=\"2.5\" file=\"c.vtu\"/>\n\
                 \x20   <DataSet timestep=\"0.0\" file=\"a.vtu\"/>\n\
                 \x20   <DataSet timestep=\"1.0\" file=\"b.vtu\"/>\n",
            ),
        );

        let series = read_pvd(&path).expect("read pvd");
        let times: Vec<f64> = series.timesteps.iter().map(|t| t.time).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.5]);

        let names: Vec<&str> = series
            .timesteps
            .iter()
            .map(|t| t.file.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.vtu", "b.vtu", "c.vtu"]);
    }

    /// A relative `file` is resolved against the collection's own directory,
    /// not the process working directory, so a series opened from anywhere
    /// still points at its data.
    #[test]
    fn relative_paths_resolve_against_the_collection() {
        let dir = synth::temp_dir("pvd_relative");
        let path = dir.join("series.pvd");
        synth::write(
            &path,
            collection("    <DataSet timestep=\"0\" file=\"steps/step0.vtu\"/>\n"),
        );

        let series = read_pvd(&path).expect("read pvd");
        assert_eq!(series.timesteps[0].file, dir.join("steps/step0.vtu"));
    }

    /// An absolute `file` is taken as written rather than joined onto the
    /// collection's directory.
    #[test]
    fn absolute_paths_are_left_alone() {
        let path = synth::temp_path("pvd_absolute", "series.pvd");
        synth::write(
            &path,
            collection("    <DataSet timestep=\"0\" file=\"/data/step0.vtu\"/>\n"),
        );

        let series = read_pvd(&path).expect("read pvd");
        assert_eq!(series.timesteps[0].file, PathBuf::from("/data/step0.vtu"));
    }

    /// An entry missing either attribute names no data at no time, so it is
    /// skipped rather than landing in the timeline as a zero-time placeholder.
    #[test]
    fn incomplete_entries_are_skipped() {
        let path = synth::temp_path("pvd_incomplete", "series.pvd");
        synth::write(
            &path,
            collection(
                "    <DataSet timestep=\"0\"/>\n\
                 \x20   <DataSet file=\"orphan.vtu\"/>\n\
                 \x20   <DataSet timestep=\"1\" file=\"good.vtu\"/>\n",
            ),
        );

        let series = read_pvd(&path).expect("read pvd");
        assert_eq!(series.timesteps.len(), 1);
        assert_eq!(series.timesteps[0].time, 1.0);
    }

    /// A collection with no usable entry reports rather than handing back an
    /// empty timeline a caller would index into.
    #[test]
    fn an_empty_collection_is_an_error() {
        let path = synth::temp_path("pvd_empty", "series.pvd");
        synth::write(&path, collection(""));

        assert!(read_pvd(&path).is_err());
    }

    /// A missing file reports through the same error type as a parse failure.
    #[test]
    fn a_missing_file_is_an_error() {
        let path = synth::temp_path("pvd_missing", "not-here.pvd");
        assert!(read_pvd(&path).is_err());
    }
}
