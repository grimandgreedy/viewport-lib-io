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
