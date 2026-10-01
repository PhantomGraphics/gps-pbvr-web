//! 3DGS PLY loader (ASCII and binary little-endian, SH degree 0..=3).
//!
//! Interpretation matches `GSPointCloud::readStandard` in the C++ code base:
//! `scale_*` are log-scales, `opacity` is a logit, `rot_0..3` is (w,x,y,z),
//! `f_rest_*` are channel-major. Activations are applied by [`Gaussians::activated`],
//! never on load, so the file values stay inspectable.
//!
//! No panics on bad input: everything returns [`Result`].

use glam::{DQuat, DVec3};
use std::fmt;

/// Hard cap against absurd header counts (memory-bomb protection).
pub const MAX_VERTICES: usize = 200_000_000;

#[derive(Debug, PartialEq)]
pub enum PlyError {
    NotPly,
    BadHeader(String),
    UnsupportedFormat(String),
    MissingProperty(&'static str),
    TooManyVertices(usize),
    Truncated { expected: usize, got: usize },
    BadAscii { vertex: usize, msg: String },
    NonFinite { vertex: usize, field: &'static str },
    BadShCount(usize),
}

impl fmt::Display for PlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PlyError::*;
        match self {
            NotPly => write!(f, "not a PLY file"),
            BadHeader(m) => write!(f, "bad PLY header: {m}"),
            UnsupportedFormat(m) => write!(f, "unsupported PLY format: {m}"),
            MissingProperty(p) => write!(f, "missing required property '{p}'"),
            TooManyVertices(n) => write!(f, "vertex count {n} exceeds limit {MAX_VERTICES}"),
            Truncated { expected, got } => {
                write!(f, "file truncated: need {expected} bytes of vertex data, have {got}")
            }
            BadAscii { vertex, msg } => write!(f, "ASCII parse error at vertex {vertex}: {msg}"),
            NonFinite { vertex, field } => write!(f, "non-finite value in '{field}' at vertex {vertex}"),
            BadShCount(n) => write!(f, "{n} f_rest_* properties is not 9, 24 or 45"),
        }
    }
}
impl std::error::Error for PlyError {}

/// Parsed Gaussians in file (pre-activation) form, struct-of-arrays.
#[derive(Debug, Default, Clone)]
pub struct Gaussians {
    pub count: usize,
    /// 0..=3
    pub sh_degree: usize,
    pub position: Vec<[f32; 3]>,
    /// log of std-dev per axis (file value)
    pub log_scale: Vec<[f32; 3]>,
    /// (w, x, y, z), not normalised (file value)
    pub rot_wxyz: Vec<[f32; 4]>,
    /// logit (file value)
    pub opacity_logit: Vec<f32>,
    pub f_dc: Vec<[f32; 3]>,
    /// `count * 3 * sh_rest_per_channel`, per-splat channel-major (R.., G.., B..).
    pub sh_rest: Vec<f32>,
}

/// One splat after activation, in f64 (oracle form).
#[derive(Debug, Clone, Copy)]
pub struct ActivatedGaussian {
    pub position: DVec3,
    /// log-scale kept (covariance uses exp(2*log)), so only opacity/rotation are activated.
    pub log_scale: DVec3,
    pub rotation: DQuat,
    pub opacity: f64,
    /// linear SH0 colour = f_dc*C0 + 0.5 (unclamped)
    pub dc: DVec3,
}

impl Gaussians {
    pub fn sh_rest_per_channel(&self) -> usize {
        gps_core::sh_rest_per_channel(self.sh_degree)
    }

    pub fn activated(&self, i: usize) -> ActivatedGaussian {
        let p = self.position[i];
        let s = self.log_scale[i];
        let q = self.rot_wxyz[i];
        let d = self.f_dc[i];
        ActivatedGaussian {
            position: DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64),
            log_scale: DVec3::new(s[0] as f64, s[1] as f64, s[2] as f64),
            rotation: DQuat::from_xyzw(q[1] as f64, q[2] as f64, q[3] as f64, q[0] as f64),
            opacity: gps_core::sigmoid(self.opacity_logit[i] as f64),
            dc: DVec3::new(d[0] as f64, d[1] as f64, d[2] as f64),
        }
    }

    /// `3 * rest_per_channel` SH-rest coefficients of splat `i` as f64 (channel-major).
    pub fn sh_rest_f64(&self, i: usize) -> Vec<f64> {
        let n = 3 * self.sh_rest_per_channel();
        self.sh_rest[i * n..(i + 1) * n].iter().map(|&v| v as f64).collect()
    }
}

// ------------------------------------------------------------------- header

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    /// Caller guarantees `b.len() >= self.size()`.
    fn read_le(self, b: &[u8]) -> f64 {
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Self::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        }
    }
}

struct Header {
    binary: bool,
    vertex_count: usize,
    /// (name, type, byte offset within a vertex record) for scalar properties
    props: Vec<(String, Scalar, usize)>,
    stride: usize,
    data_start: usize,
}

fn parse_header(bytes: &[u8]) -> Result<Header, PlyError> {
    // Header is ASCII and ends with "end_header\n" (or \r\n).
    let marker = b"end_header";
    let pos = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or_else(|| PlyError::BadHeader("no end_header".into()))?;
    let mut data_start = pos + marker.len();
    if bytes.get(data_start) == Some(&b'\r') {
        data_start += 1;
    }
    if bytes.get(data_start) == Some(&b'\n') {
        data_start += 1;
    } else {
        return Err(PlyError::BadHeader("end_header not followed by newline".into()));
    }
    let text = std::str::from_utf8(&bytes[..pos]).map_err(|_| PlyError::BadHeader("non-UTF8 header".into()))?;
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(PlyError::NotPly);
    }
    let mut binary = None;
    let mut vertex_count = None;
    let mut props = Vec::new();
    let mut offset = 0usize;
    let mut in_vertex = false;
    let mut seen_other_before_vertex_end = false;
    for line in lines {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["format", f, _] => {
                binary = Some(match *f {
                    "ascii" => false,
                    "binary_little_endian" => true,
                    other => return Err(PlyError::UnsupportedFormat(other.into())),
                })
            }
            ["element", name, n] => {
                in_vertex = *name == "vertex";
                if in_vertex {
                    let n: usize = n.parse().map_err(|_| PlyError::BadHeader("vertex count".into()))?;
                    if n > MAX_VERTICES {
                        return Err(PlyError::TooManyVertices(n));
                    }
                    vertex_count = Some(n);
                } else if vertex_count.is_none() {
                    // An element before "vertex" would shift the data; unsupported.
                    seen_other_before_vertex_end = true;
                }
            }
            ["property", "list", ..] => {
                if in_vertex {
                    return Err(PlyError::UnsupportedFormat("list property in vertex element".into()));
                }
            }
            ["property", ty, name] if in_vertex => {
                let s = Scalar::parse(ty).ok_or_else(|| PlyError::BadHeader(format!("type {ty}")))?;
                props.push((name.to_string(), s, offset));
                offset += s.size();
            }
            _ => {}
        }
    }
    if seen_other_before_vertex_end {
        return Err(PlyError::UnsupportedFormat("element before vertex".into()));
    }
    Ok(Header {
        binary: binary.ok_or_else(|| PlyError::BadHeader("no format line".into()))?,
        vertex_count: vertex_count.ok_or_else(|| PlyError::BadHeader("no vertex element".into()))?,
        props,
        stride: offset,
        data_start,
    })
}

// ------------------------------------------------------------------- loading

fn sh_degree_from_rest_count(n: usize) -> Result<usize, PlyError> {
    match n {
        0 => Ok(0),
        9 => Ok(1),
        24 => Ok(2),
        45 => Ok(3),
        n => Err(PlyError::BadShCount(n)),
    }
}

/// Parse a 3DGS PLY from memory. Validates counts, required attributes and finiteness.
pub fn load_ply(bytes: &[u8]) -> Result<Gaussians, PlyError> {
    let h = parse_header(bytes)?;
    let idx = |name: &'static str| -> Result<usize, PlyError> {
        h.props.iter().position(|p| p.0 == name).ok_or(PlyError::MissingProperty(name))
    };
    let ix = [idx("x")?, idx("y")?, idx("z")?];
    let idc = [idx("f_dc_0")?, idx("f_dc_1")?, idx("f_dc_2")?];
    let iop = idx("opacity")?;
    let isc = [idx("scale_0")?, idx("scale_1")?, idx("scale_2")?];
    let irot = [idx("rot_0")?, idx("rot_1")?, idx("rot_2")?, idx("rot_3")?];
    let mut irest = Vec::new();
    while let Some(p) = h.props.iter().position(|p| p.0 == format!("f_rest_{}", irest.len())) {
        irest.push(p);
    }
    let sh_degree = sh_degree_from_rest_count(irest.len())?;
    let per_ch = gps_core::sh_rest_per_channel(sh_degree);

    let n = h.vertex_count;
    let mut g = Gaussians {
        count: n,
        sh_degree,
        position: Vec::with_capacity(n),
        log_scale: Vec::with_capacity(n),
        rot_wxyz: Vec::with_capacity(n),
        opacity_logit: Vec::with_capacity(n),
        f_dc: Vec::with_capacity(n),
        sh_rest: Vec::with_capacity(n * 3 * per_ch),
    };

    let body = &bytes[h.data_start..];
    if h.binary {
        let need = n.checked_mul(h.stride).ok_or(PlyError::TooManyVertices(n))?;
        if body.len() < need {
            return Err(PlyError::Truncated { expected: need, got: body.len() });
        }
        for v in 0..n {
            let rec = &body[v * h.stride..(v + 1) * h.stride];
            let get = |pi: usize| {
                let (_, ty, off) = &h.props[pi];
                ty.read_le(&rec[*off..*off + ty.size()]) as f32
            };
            push_vertex(&mut g, v, &get, &ix, &idc, iop, &isc, &irot, &irest)?;
        }
    } else {
        let text = std::str::from_utf8(body)
            .map_err(|_| PlyError::BadAscii { vertex: 0, msg: "non-UTF8 body".into() })?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let np = h.props.len();
        let mut vals = vec![0f32; np];
        for v in 0..n {
            let line = lines.next().ok_or(PlyError::Truncated { expected: n, got: v })?;
            let mut it = line.split_whitespace();
            for (k, slot) in vals.iter_mut().enumerate() {
                let tok = it.next().ok_or_else(|| PlyError::BadAscii {
                    vertex: v,
                    msg: format!("expected {np} values, got {k}"),
                })?;
                *slot = tok.parse::<f32>().map_err(|e| PlyError::BadAscii { vertex: v, msg: e.to_string() })?;
            }
            let get = |pi: usize| vals[pi];
            push_vertex(&mut g, v, &get, &ix, &idc, iop, &isc, &irot, &irest)?;
        }
    }
    Ok(g)
}

#[allow(clippy::too_many_arguments)]
fn push_vertex(
    g: &mut Gaussians,
    v: usize,
    get: &dyn Fn(usize) -> f32,
    ix: &[usize; 3],
    idc: &[usize; 3],
    iop: usize,
    isc: &[usize; 3],
    irot: &[usize; 4],
    irest: &[usize],
) -> Result<(), PlyError> {
    let chk = |x: f32, field: &'static str| -> Result<f32, PlyError> {
        if x.is_finite() {
            Ok(x)
        } else {
            Err(PlyError::NonFinite { vertex: v, field })
        }
    };
    g.position.push([chk(get(ix[0]), "position")?, chk(get(ix[1]), "position")?, chk(get(ix[2]), "position")?]);
    g.f_dc.push([chk(get(idc[0]), "f_dc")?, chk(get(idc[1]), "f_dc")?, chk(get(idc[2]), "f_dc")?]);
    g.opacity_logit.push(chk(get(iop), "opacity")?);
    g.log_scale.push([chk(get(isc[0]), "scale")?, chk(get(isc[1]), "scale")?, chk(get(isc[2]), "scale")?]);
    g.rot_wxyz.push([
        chk(get(irot[0]), "rot")?,
        chk(get(irot[1]), "rot")?,
        chk(get(irot[2]), "rot")?,
        chk(get(irot[3]), "rot")?,
    ]);
    for &r in irest {
        g.sh_rest.push(chk(get(r), "f_rest")?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(format: &str, n: usize, with_sh1: bool) -> String {
        let mut h = format!("ply\nformat {format} 1.0\nelement vertex {n}\n");
        for p in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] {
            h += &format!("property float {p}\n");
        }
        if with_sh1 {
            for i in 0..9 {
                h += &format!("property float f_rest_{i}\n");
            }
        }
        for p in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
            h += &format!("property float {p}\n");
        }
        h + "end_header\n"
    }

    #[test]
    fn binary_roundtrip_sh1() {
        let mut b = header("binary_little_endian", 2, true).into_bytes();
        for v in 0..2 {
            let vals: Vec<f32> = (0..(6 + 9 + 8)).map(|k| (v * 100 + k) as f32 * 0.01).collect();
            for x in vals {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        let g = load_ply(&b).unwrap();
        assert_eq!((g.count, g.sh_degree), (2, 1));
        assert_eq!(g.position[1], [1.0, 1.01, 1.02]);
        assert_eq!(g.sh_rest.len(), 2 * 9);
        // property order: opacity follows f_rest_8 => index 6+9
        assert!((g.opacity_logit[0] - 0.15).abs() < 1e-6);
        assert!((g.rot_wxyz[0][0] - 0.19).abs() < 1e-6);
    }

    #[test]
    fn ascii_and_activation() {
        let mut s = header("ascii", 1, false);
        s += "1 2 3  0.5 0 0  0  0 0 0  1 0 0 0\n";
        let g = load_ply(s.as_bytes()).unwrap();
        let a = g.activated(0);
        assert!((a.opacity - 0.5).abs() < 1e-12);
        assert_eq!(a.rotation, DQuat::IDENTITY);
    }

    #[test]
    fn rejects_bad_input_without_panicking() {
        assert_eq!(load_ply(b"hello").unwrap_err(), PlyError::BadHeader("no end_header".into()));
        assert_eq!(load_ply(b"nope\nend_header\n").unwrap_err(), PlyError::NotPly);
        let mut b = header("binary_little_endian", 5, false).into_bytes();
        b.extend_from_slice(&[0u8; 10]);
        assert!(matches!(load_ply(&b), Err(PlyError::Truncated { .. })));
        let huge = header("binary_little_endian", MAX_VERTICES + 1, false);
        assert!(matches!(load_ply(huge.as_bytes()), Err(PlyError::TooManyVertices(_))));
        let mut nan = header("ascii", 1, false);
        nan += "nan 0 0 0 0 0 0 0 0 0 0 0 0 0\n";
        assert!(load_ply(nan.as_bytes()).is_err());
        let missing = "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n";
        assert_eq!(load_ply(missing.as_bytes()).unwrap_err(), PlyError::MissingProperty("y"));
    }

    #[test]
    fn rejects_bad_sh_count() {
        let mut h = "ply\nformat ascii 1.0\nelement vertex 0\n".to_string();
        for p in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_rest_0"] {
            h += &format!("property float {p}\n");
        }
        h += "end_header\n";
        assert_eq!(load_ply(h.as_bytes()).unwrap_err(), PlyError::BadShCount(1));
    }
}
