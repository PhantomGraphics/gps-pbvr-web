//! CPU reference maths for Gaussian-Point Splatting (GPS) x PBVR.
//!
//! Direct port of `Phantom/PointCloud/GSView/GaussianPointMath.{h,cpp}`
//! (namespace `GSView::gpm`). Everything is f64 and deterministic; the GPU
//! (WGSL) implementation is checked against this.

use glam::{DMat2, DMat3, DQuat, DVec2, DVec3};
use std::f64::consts::PI;

pub mod camera;
pub mod oracle;
pub mod vectors;

/// SH band-0 basis: color = f_dc * SH_C0 + 0.5.
pub const SH_C0: f64 = 0.282_094_791_773_878_14;
/// pi^2 / 6 = Li2(1).
pub const DILOG_AT_ONE: f64 = 1.644_934_066_848_226_4;

// ---------------------------------------------------------------- activations

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sh_dc_to_color(sh: f64) -> f64 {
    (sh * SH_C0 + 0.5).clamp(0.0, 1.0)
}

pub const fn sh_rest_per_channel(degree: usize) -> usize {
    (degree + 1) * (degree + 1) - 1
}

/// Linear RGB for view direction `dir` (camera -> splat, normalised).
/// `rest` is channel-major (all R, then G, then B) with `storage_stride`
/// coefficients per channel (0 = tightly packed for `degree`).
pub fn eval_sh(
    degree: usize,
    dc: DVec3,
    rest: Option<&[f64]>,
    dir: DVec3,
    storage_stride: usize,
) -> DVec3 {
    const C1: f64 = 0.488_602_511_902_919_9;
    const C2: [f64; 5] = [
        1.092_548_430_592_079_2,
        -1.092_548_430_592_079_2,
        0.315_391_565_252_520_05,
        -1.092_548_430_592_079_2,
        0.546_274_215_296_039_6,
    ];
    const C3: [f64; 7] = [
        -0.590_043_589_926_643_5,
        2.890_611_442_640_554,
        -0.457_045_799_464_465_8,
        0.373_176_332_590_115_4,
        -0.457_045_799_464_465_8,
        1.445_305_721_320_277,
        -0.590_043_589_926_643_5,
    ];
    let mut result = SH_C0 * dc;
    let degree = degree.min(3);
    if let (true, Some(rest)) = (degree >= 1, rest) {
        let r = storage_stride.max(sh_rest_per_channel(degree));
        let sh = |k: usize| DVec3::new(rest[k], rest[r + k], rest[2 * r + k]);
        let (x, y, z) = (dir.x, dir.y, dir.z);
        result += -C1 * y * sh(0) + C1 * z * sh(1) - C1 * x * sh(2);
        if degree >= 2 {
            let (xx, yy, zz) = (x * x, y * y, z * z);
            let (xy, yz, xz) = (x * y, y * z, x * z);
            result += C2[0] * xy * sh(3)
                + C2[1] * yz * sh(4)
                + C2[2] * (2.0 * zz - xx - yy) * sh(5)
                + C2[3] * xz * sh(6)
                + C2[4] * (xx - yy) * sh(7);
            if degree >= 3 {
                result += C3[0] * y * (3.0 * xx - yy) * sh(8)
                    + C3[1] * xy * z * sh(9)
                    + C3[2] * y * (4.0 * zz - xx - yy) * sh(10)
                    + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh(11)
                    + C3[4] * x * (4.0 * zz - xx - yy) * sh(12)
                    + C3[5] * z * (xx - yy) * sh(13)
                    + C3[6] * x * (xx - 3.0 * yy) * sh(14);
            }
        }
    }
    (result + DVec3::splat(0.5)).max(DVec3::ZERO)
}

// ----------------------------------------------------------------- covariance

/// Rotation of a (w,x,y,z) quaternion, normalised internally.
/// `q` is passed as glam's (x,y,z,w) `DQuat`.
pub fn quat_to_rotation(q: DQuat) -> DMat3 {
    DMat3::from_quat(q.normalize())
}

/// Sigma = R S^2 R^T with S = diag(exp(log_scale)).
pub fn covariance_3d(log_scale: DVec3, rot: DQuat) -> DMat3 {
    let r = quat_to_rotation(rot);
    let s2 = DMat3::from_diagonal(DVec3::new(
        (2.0 * log_scale.x).exp(),
        (2.0 * log_scale.y).exp(),
        (2.0 * log_scale.z).exp(),
    ));
    r * s2 * r.transpose()
}

/// Pinhole camera: `cam = view_rot * (world - view_pos)`, looking along +Z.
#[derive(Clone, Copy, Debug)]
pub struct PinholeCamera {
    pub view_rot: DMat3,
    pub view_pos: DVec3,
    pub focal_x: f64,
    pub focal_y: f64,
    pub tan_fov_x: f64,
    pub tan_fov_y: f64,
}

impl Default for PinholeCamera {
    fn default() -> Self {
        Self {
            view_rot: DMat3::IDENTITY,
            view_pos: DVec3::ZERO,
            focal_x: 1.0,
            focal_y: 1.0,
            tan_fov_x: 1.0,
            tan_fov_y: 1.0,
        }
    }
}

pub fn world_to_camera(cam: &PinholeCamera, world: DVec3) -> DVec3 {
    cam.view_rot * (world - cam.view_pos)
}

/// EWA screen-space covariance with 3DGS frustum clamp and low-pass dilation.
pub fn covariance_2d(cov3d: DMat3, mean_world: DVec3, cam: &PinholeCamera, low_pass: f64) -> DMat2 {
    let mut t = world_to_camera(cam, mean_world);
    let limx = 1.3 * cam.tan_fov_x;
    let limy = 1.3 * cam.tan_fov_y;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    t.x = txtz.clamp(-limx, limx) * t.z;
    t.y = tytz.clamp(-limy, limy) * t.z;

    // glam (like glm) is column-major: element(row r, col c) = col(c)[r].
    let j = DMat3::from_cols(
        DVec3::new(cam.focal_x / t.z, 0.0, 0.0),
        DVec3::new(0.0, cam.focal_y / t.z, 0.0),
        DVec3::new(
            -cam.focal_x * t.x / (t.z * t.z),
            -cam.focal_y * t.y / (t.z * t.z),
            0.0,
        ),
    );
    let tm = j * cam.view_rot;
    let cov = tm * cov3d * tm.transpose();
    let mut c2 = DMat2::from_cols(
        DVec2::new(cov.col(0)[0], cov.col(0)[1]),
        DVec2::new(cov.col(1)[0], cov.col(1)[1]),
    );
    c2.x_axis.x += low_pass;
    c2.y_axis.y += low_pass;
    c2
}

/// Lower-triangular L with L L^T = A (2x2 SPD).
pub fn cholesky_2d(a: DMat2) -> DMat2 {
    let eps = 1e-300;
    let l00 = a.x_axis.x.max(eps).sqrt();
    let l10 = a.x_axis.y / l00;
    let l11 = (a.y_axis.y - l10 * l10).max(eps).sqrt();
    DMat2::from_cols(DVec2::new(l00, l10), DVec2::new(0.0, l11))
}

// ------------------------------------------------------------------ dilogarithm

fn dilog_series(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for k in 2..1000 {
        term *= x;
        let add = term / (k as f64 * k as f64);
        sum += add;
        if add.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

/// Li2(x) on [-1, 1] (clamped outside).
pub fn dilog(x: f64) -> f64 {
    if x >= 1.0 {
        return DILOG_AT_ONE;
    }
    if x <= -1.0 {
        return -DILOG_AT_ONE / 2.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < 0.0 {
        let y = x / (x - 1.0);
        let l = (1.0 - x).ln();
        return -dilog_series(y) - 0.5 * l * l;
    }
    if x <= 0.5 {
        return dilog_series(x);
    }
    DILOG_AT_ONE - x.ln() * (1.0 - x).ln() - dilog_series(1.0 - x)
}

fn next_down_one_pi2_6() -> f64 {
    f64::from_bits(DILOG_AT_ONE.to_bits() - 1)
}

/// Inverse of [`dilog`] on [0,1) -> [0, pi^2/6).
pub fn inv_dilog(target: f64) -> f64 {
    let target = target.clamp(0.0, next_down_one_pi2_6());
    if target <= 0.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let mut y = target / DILOG_AT_ONE;
    for _ in 0..40 {
        let f = dilog(y) - target;
        if f > 0.0 {
            hi = y;
        } else {
            lo = y;
        }
        let dfdy = if y > 1e-12 && y < 1.0 - 1e-15 {
            -(1.0 - y).ln() / y
        } else {
            1.0
        };
        let mut ny = y - f / dfdy;
        if !(ny > lo && ny < hi) {
            ny = 0.5 * (lo + hi);
        }
        if (ny - y).abs() <= 1e-14 * (1.0 + y) {
            return ny;
        }
        y = ny;
    }
    y
}

// ---------------------------------------------------------- expected point count

/// E[N] = 2*pi*sqrt(det) * Li2(o) per unit pixel area and pass.
pub fn expected_point_count(det_cov2d: f64, opacity: f64) -> f64 {
    if det_cov2d <= 0.0 || opacity <= 0.0 {
        return 0.0;
    }
    2.0 * PI * det_cov2d.sqrt() * dilog(opacity.min(1.0))
}

pub fn pixel_density_scale(
    depth: f64,
    focal_x: f64,
    focal_y: f64,
    reference_pixel_length: f64,
    near_z: f64,
) -> f64 {
    let finite = [depth, focal_x, focal_y, reference_pixel_length, near_z]
        .iter()
        .all(|v| v.is_finite());
    if !finite
        || depth <= near_z.max(0.0)
        || focal_x <= 0.0
        || focal_y <= 0.0
        || reference_pixel_length <= 0.0
    {
        return 0.0;
    }
    let ratio = reference_pixel_length / depth;
    (ratio * ratio * focal_x * focal_y).min(1.0e12)
}

// -------------------------------------------------- footprint-aware calibration

const MAX_OPACITY: f64 = 1.0 - 1e-12;

/// 2*pi*sqrt(det cov2d); 0 for degenerate / non-finite.
pub fn projected_footprint_area(cov2d: DMat2) -> f64 {
    let det = cov2d.x_axis.x * cov2d.y_axis.y - cov2d.y_axis.x * cov2d.x_axis.y;
    if !det.is_finite() || det <= 0.0 {
        return 0.0;
    }
    2.0 * PI * det.sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpacityRule {
    Proportional,
    Extinction,
    Dilog,
}

pub fn opacity_term(rule: OpacityRule, opacity: f64) -> f64 {
    if !opacity.is_finite() || opacity <= 0.0 {
        return 0.0;
    }
    let o = opacity.min(MAX_OPACITY);
    match rule {
        OpacityRule::Proportional => o,
        OpacityRule::Extinction => -(-o).ln_1p(),
        OpacityRule::Dilog => dilog(o),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationLevel {
    None = 0,
    ObjectZoom = 1,
    PerSplatDepth = 2,
    PerSplatFootprint = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct CalibrationInputs {
    pub object_depth: f64,
    pub splat_depth: f64,
    pub focal_x: f64,
    pub focal_y: f64,
    pub near_z: f64,
    pub reference_pixel_length: f64,
    pub footprint_area: f64,
    pub spp: f64,
}

impl Default for CalibrationInputs {
    fn default() -> Self {
        Self {
            object_depth: 0.0,
            splat_depth: 0.0,
            focal_x: 1.0,
            focal_y: 1.0,
            near_z: 0.0,
            reference_pixel_length: 0.01,
            footprint_area: 0.0,
            spp: 1.0,
        }
    }
}

pub fn calibrated_count(
    level: CalibrationLevel,
    rule: OpacityRule,
    opacity: f64,
    base_k: f64,
    density_scale: f64,
    inp: &CalibrationInputs,
) -> f64 {
    let g = opacity_term(rule, opacity);
    if !(g > 0.0) || !density_scale.is_finite() || density_scale <= 0.0 {
        return 0.0;
    }
    let scale = match level {
        CalibrationLevel::None => base_k,
        CalibrationLevel::ObjectZoom => {
            base_k
                * pixel_density_scale(
                    inp.object_depth,
                    inp.focal_x,
                    inp.focal_y,
                    inp.reference_pixel_length,
                    inp.near_z,
                )
        }
        CalibrationLevel::PerSplatDepth => {
            base_k
                * pixel_density_scale(
                    inp.splat_depth,
                    inp.focal_x,
                    inp.focal_y,
                    inp.reference_pixel_length,
                    inp.near_z,
                )
        }
        CalibrationLevel::PerSplatFootprint => inp.spp * inp.footprint_area,
    };
    let n = density_scale * g * scale;
    if n.is_finite() && n > 0.0 {
        n
    } else {
        0.0
    }
}

fn extinction_over_linear(x: f64) -> f64 {
    if x < 1e-8 {
        1.0 + 0.5 * x
    } else {
        -(-x).ln_1p() / x
    }
}

/// Radial correction keep-probability (C3+R), in (0, 1].
pub fn radial_keep_probability(opacity: f64, r2: f64) -> f64 {
    if !opacity.is_finite() || opacity <= 0.0 {
        return 1.0;
    }
    let r2 = if r2.is_nan() || r2 < 0.0 { 0.0 } else { r2 };
    let o = opacity.min(MAX_OPACITY);
    let y = o * (-0.5 * r2).exp();
    (extinction_over_linear(y) / extinction_over_linear(o)).clamp(0.0, 1.0)
}

pub fn footprint_point_count(expected: f64, footprint_subpixels: u32) -> f64 {
    if !expected.is_finite() || expected <= 0.0 {
        return 0.0;
    }
    let s = footprint_subpixels.max(1) as f64;
    expected / (s * s)
}

pub fn adaptive_footprint(sigma_min_subpixels: f64, kappa: f64, s_max: u32) -> u32 {
    if !(kappa > 0.0) || !sigma_min_subpixels.is_finite() || sigma_min_subpixels <= 0.0 {
        return 1;
    }
    let hi = s_max.max(1);
    let s = (kappa * sigma_min_subpixels).floor();
    if !(s >= 1.0) {
        1
    } else if s >= hi as f64 {
        hi
    } else {
        s as u32
    }
}

// --------------------------------------------------------------------- sampling

/// PCG hash, bit-identical to `pcg()` in the gps_*.comp shaders.
pub fn pcg_hash(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedMode {
    Deterministic,
    FrameVarying,
}

pub fn particle_seed(mode: SeedMode, splat_id: u32, particle_index: u32, frame_index: u32) -> u32 {
    let mut s = pcg_hash(
        splat_id
            .wrapping_mul(747_796_405)
            .wrapping_add(particle_index)
            .wrapping_add(1),
    );
    if mode == SeedMode::FrameVarying {
        s = pcg_hash(s ^ frame_index.wrapping_mul(2_654_435_761).wrapping_add(1));
    }
    if s != 0 {
        s
    } else {
        1
    }
}

/// Small stateful uniform stream.
#[derive(Clone, Debug)]
pub struct RandStream {
    state: u32,
    spare: Option<f64>,
}

impl RandStream {
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed != 0 { seed } else { 0x9E37_79B9 },
            spare: None,
        }
    }
    /// U[0,1)
    pub fn next(&mut self) -> f64 {
        self.state = pcg_hash(self.state);
        self.state as f64 * (1.0 / 4_294_967_296.0)
    }
    /// N(0,1), Box-Muller (one value per call).
    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.next().max(1e-300);
        let u2 = self.next();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

pub fn poisson_sample(lambda: f64, rng: &mut RandStream) -> i32 {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next();
            if p <= l {
                break;
            }
        }
        return k - 1;
    }
    let n = (lambda + lambda.sqrt() * rng.normal()).round();
    n.max(0.0) as i32
}

/// Whitened radius of the corrected radial law (density ∝ r * -log(1 - o e^{-r²/2})).
pub fn sample_corrected_radius(opacity: f64, u: f64) -> f64 {
    let o = opacity.clamp(0.0, 1.0);
    if o <= 0.0 {
        return 0.0;
    }
    let u = u.clamp(0.0, 1.0);
    let d = dilog(o);
    let target = (1.0 - u) * d;
    let y = inv_dilog(target).min(o);
    if y <= 0.0 {
        return 40.0;
    }
    let r2 = -2.0 * (y / o).ln();
    r2.max(0.0).sqrt()
}

pub fn sample_corrected_offset(opacity: f64, cov2d: DMat2, rng: &mut RandStream) -> DVec2 {
    let r = sample_corrected_radius(opacity, rng.next());
    let theta = 2.0 * PI * rng.next();
    cholesky_2d(cov2d) * DVec2::new(r * theta.cos(), r * theta.sin())
}

// ------------------------------------------------------ analytic coverage oracle

#[derive(Clone, Copy, Debug)]
pub struct Gaussian2D {
    pub mean: DVec2,
    pub cov: DMat2,
    pub opacity: f64,
    pub color: DVec3,
    pub depth: f64,
}

pub fn alpha_at(g: &Gaussian2D, pixel: DVec2) -> f64 {
    let d = pixel - g.mean;
    let m = d.dot(g.cov.inverse() * d);
    g.opacity.clamp(0.0, 1.0) * (-0.5 * m).exp()
}

pub fn expected_coverage(alpha: f64, pixel_area: f64) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0 - 1e-15);
    1.0 - (1.0 - alpha).powf(pixel_area)
}

/// Expected composite of projected Gaussians at one pixel (front-to-back).
pub fn composite_expected(
    mut layers: Vec<Gaussian2D>,
    pixel: DVec2,
    background: DVec3,
    pixel_area: f64,
) -> DVec3 {
    layers.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    let mut accum = DVec3::ZERO;
    let mut t = 1.0;
    for g in &layers {
        let a = expected_coverage(alpha_at(g, pixel), pixel_area);
        accum += t * a * g.color;
        t *= 1.0 - a;
        if t <= 1e-9 {
            break;
        }
    }
    accum + t * background
}

// -------------------------------------------------------- depth / colour packing

/// Order-preserving float -> u32 key so an unsigned atomicMin keeps the nearest.
pub fn ordered_depth_key(depth: f32) -> u32 {
    let bits = depth.to_bits();
    let mask = if bits & 0x8000_0000 != 0 { 0xFFFF_FFFF } else { 0x8000_0000 };
    bits ^ mask
}

pub fn ordered_depth_key_inverse(key: u32) -> f32 {
    let mask = if key & 0x8000_0000 != 0 { 0x8000_0000 } else { 0xFFFF_FFFF };
    f32::from_bits(key ^ mask)
}

pub fn pack_depth_color(depth: f32, rgba8: u32) -> u64 {
    ((ordered_depth_key(depth) as u64) << 32) | rgba8 as u64
}
pub fn unpack_color(p: u64) -> u32 {
    p as u32
}
pub fn unpack_depth_key(p: u64) -> u32 {
    (p >> 32) as u32
}
