//! CPU oracle renderers. Port of `Phantom/PointCloud/GSView/GaussianPointOracle.{h,cpp}`.
//!
//! * [`render_analytic`]      - ground truth (expected-coverage composite per pixel)
//! * [`render_monte_carlo`]   - CPU port of the GPS GPU pipeline
//! * [`render_particles_3d`]  - CPU model of the PBVR3D path
//!
//! The RNG streams, seeds and sample ordering are identical to the C++ code, so
//! for the same inputs both implementations draw the same particles.

use crate::*;
use glam::{DMat2, DMat3, DQuat, DVec2, DVec3};

#[derive(Clone, Copy, Debug)]
pub struct Gaussian3D {
    pub pos: DVec3,
    /// log of the per-axis std-dev
    pub log_scale: DVec3,
    /// (x, y, z, w) glam order; the C++ side is (w, x, y, z)
    pub rot: DQuat,
    pub opacity: f64,
    /// linear RGB
    pub color: DVec3,
}

impl Default for Gaussian3D {
    fn default() -> Self {
        Self { pos: DVec3::ZERO, log_scale: DVec3::ZERO, rot: DQuat::IDENTITY, opacity: 1.0, color: DVec3::ONE }
    }
}

/// Camera looks along +Z: `cam = view_rot * (world - view_pos)`; pixel y grows downward
/// with camera +Y (no flip), principal point (cx, cy) in pixels.
#[derive(Clone, Copy, Debug)]
pub struct OracleCamera {
    pub view_rot: DMat3,
    pub view_pos: DVec3,
    pub focal_x: f64,
    pub focal_y: f64,
    pub cx: f64,
    pub cy: f64,
    pub low_pass: f64,
}

impl Default for OracleCamera {
    fn default() -> Self {
        Self { view_rot: DMat3::IDENTITY, view_pos: DVec3::ZERO, focal_x: 200.0, focal_y: 200.0, cx: 0.0, cy: 0.0, low_pass: 0.3 }
    }
}

impl OracleCamera {
    /// Look-at camera (+Z forward, +Y down in image space) for a `w`x`h` image with a
    /// vertical field of view `fov_y` (radians). `up` is the world up direction.
    pub fn look_at(eye: DVec3, target: DVec3, up: DVec3, fov_y: f64, w: u32, h: u32) -> Self {
        let f = (target - eye).normalize();
        let right = f.cross(up).normalize();
        // image +Y points down: down = f x right... chosen so right x down = forward.
        let down = f.cross(right);
        // rows of view_rot are the camera axes in world space
        let view_rot = DMat3::from_cols(right, down, f).transpose();
        let focal = 0.5 * h as f64 / (0.5 * fov_y).tan();
        Self {
            view_rot,
            view_pos: eye,
            focal_x: focal,
            focal_y: focal,
            cx: 0.5 * w as f64,
            cy: 0.5 * h as f64,
            low_pass: 0.3,
        }
    }

    fn to_pinhole(&self) -> PinholeCamera {
        // Generous tan(fov/2): only feeds the EWA frustum clamp (same as C++).
        PinholeCamera {
            view_rot: self.view_rot,
            view_pos: self.view_pos,
            focal_x: self.focal_x,
            focal_y: self.focal_y,
            tan_fov_x: 4.0,
            tan_fov_y: 4.0,
        }
    }

    fn with_default_principal_point(mut self, w: usize, h: usize) -> Self {
        if self.cx == 0.0 && self.cy == 0.0 {
            self.cx = 0.5 * w as f64;
            self.cy = 0.5 * h as f64;
        }
        self
    }
}

/// Row-major, `w * h` linear-RGB pixels.
pub type Image = Vec<DVec3>;

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    pub points: f64,
    pub candidates: f64,
    pub subpixel_writes: f64,
}

#[derive(Clone, Copy, Debug)]
struct Projected {
    mean: DVec2,
    cov2d: DMat2,
    det: f64,
    depth: f64,
    opacity: f64,
    color: DVec3,
}

fn project(g: &Gaussian3D, c: &OracleCamera) -> Option<Projected> {
    let pin = c.to_pinhole();
    let cam_mean = world_to_camera(&pin, g.pos);
    if cam_mean.z <= 1e-4 {
        return None;
    }
    let cov3d = covariance_3d(g.log_scale, g.rot);
    let cov2d = covariance_2d(cov3d, g.pos, &pin, c.low_pass);
    let det = cov2d.determinant();
    if det <= 0.0 {
        return None;
    }
    Some(Projected {
        mean: DVec2::new(c.focal_x * cam_mean.x / cam_mean.z + c.cx, c.focal_y * cam_mean.y / cam_mean.z + c.cy),
        cov2d,
        det,
        depth: cam_mean.z,
        opacity: g.opacity.clamp(0.0, 1.0),
        color: g.color,
    })
}

fn project_layers(scene: &[Gaussian3D], cam: &OracleCamera) -> Vec<Gaussian2D> {
    scene
        .iter()
        .filter_map(|g| project(g, cam))
        .map(|p| Gaussian2D { mean: p.mean, cov: p.cov2d, opacity: p.opacity, color: p.color, depth: p.depth })
        .collect()
}

/// Nearest-wins test on one subpixel (CPU stand-in for the GPU atomicMin).
fn depth_test(idx: usize, d: f64, c: DVec3, depth: &mut [f64], color: &mut [DVec3], stats: &mut RenderStats) {
    stats.subpixel_writes += 1.0;
    if d < depth[idx] {
        depth[idx] = d;
        color[idx] = c;
    }
}

#[allow(clippy::too_many_arguments)]
fn write_block(
    pixel: DVec2, s: i64, w: usize, h: usize, spp_side: i64, d: f64, c: DVec3,
    depth: &mut [f64], color: &mut [DVec3], stats: &mut RenderStats,
) {
    let spp = spp_side * spp_side;
    let gx0 = (pixel.x * spp_side as f64 - 0.5 * s as f64 + 0.5).floor() as i64;
    let gy0 = (pixel.y * spp_side as f64 - 0.5 * s as f64 + 0.5).floor() as i64;
    let gw = w as i64 * spp_side;
    let gh = h as i64 * spp_side;
    for gy in gy0.max(0)..gh.min(gy0 + s) {
        for gx in gx0.max(0)..gw.min(gx0 + s) {
            let (px, py) = (gx / spp_side, gy / spp_side);
            let (sx, sy) = (gx % spp_side, gy % spp_side);
            let idx = ((py as usize * w + px as usize) as i64 * spp + sy * spp_side + sx) as usize;
            depth_test(idx, d, c, depth, color, stats);
        }
    }
}

fn resolve_set(w: usize, h: usize, spp: usize, depth: &[f64], color: &[DVec3], bg: DVec3, accum: &mut Image) {
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * spp;
            let mut sum = DVec3::ZERO;
            for k in 0..spp {
                sum += if depth[base + k] < 1e299 { color[base + k] } else { bg };
            }
            accum[y * w + x] += sum / spp as f64;
        }
    }
}

// ------------------------------------------------------------------- analytic

pub fn render_analytic(scene: &[Gaussian3D], cam: OracleCamera, w: usize, h: usize, bg: DVec3) -> Image {
    let cam = cam.with_default_principal_point(w, h);
    let layers = project_layers(scene, &cam);
    let mut img = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            img.push(composite_expected(layers.clone(), DVec2::new(x as f64 + 0.5, y as f64 + 0.5), bg, 1.0));
        }
    }
    img
}

// ---------------------------------------------------------------- Monte Carlo

#[derive(Clone, Copy, Debug)]
pub struct MonteCarloOptions {
    pub footprint_subpixels: u32,
    /// > 0: per-splat adaptive footprint
    pub adaptive_kappa: f64,
    pub footprint_max: u32,
    pub compensate_blur: bool,
    /// Multiplies the expected point count (the renderer's `density_scale`).
    pub density_scale: f64,
    /// Rust/GPU-only sampling scheme (no C++ counterpart): particle `j` of splat `g` draws from its own
    /// stream seeded `particle_seed(FrameVarying, g, j + 1, frame)`; the count still comes from the
    /// `particle_seed(.., g, 0, ..)` stream. Random access like this is what lets the GPU generate
    /// particles in parallel. Statistically identical to the per-splat stream.
    pub per_particle_seed: bool,
    /// Per-splat cap on particles (the GPU's `max_points_per_splat`); `u32::MAX` = none.
    pub max_points_per_splat: u32,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        Self { footprint_subpixels: 1, adaptive_kappa: 0.0, footprint_max: 4, compensate_blur: false, density_scale: 1.0, per_particle_seed: false, max_points_per_splat: u32::MAX }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_monte_carlo(
    scene: &[Gaussian3D], cam: OracleCamera, w: usize, h: usize, spp_side: usize, num_sets: usize,
    bg: DVec3, base_seed: u32, opt: &MonteCarloOptions,
) -> (Image, RenderStats) {
    let cam = cam.with_default_principal_point(w, h);
    let spp_side = spp_side.clamp(1, 4);
    let spp = spp_side * spp_side;
    let n_sub = w * h * spp;
    let mut stats = RenderStats::default();

    let proj: Vec<Projected> = scene.iter().filter_map(|g| project(g, &cam)).collect();
    let mut footprint = vec![1u32; proj.len()];
    let mut sample_cov: Vec<DMat2> = proj.iter().map(|p| p.cov2d).collect();
    for (g, pr) in proj.iter().enumerate() {
        let mut fp = opt.footprint_subpixels.max(1);
        if opt.adaptive_kappa > 0.0 {
            let tr = pr.cov2d.x_axis.x + pr.cov2d.y_axis.y;
            let disc = (0.25 * tr * tr - pr.det).max(0.0).sqrt();
            let sigma_min_px = (0.5 * tr - disc).max(0.0).sqrt();
            fp = adaptive_footprint(sigma_min_px * spp_side as f64, opt.adaptive_kappa, opt.footprint_max);
        }
        footprint[g] = fp;
        if opt.compensate_blur && fp > 1 {
            let v = (fp as f64 / spp_side as f64) * (fp as f64 / spp_side as f64) / 12.0;
            let shrunk = pr.cov2d - DMat2::from_diagonal(DVec2::splat(v));
            let t = shrunk.x_axis.x + shrunk.y_axis.y;
            let d = shrunk.determinant();
            if d > 0.0 && t > 0.0 && 0.5 * t - (0.25 * t * t - d).max(0.0).sqrt() >= v {
                sample_cov[g] = shrunk;
            }
        }
    }

    let mut accum = vec![DVec3::ZERO; w * h];
    let mut depth = vec![0.0; n_sub];
    let mut color = vec![DVec3::ZERO; n_sub];

    for s in 0..num_sets {
        depth.fill(1e300);
        color.fill(DVec3::ZERO);
        for (gid, pr) in proj.iter().enumerate() {
            let fp = footprint[gid];
            let en = footprint_point_count(expected_point_count(pr.det, pr.opacity) * spp as f64 * opt.density_scale, fp);
            if en <= 0.0 {
                continue;
            }
            let seed = particle_seed(SeedMode::FrameVarying, gid as u32, 0, base_seed.wrapping_add(s as u32));
            let mut rng = RandStream::new(seed);
            let n_full = poisson_sample(en, &mut rng);
            stats.points += n_full as f64;
            let n = (n_full as u32).min(opt.max_points_per_splat) as i32;
            for j in 0..n {
                if opt.per_particle_seed {
                    rng = RandStream::new(particle_seed(SeedMode::FrameVarying, gid as u32, j as u32 + 1, base_seed.wrapping_add(s as u32)));
                }
                let off = sample_corrected_offset(pr.opacity, sample_cov[gid], &mut rng);
                let pixel = pr.mean + off;
                if fp == 1 {
                    if pixel.x < 0.0 || pixel.x >= w as f64 || pixel.y < 0.0 || pixel.y >= h as f64 {
                        continue;
                    }
                    let (px, py) = (pixel.x as usize, pixel.y as usize);
                    let sx = (spp_side - 1).min(((pixel.x - px as f64) * spp_side as f64) as usize);
                    let sy = (spp_side - 1).min(((pixel.y - py as f64) * spp_side as f64) as usize);
                    depth_test((py * w + px) * spp + sy * spp_side + sx, pr.depth, pr.color, &mut depth, &mut color, &mut stats);
                } else {
                    write_block(pixel, fp as i64, w, h, spp_side as i64, pr.depth, pr.color, &mut depth, &mut color, &mut stats);
                }
            }
        }
        resolve_set(w, h, spp, &depth, &color, bg, &mut accum);
    }
    for c in &mut accum {
        *c /= num_sets as f64;
    }
    (accum, stats)
}

// ------------------------------------------------------------ PBVR3D particles

#[derive(Clone, Copy, Debug)]
pub struct ParticleOptions {
    pub level: CalibrationLevel,
    pub rule: OpacityRule,
    pub radial_correction: bool,
    pub base_k: f64,
    pub reference_pixel_length: f64,
    pub near_z: f64,
    pub low_pass_jitter: bool,
    pub centre_depth: bool,
    pub linearized_projection: bool,
    pub density_scale: f64,
    /// Every particle draws from its own stream `particle_seed(scene index, k + 1, set)` and the
    /// count from `(scene index, 0, set)`: the scheme of the GPU particle-parallel path (random access,
    /// no sequential dependency). Off = the C++ reference's single sequential stream per splat.
    pub per_particle_seed: bool,
}

impl Default for ParticleOptions {
    fn default() -> Self {
        Self {
            level: CalibrationLevel::PerSplatFootprint,
            rule: OpacityRule::Extinction,
            radial_correction: false,
            base_k: 512.0,
            reference_pixel_length: 0.01,
            near_z: 0.05,
            low_pass_jitter: true,
            centre_depth: false,
            linearized_projection: false,
            density_scale: 1.0,
            per_particle_seed: false,
        }
    }
}

fn cholesky_3d(c: DMat3) -> DMat3 {
    let (a00, a10, a20) = (c.col(0)[0], c.col(0)[1], c.col(0)[2]);
    let (a11, a21, a22) = (c.col(1)[1], c.col(1)[2], c.col(2)[2]);
    let l00 = a00.max(1e-30).sqrt();
    let (l10, l20) = (a10 / l00, a20 / l00);
    let l11 = (a11 - l10 * l10).max(1e-30).sqrt();
    let l21 = (a21 - l20 * l10) / l11;
    let l22 = (a22 - l20 * l20 - l21 * l21).max(1e-30).sqrt();
    DMat3::from_cols(DVec3::new(l00, l10, l20), DVec3::new(0.0, l11, l21), DVec3::new(0.0, 0.0, l22))
}

struct Prep {
    index: u32,
    pos: DVec3,
    pr: Projected,
    l3: DMat3,
    cov2d_inv: DMat2,
    jw: DMat3,
    lambda: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn render_particles_3d(
    scene: &[Gaussian3D], cam: OracleCamera, w: usize, h: usize, spp_side: usize, num_sets: usize,
    bg: DVec3, base_seed: u32, opt: &ParticleOptions,
) -> (Image, RenderStats) {
    let cam = cam.with_default_principal_point(w, h);
    let spp_side = spp_side.clamp(1, 4);
    let spp = spp_side * spp_side;
    let n_sub = w * h * spp;
    let pin = cam.to_pinhole();
    let mut stats = RenderStats::default();

    let mut centre = DVec3::ZERO;
    for g in scene {
        centre += g.pos;
    }
    if !scene.is_empty() {
        centre /= scene.len() as f64;
    }
    let object_depth = world_to_camera(&pin, centre).z;

    let mut preps = Vec::new();
    for (scene_index, g) in scene.iter().enumerate() {
        let Some(pr) = project(g, &cam) else { continue };
        let cov3 = covariance_3d(g.log_scale, g.rot);
        let l3 = cholesky_3d(cov3);
        let cm = world_to_camera(&pin, g.pos);
        let iz = 1.0 / cm.z;
        let jx = DVec3::new(cam.focal_x * iz, 0.0, -cam.focal_x * cm.x * iz * iz);
        let jy = DVec3::new(0.0, cam.focal_y * iz, -cam.focal_y * cm.y * iz * iz);
        let wt = cam.view_rot.transpose();
        let (rx, ry) = (wt * jx, wt * jy);
        let jw = DMat3::from_cols(DVec3::new(rx.x, ry.x, 0.0), DVec3::new(rx.y, ry.y, 0.0), DVec3::new(rx.z, ry.z, 0.0));
        let inp = CalibrationInputs {
            object_depth,
            splat_depth: cm.z,
            focal_x: cam.focal_x,
            focal_y: cam.focal_y,
            near_z: opt.near_z,
            reference_pixel_length: opt.reference_pixel_length,
            footprint_area: projected_footprint_area(pr.cov2d),
            spp: spp as f64,
        };
        let lambda = calibrated_count(opt.level, opt.rule, pr.opacity, opt.base_k, opt.density_scale, &inp);
        preps.push(Prep { index: scene_index as u32, pos: g.pos, pr, l3, cov2d_inv: pr.cov2d.inverse(), jw, lambda });
    }

    let mut accum = vec![DVec3::ZERO; w * h];
    let mut depth = vec![0.0; n_sub];
    let mut color = vec![DVec3::ZERO; n_sub];
    let forward = cam.view_rot.transpose().col(2);
    let jitter = if opt.low_pass_jitter { cam.low_pass.max(0.0).sqrt() } else { 0.0 };

    for s in 0..num_sets {
        depth.fill(1e300);
        color.fill(DVec3::ZERO);
        for (gid, p) in preps.iter().enumerate() {
            if p.lambda <= 0.0 {
                continue;
            }
            let set_seed = base_seed.wrapping_add(s as u32);
            let mut rng = if opt.per_particle_seed {
                RandStream::new(particle_seed(SeedMode::FrameVarying, p.index, 0, set_seed))
            } else {
                RandStream::new(particle_seed(SeedMode::FrameVarying, gid as u32, 1, set_seed))
            };
            let n = poisson_sample(p.lambda, &mut rng);
            stats.candidates += n as f64;
            for k in 0..n {
                if opt.per_particle_seed {
                    rng = RandStream::new(particle_seed(SeedMode::FrameVarying, p.index, k as u32 + 1, set_seed));
                }
                // C++ writes dvec3(rng.normal(), rng.normal(), rng.normal()); argument evaluation
                // order is unspecified there and MSVC evaluates right-to-left (z, y, x). Rust is
                // explicit and follows the MSVC reference so the two implementations draw the
                // same particles.
                let uz = rng.normal();
                let uy = rng.normal();
                let ux = rng.normal();
                let u = DVec3::new(ux, uy, uz);
                let world = p.pos + p.l3 * u;
                let (mut pixel, mut d);
                if opt.linearized_projection {
                    let off = p.jw * (world - p.pos);
                    pixel = p.pr.mean + DVec2::new(off.x, off.y);
                    d = p.pr.depth + forward.dot(world - p.pos);
                } else {
                    let cp = world_to_camera(&pin, world);
                    if cp.z <= opt.near_z {
                        continue;
                    }
                    pixel = DVec2::new(cam.focal_x * cp.x / cp.z + cam.cx, cam.focal_y * cp.y / cp.z + cam.cy);
                    d = cp.z;
                }
                if jitter > 0.0 {
                    // Same MSVC right-to-left order as above (y first, then x).
                    let jy = rng.normal();
                    let jx = rng.normal();
                    pixel += jitter * DVec2::new(jx, jy);
                }
                if opt.radial_correction {
                    let dv = pixel - p.pr.mean;
                    let r2 = dv.dot(p.cov2d_inv * dv);
                    if rng.next() >= radial_keep_probability(p.pr.opacity, r2) {
                        continue;
                    }
                }
                if opt.centre_depth {
                    d = p.pr.depth;
                }
                stats.points += 1.0;
                if pixel.x < 0.0 || pixel.x >= w as f64 || pixel.y < 0.0 || pixel.y >= h as f64 {
                    continue;
                }
                let (px, py) = (pixel.x as usize, pixel.y as usize);
                let sx = (spp_side - 1).min(((pixel.x - px as f64) * spp_side as f64) as usize);
                let sy = (spp_side - 1).min(((pixel.y - py as f64) * spp_side as f64) as usize);
                depth_test((py * w + px) * spp + sy * spp_side + sx, d, p.pr.color, &mut depth, &mut color, &mut stats);
            }
        }
        resolve_set(w, h, spp, &depth, &color, bg, &mut accum);
    }
    for c in &mut accum {
        *c /= num_sets.max(1) as f64;
    }
    (accum, stats)
}

// -------------------------------------------------------------------- metrics

pub fn psnr(a: &[DVec3], b: &[DVec3]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).length_squared()).sum::<f64>() / (n * 3) as f64;
    if mse <= 1e-20 {
        120.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// Mean SSIM over 8x8 windows (step 4) on luminance.
pub fn ssim(a: &[DVec3], b: &[DVec3], w: usize, h: usize) -> f64 {
    let lum = |c: DVec3| 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let (c1, c2) = (0.01 * 0.01, 0.03 * 0.03);
    let (mut total, mut count) = (0.0, 0);
    let mut y0 = 0;
    while y0 + 8 <= h {
        let mut x0 = 0;
        while x0 + 8 <= w {
            let (mut ma, mut mb) = (0.0, 0.0);
            for y in 0..8 {
                for x in 0..8 {
                    let i = (y0 + y) * w + x0 + x;
                    ma += lum(a[i]);
                    mb += lum(b[i]);
                }
            }
            ma /= 64.0;
            mb /= 64.0;
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for y in 0..8 {
                for x in 0..8 {
                    let i = (y0 + y) * w + x0 + x;
                    let (da, db) = (lum(a[i]) - ma, lum(b[i]) - mb);
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= 63.0;
            vb /= 63.0;
            cov /= 63.0;
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            count += 1;
            x0 += 4;
        }
        y0 += 4;
    }
    if count > 0 {
        total / count as f64
    } else {
        1.0
    }
}
