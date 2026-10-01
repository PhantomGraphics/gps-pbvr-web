//! Phase 2 verification of the GPU GPS renderer against the CPU oracle (`gps-core`).
//! Shared by native `gps-verify` and the browser (`gps-web`). Thresholds are the ones fixed
//! in the plan / C++ tests (PSNR >= 35 dB, SSIM >= 0.98) and are not tuned to pass.

use crate::gps::*;
use glam::{DQuat, DVec3};
use gps_core::oracle::*;

const W: usize = 96;
const H: usize = 96;
const SPP_SIDE: u32 = 2;
const SETS: u32 = 300;
const BG: [f32; 3] = [0.05, 0.05, 0.08];

fn camera() -> OracleCamera {
    OracleCamera { focal_x: 200.0, focal_y: 200.0, ..Default::default() }
}
fn iso(z: f64, o: f64, col: DVec3) -> Gaussian3D {
    Gaussian3D { pos: DVec3::new(0.0, 0.0, z), log_scale: DVec3::splat(0.15_f64.ln()), opacity: o, color: col, ..Default::default() }
}
fn scenes() -> Vec<(&'static str, Vec<Gaussian3D>)> {
    let c = DVec3::new(0.9, 0.4, 0.2);
    vec![
        ("single o=0.1", vec![iso(6.0, 0.1, c)]),
        ("single o=0.5", vec![iso(6.0, 0.5, c)]),
        ("single o=0.9", vec![iso(6.0, 0.9, c)]),
        ("front/back", vec![iso(9.0, 0.8, DVec3::new(0.15, 0.7, 0.25)), iso(5.0, 0.6, DVec3::new(0.9, 0.2, 0.2))]),
        (
            "anisotropic",
            vec![Gaussian3D {
                pos: DVec3::new(0.0, 0.0, 6.0),
                log_scale: DVec3::new(0.28_f64.ln(), 0.10_f64.ln(), 0.06_f64.ln()),
                rot: DQuat::from_xyzw(0.20, -0.45, 0.30, 0.82).normalize(),
                opacity: 0.7,
                color: DVec3::new(0.3, 0.5, 0.95),
            }],
        ),
    ]
}

async fn render_mean(r: &mut GpsRenderer, scene: &[Gaussian3D], params: &RenderParams, sets: u32, seed0: u32) -> Result<Vec<[f32; 4]>, String> {
    let gg: Vec<_> = scene.iter().map(GpuGaussian::from_oracle).collect();
    r.set_gaussians(&gg)?;
    r.reset_stats();
    let mut done = 0;
    while done < sets {
        let n = (sets - done).min(MAX_ENSEMBLES_PER_SUBMIT);
        r.render_ensembles(&camera(), params, seed0 + done, n)?;
        done += n;
    }
    Ok(r.read_accum().await)
}

fn to_image(acc: &[[f32; 4]]) -> Image {
    acc.iter().map(|p| DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64) / p[3].max(1.0) as f64).collect()
}

async fn read_texture(r: &GpsRenderer, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let bpr = (w * 4).next_multiple_of(256);
    let staging = r.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex staging"),
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r.device().create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    r.queue().submit([enc.finish()]);
    let raw = map_read(r.device(), &staging).await;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&raw[y * bpr as usize..y * bpr as usize + (w * 4) as usize]);
    }
    out
}

fn srgb_oetf(c: f64) -> f64 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}


/// Renders `scene` with both GPU paths and checks each against the CPU oracle (per-splat stream for the
/// primitive path, per-particle seeds for the particle path). Thresholds are fixed, not tuned.
async fn compare_case(
    r: &mut GpsRenderer, log: &mut dyn FnMut(String), name: &str, scene: &[Gaussian3D], base: &RenderParams, sets: u32,
    analytic: bool, primitive_too: bool,
) -> bool {
    let bg = DVec3::new(BG[0] as f64, BG[1] as f64, BG[2] as f64);
    let ana = if analytic { Some(render_analytic(scene, camera(), W, H, bg)) } else { None };
    let mut ok = true;
    let mut seen: Vec<GpuStats> = Vec::new();
    let paths: &[RenderPath] = if primitive_too { &[RenderPath::Primitive, RenderPath::Particle] } else { &[RenderPath::Particle] };
    for &path in paths {
        let params = RenderParams { path, ..*base };
        let acc = match render_mean(r, scene, &params, sets, 1).await {
            Ok(a) => a,
            Err(e) => {
                log(format!("[{name} / {path:?}] render failed: {e}"));
                ok = false;
                continue;
            }
        };
        let gpu = to_image(&acc);
        let st = r.read_stats().await;
        let opt = MonteCarloOptions {
            per_particle_seed: path == RenderPath::Particle,
            density_scale: base.density_scale as f64,
            max_points_per_splat: base.max_points_per_splat,
            ..Default::default()
        };
        let (cpu, cst) = render_monte_carlo(scene, camera(), W, H, SPP_SIDE as usize, sets as usize, bg, 1, &opt);
        let p_cpu = psnr(&gpu, &cpu);
        let diff = (st.points as f64 - cst.points).abs();
        let tol = 4.0 * (2.0 * cst.points).sqrt().max(1.0);
        let mut pass = p_cpu >= 60.0 && diff <= tol && st.orphan_subpixels == 0;
        let mut ana_txt = String::new();
        if let Some(a) = &ana {
            let (p_a, s_a) = (psnr(&gpu, a), ssim(&gpu, a, W, H));
            pass &= p_a >= 35.0 && s_a >= 0.98 && st.truncated_splats == 0;
            ana_txt = format!(" | analytic PSNR={p_a:.2} SSIM={s_a:.4}");
        }
        log(format!(
            "[{name} / {path:?}] vs CPU PSNR={p_cpu:.1} dB{ana_txt} | points gpu={} cpu={:.0} orphan={} truncated={} dropped={} {}",
            st.points, cst.points, st.orphan_subpixels, st.truncated_splats, st.dropped_points, if pass { "ok" } else { "FAIL" }
        ));
        ok &= pass;
        seen.push(st);
    }
    if seen.len() == 2 {
        // Same Poisson stream, same cap: the two paths must agree on every counter exactly.
        let same = seen[0] == seen[1];
        log(format!("[{name}] primitive and particle counters identical: {}", if same { "ok" } else { "FAIL" }));
        ok &= same;
    }
    ok
}

pub async fn run_phase2(adapter: &wgpu::Adapter, log: &mut dyn FnMut(String)) -> bool {
    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor { label: Some("gps-phase2"), required_limits: adapter.limits(), ..Default::default() })
        .await
    {
        Ok(d) => d,
        Err(e) => {
            log(format!("request_device failed: {e}"));
            return false;
        }
    };
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let mut r = match GpsRenderer::new(device, queue, fmt, W as u32, H as u32, SPP_SIDE) {
        Ok(r) => r,
        Err(e) => {
            log(format!("renderer init failed: {e}"));
            return false;
        }
    };
    let params = RenderParams { spp_side: SPP_SIDE, background: BG, ..Default::default() };
    let mut ok = true;

    // Both GPU paths against the CPU oracle (and the analytic solution where it applies).
    for (name, scene) in scenes() {
        ok &= compare_case(&mut r, log, name, &scene, &params, SETS, true, true).await;
    }
    {
        // 70,000 tiny splats: the scan needs 3 levels (70000 -> 274 -> 2); counts must match exactly.
        let mut many = Vec::new();
        for i in 0..70_000u32 {
            let (gx, gy) = ((i % 280) as f64 / 280.0 - 0.5, (i / 280) as f64 / 250.0 - 0.5);
            many.push(Gaussian3D {
                pos: DVec3::new(gx * 0.9, gy * 0.9, 6.0 + (i % 7) as f64 * 0.1),
                log_scale: DVec3::splat(0.012_f64.ln()),
                opacity: 0.3 + 0.6 * ((i * 37 % 100) as f64 / 100.0),
                color: DVec3::new((i % 5) as f64 / 5.0, (i % 3) as f64 / 3.0, 0.5),
                ..Default::default()
            });
        }
        ok &= compare_case(&mut r, log, "70k splats (3 scan levels)", &many, &params, 8, false, true).await;
    }
    {
        // A per-splat cap that truncates: counters must agree between the paths, never silently.
        let capped = RenderParams { max_points_per_splat: 50, ..params };
        ok &= compare_case(&mut r, log, "cap=50 truncation", &scenes()[2].1, &capped, 8, false, true).await;
    }
    {
        // > 4.19M particles from ONE splat in one ensemble: the particle dispatch must split into 2D
        // groups. Particle path only: the splat-parallel path would give a single thread all ~4.4M
        // particles, which exceeds the OS GPU watchdog (device lost) -- the imbalance this path removes.
        let big = vec![Gaussian3D { log_scale: DVec3::splat(0.55_f64.ln()), ..iso(6.0, 0.9, DVec3::new(0.9, 0.4, 0.2)) }];
        let p = RenderParams { density_scale: 400.0, max_points_per_splat: 1 << 24, ..params };
        ok &= compare_case(&mut r, log, "4M+ particles from one splat (2D dispatch)", &big, &p, 2, false, false).await;
    }

    // Determinism: same seeds -> bit-identical accumulation (atomicMin is order independent).
    {
        let scene = &scenes()[3].1;
        let a = render_mean(&mut r, scene, &params, 8, 500).await.unwrap_or_default();
        let b = render_mean(&mut r, scene, &params, 8, 500).await.unwrap_or_default();
        let c = render_mean(&mut r, scene, &params, 8, 600).await.unwrap_or_default();
        let pass = !a.is_empty() && a == b && a != c;
        log(format!("[determinism] same seed identical, different seed differs: {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }

    // History reset + composite (single gamma).
    {
        let scene = &scenes()[1].1;
        let acc = render_mean(&mut r, scene, &params, 16, 1).await.unwrap_or_default();
        let w_ok = !acc.is_empty() && acc.iter().all(|p| (p[3] - 16.0).abs() < 1e-3);
        r.reset_accum();
        let zero = r.read_accum().await.iter().all(|p| *p == [0.0; 4]);
        let _ = r.render_ensembles(&camera(), &params, 1, 3);
        let three = r.read_accum().await.iter().all(|p| (p[3] - 3.0).abs() < 1e-3);
        let pass = w_ok && zero && three && r.accumulated() == 3;
        log(format!("[history] count=16 -> reset=0 -> 3: {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;

        let acc = r.read_accum().await;
        let tex = r.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: W as u32, height: H as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut enc = r.device().create_command_encoder(&Default::default());
        r.composite(&mut enc, &view, &params);
        r.queue().submit([enc.finish()]);
        let px = read_texture(&r, &tex, W as u32, H as u32).await;
        let mut worst = 0.0f64;
        for i in [0usize, 47 * W + 48, 48 * W + 48, 60 * W + 30, W * H - 1] {
            let a = acc[i];
            for c in 0..3 {
                let lin = (a[c] / a[3].max(1.0)) as f64;
                let expect = srgb_oetf(lin.clamp(0.0, 1.0)) * 255.0;
                worst = worst.max((px[i * 4 + c] as f64 - expect).abs());
            }
        }
        let pass = worst <= 1.01;
        log(format!("[composite] max |8-bit error| vs single-OETF reference = {worst:.2} {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }

    // A large subpixel count forces 2D dispatch splitting (> 65535 groups).
    {
        let (w, h, s) = (2048u32, 1024u32, 4u32);
        match r.resize(w, h, s) {
            Ok(()) => {
                let gg = [GpuGaussian::from_oracle(&iso(6.0, 0.5, DVec3::new(0.9, 0.4, 0.2)))];
                let _ = r.set_gaussians(&gg);
                r.reset_stats();
                let p = RenderParams { spp_side: s, background: BG, ..Default::default() };
                let cam = OracleCamera { focal_x: 900.0, focal_y: 900.0, ..Default::default() };
                let res = r.render_ensembles(&cam, &p, 1, 1);
                let acc = r.read_accum().await;
                let corner_bg = (acc[0][0] - BG[0]).abs() < 1e-6 && (acc[acc.len() - 1][2] - BG[2]).abs() < 1e-6;
                let centre = acc[(h as usize / 2) * w as usize + w as usize / 2];
                let covered = (centre[0] - BG[0]).abs() > 1e-3;
                let st = r.read_stats().await;
                let pass = res.is_ok() && corner_bg && covered && st.orphan_subpixels == 0;
                log(format!(
                    "[2D dispatch] {w}x{h} spp={} ({} subpixels): corners=bg {corner_bg}, centre covered {covered}, orphan={} {}",
                    s * s,
                    w * h * s * s,
                    st.orphan_subpixels,
                    if pass { "ok" } else { "FAIL" }
                ));
                ok &= pass;
            }
            Err(e) => log(format!("[2D dispatch] skipped: {e}")),
        }
    }
    log(if ok { "PHASE2: PASS".into() } else { "PHASE2: FAIL".into() });
    ok
}

// ------------------------------------------------------------------------------------------------
// Phase 3: PBVR methods (Proportional / Extinction, C0..C3, C3+R, centre depth) and ViewConditioned.
// Thresholds, fixed before the first run: GPU vs CPU oracle (same per-particle streams) PSNR >= 50 dB and
// point/candidate counts within 4 sigma; C3+R and ViewConditioned vs the analytic GPS image PSNR >= 30 dB.

struct PbvrCase {
    name: &'static str,
    method: Method,
    cal: Calibration,
    radial: bool,
    centre: bool,
    jitter: bool,
}

fn pbvr_params(base: &RenderParams, c: &PbvrCase) -> RenderParams {
    RenderParams {
        method: c.method,
        pbvr: PbvrParams { calibration: c.cal, radial_correction: c.radial, centre_depth: c.centre, low_pass_jitter: c.jitter, ..PbvrParams::default() },
        max_points_per_splat: 1 << 24,
        ..*base
    }
}

fn oracle_options(c: &PbvrCase) -> ParticleOptions {
    let level = match c.cal {
        Calibration::None => gps_core::CalibrationLevel::None,
        Calibration::ObjectZoom => gps_core::CalibrationLevel::ObjectZoom,
        Calibration::PerSplatDepth => gps_core::CalibrationLevel::PerSplatDepth,
        Calibration::PerSplatFootprint => gps_core::CalibrationLevel::PerSplatFootprint,
    };
    ParticleOptions {
        level,
        rule: if c.method == Method::Proportional { gps_core::OpacityRule::Proportional } else { gps_core::OpacityRule::Extinction },
        radial_correction: c.radial,
        centre_depth: c.centre,
        low_pass_jitter: c.jitter,
        per_particle_seed: true,
        ..Default::default()
    }
}

async fn pbvr_vs_oracle(
    r: &mut GpsRenderer, log: &mut dyn FnMut(String), scene_name: &str, scene: &[Gaussian3D], base: &RenderParams, c: &PbvrCase, sets: u32,
) -> (bool, Option<Image>) {
    let bg = DVec3::new(BG[0] as f64, BG[1] as f64, BG[2] as f64);
    let params = pbvr_params(base, c);
    let acc = match render_mean(r, scene, &params, sets, 1).await {
        Ok(a) => a,
        Err(e) => {
            log(format!("[{scene_name} / {}] render failed: {e}", c.name));
            return (false, None);
        }
    };
    let gpu = to_image(&acc);
    let st = r.read_stats().await;
    let (cpu, cst) = render_particles_3d(scene, camera(), W, H, SPP_SIDE as usize, sets as usize, bg, 1, &oracle_options(c));
    let p_cpu = psnr(&gpu, &cpu);
    let tol_pts = 4.0 * (2.0 * cst.points).sqrt().max(1.0);
    let tol_cand = 4.0 * (2.0 * cst.candidates).sqrt().max(1.0);
    let pass = p_cpu >= 50.0
        && (st.points as f64 - cst.points).abs() <= tol_pts
        && (st.candidates as f64 - cst.candidates).abs() <= tol_cand
        && st.orphan_subpixels == 0
        && st.truncated_splats == 0;
    log(format!(
        "[{scene_name} / {}] vs CPU PSNR={p_cpu:.1} dB | points gpu={} cpu={:.0} | candidates gpu={} cpu={:.0} | orphan={} truncated={} {}",
        c.name,
        st.points,
        cst.points,
        st.candidates,
        cst.candidates,
        st.orphan_subpixels,
        st.truncated_splats,
        if pass { "ok" } else { "FAIL" }
    ));
    (pass, Some(gpu))
}

pub async fn run_phase3(adapter: &wgpu::Adapter, log: &mut dyn FnMut(String)) -> bool {
    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor { label: Some("gps-phase3"), required_limits: adapter.limits(), ..Default::default() })
        .await
    {
        Ok(d) => d,
        Err(e) => {
            log(format!("request_device failed: {e}"));
            return false;
        }
    };
    let mut r = match GpsRenderer::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, W as u32, H as u32, SPP_SIDE) {
        Ok(r) => r,
        Err(e) => {
            log(format!("renderer init failed: {e}"));
            return false;
        }
    };
    let base = RenderParams { spp_side: SPP_SIDE, background: BG, ..Default::default() };
    let bg = DVec3::new(BG[0] as f64, BG[1] as f64, BG[2] as f64);
    let all = scenes();
    let mut ok = true;

    let c = |name, method, cal, radial, centre, jitter| PbvrCase { name, method, cal, radial, centre, jitter };
    let cases = [
        c("Proportional C0", Method::Proportional, Calibration::None, false, false, true),
        c("Proportional C3", Method::Proportional, Calibration::PerSplatFootprint, false, false, true),
        c("Extinction C0", Method::Extinction, Calibration::None, false, false, true),
        c("Extinction C1", Method::Extinction, Calibration::ObjectZoom, false, false, true),
        c("Extinction C2", Method::Extinction, Calibration::PerSplatDepth, false, false, true),
        c("Extinction C3", Method::Extinction, Calibration::PerSplatFootprint, false, false, true),
        c("Extinction C3 no-jitter", Method::Extinction, Calibration::PerSplatFootprint, false, false, false),
        c("Extinction C3 centre-depth", Method::Extinction, Calibration::PerSplatFootprint, false, true, true),
        c("Extinction C3+R", Method::Extinction, Calibration::PerSplatFootprint, true, false, true),
        c("Extinction C3+R centre-depth", Method::Extinction, Calibration::PerSplatFootprint, true, true, true),
    ];
    // single o=0.5, front/back layers, anisotropic
    for &si in &[1usize, 3, 4] {
        let (sname, scene) = (all[si].0, &all[si].1);
        let ana = render_analytic(scene, camera(), W, H, bg);
        for case in &cases {
            let (pass, gpu) = pbvr_vs_oracle(&mut r, log, sname, scene, &base, case, 120).await;
            ok &= pass;
            if let Some(g) = gpu {
                let (p_a, s_a) = (psnr(&g, &ana), ssim(&g, &ana, W, H));
                // C3+R is built to reproduce the GPS projected intensity; the others are reported for comparison.
                let must = case.radial;
                let pass_a = !must || p_a >= 30.0;
                let tag = if !must {
                    ""
                } else if pass_a {
                    "  (required >= 30 dB) ok"
                } else {
                    "  (required >= 30 dB) FAIL"
                };
                log(format!("    vs analytic GPS: PSNR={p_a:.1} dB SSIM={s_a:.4}{tag}"));
                ok &= pass_a;
            }
        }
    }

    // ViewConditioned has no CPU oracle (the C++ one is GPU-only): check it against GPS itself.
    for &si in &[1usize, 3, 4] {
        let (sname, scene) = (all[si].0, &all[si].1);
        let ana = render_analytic(scene, camera(), W, H, bg);
        let vc = c("ViewConditioned", Method::ViewConditioned, Calibration::None, false, false, true);
        // The candidate set (base_k per unit extinction) must be large enough to reach the GPS target.
        let mut vp = pbvr_params(&base, &vc);
        vp.pbvr.base_k = 4096.0;
        let acc = render_mean(&mut r, scene, &vp, 120, 1).await;
        let st_vc = r.read_stats().await;
        let gps = render_mean(&mut r, scene, &RenderParams { max_points_per_splat: 1 << 24, ..base }, 120, 1).await;
        let st_gps = r.read_stats().await;
        let (Ok(acc), Ok(_)) = (acc, gps) else {
            log(format!("[{sname} / ViewConditioned] render failed"));
            ok = false;
            continue;
        };
        let img = to_image(&acc);
        let (p_a, s_a) = (psnr(&img, &ana), ssim(&img, &ana, W, H));
        let tol = 4.0 * (2.0 * st_gps.points as f64).sqrt();
        let count_ok = (st_vc.points as f64 - st_gps.points as f64).abs() <= tol && st_vc.undercovered_splats == 0;
        let pass = p_a >= 30.0 && count_ok && st_vc.orphan_subpixels == 0;
        log(format!(
            "[{sname} / ViewConditioned] vs analytic PSNR={p_a:.1} dB SSIM={s_a:.4} | kept points={} (GPS {}, 4-sigma {tol:.0}) candidates={} undercovered={} orphan={} {}",
            st_vc.points,
            st_gps.points,
            st_vc.candidates,
            st_vc.undercovered_splats,
            st_vc.orphan_subpixels,
            if pass { "ok" } else { "FAIL" }
        ));
        ok &= pass;
    }

    // A too-small candidate set must be reported, not silently under-covered (the default base_k = 512).
    {
        let vc = c("ViewConditioned", Method::ViewConditioned, Calibration::None, false, false, true);
        let _ = render_mean(&mut r, &all[1].1, &pbvr_params(&base, &vc), 8, 1).await;
        let st = r.read_stats().await;
        let pass = st.undercovered_splats > 0;
        log(format!("[ViewConditioned base_k=512] under-covered splats reported: {} {}", st.undercovered_splats, if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }

    // 70,000 tiny splats: three scan levels with the PBVR proj layout; counts compared with the oracle.
    {
        let mut many = Vec::new();
        for i in 0..70_000u32 {
            let (gx, gy) = ((i % 280) as f64 / 280.0 - 0.5, (i / 280) as f64 / 250.0 - 0.5);
            many.push(Gaussian3D {
                pos: DVec3::new(gx * 0.9, gy * 0.9, 6.0 + (i % 7) as f64 * 0.1),
                log_scale: DVec3::splat(0.012_f64.ln()),
                opacity: 0.3 + 0.6 * ((i * 37 % 100) as f64 / 100.0),
                color: DVec3::new((i % 5) as f64 / 5.0, (i % 3) as f64 / 3.0, 0.5),
                ..Default::default()
            });
        }
        let case = c("Extinction C3+R", Method::Extinction, Calibration::PerSplatFootprint, true, false, true);
        let (pass, _) = pbvr_vs_oracle(&mut r, log, "70k splats", &many, &base, &case, 6).await;
        ok &= pass;
    }

    // Determinism with a PBVR method.
    {
        let scene = &all[3].1;
        let case = c("Extinction C3+R", Method::Extinction, Calibration::PerSplatFootprint, true, false, true);
        let p = pbvr_params(&base, &case);
        let a = render_mean(&mut r, scene, &p, 8, 500).await.unwrap_or_default();
        let b = render_mean(&mut r, scene, &p, 8, 500).await.unwrap_or_default();
        let d = render_mean(&mut r, scene, &p, 8, 600).await.unwrap_or_default();
        let pass = !a.is_empty() && a == b && a != d;
        log(format!("[determinism / PBVR] same seed identical, different seed differs: {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }

    // Method comparison (information): per-method point counts and agreement with the analytic GPS image.
    {
        let (sname, scene) = (all[3].0, &all[3].1);
        let ana = render_analytic(scene, camera(), W, H, bg);
        log(format!("[comparison on '{sname}', 120 ensembles] method | points per ensemble | PSNR vs analytic GPS"));
        let rows = [
            ("GPS", RenderParams { max_points_per_splat: 1 << 24, ..base }),
            ("PBVR Extinction C3+R", pbvr_params(&base, &c("", Method::Extinction, Calibration::PerSplatFootprint, true, false, true))),
            ("PBVR ViewConditioned", { let mut p = pbvr_params(&base, &c("", Method::ViewConditioned, Calibration::None, false, false, true)); p.pbvr.base_k = 4096.0; p }),
            ("PBVR Extinction C3", pbvr_params(&base, &c("", Method::Extinction, Calibration::PerSplatFootprint, false, false, true))),
            ("PBVR Proportional C3", pbvr_params(&base, &c("", Method::Proportional, Calibration::PerSplatFootprint, false, false, true))),
        ];
        for (name, p) in rows {
            if let Ok(acc) = render_mean(&mut r, scene, &p, 120, 1).await {
                let st = r.read_stats().await;
                log(format!("    {name:<24} {:>9} | {:.1} dB", st.points / 120, psnr(&to_image(&acc), &ana)));
            }
        }
    }

    log(if ok { "PHASE3: PASS".into() } else { "PHASE3: FAIL".into() });
    ok
}

// ------------------------------------------------------------------------------------------------
// Phase 4: SH bands 1..3. Each degree is checked against the CPU `eval_sh` evaluated for the same
// camera -> splat direction (threshold fixed in advance: GPU vs CPU oracle PSNR >= 60 dB).

pub async fn run_phase4(adapter: &wgpu::Adapter, log: &mut dyn FnMut(String)) -> bool {
    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor { label: Some("gps-phase4"), required_limits: adapter.limits(), ..Default::default() })
        .await
    {
        Ok(d) => d,
        Err(e) => {
            log(format!("request_device failed: {e}"));
            return false;
        }
    };
    let mut r = match GpsRenderer::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, W as u32, H as u32, SPP_SIDE) {
        Ok(r) => r,
        Err(e) => {
            log(format!("renderer init failed: {e}"));
            return false;
        }
    };
    let bg = DVec3::new(BG[0] as f64, BG[1] as f64, BG[2] as f64);
    let mut ok = true;

    // Four opaque-ish splats at different view angles with deterministic pseudo-random SH coefficients.
    let stride = gps_core::sh_rest_per_channel(3);
    let mut seed = 99u32;
    let mut rnd = || {
        seed = gps_core::pcg_hash(seed);
        seed as f64 / 4_294_967_296.0 - 0.5
    };
    let mut dcs = Vec::new();
    let mut rests: Vec<f64> = Vec::new();
    let mut base = Vec::new();
    for i in 0..4 {
        let dc = DVec3::new(rnd(), rnd(), rnd()) * 1.2;
        dcs.push(dc);
        for _ in 0..3 * stride {
            rests.push(rnd() * 0.8);
        }
        base.push(Gaussian3D {
            pos: DVec3::new(-0.9 + 0.6 * i as f64, 0.25 * (i as f64 - 1.5), 5.0 + 0.7 * i as f64),
            log_scale: DVec3::splat(0.22_f64.ln()),
            opacity: 0.85,
            ..Default::default()
        });
    }
    let gg: Vec<GpuGaussian> = base
        .iter()
        .zip(&dcs)
        .map(|(g, dc)| {
            let c = *dc * gps_core::SH_C0 + DVec3::splat(0.5);
            GpuGaussian { color: [c.x as f32, c.y as f32, c.z as f32, 0.0], ..GpuGaussian::from_oracle(g) }
        })
        .collect();
    let rest32: Vec<f32> = rests.iter().map(|&v| v as f32).collect();
    if let Err(e) = r.set_gaussians_sh(&gg, &rest32, 3) {
        log(format!("set_gaussians_sh failed: {e}"));
        return false;
    }
    let cam = camera();
    let mut images = Vec::new();
    for deg in 0..=3u32 {
        let expected: Vec<Gaussian3D> = base
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let dir = (g.pos - cam.view_pos).normalize();
                let rest = &rests[i * 3 * stride..(i + 1) * 3 * stride];
                let col = gps_core::eval_sh(deg as usize, dcs[i], Some(rest), dir, stride);
                Gaussian3D { color: col, ..*g }
            })
            .collect();
        let params = RenderParams { spp_side: SPP_SIDE, background: BG, sh_degree: deg, ..Default::default() };
        r.reset_accum();
        r.reset_stats();
        let mut done = 0;
        while done < SETS {
            let n = (SETS - done).min(MAX_ENSEMBLES_PER_SUBMIT);
            if let Err(e) = r.render_ensembles(&cam, &params, 1 + done, n) {
                log(format!("render failed: {e}"));
                return false;
            }
            done += n;
        }
        let gpu = to_image(&r.read_accum().await);
        let opt = MonteCarloOptions { per_particle_seed: true, ..Default::default() };
        let (cpu, _) = render_monte_carlo(&expected, cam, W, H, SPP_SIDE as usize, SETS as usize, bg, 1, &opt);
        let p = psnr(&gpu, &cpu);
        let pass = p >= 60.0;
        log(format!("[SH degree {deg}] GPU vs CPU eval_sh oracle PSNR={p:.1} dB {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;
        images.push(gpu);
    }
    // The bands must actually change the image (otherwise the comparison above proves nothing).
    for d in 1..=3 {
        let p = psnr(&images[d], &images[d - 1]);
        let pass = p < 45.0;
        log(format!("[SH sensitivity] degree {d} vs {}: PSNR={p:.1} dB (must differ, < 45 dB) {}", d - 1, if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }
    // A scene loaded without SH must fall back to SH0 even if degree 3 is requested.
    {
        let _ = r.set_gaussians(&gg);
        let params = RenderParams { spp_side: SPP_SIDE, background: BG, sh_degree: 3, ..Default::default() };
        r.reset_accum();
        let _ = r.render_ensembles(&cam, &params, 1, 4);
        let acc = r.read_accum().await;
        let pass = r.sh_max_degree() == 0 && acc.iter().all(|p| p[0].is_finite());
        log(format!("[SH fallback] scene without SH bands renders at degree 0: {}", if pass { "ok" } else { "FAIL" }));
        ok &= pass;
    }
    log(if ok { "PHASE4: PASS".into() } else { "PHASE4: FAIL".into() });
    ok
}
