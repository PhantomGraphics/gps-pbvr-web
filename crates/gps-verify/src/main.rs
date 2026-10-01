//! Native runner.
//!   gps-verify [phase0|phase2|all]   correctness checks, exit code 0 only if every check passes
//!   gps-verify bench                 splat-parallel vs particle-parallel timing (informational)

use gps_core::camera::OrbitCamera;
use gps_render::gps::{Calibration, GpsRenderer, GpuGaussian, Method, PbvrParams, RenderParams, RenderPath};

fn adapter() -> wgpu::Adapter {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no adapter: {e}");
            std::process::exit(2);
        }
    }
}

/// The viewer's demo-like scene: `n` Gaussians on a sphere shell, plus `big` very large splats
/// (the load-imbalance case: a few splats own most of the particles).
fn scene(n: usize, big: usize) -> Vec<GpuGaussian> {
    let mut s = 12345u32;
    let mut rnd = move || {
        s = gps_core::pcg_hash(s);
        s as f32 * (1.0 / 4_294_967_296.0)
    };
    let mut v = Vec::new();
    for i in 0..n {
        let (u, w) = (rnd() * std::f32::consts::TAU, rnd() * 2.0 - 1.0);
        let ring = 0.6 + 0.25 * (i % 3) as f32;
        let (x, y, z) = (ring * (1.0 - w * w).sqrt() * u.cos(), w * ring, ring * (1.0 - w * w).sqrt() * u.sin());
        let sc = (0.05 + 0.05 * rnd()).ln();
        v.push(GpuGaussian {
            pos_opacity: [x, y, z, 0.5 + 0.45 * rnd()],
            log_scale: [sc, sc, sc, 0.0],
            rot_wxyz: [1.0, 0.0, 0.0, 0.0],
            color: [0.5 + 0.4 * x, 0.5 + 0.4 * y, 0.5 + 0.4 * z, 0.0],
        });
    }
    for i in 0..big {
        let sc = (0.9 + 0.2 * i as f32).ln();
        v.push(GpuGaussian {
            pos_opacity: [0.0, 0.0, 0.0, 0.8],
            log_scale: [sc, sc, 0.1f32.ln(), 0.0],
            rot_wxyz: [1.0, 0.0, 0.0, 0.0],
            color: [0.3, 0.3, 0.35, 0.0],
        });
    }
    v
}

fn bench() {
    let adapter = adapter();
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() })).expect("device");
    let cam_of = |w: u32, h: u32| {
        let mut c = OrbitCamera { pitch: 0.25, yaw: 0.6, ..Default::default() };
        c.frame_sphere(glam::DVec3::ZERO, 1.0);
        c.to_camera(w, h)
    };
    println!("{:<40} {:>11} {:>12} {:>13}", "case (ms per ensemble, lower is better)", "points/ens", "splat-par", "particle-par");
    // (label, splats, huge splats, w, h, spp side, run the splat-parallel path too). That path is skipped
    // for the huge-splat case: one thread would own millions of particles and trip the OS GPU watchdog.
    for (label, n, big, w, h, spp, primitive) in [
        ("320x180 spp1, 600 splats", 600, 0, 320, 180, 1, true),
        ("640x360 spp1, 600 splats", 600, 0, 640, 360, 1, true),
        ("640x360 spp4, 600 splats", 600, 0, 640, 360, 2, true),
        ("640x360 spp1, 600 + 2 huge splats", 600, 2, 640, 360, 1, false),
        ("640x360 spp1, 20000 splats", 20_000, 0, 640, 360, 1, true),
    ] {
        let mut r = GpsRenderer::new(device.clone(), queue.clone(), wgpu::TextureFormat::Rgba8Unorm, w, h, spp).expect("renderer");
        r.set_gaussians(&scene(n, big)).expect("scene");
        let cam = cam_of(w, h);
        let mut ms = [0.0f64; 2];
        let mut points = 0u64;
        for (k, path) in [RenderPath::Primitive, RenderPath::Particle].into_iter().enumerate() {
            if k == 0 && !primitive {
                ms[0] = f64::NAN;
                continue;
            }
            let p = RenderParams { spp_side: spp, path, ..Default::default() };
            r.render_ensembles(&cam, &p, 1, 1).expect("warmup");
            r.wait_idle();
            r.reset_stats();
            let t = std::time::Instant::now();
            let reps = 12;
            for i in 0..reps {
                r.render_ensembles(&cam, &p, 100 + i, 1).expect("render");
                r.wait_idle();
            }
            ms[k] = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            points = pollster::block_on(r.read_stats()).points / reps as u64;
        }
        if primitive {
            println!("{label:<40} {points:>11} {:>10.2}ms {:>11.2}ms  ({:.1}x)", ms[0], ms[1], ms[0] / ms[1]);
        } else {
            println!("{label:<40} {points:>11} {:>12} {:>11.2}ms", "(watchdog)", ms[1]);
        }
    }
}

/// `gps-verify bench-ply <file.ply> [WxH] [spp_side]`: per-ensemble GPU time on a real 3DGS file for each method.
fn bench_ply(path: &str, size: &str, spp: u32) {
    let bytes = std::fs::read(path).expect("read ply");
    let g = gps_io::load_ply(&bytes).expect("load ply");
    let gpu: Vec<GpuGaussian> = (0..g.count).map(|i| GpuGaussian::from_activated(&g.activated(i))).collect();
    let adapter = adapter();
    println!("adapter: {} | {} Gaussians, SH degree {}", adapter.get_info().name, g.count, g.sh_degree);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() })).expect("device");
    let (w, h) = size.split_once('x').map(|(a, b)| (a.parse().unwrap_or(480), b.parse().unwrap_or(270))).unwrap_or((480, 270));
    let mut r = GpsRenderer::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, w, h, spp).expect("renderer");
    if g.sh_degree > 0 {
        r.set_gaussians_sh(&gpu, &g.sh_rest, g.sh_degree as u32).expect("scene");
    } else {
        r.set_gaussians(&gpu).expect("scene");
    }
    let pts = gpu.iter().map(|p| glam::DVec3::new(p.pos_opacity[0] as f64, p.pos_opacity[1] as f64, p.pos_opacity[2] as f64));
    let (c, rad) = OrbitCamera::robust_sphere(pts).expect("non-empty");
    println!("{:<34} {:>13} {:>10} {:>12}", "view / method", "points/ens", "ms/ens", "truncated");
    let env_max: u32 = std::env::var("MAXPTS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
    for (vname, dist_mul) in [("near (x1.25)", 1.0), ("mid (x2.3)", 1.85), ("far (x4)", 3.2)] {
        let mut cam = OrbitCamera { up: -glam::DVec3::Y, pitch: 0.25, yaw: 0.6, ..Default::default() };
        cam.frame_sphere(c, rad);
        cam.distance *= dist_mul;
        let cam = cam.to_camera(w, h);
        for (mname, method, cal, radial) in [("GPS", Method::Gps, Calibration::PerSplatFootprint, false), ("PBVR Ext C3+R", Method::Extinction, Calibration::PerSplatFootprint, true)] {
            let p = RenderParams { spp_side: spp, max_points_per_splat: env_max, method, pbvr: PbvrParams { calibration: cal, radial_correction: radial, ..Default::default() }, ..Default::default() };
            r.render_ensembles(&cam, &p, 1, 1).expect("warmup");
            r.wait_idle();
            r.reset_stats();
            let t = std::time::Instant::now();
            let reps = 6;
            for i in 0..reps {
                r.render_ensembles(&cam, &p, 100 + i, 1).expect("render");
                r.wait_idle();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            let st = pollster::block_on(r.read_stats());
            println!("{:<34} {:>13} {:>8.1}ms {:>12}", format!("{vname} {mname}"), st.points / reps as u64, ms, st.truncated_splats / reps as u64);
        }
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if which == "bench-ply" {
        let a: Vec<String> = std::env::args().collect();
        bench_ply(a.get(2).expect("path"), a.get(3).map(String::as_str).unwrap_or("480x270"), a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1));
        return;
    }
    if which == "bench" {
        bench();
        return;
    }
    let adapter = adapter();
    let mut ok = true;
    if which == "all" || which == "phase0" {
        ok &= pollster::block_on(gps_render::run_phase0(&adapter, &mut |l| println!("{l}")));
    }
    if which == "phase4" || which == "all" {
        ok &= pollster::block_on(gps_render::verify::run_phase4(&adapter, &mut |l| println!("{l}")));
    }
    if which == "phase3" || which == "all" {
        ok &= pollster::block_on(gps_render::verify::run_phase3(&adapter, &mut |l| println!("{l}")));
    }
    if which == "all" || which == "phase2" {
        ok &= pollster::block_on(gps_render::verify::run_phase2(&adapter, &mut |l| println!("{l}")));
    }
    std::process::exit(if ok { 0 } else { 1 });
}
