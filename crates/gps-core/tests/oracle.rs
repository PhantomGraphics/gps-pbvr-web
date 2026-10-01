//! Rust port of PointCloudTest/GaussianPointOracleTest.cpp (same scenes, same thresholds:
//! PSNR >= 35 dB, SSIM >= 0.98), plus camera checks.
use glam::{DQuat, DVec3};
use gps_core::oracle::*;

const W: usize = 96;
const H: usize = 96;
const SPP_SIDE: usize = 2;
const SETS: usize = 300;
const BG: DVec3 = DVec3::new(0.05, 0.05, 0.08);

fn cam() -> OracleCamera {
    OracleCamera { focal_x: 200.0, focal_y: 200.0, ..Default::default() }
}
fn iso(z: f64, o: f64) -> Gaussian3D {
    Gaussian3D { pos: DVec3::new(0.0, 0.0, z), log_scale: DVec3::splat(0.15_f64.ln()), opacity: o, color: DVec3::new(0.9, 0.4, 0.2), ..Default::default() }
}
fn mc(scene: &[Gaussian3D]) -> Image {
    render_monte_carlo(scene, cam(), W, H, SPP_SIDE, SETS, BG, 1, &Default::default()).0
}

#[test]
fn single_gaussian_converges() {
    for o in [0.1, 0.5, 0.9] {
        let scene = [iso(6.0, o)];
        let (r, m) = (render_analytic(&scene, cam(), W, H, BG), mc(&scene));
        let (p, s) = (psnr(&m, &r), ssim(&m, &r, W, H));
        println!("single o={o} PSNR={p:.2} SSIM={s:.4}");
        assert!(p >= 35.0 && s >= 0.98, "o={o} psnr={p} ssim={s}");
    }
}

#[test]
fn front_back_layers_converge_and_are_order_independent() {
    let back = Gaussian3D { color: DVec3::new(0.15, 0.7, 0.25), ..iso(9.0, 0.8) };
    let front = Gaussian3D { color: DVec3::new(0.9, 0.2, 0.2), ..iso(5.0, 0.6) };
    let scene = [back, front];
    let (r, m) = (render_analytic(&scene, cam(), W, H, BG), mc(&scene));
    assert!(psnr(&m, &r) >= 35.0 && ssim(&m, &r, W, H) >= 0.98);
    assert!(psnr(&m, &mc(&[front, back])) >= 40.0);
}

#[test]
fn anisotropic_rotated_converges() {
    // C++ dquat(w,x,y,z) = (0.82, 0.20, -0.45, 0.30)
    let g = Gaussian3D {
        pos: DVec3::new(0.0, 0.0, 6.0),
        log_scale: DVec3::new(0.28_f64.ln(), 0.10_f64.ln(), 0.06_f64.ln()),
        rot: DQuat::from_xyzw(0.20, -0.45, 0.30, 0.82).normalize(),
        opacity: 0.7,
        color: DVec3::new(0.3, 0.5, 0.95),
    };
    let scene = [g];
    let (r, m) = (render_analytic(&scene, cam(), W, H, BG), mc(&scene));
    assert!(psnr(&m, &r) >= 35.0 && ssim(&m, &r, W, H) >= 0.98);
}

#[test]
fn transparent_scene_is_background() {
    let (m, _) = render_monte_carlo(&[iso(6.0, 0.0)], cam(), W, H, SPP_SIDE, 8, BG, 1, &Default::default());
    assert!(m.iter().all(|p| (*p - BG).abs().max_element() < 1e-9));
}

#[test]
fn metrics_sanity() {
    let a = vec![DVec3::splat(0.5); 64 * 64];
    assert!(psnr(&a, &a) >= 100.0 && (ssim(&a, &a, 64, 64) - 1.0).abs() < 1e-9);
    let b: Vec<_> = a.iter().map(|p| *p + DVec3::splat(0.1)).collect();
    assert!(psnr(&a, &b) < 25.0);
}

#[test]
fn look_at_camera_projects_target_to_centre() {
    let eye = DVec3::new(3.0, 2.0, -5.0);
    let target = DVec3::new(0.0, 0.5, 1.0);
    let c = OracleCamera::look_at(eye, target, DVec3::Y, 0.9, 128, 96);
    let cam_space = c.view_rot * (target - eye);
    assert!(cam_space.x.abs() < 1e-9 && cam_space.y.abs() < 1e-9 && cam_space.z > 0.0);
    assert!((c.view_rot * c.view_rot.transpose() - glam::DMat3::IDENTITY).abs_diff_eq(glam::DMat3::ZERO, 1e-12));
    // world up maps to image-up (negative camera y)
    let up_cam = c.view_rot * DVec3::Y;
    assert!(up_cam.y < 0.0);
    // vertical fov: a point at tan(fov/2) above centre lands on the top edge
    let d = 4.0;
    let p = eye + (target - eye).normalize() * d;
    let top = p + c.view_rot.transpose() * DVec3::new(0.0, -d * (0.45f64).tan(), 0.0);
    let t = c.view_rot * (top - eye);
    let v = c.focal_y * t.y / t.z + c.cy;
    assert!(v.abs() < 1e-9, "v={v}");
}

#[test]
fn particles_3d_c3r_matches_gps_statistically() {
    // C3+R (footprint calibration + radial correction) reproduces the GPS image (plan NOTE).
    let scene = [iso(6.0, 0.7)];
    let ana = render_analytic(&scene, cam(), W, H, BG);
    let opt = ParticleOptions { radial_correction: true, linearized_projection: true, centre_depth: true, ..Default::default() };
    let (img, st) = render_particles_3d(&scene, cam(), W, H, SPP_SIDE, SETS, BG, 1, &opt);
    let p = psnr(&img, &ana);
    println!("particles C3+R PSNR={p:.2} candidates={}", st.candidates);
    assert!(p >= 30.0, "psnr={p}");
}

#[test]
fn deterministic_for_same_seed_and_distinct_across_seeds() {
    let scene = [iso(6.0, 0.5)];
    let a = render_monte_carlo(&scene, cam(), 32, 32, 2, 4, BG, 7, &Default::default()).0;
    let b = render_monte_carlo(&scene, cam(), 32, 32, 2, 4, BG, 7, &Default::default()).0;
    let c = render_monte_carlo(&scene, cam(), 32, 32, 2, 4, BG, 8, &Default::default()).0;
    assert_eq!(a, b);
    assert_ne!(a, c);
}
