//! Compares the Rust oracle against images produced by the ORIGINAL C++ oracle on identical
//! scenes/seeds (tests/cpp_reference/dump_oracle.cpp -> tests/fixtures/cpp_*.f64).
//! RNG streams are bit-identical, so only libm last-ulp differences may flip a rare particle.
use glam::{DQuat, DVec3};
use gps_core::oracle::*;
use gps_core::{CalibrationLevel, OpacityRule};

const W: usize = 48;
const H: usize = 48;
const SPP: usize = 2;
const SETS: usize = 20;
const BG: DVec3 = DVec3::new(0.05, 0.05, 0.08);

fn cam() -> OracleCamera {
    OracleCamera { focal_x: 100.0, focal_y: 100.0, ..Default::default() }
}
fn iso(z: f64, o: f64, col: DVec3, x: f64) -> Gaussian3D {
    Gaussian3D { pos: DVec3::new(x, 0.0, z), log_scale: DVec3::splat(0.15_f64.ln()), opacity: o, color: col, ..Default::default() }
}
fn aniso() -> Vec<Gaussian3D> {
    vec![Gaussian3D {
        pos: DVec3::new(0.1, -0.05, 6.0),
        log_scale: DVec3::new(0.28_f64.ln(), 0.10_f64.ln(), 0.06_f64.ln()),
        rot: DQuat::from_xyzw(0.20, -0.45, 0.30, 0.82).normalize(),
        opacity: 0.7,
        color: DVec3::new(0.3, 0.5, 0.95),
    }]
}
fn layers() -> Vec<Gaussian3D> {
    vec![
        iso(9.0, 0.8, DVec3::new(0.15, 0.7, 0.25), 0.1),
        iso(5.0, 0.6, DVec3::new(0.9, 0.2, 0.2), -0.1),
        iso(7.0, 0.4, DVec3::new(0.2, 0.3, 0.9), 0.0),
    ]
}
fn load(name: &str) -> Vec<DVec3> {
    let p = format!("{}/../../tests/fixtures/cpp_{name}.f64", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e} (run tests/cpp_reference/run.ps1)"));
    b.chunks_exact(24)
        .map(|c| {
            let f = |i: usize| f64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
            DVec3::new(f(0), f(1), f(2))
        })
        .collect()
}
/// (max abs error, pixels differing by > 1e-9, PSNR)
fn diff(name: &str, rust: &[DVec3]) -> (f64, usize, f64) {
    let cpp = load(name);
    assert_eq!(cpp.len(), rust.len());
    let max = cpp.iter().zip(rust).map(|(a, b)| (*a - *b).abs().max_element()).fold(0.0, f64::max);
    let bad = cpp.iter().zip(rust).filter(|(a, b)| (**a - **b).abs().max_element() > 1e-9).count();
    let p = psnr(&cpp, rust);
    println!("{name}: max_abs={max:.3e} pixels>1e-9={bad}/{} psnr={p:.1}", cpp.len());
    (max, bad, p)
}

#[test]
fn analytic_matches_cpp() {
    let (max, _, _) = diff("analytic_layers", &render_analytic(&layers(), cam(), W, H, BG));
    assert!(max < 1e-9, "max={max}");
}
#[test]
fn mc_aniso_matches_cpp() {
    let (img, _) = render_monte_carlo(&aniso(), cam(), W, H, SPP, SETS, BG, 11, &Default::default());
    let (_, bad, p) = diff("mc_aniso", &img);
    assert!(bad <= 23 && p >= 60.0); // <= 1% of pixels
}
#[test]
fn mc_layers_matches_cpp() {
    let (img, _) = render_monte_carlo(&layers(), cam(), W, H, SPP, SETS, BG, 5, &Default::default());
    let (_, bad, p) = diff("mc_layers", &img);
    assert!(bad <= 23 && p >= 60.0);
}
#[test]
fn mc_footprint2_matches_cpp() {
    let opt = MonteCarloOptions { footprint_subpixels: 2, compensate_blur: true, ..Default::default() };
    let (img, _) = render_monte_carlo(&layers(), cam(), W, H, SPP, SETS, BG, 5, &opt);
    let (_, bad, p) = diff("mc_footprint2", &img);
    assert!(bad <= 23 && p >= 60.0);
}
#[test]
fn particles_c3r_matches_cpp() {
    let opt = ParticleOptions { radial_correction: true, ..Default::default() };
    let (img, _) = render_particles_3d(&layers(), cam(), W, H, SPP, SETS, BG, 3, &opt);
    let (_, bad, p) = diff("particles_c3r", &img);
    assert!(bad <= 23 && p >= 60.0);
}
#[test]
fn particles_c1_linearized_matches_cpp() {
    let opt = ParticleOptions {
        level: CalibrationLevel::ObjectZoom,
        rule: OpacityRule::Proportional,
        linearized_projection: true,
        centre_depth: true,
        ..Default::default()
    };
    let (img, _) = render_particles_3d(&aniso(), cam(), W, H, SPP, SETS, BG, 3, &opt);
    let (_, bad, p) = diff("particles_c1_lin", &img);
    assert!(bad <= 23 && p >= 60.0);
}
