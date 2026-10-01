use glam::{DMat2, DMat3, DQuat, DVec2, DVec3};
use gps_core::vectors::*;
use gps_core::*;

#[test]
fn dilog_matches_vectors() {
    for c in DILOG {
        assert!((dilog(c.x) - c.li2).abs() < 1e-12, "x={} got {}", c.x, dilog(c.x));
    }
}

#[test]
fn inv_dilog_roundtrip() {
    for &y in &[1e-6, 0.01, 0.3, 0.5, 0.77, 0.99, 0.999999] {
        assert!((inv_dilog(dilog(y)) - y).abs() < 1e-9, "y={y}");
    }
}

#[test]
fn sigmoid_matches_vectors() {
    for c in SIGMOID {
        assert!((sigmoid(c.x) - c.s).abs() < 1e-12);
    }
}

#[test]
fn expected_count_matches_vectors() {
    for c in EXPECTED_COUNT {
        let det = c.cov2d_isotropic * c.cov2d_isotropic;
        assert!((expected_point_count(det, c.opacity) - c.expected_n).abs() < 1e-9);
    }
}

#[test]
fn depth_key_vectors_and_order() {
    for c in DEPTH_KEY {
        assert_eq!(ordered_depth_key(c.depth), c.key);
        assert_eq!(ordered_depth_key_inverse(c.key), c.depth);
    }
    assert!(ordered_depth_key(1.5) < ordered_depth_key(2.5));
}

#[test]
fn pcg_known_values_are_stable() {
    // Independently computed (Python) from the C++ pcgHash formula; the WGSL pcg() must match.
    let v: Vec<u32> = [0, 1, 2, 3, 0xdead_beef].iter().map(|&x| pcg_hash(x)).collect();
    assert_eq!(v, vec![129708002, 2831084092, 2055130248, 2131687100, 1730779506]);
    assert_ne!(pcg_hash(0), pcg_hash(1));
    assert_eq!(particle_seed(SeedMode::Deterministic, 3, 4, 0), particle_seed(SeedMode::Deterministic, 3, 4, 99));
    assert_ne!(particle_seed(SeedMode::FrameVarying, 3, 4, 0), particle_seed(SeedMode::FrameVarying, 3, 4, 1));
}

#[test]
fn covariance_isotropic_projection() {
    // Isotropic sigma=0.1 at depth 10, focal 500: pixel sigma = 500*0.1/10 = 5.
    let cam = PinholeCamera { focal_x: 500.0, focal_y: 500.0, tan_fov_x: 1.0, tan_fov_y: 1.0, ..Default::default() };
    let cov3 = covariance_3d(DVec3::splat(0.1_f64.ln()), DQuat::IDENTITY);
    let c2 = covariance_2d(cov3, DVec3::new(0.0, 0.0, 10.0), &cam, 0.0);
    assert!((c2.x_axis.x - 25.0).abs() < 1e-9 && (c2.y_axis.y - 25.0).abs() < 1e-9);
    assert!(c2.x_axis.y.abs() < 1e-9);
}

#[test]
fn cholesky_reconstructs() {
    let a = DMat2::from_cols(DVec2::new(4.0, 1.0), DVec2::new(1.0, 3.0));
    let l = cholesky_2d(a);
    let r = l * l.transpose();
    assert!((r - a).abs_diff_eq(DMat2::ZERO, 1e-12));
}

#[test]
fn rotation_is_orthonormal() {
    let q = DQuat::from_xyzw(0.1, 0.2, 0.3, 0.9);
    let r = quat_to_rotation(q);
    assert!((r * r.transpose()).abs_diff_eq(DMat3::IDENTITY, 1e-12));
}

#[test]
fn poisson_mean_and_variance() {
    for &lam in &[0.5, 4.0, 12.0, 100.0] {
        let mut rng = RandStream::new(12345);
        let n = 20000;
        let xs: Vec<f64> = (0..n).map(|_| poisson_sample(lam, &mut rng) as f64).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        assert!((mean - lam).abs() < 5.0 * (lam / n as f64).sqrt() + 0.02, "lam={lam} mean={mean}");
    }
}

#[test]
fn corrected_radius_cdf_matches_analytic() {
    // F(r) = 1 - Li2(o e^{-r^2/2}) / Li2(o). Inverse sampling must round-trip.
    let o = 0.8;
    for &u in &[0.05, 0.3, 0.5, 0.9] {
        let r = sample_corrected_radius(o, u);
        let f = 1.0 - dilog(o * (-0.5 * r * r).exp()) / dilog(o);
        assert!((f - u).abs() < 1e-9, "u={u} f={f}");
    }
}

#[test]
fn coverage_identity_at_unit_pixel() {
    for &a in &[0.0, 0.2, 0.9] {
        assert!((expected_coverage(a, 1.0) - a).abs() < 1e-12);
    }
}

#[test]
fn radial_keep_bounds() {
    let o = 0.7;
    assert!((radial_keep_probability(o, 0.0) - 1.0).abs() < 1e-12);
    let far = radial_keep_probability(o, 1e6);
    assert!((far - o / -(1.0_f64 - o).ln()).abs() < 1e-9);
}

#[test]
fn sh_dc_only_and_clamp() {
    let c = eval_sh(0, DVec3::new(1.0, 0.0, -10.0), None, DVec3::Z, 0);
    assert!((c.x - (SH_C0 + 0.5)).abs() < 1e-12);
    assert_eq!(c.z, 0.0);
}

#[test]
fn calibration_c3_ignores_base_k() {
    let inp = CalibrationInputs { footprint_area: 10.0, spp: 4.0, ..Default::default() };
    let a = calibrated_count(CalibrationLevel::PerSplatFootprint, OpacityRule::Proportional, 0.5, 1.0, 1.0, &inp);
    let b = calibrated_count(CalibrationLevel::PerSplatFootprint, OpacityRule::Proportional, 0.5, 999.0, 1.0, &inp);
    assert_eq!(a, b);
    assert!((a - 20.0).abs() < 1e-12);
}
