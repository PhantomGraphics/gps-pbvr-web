//! Statistical tests with tolerances fixed a priori (plan section 7: confidence interval AND a
//! practical error bound; never loosened after a failure).
//!   * particle count: mean over seeds within 4 sigma of E[N] (Poisson, sigma = sqrt(E/n))
//!   * per-pixel linear RGB: mean over seeds within 4.5 standard errors of the analytic value,
//!     and the image-wide RMS error bounded by 0.01.
use glam::DVec3;
use gps_core::oracle::*;
use gps_core::{dilog, expected_point_count};

const W: usize = 40;
const H: usize = 40;
const BG: DVec3 = DVec3::new(0.05, 0.05, 0.08);
const SEEDS: u32 = 120;

fn cam() -> OracleCamera {
    OracleCamera { focal_x: 100.0, focal_y: 100.0, ..Default::default() }
}
fn splat(o: f64) -> Gaussian3D {
    Gaussian3D { pos: DVec3::new(0.0, 0.0, 6.0), log_scale: DVec3::splat(0.2_f64.ln()), opacity: o, color: DVec3::new(0.9, 0.4, 0.2), ..Default::default() }
}

fn area_integrated_reference(g: &Gaussian3D) -> Vec<DVec3> {
    let c = cam();
    let pin = gps_core::PinholeCamera { view_rot: c.view_rot, view_pos: c.view_pos, focal_x: c.focal_x, focal_y: c.focal_y, tan_fov_x: 4.0, tan_fov_y: 4.0 };
    let cam_mean = gps_core::world_to_camera(&pin, g.pos);
    let layer = gps_core::Gaussian2D {
        mean: glam::DVec2::new(c.focal_x * cam_mean.x / cam_mean.z + 0.5 * W as f64, c.focal_y * cam_mean.y / cam_mean.z + 0.5 * H as f64),
        cov: gps_core::covariance_2d(gps_core::covariance_3d(g.log_scale, g.rot), g.pos, &pin, c.low_pass),
        opacity: g.opacity,
        color: g.color,
        depth: cam_mean.z,
    };
    const N: usize = 8;
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let mut acc = DVec3::ZERO;
            for j in 0..N {
                for i in 0..N {
                    let p = glam::DVec2::new(x as f64 + (i as f64 + 0.5) / N as f64, y as f64 + (j as f64 + 0.5) / N as f64);
                    acc += gps_core::composite_expected(vec![layer], p, BG, 1.0);
                }
            }
            out.push(acc / (N * N) as f64);
        }
    }
    out
}

#[test]
fn point_count_mean_within_4_sigma() {
    for o in [0.05, 0.3, 0.7, 0.99] {
        let g = splat(o);
        let c = cam();
        // expected total points per set = spp * 2*pi*sqrt(det) * Li2(o), det from the projection
        let pin_cov = {
            let pin = gps_core::PinholeCamera { view_rot: c.view_rot, view_pos: c.view_pos, focal_x: c.focal_x, focal_y: c.focal_y, tan_fov_x: 4.0, tan_fov_y: 4.0 };
            gps_core::covariance_2d(gps_core::covariance_3d(g.log_scale, g.rot), g.pos, &pin, c.low_pass)
        };
        let spp = 4.0;
        let e = spp * expected_point_count(pin_cov.determinant(), o);
        let mut total = 0.0;
        for seed in 0..SEEDS {
            let (_, st) = render_monte_carlo(&[g], cam(), W, H, 2, 1, BG, 1000 + seed, &Default::default());
            total += st.points;
        }
        let mean = total / SEEDS as f64;
        let sigma = (e / SEEDS as f64).sqrt();
        assert!((mean - e).abs() <= 4.0 * sigma, "o={o} mean={mean} expected={e} 4sigma={}", 4.0 * sigma);
        assert!(dilog(o) > 0.0);
    }
}

#[test]
fn pixel_mean_converges_to_analytic_with_ci_and_rms_bound() {
    let scene = [splat(0.6)];
    // The MC estimator integrates coverage over each (sub)pixel area, whereas render_analytic
    // point-samples the pixel centre. The CI is therefore centred on the area-integrated
    // analytic value (8x8 supersampling of the same expected-coverage composite).
    let ana = area_integrated_reference(&scene[0]);
    let n = SEEDS as usize;
    let mut sum = vec![DVec3::ZERO; W * H];
    for seed in 0..SEEDS {
        let (img, _) = render_monte_carlo(&scene, cam(), W, H, 2, 1, BG, 77 + seed, &Default::default());
        for (i, p) in img.iter().enumerate() {
            sum[i] += *p;
        }
    }
    // The standard error comes from the THEORETICAL variance, not the sample variance: in the
    // Gaussian tails a pixel gets no hit in any seed (sample variance 0) although its expected
    // coverage is positive, which made a sample-SE interval invalid (diagnosed, not loosened).
    // One seed's pixel = bg + (c - bg) * k/4 with k ~ Binomial(4, a) (independent subpixels).
    let (c, bg) = (scene[0].color, BG);
    let mut outside_ci = 0;
    let mut se2 = 0.0;
    for i in 0..W * H {
        let mean = sum[i] / n as f64;
        let a = ((ana[i].x - bg.x) / (c.x - bg.x)).clamp(0.0, 1.0);
        let se = (c - bg).abs() * (a * (1.0 - a) / (4.0 * n as f64)).sqrt();
        let err = (mean - ana[i]).abs();
        if (0..3).any(|k| err[k] > 4.5 * se[k] + 1e-4) {
            outside_ci += 1;
        }
        se2 += (mean - ana[i]).length_squared();
    }
    let rms = (se2 / (W * H * 3) as f64).sqrt();
    println!("outside 4.5-SE CI: {outside_ci}/{} rms={rms:.5}", W * H);
    // 4.5 sigma per channel: a handful of multiple-testing excursions are allowed (<= 0.5%).
    assert!(outside_ci <= W * H / 200, "outside={outside_ci}");
    assert!(rms <= 0.01, "rms={rms}");
}
