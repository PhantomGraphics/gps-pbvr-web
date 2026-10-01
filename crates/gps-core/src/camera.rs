//! Orbit camera for the viewer. Produces the same [`OracleCamera`] the CPU oracle uses, so the
//! GPU renderer and the oracle always see identical projection conventions
//! (+Z forward, image +Y down, principal point at the image centre).

use crate::oracle::OracleCamera;
use glam::DVec3;

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: DVec3,
    pub distance: f64,
    /// radians, rotation around the up axis
    pub yaw: f64,
    /// radians, clamped away from the poles
    pub pitch: f64,
    /// World up direction. 3DGS/COLMAP scenes are usually Y-down: (0,-1,0).
    pub up: DVec3,
    /// vertical field of view, radians
    pub fov_y: f64,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: DVec3::ZERO, distance: 5.0, yaw: 0.0, pitch: 0.0, up: DVec3::Y, fov_y: 0.9 }
    }
}

const PITCH_LIMIT: f64 = 1.553; // ~89 degrees

impl OrbitCamera {
    pub fn eye(&self) -> DVec3 {
        // Build an orthonormal frame around `up`, then orbit.
        let up = self.up.normalize();
        let helper = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
        let a = up.cross(helper).normalize();
        let b = up.cross(a);
        let horiz = a * self.yaw.cos() + b * self.yaw.sin();
        let dir = horiz * self.pitch.cos() + up * self.pitch.sin();
        self.target + dir * self.distance
    }

    pub fn orbit(&mut self, d_yaw: f64, d_pitch: f64) {
        self.yaw += d_yaw;
        self.pitch = (self.pitch + d_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Translates the orbit target in the view plane so the scene follows the pointer: a drag of
    /// (`dx`, `dy`) pixels on a view `view_height` pixels tall moves the point under the target by exactly that
    /// much on screen (the scale is distance * 2 tan(fov / 2) / view_height).
    pub fn pan(&mut self, dx: f64, dy: f64, view_height: f64) {
        let cam = self.to_camera(1, 1);
        let r = cam.view_rot.transpose();      // columns = camera axes in world space
        let (right, down) = (r.col(0), r.col(1));
        let s = self.distance * 2.0 * (0.5 * self.fov_y).tan() / view_height.max(1.0);
        self.target -= right * (dx * s) + down * (dy * s);
    }

    /// Multiplicative zoom; `factor` > 1 moves away.
    pub fn zoom(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(1e-4, 1e6);
    }

    pub fn to_camera(&self, w: u32, h: u32) -> OracleCamera {
        OracleCamera::look_at(self.eye(), self.target, self.up, self.fov_y, w, h)
    }

    /// Robust sphere (centre, radius) of a point set for framing: per-axis median and the 75th percentile of the
    /// distance to it. Real captures carry far background Gaussians / floaters that dominate a mean or a
    /// 90th percentile. Returns `None` for an empty set.
    pub fn robust_sphere(points: impl Iterator<Item = DVec3>) -> Option<(DVec3, f64)> {
        let pts: Vec<DVec3> = points.collect();
        if pts.is_empty() {
            return None;
        }
        let median = |axis: usize| {
            let mut v: Vec<f64> = pts.iter().map(|p| p[axis]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let c = DVec3::new(median(0), median(1), median(2));
        let mut d: Vec<f64> = pts.iter().map(|p| (*p - c).length()).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        Some((c, d[((d.len() as f64 * 0.75) as usize).min(d.len() - 1)]))
    }

    /// Frame a bounding sphere so it fills roughly 80% of the view.
    pub fn frame_sphere(&mut self, centre: DVec3, radius: f64) {
        self.target = centre;
        self.distance = (radius.max(1e-6) / (0.5 * self.fov_y).sin()) * 1.25;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_is_at_distance_and_target_is_centred() {
        for up in [DVec3::Y, -DVec3::Y, DVec3::Z] {
            let mut c = OrbitCamera { target: DVec3::new(1.0, 2.0, 3.0), distance: 7.0, up, ..Default::default() };
            c.orbit(0.7, 0.3);
            assert!(((c.eye() - c.target).length() - 7.0).abs() < 1e-12);
            let cam = c.to_camera(64, 48);
            let t = cam.view_rot * (c.target - cam.view_pos);
            assert!(t.x.abs() < 1e-9 && t.y.abs() < 1e-9 && (t.z - 7.0).abs() < 1e-9);
        }
    }

    #[test]
    fn pan_moves_the_scene_with_the_pointer() {
        for up in [DVec3::Y, -DVec3::Y] {
            let mut c = OrbitCamera { target: DVec3::new(0.3, -0.2, 1.0), distance: 4.0, yaw: 0.7, pitch: 0.4, up, ..Default::default() };
            let h = 720.0;
            // pixel of the old target before and after a (+40, -25) pixel drag
            let project = |c: &OrbitCamera, p: DVec3| {
                let cam = c.to_camera(1280, 720);
                let q = cam.view_rot * (p - cam.view_pos);
                (cam.focal_x * q.x / q.z + 640.0, cam.focal_y * q.y / q.z + 360.0)
            };
            let p0 = c.target;
            let before = project(&c, p0);
            c.pan(40.0, -25.0, h);
            let after = project(&c, p0);
            assert!((after.0 - before.0 - 40.0).abs() < 1e-6 && (after.1 - before.1 + 25.0).abs() < 1e-6, "{before:?} -> {after:?}");
        }
    }

    #[test]
    fn pitch_is_clamped_and_world_up_stays_up() {
        let mut c = OrbitCamera::default();
        c.orbit(0.0, 10.0);
        assert!(c.pitch <= PITCH_LIMIT);
        let cam = OrbitCamera { pitch: 0.2, ..Default::default() }.to_camera(64, 64);
        assert!((cam.view_rot * DVec3::Y).y < 0.0); // world up -> image up (negative y)
    }
}
