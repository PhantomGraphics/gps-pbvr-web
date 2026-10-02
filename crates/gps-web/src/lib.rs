#![cfg(target_arch = "wasm32")]
//! Browser entry. The JS/WASM boundary carries load / settings / input events in and a small
//! stats string out; nothing per-particle ever crosses it.

use glam::DVec3;
use gps_core::camera::OrbitCamera;
use gps_render::gps::{estimate_memory, Calibration, GpsRenderer, GpuGaussian, Method, PbvrParams, RenderParams, MAX_ENSEMBLES_PER_SUBMIT};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn browser_instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    wgpu::Instance::new(desc)
}

/// Runs the Phase 0 GPU checks on the browser's WebGPU adapter.
/// Returns the report; last line is `PHASE0: PASS` or `PHASE0: FAIL`.
#[wasm_bindgen]
pub async fn run_phase0() -> String {
    let mut out = Vec::<String>::new();
    match browser_instance().request_adapter(&wgpu::RequestAdapterOptions::default()).await {
        Ok(adapter) => {
            gps_render::run_phase0(&adapter, &mut |l| out.push(l)).await;
        }
        Err(e) => out.push(format!("NO WEBGPU ADAPTER: {e}\nPHASE0: FAIL")),
    }
    out.join("\n")
}

/// Runs the Phase 2 GPU-vs-oracle verification in the browser. Last line: `PHASE2: PASS|FAIL`.
#[wasm_bindgen]
pub async fn run_phase2() -> String {
    let mut out = Vec::<String>::new();
    match browser_instance().request_adapter(&wgpu::RequestAdapterOptions::default()).await {
        Ok(adapter) => {
            gps_render::verify::run_phase2(&adapter, &mut |l| out.push(l)).await;
        }
        Err(e) => out.push(format!("NO WEBGPU ADAPTER: {e}\nPHASE2: FAIL")),
    }
    out.join("\n")
}

/// Runs the Phase 3 (PBVR methods) GPU-vs-oracle verification in the browser. Last line: `PHASE3: PASS|FAIL`.
#[wasm_bindgen]
pub async fn run_phase3() -> String {
    let mut out = Vec::<String>::new();
    match browser_instance().request_adapter(&wgpu::RequestAdapterOptions::default()).await {
        Ok(adapter) => {
            gps_render::verify::run_phase3(&adapter, &mut |l| out.push(l)).await;
        }
        Err(e) => out.push(format!("NO WEBGPU ADAPTER: {e}
PHASE3: FAIL")),
    }
    out.join("
")
}

/// Runs the Phase 4 (SH bands) GPU-vs-CPU verification in the browser. Last line: `PHASE4: PASS|FAIL`.
#[wasm_bindgen]
pub async fn run_phase4() -> String {
    let mut out = Vec::<String>::new();
    match browser_instance().request_adapter(&wgpu::RequestAdapterOptions::default()).await {
        Ok(adapter) => {
            gps_render::verify::run_phase4(&adapter, &mut |l| out.push(l)).await;
        }
        Err(e) => out.push(format!("NO WEBGPU ADAPTER: {e}
PHASE4: FAIL")),
    }
    out.join("
")
}

/// Interactive GPS viewer bound to a canvas.
/// Auto path: scenes above this many splats never use the sorted path (its cost grows with the splat count).
const SORT_MAX_SPLATS: u32 = 8_000_000;
/// Auto path: a moving sorted frame slower than this (EMA, ms) for 20 frames hands motion over to GPS particles.
const SORT_SLOW_MS: f64 = 60.0;

#[wasm_bindgen]
pub struct Viewer {
    renderer: GpsRenderer,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    cam: OrbitCamera,
    /// framing chosen when the scene was loaded; `reset_view` returns here
    home_cam: OrbitCamera,
    params: RenderParams,
    width: u32,
    height: u32,
    output_width: u32,
    output_height: u32,
    /// first ensemble seed after a reset (part of the reproducible session state)
    seed: u32,
    next_seed: u32,
    /// set by the device-lost callback; the page recreates the viewer
    lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    per_frame: u32,
    target: u32,
    /// ensemble LOD: false = Manual (`per_frame` every frame), true = Adaptive (frame-time driven, 1 while moving)
    lod_adaptive: bool,
    lod_target_ms: f64,
    last_input_ms: f64,
    last_frame_ms: f64,
    ema_ms: f64,
    effective_per_frame: u32,
    lod_n: u32,
    lod_quiet: u32,
    /// points-per-ensemble budget (0 = off) and the automatic density factor that enforces it
    point_budget: f64,
    auto_density: f32,
    /// render path: 0 = GPS particles, 1 = depth-sorted alpha, 2 = Auto (sorted when the GPS demand explodes)
    path_mode: u32,
    /// Auto: GPS particle demand (points per ensemble) above which the sorted path takes over
    demand_threshold: f64,
    /// the sorted path is in use for the current history window
    sorted_active: bool,
    /// the device cannot hold the sorted buffers, or sorting proved too slow: Auto stays on GPS
    sorted_ok: bool,
    /// Auto decision bookkeeping: epoch of the last probe / decision and frames waited for the probe readback
    probe_epoch: u32,
    decided_epoch: u32,
    probe_wait: u32,
    /// the demand probe's readback is valid once a snapshot with `issue` above this arrives
    probe_issue: u32,
    slow_frames: u32,
    /// Auto: sorting every frame while the camera moves proved too slow, so motion uses GPS particles
    /// (their capped preview); a still view is still decided by the demand probe
    moving_prefers_gps: bool,
    last_demand: u64,
    /// frames since the history last restarted (the stats readback lags by a few frames)
    frames_since_reset: u32,
    /// monotonic counters for the page's frame-time statistics (see `ensemble_counter`/`history_epoch`)
    ensemble_counter: u32,
    history_epoch: u32,
    /// Adaptive LOD: per-splat particle cap used while the camera moves (the full cap applies once still)
    moving_cap: u32,
    lod_moving: bool,
    dirty: bool,
    needs_present: bool,
    scene_info: String,
}

fn pick_format(caps: &wgpu::SurfaceCapabilities) -> Result<wgpu::TextureFormat, String> {
    // Gamma is applied manually in the composite shader, so the canvas format must be non-sRGB.
    caps.formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb() && matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
        .ok_or_else(|| "canvas offers no non-sRGB 8-bit format".to_string())
}

#[wasm_bindgen]
impl Viewer {
    /// Creates the viewer on `canvas` (its width/height attributes are the internal resolution).
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Viewer, JsValue> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = browser_instance();
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(js_err)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() })
            .await
            .map_err(|e| js_err(format!("no WebGPU adapter: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("gps-viewer"), required_limits: adapter.limits(), ..Default::default() })
            .await
            .map_err(js_err)?;
        let surface_format = pick_format(&surface.get_capabilities(&adapter)).map_err(js_err)?;
        let lost = std::sync::Arc::new(std::sync::Mutex::new(None));
        {
            let lost = lost.clone();
            device.set_device_lost_callback(move |reason, msg| {
                if let Ok(mut g) = lost.lock() {
                    *g = Some(format!("{reason:?}: {msg}"));
                }
            });
        }
        let params = RenderParams::default();
        let renderer = GpsRenderer::new(device, queue, surface_format, width, height, params.spp_side).map_err(js_err)?;
        let mut v = Viewer {
            renderer,
            surface,
            surface_format,
            cam: OrbitCamera::default(),
            home_cam: OrbitCamera::default(),
            params,
            width,
            height,
            output_width: width,
            output_height: height,
            seed: 1,
            next_seed: 1,
            lost,
            per_frame: 1,
            target: 256,
            lod_adaptive: false,
            lod_target_ms: 33.3,
            last_input_ms: f64::NEG_INFINITY,
            last_frame_ms: 0.0,
            ema_ms: 16.7,
            effective_per_frame: 1,
            lod_n: 1,
            lod_quiet: 0,
            point_budget: 0.0,
            auto_density: 1.0,
            path_mode: 2,
            demand_threshold: 30.0e6,
            sorted_active: false,
            sorted_ok: true,
            probe_epoch: u32::MAX,
            decided_epoch: u32::MAX,
            probe_wait: 0,
            probe_issue: 0,
            slow_frames: 0,
            moving_prefers_gps: false,
            last_demand: 0,
            frames_since_reset: 0,
            ensemble_counter: 0,
            history_epoch: 0,
            moving_cap: 2048,
            lod_moving: false,
            dirty: true,
            needs_present: true,
            scene_info: String::new(),
        };
        v.configure_surface();
        Ok(v)
    }

    fn configure_surface(&mut self) {
        self.surface.configure(
            self.renderer.device(),
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                width: self.output_width,
                height: self.output_height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
            },
        );
    }

    fn invalidate(&mut self) {
        self.dirty = true;
    }

    /// Loads a 3DGS PLY (SH bands 1..3 are kept when present).
    pub fn load_ply(&mut self, bytes: &[u8]) -> Result<String, JsValue> {
        let g = gps_io::load_ply(bytes).map_err(js_err)?;
        if g.count == 0 {
            return Err(js_err("the file contains no Gaussians"));
        }
        let gpu: Vec<GpuGaussian> = (0..g.count).map(|i| GpuGaussian::from_activated(&g.activated(i))).collect();
        if g.sh_degree > 0 {
            self.renderer.set_gaussians_sh(&gpu, &g.sh_rest, g.sh_degree as u32).map_err(js_err)?;
        } else {
            self.renderer.set_gaussians(&gpu).map_err(js_err)?;
        }
        self.frame_scene(&gpu, true);
        // 3DGS PLY colours are display-referred (sRGB-space) values: show them unchanged. The sRGB OETF is for linear data.
        self.params.srgb_output = false;
        self.needs_present = true;
        let note = if g.sh_degree > 0 { format!(" (SH degree {})", g.sh_degree) } else { String::new() };
        self.scene_info = format!("{} Gaussians{note}", g.count);
        self.invalidate();
        Ok(self.scene_info.clone())
    }

    /// A deterministic synthetic scene (Y-up) for checking the viewer without data.
    pub fn load_demo(&mut self) -> Result<String, JsValue> {
        let mut gpu = Vec::new();
        let mut s = 12345u32;
        let mut rnd = || {
            s = gps_core::pcg_hash(s);
            s as f32 * (1.0 / 4_294_967_296.0)
        };
        for i in 0..600 {
            let (u, v) = (rnd() * std::f32::consts::TAU, rnd() * 2.0 - 1.0);
            let ring = 0.6 + 0.25 * (i % 3) as f32;
            let (x, y, z) = (ring * (1.0 - v * v).sqrt() * u.cos(), v * ring, ring * (1.0 - v * v).sqrt() * u.sin());
            let sc = (0.05 + 0.05 * rnd()).ln();
            gpu.push(GpuGaussian {
                pos_opacity: [x, y, z, 0.5 + 0.45 * rnd()],
                log_scale: [sc, sc + 0.3 * rnd(), sc - 0.3 * rnd(), 0.0],
                rot_wxyz: [rnd() + 0.1, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5],
                color: [0.5 + 0.5 * x / 1.1, 0.5 + 0.5 * y / 1.1, 0.5 + 0.5 * z / 1.1, 0.0],
            });
        }
        self.renderer.set_gaussians(&gpu).map_err(js_err)?;
        self.frame_scene(&gpu, false);
        self.scene_info = format!("{} Gaussians (demo)", gpu.len());
        self.invalidate();
        Ok(self.scene_info.clone())
    }

    fn frame_scene(&mut self, g: &[GpuGaussian], y_down: bool) {
        let pts = g.iter().map(|p| DVec3::new(p.pos_opacity[0] as f64, p.pos_opacity[1] as f64, p.pos_opacity[2] as f64));
        let (c, radius) = OrbitCamera::robust_sphere(pts).unwrap_or((DVec3::ZERO, 1.0));
        self.cam = OrbitCamera { up: if y_down { -DVec3::Y } else { DVec3::Y }, pitch: 0.25, yaw: 0.6, ..Default::default() };
        self.cam.frame_sphere(c, radius.max(1e-3));
        self.home_cam = self.cam;
    }

    /// Flip the world up direction (3DGS/COLMAP data is usually Y-down).
    pub fn set_y_down(&mut self, y_down: bool) {
        self.cam.up = if y_down { -DVec3::Y } else { DVec3::Y };
        self.home_cam.up = self.cam.up;
        self.invalidate();
    }

    /// Sets resolution / quality. Any change resets accumulation. Returns an error string if
    /// the device cannot hold the requested buffers (the previous settings stay active).
    pub fn configure(&mut self, width: u32, height: u32, spp_side: u32, density: f32, per_frame: u32, target: u32, max_points: u32) -> Result<(), JsValue> {
        self.configure_scaled(width, height, 1, spp_side, density, per_frame, target, max_points)
    }

    /// Experimental low-resolution rendering, with an unchanged full-size canvas.
    /// Density/opacity parameters are unchanged: generate particles for the new
    /// pixel footprints instead of thinning an already generated particle set.
    #[allow(clippy::too_many_arguments)]
    pub fn configure_scaled(&mut self, width: u32, height: u32, divisor: u32, spp_side: u32, density: f32, per_frame: u32, target: u32, max_points: u32) -> Result<(), JsValue> {
        let output = (width.max(1), height.max(1));
        let divisor = divisor.clamp(1, 2);
        let (width, height) = (output.0.div_ceil(divisor), output.1.div_ceil(divisor));
        let spp_side = spp_side.clamp(1, 4);
        if (width, height) != (self.width, self.height) || spp_side != self.params.spp_side {
            self.renderer.resize(width, height, spp_side).map_err(js_err)?;
            (self.width, self.height) = (width, height);
        }
        if output != (self.output_width, self.output_height) {
            (self.output_width, self.output_height) = output;
            self.configure_surface();
        }
        self.params.spp_side = spp_side;
        self.params.density_scale = density.clamp(0.01, 100.0);
        self.params.max_points_per_splat = max_points.max(1);
        self.per_frame = per_frame.clamp(1, MAX_ENSEMBLES_PER_SUBMIT);
        self.target = target.max(1);
        self.invalidate();
        Ok(())
    }

    /// Selects the particle model: `method` 0 = GPS, 1 = PBVR Proportional, 2 = PBVR Extinction,
    /// 3 = PBVR ViewConditioned, 4 = clipped screen occupancy (centre depth);
    /// `calibration` 0..3 = C0..C3. `radial` (C3+R) only takes effect for
    /// Extinction at C3 (the same rule as the C++ renderer). Resets accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn set_method(&mut self, method: u32, calibration: u32, radial: bool, centre_depth: bool, jitter: bool, base_k: f32, reference_pixel_length: f32) {
        let method = Method::from_index(method);
        let calibration = Calibration::from_index(calibration);
        self.params.method = method;
        self.params.pbvr = PbvrParams {
            calibration,
            radial_correction: radial && method == Method::Extinction && calibration == Calibration::PerSplatFootprint,
            centre_depth,
            low_pass_jitter: jitter,
            base_k: base_k.clamp(1.0, 1.0e6),
            reference_pixel_length: reference_pixel_length.clamp(1.0e-6, 1.0e3),
        };
        self.invalidate();
    }

    /// First ensemble seed after every reset. Same seed + same settings + same camera = identical image.
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
        self.invalidate();
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Camera as JSON (`target`, `distance`, `yaw`, `pitch`, `fov_y`, `up_y`: +1 = Y up, -1 = Y down).
    pub fn camera_json(&self) -> String {
        let c = &self.cam;
        format!(
            "{{\"target\":[{},{},{}],\"distance\":{},\"yaw\":{},\"pitch\":{},\"fov_y\":{},\"up_y\":{}}}",
            c.target.x, c.target.y, c.target.z, c.distance, c.yaw, c.pitch, c.fov_y, if c.up.y < 0.0 { -1 } else { 1 }
        )
    }

    /// Sets the orbit camera (values are validated; non-finite input is rejected). Resets accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn set_camera(&mut self, tx: f64, ty: f64, tz: f64, distance: f64, yaw: f64, pitch: f64, fov_y: f64) -> Result<(), JsValue> {
        if ![tx, ty, tz, distance, yaw, pitch, fov_y].iter().all(|v| v.is_finite()) || distance <= 0.0 || !(0.05..3.0).contains(&fov_y) {
            return Err(js_err("invalid camera values"));
        }
        self.cam.target = DVec3::new(tx, ty, tz);
        self.cam.distance = distance.clamp(1e-4, 1e6);
        self.cam.yaw = yaw;
        self.cam.pitch = pitch.clamp(-1.553, 1.553);
        self.cam.fov_y = fov_y;
        self.invalidate();
        Ok(())
    }

    /// Why the GPU device was lost (None while healthy). The page recreates the viewer.
    pub fn device_lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|g| g.clone())
    }

    /// Test hook: destroys the device to exercise the device-lost path.
    pub fn debug_lose_device(&self) {
        self.renderer.device().destroy();
    }

    /// Active SH degree 0..=3 (clamped to what the scene provides). Resets accumulation.
    pub fn set_sh_degree(&mut self, degree: u32) {
        self.params.sh_degree = degree.min(3);
        self.invalidate();
    }

    /// Highest SH degree of the loaded scene.
    pub fn sh_max_degree(&self) -> u32 {
        self.renderer.sh_max_degree()
    }

    /// Ensemble LOD. `adaptive = false`: always `per_frame` ensembles (Manual). `adaptive = true`:
    /// one ensemble while the camera moves, and while still as many as fit in `target_ms` of frame time
    /// (up to the engine maximum). Does not change the image statistics, only how fast it converges.
    ///
    /// Adaptive also lowers the per-splat particle cap to `moving_cap` while the camera moves (an explicitly coarser,
    /// biased preview: very large low-opacity splats are truncated) and restarts accumulation with the full cap once
    /// the camera is still, so the converged image is the same as with Manual.
    pub fn set_lod(&mut self, adaptive: bool, target_ms: f64, moving_cap: u32) {
        self.lod_adaptive = adaptive;
        self.lod_target_ms = target_ms.clamp(8.0, 200.0);
        self.moving_cap = moving_cap.clamp(64, 1 << 24);
        self.invalidate();
    }

    /// Per-ensemble point budget in millions (0 = off). When the view would need more points (zooming in makes every
    /// splat cover more pixels), the density is lowered automatically to stay within the budget. The mean image is
    /// unchanged; only the sampling noise rises, which more accumulated ensembles average out.
    pub fn set_point_budget(&mut self, millions: f64) {
        let b = if millions.is_finite() { millions.max(0.0) * 1.0e6 } else { 0.0 };
        if b != self.point_budget {
            self.point_budget = b;
            self.auto_density = 1.0;
            self.invalidate();
        }
    }

    /// Render path: 0 = GPS particles, 1 = depth-sorted alpha compositing, 2 = Auto. Auto switches to the sorted path
    /// when the GPS particle demand of the view exceeds `demand_threshold_millions` (million points per ensemble:
    /// close views full of large overlapping splats) and falls back to GPS if sorting is unavailable or too slow.
    /// The sorted path renders the expectation of the GPS process, without sampling noise. Only for the GPS particle
    /// model: PBVR methods always use particles.
    pub fn set_render_path(&mut self, mode: u32, demand_threshold_millions: f64) {
        let mode = mode.min(2);
        let thr = if demand_threshold_millions.is_finite() { demand_threshold_millions.clamp(0.1, 100000.0) * 1.0e6 } else { 30.0e6 };
        if mode != self.path_mode || thr != self.demand_threshold {
            self.path_mode = mode;
            self.demand_threshold = thr;
            self.sorted_ok = true;
            self.moving_prefers_gps = false;
            self.slow_frames = 0;
            self.decided_epoch = u32::MAX;
            self.invalidate();
        }
    }

    /// Presentation only (no history reset): true = apply the sRGB OETF to the linear colours (default, the C++
    /// reference convention), false = show the stored colour values unchanged (how 3DGS viewers display PLY data).
    pub fn srgb_output(&self) -> bool {
        self.params.srgb_output
    }

    pub fn set_srgb_output(&mut self, on: bool) {
        self.params.srgb_output = on;
        self.needs_present = true;
    }

    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.last_input_ms = self.last_frame_ms;
        self.cam.orbit(-dx * 0.005, dy * 0.005);
        self.invalidate();
    }

    /// Translates the camera target in the view plane (`dx`, `dy` in CSS pixels, `view_height` = CSS height of the canvas).
    pub fn pan(&mut self, dx: f64, dy: f64, view_height: f64) {
        self.last_input_ms = self.last_frame_ms;
        self.cam.pan(dx, dy, view_height);
        self.invalidate();
    }

    pub fn zoom(&mut self, wheel: f64) {
        self.last_input_ms = self.last_frame_ms;
        self.cam.zoom((wheel * 0.001).exp());
        self.invalidate();
    }

    /// Clears accumulation and statistics (history reset) without touching the camera.
    pub fn reset(&mut self) {
        self.invalidate();
    }

    /// Returns the camera to the framing chosen at load time and restarts accumulation.
    pub fn reset_view(&mut self) {
        self.cam = self.home_cam;
        self.invalidate();
    }

    /// One animation tick: renders `per_frame` ensembles until `target` is reached, then
    /// presents. Returns true if anything was drawn.
    /// `now_ms` is a monotonic clock (performance.now()); it drives the adaptive LOD only.
    pub fn frame(&mut self, now_ms: f64) -> Result<bool, JsValue> {
        let dt = if self.last_frame_ms > 0.0 { (now_ms - self.last_frame_ms).clamp(1.0, 500.0) } else { 16.7 };
        self.last_frame_ms = now_ms;
        self.ema_ms += 0.1 * (dt - self.ema_ms);
        let moving = self.lod_adaptive && now_ms - self.last_input_ms < 250.0;
        if moving != self.lod_moving {
            self.lod_moving = moving;
            self.dirty = true;                   // the particle cap changes, so the history restarts
        }
        if self.dirty {
            self.renderer.reset_accum();
            self.renderer.reset_stats();
            self.history_epoch = self.history_epoch.wrapping_add(1);
            self.frames_since_reset = 0;
            self.next_seed = self.seed;
            self.dirty = false;
            self.needs_present = true;
        }
        // The valid count needs the skip counter; keep the (non-blocking, one-in-flight) readback going
        // while converging. If every ensemble so far overflowed u32 (a deterministic property of the
        // view/settings), stop submitting more of them: the UI reports the skipped count.
        if self.renderer.submitted() <= self.target + 8 {
            self.renderer.request_stats();
        }
        let acc = self.renderer.accumulated();
        // Budget control: demand = points an ensemble would need at density factor 1, measured from the
        // GPU counters of the current history window (reset on every change, so the measure is fresh).
        self.frames_since_reset = self.frames_since_reset.saturating_add(1);
        // wait until a stats readback of the CURRENT window has certainly arrived, or stale counters would flip-flop the factor
        if self.point_budget > 0.0 && !self.lod_moving && self.frames_since_reset >= 8 && self.renderer.latest_stats().ensembles >= 2 {
            // Truncated splats drop part of their demand (`dropped_points`); count it, or a view dominated by huge
            // splats would look cheap while its image is biased dark.
            let st = self.renderer.latest_stats();
            let demand = (st.points + st.dropped_points) as f64 / st.ensembles as f64 / self.auto_density as f64;
            if demand > 0.0 {
                let want = (self.point_budget / demand).clamp(0.1, 1.0) as f32;
                if (want / self.auto_density).ln().abs() > 0.25 {
                    self.auto_density = want;
                    self.invalidate();
                }
            }
        }
        // ---- render path (GPS particles / depth-sorted alpha) ----
        let moving_now = now_ms - self.last_input_ms < 250.0;
        let cam = self.cam.to_camera(self.width, self.height);
        let sorted_eligible = self.params.method == Method::Gps && self.renderer.gaussian_count() > 0 && self.renderer.gaussian_count() <= SORT_MAX_SPLATS;
        let mut wait_for_probe = false;
        match self.path_mode {
            0 => self.sorted_active = false,
            1 => self.sorted_active = sorted_eligible && self.sorted_ok,
            _ => {
                if !(sorted_eligible && self.sorted_ok) || (moving_now && self.moving_prefers_gps) {
                    self.sorted_active = false;
                } else if !moving_now && self.decided_epoch != self.history_epoch {
                    // Still view: measure the GPS demand of this exact view first (cheap projection pass, no rendering),
                    // then choose. Moving views keep the last choice (a probe per frame would stall the motion).
                    let mut p = self.params;
                    p.density_scale *= self.auto_density;
                    if self.probe_epoch != self.history_epoch {
                        match self.renderer.probe_demand(&cam, &p) {
                            Ok(()) => {
                                self.probe_epoch = self.history_epoch;
                                self.probe_wait = 0;
                                self.probe_issue = self.renderer.stats_issued();
                            }
                            Err(_) => {
                                self.sorted_ok = false;
                                self.sorted_active = false;
                            }
                        }
                    }
                    if self.probe_epoch == self.history_epoch {
                        self.renderer.request_stats();
                        self.probe_wait += 1;
                        // a readback issued after the probe (an earlier one in flight would still show the old window)
                        if self.renderer.latest_stats().issue > self.probe_issue {
                            let demand = self.renderer.latest_stats().probe_demand as f64;
                            self.last_demand = demand as u64;
                            let keep = if self.sorted_active { 0.5 } else { 1.0 }; // hysteresis against flip-flopping near the threshold
                            self.sorted_active = demand > keep * self.demand_threshold;
                            self.decided_epoch = self.history_epoch;
                        } else {
                            wait_for_probe = true;
                        }
                    }
                }
            }
        }
        if self.sorted_active {
            // Sorting too slow while the camera moves (every moving frame sorts again): give up, GPS particles take over.
            if moving_now && self.path_mode == 2 {
                self.slow_frames = if self.ema_ms > SORT_SLOW_MS { self.slow_frames + 1 } else { 0 };
                if self.slow_frames >= 20 {
                    self.moving_prefers_gps = true;
                    self.sorted_active = false;
                    self.slow_frames = 0;
                    self.invalidate();
                }
            }
        } else {
            self.slow_frames = 0;
        }

        let stalled = acc == 0 && self.renderer.submitted() >= 8;
        if self.sorted_active && !wait_for_probe {
            if acc < self.target {
                let mut p = self.params;
                p.density_scale *= self.auto_density;
                match self.renderer.render_sorted(&cam, &p, self.target) {
                    Ok(()) => {
                        self.effective_per_frame = 0;
                        self.ensemble_counter = self.ensemble_counter.wrapping_add(1);
                        self.needs_present = true;
                    }
                    Err(_) => {
                        // cannot hold the buffers: fall back to GPS particles
                        self.sorted_ok = false;
                        self.sorted_active = false;
                        self.invalidate();
                    }
                }
            }
        } else if wait_for_probe {
            // waiting for the demand probe; keep the previous picture on screen
        } else if acc < self.target && !stalled {
            let moving = now_ms - self.last_input_ms < 250.0;
            if moving {
                self.lod_n = 1;
                self.lod_quiet = 0;
            }
            let want = if !self.lod_adaptive {
                self.per_frame
            } else if moving {
                1
            } else {
                // The frame interval is vsync-quantised (16.7 ms at best), so "fits N more" cannot be derived
                // from it. Feedback instead: back off at once when over the target, probe one more ensemble
                // per ~30 quiet frames while well below it.
                self.lod_quiet += 1;
                if self.ema_ms > self.lod_target_ms {
                    self.lod_n = (self.lod_n.saturating_sub(1)).max(1);
                    self.lod_quiet = 0;
                } else if self.ema_ms < 0.6 * self.lod_target_ms && self.lod_quiet >= 30 {
                    self.lod_n = (self.lod_n + 1).min(MAX_ENSEMBLES_PER_SUBMIT);
                    self.lod_quiet = 0;
                }
                self.lod_n
            };
            self.effective_per_frame = want;
            let n = want.min(self.target - acc);
            let mut p = self.params;
            p.density_scale *= self.auto_density;
            if self.lod_moving {
                p.max_points_per_splat = p.max_points_per_splat.min(self.moving_cap);
            }
            self.renderer.render_ensembles(&cam, &p, self.next_seed, n).map_err(js_err)?;
            self.next_seed = self.next_seed.wrapping_add(n);
            self.ensemble_counter = self.ensemble_counter.wrapping_add(n);
            self.needs_present = true;
        }
        if !self.needs_present {
            return Ok(false);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                // Lost / outdated / timeout: reconfigure and retry next tick.
                self.configure_surface();
                return Err(js_err(format!("surface not ready: {other:?}")));
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.renderer.device().create_command_encoder(&Default::default());
        self.renderer.composite(&mut enc, &view, &self.params);
        self.renderer.queue().submit([enc.finish()]);
        self.renderer.queue().present(frame);
        self.needs_present = false;
        Ok(true)
    }

    /// Re-presents the current accumulation without resetting it (used before saving a PNG,
    /// because a WebGPU canvas can only be read in the task that presented it).
    pub fn redraw(&mut self) {
        self.needs_present = true;
    }

    /// Total ensembles submitted over the viewer's lifetime (wrapping). A change between two frames
    /// means that frame did render work, as opposed to idling on a converged image.
    pub fn ensemble_counter(&self) -> u32 {
        self.ensemble_counter
    }

    /// Bumps whenever the history restarts (camera / data / parameter change).
    pub fn history_epoch(&self) -> u32 {
        self.history_epoch
    }

    pub fn accumulated(&self) -> u32 {
        self.renderer.accumulated()
    }

    /// JSON with the current state and the most recent GPU counters. Synchronous on purpose:
    /// an async method would keep `&self` borrowed across an await and make concurrent input
    /// events (orbit, load) fail with wasm-bindgen's re-entrancy check. Also requests the next
    /// counter readback, so values lag by one call.
    pub fn stats_json(&self) -> String {
        self.renderer.request_stats();
        let s = self.renderer.latest_stats();
        let m = estimate_memory(self.width, self.height, self.params.spp_side, self.renderer.gaussian_count() as usize);
        format!(
            "{{\"scene\":\"{}\",\"gaussians\":{},\"width\":{},\"height\":{},\"spp\":{},\"accumulated\":{},\"target\":{},\"points\":{},\"visible\":{},\"truncated_splats\":{},\"dropped_points\":{},\"seed\":{},\"lod_moving\":{},\"ensembles_per_frame\":{},\"sh_max\":{},\"sh_degree\":{},\"orphan\":{},\"candidates\":{},\"undercovered\":{},\"skipped\":{},\"density_auto\":{:.3},\"path\":\"{}\",\"demand\":{},\"method\":\"{}\",\"memory_mib\":{:.1},\"yaw\":{:.4},\"pitch\":{:.4},\"distance\":{:.4}}}",
            self.scene_info.replace('"', "'"),
            self.renderer.gaussian_count(),
            self.width,
            self.height,
            self.params.spp_side * self.params.spp_side,
            self.renderer.accumulated(),
            self.target,
            s.points,
            s.visible_splats,
            s.truncated_splats,
            s.dropped_points,
            self.seed,
            self.lod_moving,
            self.effective_per_frame,
            self.renderer.sh_max_degree(),
            self.params.sh_degree.min(self.renderer.sh_max_degree()),
            s.orphan_subpixels,
            s.candidates,
            s.undercovered_splats,
            s.skipped_ensembles,
            self.auto_density,
            if self.sorted_active { "sorted" } else { "gps" },
            self.last_demand,
            self.params.method.name(),
            (m.total() + self.renderer.sh_bytes()) as f64 / 1048576.0,
            self.cam.yaw,
            self.cam.pitch,
            self.cam.distance
        )
    }
}
