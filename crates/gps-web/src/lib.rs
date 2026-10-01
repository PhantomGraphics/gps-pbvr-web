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
    next_seed: u32,
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
            next_seed: 1,
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
                width: self.width,
                height: self.height,
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
        let n = g.len() as f64;
        let c = g.iter().fold(DVec3::ZERO, |a, p| a + DVec3::new(p.pos_opacity[0] as f64, p.pos_opacity[1] as f64, p.pos_opacity[2] as f64)) / n;
        let mut d: Vec<f64> = g.iter().map(|p| (DVec3::new(p.pos_opacity[0] as f64, p.pos_opacity[1] as f64, p.pos_opacity[2] as f64) - c).length()).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        let radius = d[((d.len() as f64 * 0.9) as usize).min(d.len() - 1)]; // robust to outliers
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
        let spp_side = spp_side.clamp(1, 4);
        if (width, height) != (self.width, self.height) || spp_side != self.params.spp_side {
            self.renderer.resize(width, height, spp_side).map_err(js_err)?;
            (self.width, self.height) = (width, height);
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
    /// 3 = PBVR ViewConditioned; `calibration` 0..3 = C0..C3. `radial` (C3+R) only takes effect for
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
    pub fn set_lod(&mut self, adaptive: bool, target_ms: f64) {
        self.lod_adaptive = adaptive;
        self.lod_target_ms = target_ms.clamp(8.0, 200.0);
    }

    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.last_input_ms = self.last_frame_ms;
        self.cam.orbit(-dx * 0.005, dy * 0.005);
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
        if self.dirty {
            self.renderer.reset_accum();
            self.renderer.reset_stats();
            self.next_seed = 1;
            self.dirty = false;
            self.needs_present = true;
        }
        let acc = self.renderer.accumulated();
        if acc < self.target {
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
            let cam = self.cam.to_camera(self.width, self.height);
            self.renderer.render_ensembles(&cam, &self.params, self.next_seed, n).map_err(js_err)?;
            self.next_seed = self.next_seed.wrapping_add(n);
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
            "{{\"scene\":\"{}\",\"gaussians\":{},\"width\":{},\"height\":{},\"spp\":{},\"accumulated\":{},\"target\":{},\"points\":{},\"visible\":{},\"truncated_splats\":{},\"dropped_points\":{},\"ensembles_per_frame\":{},\"sh_max\":{},\"sh_degree\":{},\"orphan\":{},\"candidates\":{},\"undercovered\":{},\"method\":\"{}\",\"memory_mib\":{:.1},\"yaw\":{:.4},\"pitch\":{:.4},\"distance\":{:.4}}}",
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
            self.effective_per_frame,
            self.renderer.sh_max_degree(),
            self.params.sh_degree.min(self.renderer.sh_max_degree()),
            s.orphan_subpixels,
            s.candidates,
            s.undercovered_splats,
            self.params.method.name(),
            m.total() as f64 / 1048576.0,
            self.cam.yaw,
            self.cam.pitch,
            self.cam.distance
        )
    }
}
