//! GPS MVP renderer (SH0, primitive path). See `shaders/gps_splat.wgsl`.
//!
//! Ownership: the renderer owns all GPU resources; the caller owns the `Device`/`Queue`
//! clones. Errors are returned as `Result<_, String>` for the UI (no panics on user input).

use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use gps_core::oracle::OracleCamera;

const SPLAT_WGSL: &str = include_str!("../../../shaders/gps_splat.wgsl");
const COMPOSITE_WGSL: &str = include_str!("../../../shaders/gps_composite.wgsl");
const SCAN_WGSL: &str = include_str!("../../../shaders/gps_scan.wgsl");

/// 256 satisfies `min_uniform_buffer_offset_alignment` on every adapter seen (native 64, web 256).
const SLOT: u64 = 256;
/// Ensembles encoded per submission (one uniform slot each). Kept small on purpose: with 16-64 the
/// command buffer of a 70k-splat scene made `finish()` fail with Out of Memory on an integrated GPU.
pub const MAX_ENSEMBLES_PER_SUBMIT: u32 = 4;
const STATS_WORDS: usize = 24;
/// Scan levels: 256^5 > u32::MAX splats, so 6 slots are more than enough.
const MAX_SCAN_LEVELS: usize = 6;
/// Splat count limit of the particle path: every scan level dispatches < 65536 groups (256^3 = 16.7M).
pub const MAX_SPLATS: usize = 16_000_000;

/// One Gaussian, SH0. 4 x vec4 = 64 bytes, all vec4 so there is no vec3 padding ambiguity.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuGaussian {
    /// xyz position, w = opacity (activated)
    pub pos_opacity: [f32; 4],
    /// xyz log-scale
    pub log_scale: [f32; 4],
    /// (w, x, y, z), normalised in the shader
    pub rot_wxyz: [f32; 4],
    /// xyz linear SH0 colour (f_dc * C0 + 0.5)
    pub color: [f32; 4],
}

impl GpuGaussian {
    pub fn from_activated(a: &gps_io::ActivatedGaussian) -> Self {
        let c = a.dc * gps_core::SH_C0 + glam::DVec3::splat(0.5);
        Self {
            pos_opacity: [a.position.x as f32, a.position.y as f32, a.position.z as f32, a.opacity as f32],
            log_scale: [a.log_scale.x as f32, a.log_scale.y as f32, a.log_scale.z as f32, 0.0],
            rot_wxyz: [a.rotation.w as f32, a.rotation.x as f32, a.rotation.y as f32, a.rotation.z as f32],
            color: [c.x as f32, c.y as f32, c.z as f32, 0.0],
        }
    }

    pub fn from_oracle(g: &gps_core::oracle::Gaussian3D) -> Self {
        Self {
            pos_opacity: [g.pos.x as f32, g.pos.y as f32, g.pos.z as f32, g.opacity as f32],
            log_scale: [g.log_scale.x as f32, g.log_scale.y as f32, g.log_scale.z as f32, 0.0],
            rot_wxyz: [g.rot.w as f32, g.rot.x as f32, g.rot.y as f32, g.rot.z as f32],
            color: [g.color.x as f32, g.color.y as f32, g.color.z as f32, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    rot: [[f32; 4]; 3],
    pos: [f32; 4],
    intr: [f32; 4],
    dims: [u32; 4],
    cfg: [u32; 4],
    cfg2: [u32; 4],
    prm: [f32; 4],
    bg: [f32; 4],
    /// method, calibration level, flags, _
    pbvr: [u32; 4],
    /// base_k, reference pixel length, object depth, _
    pbp: [f32; 4],
}

const FLAG_RADIAL: u32 = 1;
const FLAG_CENTRE: u32 = 2;
const FLAG_JITTER: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeUniform {
    width: u32,
    height: u32,
    exposure: f32,
    gamma_on: u32,
}

/// How particles are generated on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RenderPath {
    /// One thread per splat loops over all of its particles (simple; load-imbalanced). Kept as the
    /// reference the parallel path is validated against.
    Primitive,
    /// prepare -> hierarchical scan -> indirect dispatch -> one thread per particle.
    #[default]
    Particle,
}

/// How the per-splat particle set is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    /// Gaussian Point Splatting: screen-space points drawn from the corrected 2D footprint law.
    #[default]
    Gps,
    /// PBVR: 3D particles, count proportional to opacity.
    Proportional,
    /// PBVR: 3D particles, count proportional to the extinction -ln(1 - opacity).
    Extinction,
    /// PBVR: extinction candidates thinned so the on-screen count equals GPS (no calibration level).
    ViewConditioned,
}

impl Method {
    pub fn from_index(i: u32) -> Self {
        match i {
            1 => Self::Proportional,
            2 => Self::Extinction,
            3 => Self::ViewConditioned,
            _ => Self::Gps,
        }
    }
    pub fn index(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Gps => "GPS",
            Self::Proportional => "PBVR Proportional",
            Self::Extinction => "PBVR Extinction",
            Self::ViewConditioned => "PBVR ViewConditioned",
        }
    }
}

/// Density calibration level of the PBVR count (C0..C3, plan section 6 / footprint-aware calibration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Calibration {
    /// C0: base_k particles per unit opacity term
    None = 0,
    /// C1: scaled once by the object (scene centroid) depth
    ObjectZoom = 1,
    /// C2: scaled by each splat's own depth
    PerSplatDepth = 2,
    /// C3: spp x projected footprint area (matches the GPS on-screen density)
    #[default]
    PerSplatFootprint = 3,
}

impl Calibration {
    pub fn from_index(i: u32) -> Self {
        match i {
            0 => Self::None,
            1 => Self::ObjectZoom,
            2 => Self::PerSplatDepth,
            _ => Self::PerSplatFootprint,
        }
    }
}

/// PBVR-only settings (ignored for [`Method::Gps`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbvrParams {
    pub calibration: Calibration,
    /// C3+R: per-particle radial thinning; only effective for Extinction at C3
    pub radial_correction: bool,
    /// use the splat centre depth for every particle of the splat instead of the particle depth
    pub centre_depth: bool,
    /// jitter particles by the low-pass variance so they share the GPS footprint
    pub low_pass_jitter: bool,
    pub base_k: f32,
    pub reference_pixel_length: f32,
}

impl Default for PbvrParams {
    fn default() -> Self {
        Self { calibration: Calibration::PerSplatFootprint, radial_correction: false, centre_depth: false, low_pass_jitter: true, base_k: 512.0, reference_pixel_length: 0.01 }
    }
}

/// User-visible render settings. Any change that affects the image requires `reset_accum()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderParams {
    /// subpixels per pixel = spp_side^2, 1..=4
    pub spp_side: u32,
    /// multiplies the expected point count (1 = the GPS-exact density)
    pub density_scale: f32,
    /// per-splat particle cap; overflow is counted in the stats, never silent
    pub max_points_per_splat: u32,
    pub near: f32,
    pub background: [f32; 3],
    pub exposure: f32,
    /// evaluate the corrected radius exactly per point (the reference) instead of through the table
    pub exact_radius: bool,
    /// active SH degree 0..=3 (clamped to what the loaded scene provides)
    pub sh_degree: u32,
    /// ignored for the PBVR methods, which always use the particle-parallel path
    pub path: RenderPath,
    pub method: Method,
    pub pbvr: PbvrParams,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            spp_side: 2,
            density_scale: 1.0,
            max_points_per_splat: 1 << 16,
            near: 0.05,
            background: [0.05, 0.05, 0.08],
            exposure: 1.0,
            sh_degree: 3,
            exact_radius: false,
            path: RenderPath::Particle,
            method: Method::Gps,
            pbvr: PbvrParams::default(),
        }
    }
}

/// Counters read back from the GPU for the last `read_stats()` window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuStats {
    pub points: u64,
    pub truncated_splats: u64,
    pub dropped_points: u64,
    pub visible_splats: u64,
    /// Must be 0; non-zero means the depth and colour passes generated different samples.
    pub orphan_subpixels: u64,
    /// PBVR: Poisson candidates before the keep tests
    pub candidates: u64,
    /// ViewConditioned: splats whose candidate set was too small to reach the GPS target density
    pub undercovered_splats: u64,
    /// ensembles skipped because their particle total exceeded u32 (lower the per-splat cap / density)
    pub skipped_ensembles: u64,
    /// ensembles the GPU has started in this counter window (the readback lags submissions)
    pub ensembles: u64,
    /// GPS particle demand of the last `probe_demand` / `render_sorted` view (expected points per ensemble)
    pub probe_demand: u64,
    /// 1-based number of the readback this snapshot came from (0 = none yet); see `stats_issued`
    pub issue: u32,
}

impl GpuStats {
    fn from_words(w: &[u32]) -> Self {
        // counters 0..=6 are 64-bit: low word at k, high (carry) word at 16 + k
        let c = |k: usize| w[k] as u64 | (w[16 + k] as u64) << 32;
        Self {
            points: c(0),
            truncated_splats: c(1),
            dropped_points: c(2),
            visible_splats: c(3),
            orphan_subpixels: c(4),
            candidates: c(5),
            undercovered_splats: c(6),
            skipped_ensembles: w[7] as u64,
            ensembles: w[10] as u64,
            probe_demand: w[11] as u64 | (w[12] as u64) << 32,
            issue: 0,
        }
    }
}

/// Estimated GPU memory of the per-frame buffers (plan section 5).
#[derive(Clone, Copy, Debug)]
pub struct MemoryEstimate {
    pub depth_winner_bytes: u64,
    pub accum_bytes: u64,
    pub splat_bytes: u64,
}

impl MemoryEstimate {
    pub fn total(&self) -> u64 {
        self.depth_winner_bytes + self.accum_bytes + self.splat_bytes
    }
}

pub fn estimate_memory(width: u32, height: u32, spp_side: u32, n_splats: usize) -> MemoryEstimate {
    let sub = width as u64 * height as u64 * (spp_side * spp_side) as u64;
    MemoryEstimate { depth_winner_bytes: 8 * sub, accum_bytes: 16 * width as u64 * height as u64, splat_bytes: 164 * n_splats as u64 }
}

/// (groups_x, groups_y, stride in threads) covering `n` threads with 64-thread groups, staying
/// within the 65535-groups-per-dimension limit that browsers enforce.
pub(crate) fn split_dispatch(n: u64) -> (u32, u32, u32) {
    let groups = n.div_ceil(64).max(1);
    let gx = groups.min(65535);
    let gy = groups.div_ceil(gx);
    (gx as u32, gy as u32, (gx * 64) as u32)
}

pub struct GpsRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    spp_side: u32,
    n_splats: u32,
    /// mean splat position, the "object" whose depth drives calibration level C1
    centroid: glam::DVec3,
    /// SH bands 1..3 (channel-major, `sh_stride` coefficients per channel); a dummy when the scene has none
    sh_rest: wgpu::Buffer,
    /// tabulated corrected-radius law (see `gps_core::radius_lut`)
    radius_lut: wgpu::Buffer,
    sh_max_degree: u32,
    sh_stride: u32,
    target_format: wgpu::TextureFormat,

    bgl: wgpu::BindGroupLayout,
    pl_clear: wgpu::ComputePipeline,
    pl_depth: wgpu::ComputePipeline,
    pl_color: wgpu::ComputePipeline,
    pl_resolve: wgpu::ComputePipeline,
    pl_composite: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,

    uniform: wgpu::Buffer,
    splats: wgpu::Buffer,
    depth: wgpu::Buffer,
    winner: wgpu::Buffer,
    accum: wgpu::Buffer,
    stats: wgpu::Buffer,
    composite_uniform: wgpu::Buffer,
    bind_groups: Vec<wgpu::BindGroup>,
    composite_bg: wgpu::BindGroup,
    /// ensembles submitted since the last reset (valid count = this minus skipped, see `accumulated()`)
    accumulated: u32,
    /// non-blocking stats readback (one request in flight at a time)
    // particle-parallel path
    pl_prepare: wgpu::ComputePipeline,
    pl_finalize: wgpu::ComputePipeline,
    pl_pdepth: wgpu::ComputePipeline,
    pl_pcolor: wgpu::ComputePipeline,
    pl_scan: wgpu::ComputePipeline,
    pl_scan_add: wgpu::ComputePipeline,
    scan_bgl: wgpu::BindGroupLayout,
    offsets: wgpu::Buffer,
    proj: wgpu::Buffer,
    info: wgpu::Buffer,
    /// indirect dispatch arguments, copied from `info` between passes (never bound as storage)
    args: wgpu::Buffer,
    scan_params: wgpu::Buffer,
    scan_dummy: wgpu::Buffer,
    scan_sizes: Vec<u32>,
    scan_bufs: Vec<wgpu::Buffer>,
    scan_bgs: Vec<wgpu::BindGroup>,
    stats_staging: wgpu::Buffer,
    stats_pending: Arc<AtomicBool>,
    stats_latest: Arc<Mutex<GpuStats>>,
    /// readbacks issued so far; `GpuStats::issue` of a delivered snapshot is its 1-based issue number
    stats_issued: Arc<AtomicU32>,
    sorted: crate::sorted::SortedPath,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpsRenderer {
    /// `target_format` must be a non-sRGB colour format (gamma is applied in the shader).
    pub fn new(
        device: wgpu::Device, queue: wgpu::Queue, target_format: wgpu::TextureFormat, width: u32, height: u32, spp_side: u32,
    ) -> Result<Self, String> {
        if target_format.is_srgb() {
            return Err(format!("target format {target_format:?} is sRGB; use a non-sRGB format (gamma is applied manually)"));
        }
        let spp_side = spp_side.clamp(1, 4);
        Self::check_limits(&device, width, height, spp_side, 0)?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gps_splat"), source: wgpu::ShaderSource::Wgsl(SPLAT_WGSL.into()) });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gps bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(9, true),
                storage_entry(10, true),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("gps layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (pl_clear, pl_depth, pl_color, pl_resolve) = (mk("clear_buffers"), mk("depth_pass"), mk("color_pass"), mk("resolve"));
        let (pl_prepare, pl_finalize, pl_pdepth, pl_pcolor) = (mk("prepare"), mk("finalize_dispatch"), mk("particle_depth"), mk("particle_color"));

        // hierarchical scan (workgroup barriers are not cross-workgroup syncs: one pass per level)
        let scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gps_scan"), source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()) });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });
        let scan_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scan layout"), bind_group_layouts: &[Some(&scan_bgl)], immediate_size: 0 });
        let mk_scan = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&scan_layout),
                module: &scan_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (pl_scan, pl_scan_add) = (mk_scan("scan_block"), mk_scan("add_offsets"));
        let sorted = crate::sorted::SortedPath::new(&device, &module, &bgl);

        // composite
        let cmod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gps_composite"), source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()) });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let clayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("composite layout"), bind_group_layouts: &[Some(&composite_bgl)], immediate_size: 0 });
        let pl_composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
            layout: Some(&clayout),
            vertex: wgpu::VertexState { module: &cmod, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &cmod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: target_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame slots"),
            size: SLOT * MAX_ENSEMBLES_PER_SUBMIT as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats"),
            size: (STATS_WORDS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Placeholder buffers, replaced below by the sized ones.
        let tiny = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (splats, depth, winner, accum) = (tiny("splats"), tiny("depth"), tiny("winner"), tiny("accum"));
        let (offsets, proj, info) = (tiny("offsets"), tiny("proj"), tiny("info"));
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect args"),
            size: 16,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scan_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan params"),
            size: SLOT * MAX_SCAN_LEVELS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scan_dummy = tiny("scan dummy");
        let sh_rest = tiny("sh");
        let lut = gps_core::radius_lut();
        let radius_lut = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radius lut"),
            size: (lut.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&radius_lut, 0, bytemuck::cast_slice(&lut));
        let stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats staging"),
            size: (STATS_WORDS * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: composite_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum.as_entire_binding() },
            ],
        });
        let mut r = Self {
            device, queue, width, height, spp_side, n_splats: 0, centroid: glam::DVec3::ZERO, sh_rest, radius_lut, sh_max_degree: 0, sh_stride: 0, target_format, bgl, pl_clear, pl_depth, pl_color, pl_resolve, pl_composite,
            composite_bgl, uniform, splats, depth, winner, accum, stats, composite_uniform, bind_groups: Vec::new(), composite_bg, accumulated: 0,
            pl_prepare, pl_finalize, pl_pdepth, pl_pcolor, pl_scan, pl_scan_add, scan_bgl, offsets, proj, info, args, scan_params, scan_dummy,
            scan_sizes: Vec::new(), scan_bufs: Vec::new(), scan_bgs: Vec::new(),
            stats_staging,
            stats_pending: Arc::new(AtomicBool::new(false)),
            stats_latest: Arc::new(Mutex::new(GpuStats::default())),
            stats_issued: Arc::new(AtomicU32::new(0)),
            sorted,
        };
        r.realloc_frame_buffers()?;
        r.set_gaussians(&[])?;
        Ok(r)
    }

    fn check_limits(device: &wgpu::Device, w: u32, h: u32, spp_side: u32, n_splats: usize) -> Result<(), String> {
        let lim = device.limits();
        if w == 0 || h == 0 {
            return Err("width/height must be > 0".into());
        }
        let m = estimate_memory(w, h, spp_side, n_splats);
        let bind_max = lim.max_storage_buffer_binding_size as u64;
        let buf_max = lim.max_buffer_size;
        for (name, bytes) in [("depth", m.depth_winner_bytes / 2), ("accum", m.accum_bytes), ("splats", 64 * n_splats as u64), ("projection", 96 * n_splats as u64)] {
            if bytes > bind_max.min(buf_max) {
                return Err(format!(
                    "{name} buffer needs {:.0} MiB, above this device's limit of {:.0} MiB; reduce resolution / SPP / splat count",
                    bytes as f64 / 1048576.0,
                    bind_max.min(buf_max) as f64 / 1048576.0
                ));
            }
        }
        Ok(())
    }

    fn buffer(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn realloc_frame_buffers(&mut self) -> Result<(), String> {
        Self::check_limits(&self.device, self.width, self.height, self.spp_side, self.n_splats as usize)?;
        let sub = self.width as u64 * self.height as u64 * (self.spp_side * self.spp_side) as u64;
        self.depth = self.buffer("depth", sub * 4);
        self.winner = self.buffer("winner", sub * 4);
        self.accum = self.buffer("accum", self.width as u64 * self.height as u64 * 16);
        self.rebuild_bind_groups();
        self.reset_accum();
        Ok(())
    }

    fn rebuild_bind_groups(&mut self) {
        self.bind_groups = (0..MAX_ENSEMBLES_PER_SUBMIT)
            .map(|e| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gps slot"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.uniform,
                                offset: e as u64 * SLOT,
                                size: wgpu::BufferSize::new(std::mem::size_of::<FrameUniform>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: self.splats.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.depth.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.winner.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.accum.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.stats.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: self.offsets.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: self.proj.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: self.info.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: self.sh_rest.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: self.radius_lut.as_entire_binding() },
                    ],
                })
            })
            .collect();
        self.composite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.composite_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.accum.as_entire_binding() },
            ],
        });
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn spp_side(&self) -> u32 {
        self.spp_side
    }
    /// GPU bytes used by the SH bands of the loaded scene (0 without SH).
    pub fn sh_bytes(&self) -> u64 {
        self.n_splats as u64 * 3 * self.sh_stride as u64 * 4
    }
    /// Highest SH degree the loaded scene provides (0 = SH0 only).
    pub fn sh_max_degree(&self) -> u32 {
        self.sh_max_degree
    }
    pub fn gaussian_count(&self) -> u32 {
        self.n_splats
    }
    /// Ensembles submitted since the last reset, including any the GPU skipped.
    pub fn submitted(&self) -> u32 {
        self.accumulated
    }

    /// Valid ensembles in the history: submitted minus those the GPU skipped on u32 particle-total
    /// overflow (they add nothing to `accum`). The skip count arrives with the async stats readback.
    pub fn accumulated(&self) -> u32 {
        self.accumulated.saturating_sub(self.latest_stats().skipped_ensembles.min(u32::MAX as u64) as u32)
    }
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Uploads the scene. Fails (leaving the old scene in place) if it exceeds device limits.
    pub fn set_gaussians(&mut self, data: &[GpuGaussian]) -> Result<(), String> {
        self.set_scene(data, None)
    }

    /// Like [`set_gaussians`](Self::set_gaussians) with SH bands 1..3: `sh_rest` holds
    /// `3 * rest_per_channel(degree)` floats per splat, channel-major (the PLY loader's layout).
    pub fn set_gaussians_sh(&mut self, data: &[GpuGaussian], sh_rest: &[f32], degree: u32) -> Result<(), String> {
        self.set_scene(data, Some((sh_rest, degree)))
    }

    fn set_scene(&mut self, data: &[GpuGaussian], sh: Option<(&[f32], u32)>) -> Result<(), String> {
        Self::check_limits(&self.device, self.width, self.height, self.spp_side, data.len())?;
        // The shader indexes splat ids as u32 and dispatches in 2D groups.
        if data.len() as u64 >= u32::MAX as u64 / 4 {
            return Err("too many Gaussians for one dispatch".into());
        }
        if data.len() > MAX_SPLATS {
            return Err(format!("{} Gaussians exceeds the current limit of {MAX_SPLATS} (chunked rendering comes in Phase 4)", data.len()));
        }
        let (sh_data, sh_degree, sh_stride) = match sh {
            Some((v, d)) if d >= 1 && !data.is_empty() => {
                let stride = gps_core::sh_rest_per_channel(d.min(3) as usize) as u32;
                if v.len() as u64 != data.len() as u64 * 3 * stride as u64 {
                    return Err(format!("SH data has {} values, expected {} for {} splats at degree {d}", v.len(), data.len() as u64 * 3 * stride as u64, data.len()));
                }
                let lim = (self.device.limits().max_storage_buffer_binding_size as u64).min(self.device.limits().max_buffer_size);
                if v.len() as u64 * 4 > lim {
                    return Err(format!("SH data needs {:.0} MiB, above this device's limit of {:.0} MiB; load a smaller scene or a lower-degree file", v.len() as f64 * 4.0 / 1048576.0, lim as f64 / 1048576.0));
                }
                (Some(v), d.min(3), stride)
            }
            _ => (None, 0, 0),
        };
        let bytes = (data.len() as u64 * 64).max(64);
        self.sh_rest = self.buffer("sh rest", sh_data.map_or(64, |v| v.len() as u64 * 4));
        if let Some(v) = sh_data {
            self.queue.write_buffer(&self.sh_rest, 0, bytemuck::cast_slice(v));
        }
        self.sh_max_degree = sh_degree;
        self.sh_stride = sh_stride;
        self.splats = self.buffer("splats", bytes);
        self.offsets = self.buffer("offsets", data.len() as u64 * 4);
        self.proj = self.buffer("proj", data.len() as u64 * 96);
        self.info = self.buffer("info", 64);
        if !data.is_empty() {
            self.queue.write_buffer(&self.splats, 0, bytemuck::cast_slice(data));
        }
        self.n_splats = data.len() as u32;
        self.centroid = if data.is_empty() {
            glam::DVec3::ZERO
        } else {
            data.iter().fold(glam::DVec3::ZERO, |a, g| a + glam::DVec3::new(g.pos_opacity[0] as f64, g.pos_opacity[1] as f64, g.pos_opacity[2] as f64)) / data.len() as f64
        };
        self.rebuild_scan();
        self.rebuild_bind_groups();
        self.reset_accum();
        Ok(())
    }

    /// Scan level buffers and bind groups for the current splat count. Level 0 is `offsets`.
    fn rebuild_scan(&mut self) {
        let mut sizes = vec![self.n_splats.max(1)];
        while *sizes.last().unwrap_or(&1) > 256 {
            let next = sizes.last().unwrap_or(&1).div_ceil(256);
            sizes.push(next);
        }
        debug_assert!(sizes.len() <= MAX_SCAN_LEVELS);
        // scan_bufs[l - 1] holds the block sums written by level l - 1 (= the data of level l).
        self.scan_bufs = sizes.iter().skip(1).map(|&n| self.buffer("scan level", n as u64 * 4)).collect();
        let mut params = vec![0u8; SLOT as usize * sizes.len()];
        for (l, &n) in sizes.iter().enumerate() {
            params[l * SLOT as usize..l * SLOT as usize + 4].copy_from_slice(&n.to_le_bytes());
        }
        self.queue.write_buffer(&self.scan_params, 0, &params);
        self.scan_bgs = (0..sizes.len())
            .map(|l| {
                let data = if l == 0 { &self.offsets } else { &self.scan_bufs[l - 1] };
                // the top level's block total has nowhere to go; a separate scratch word keeps it
                // from clobbering data[0]
                let sums = if l + 1 < sizes.len() { &self.scan_bufs[l] } else { &self.scan_dummy };
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scan level"),
                    layout: &self.scan_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.scan_params, offset: l as u64 * SLOT, size: wgpu::BufferSize::new(16) }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: sums.as_entire_binding() },
                    ],
                })
            })
            .collect();
        self.scan_sizes = sizes;
    }

    /// Changing resolution or SPP reallocates the per-frame buffers and resets history.
    pub fn resize(&mut self, width: u32, height: u32, spp_side: u32) -> Result<(), String> {
        let spp_side = spp_side.clamp(1, 4);
        if (width, height, spp_side) == (self.width, self.height, self.spp_side) {
            return Ok(());
        }
        Self::check_limits(&self.device, width, height, spp_side, self.n_splats as usize)?;
        (self.width, self.height, self.spp_side) = (width, height, spp_side);
        self.realloc_frame_buffers()
    }

    /// Clears accumulated history (call on any camera / data / parameter change).
    pub fn reset_accum(&mut self) {
        let zeros = vec![0u8; self.width as usize * self.height as usize * 16];
        self.queue.write_buffer(&self.accum, 0, &zeros);
        self.accumulated = 0;
        // the skip counter belongs to the history window: restart it with the history
        self.queue.write_buffer(&self.stats, 7 * 4, &[0u8; 4]);
        if let Ok(mut g) = self.stats_latest.lock() {
            g.skipped_ensembles = 0;
        }
    }

    pub fn reset_stats(&self) {
        self.queue.write_buffer(&self.stats, 0, &[0u8; STATS_WORDS * 4]);
        // the CPU copy of the counters belongs to the window being discarded as well
        if let Ok(mut g) = self.stats_latest.lock() {
            *g = GpuStats::default();
        }
    }

    /// Writes the per-ensemble uniform slots (ensemble `e` uses seed `first_seed + e`).
    fn upload_frame(&self, cam: &OracleCamera, params: &RenderParams, first_seed: u32, count: u32) {
        let n_sub = self.width as u64 * self.height as u64 * (self.spp_side * self.spp_side) as u64;
        let (_, _, stride_sub) = split_dispatch(n_sub);
        let (_, _, stride_pix) = split_dispatch(self.width as u64 * self.height as u64);
        let (_, _, stride_splat) = split_dispatch(self.n_splats as u64);
        // Same convention as the CPU oracle: a principal point of exactly (0,0) means "image centre".
        let (cx, cy) = if cam.cx == 0.0 && cam.cy == 0.0 { (0.5 * self.width as f64, 0.5 * self.height as f64) } else { (cam.cx, cam.cy) };
        let max_points = params.max_points_per_splat.max(1);
        let r = cam.view_rot.transpose(); // columns of the transpose = rows of view_rot
        let rows = [r.col(0), r.col(1), r.col(2)];
        let (method, pb) = (params.method, &params.pbvr);
        let mut flags = 0;
        if pb.radial_correction {
            flags |= FLAG_RADIAL;
        }
        if pb.centre_depth {
            flags |= FLAG_CENTRE;
        }
        if pb.low_pass_jitter {
            flags |= FLAG_JITTER;
        }
        let object_depth = (cam.view_rot * (self.centroid - cam.view_pos)).z;
        let mut slots = vec![0u8; SLOT as usize * count as usize];
        for e in 0..count {
            let u = FrameUniform {
                rot: [
                    [rows[0].x as f32, rows[0].y as f32, rows[0].z as f32, 0.0],
                    [rows[1].x as f32, rows[1].y as f32, rows[1].z as f32, 0.0],
                    [rows[2].x as f32, rows[2].y as f32, rows[2].z as f32, 0.0],
                ],
                pos: [cam.view_pos.x as f32, cam.view_pos.y as f32, cam.view_pos.z as f32, cam.low_pass as f32],
                intr: [cam.focal_x as f32, cam.focal_y as f32, cx as f32, cy as f32],
                dims: [self.width, self.height, self.spp_side, self.n_splats],
                cfg: [first_seed.wrapping_add(e), max_points, stride_splat, stride_sub],
                cfg2: [stride_pix, self.sh_stride, params.exact_radius as u32, 0],
                prm: [params.density_scale, params.near, 0.0, 0.0],
                bg: [params.background[0], params.background[1], params.background[2], 0.0],
                pbvr: [method.index(), pb.calibration as u32, flags, params.sh_degree.min(self.sh_max_degree)],
                pbp: [pb.base_k, pb.reference_pixel_length, object_depth as f32, 0.0],
            };
            let o = e as usize * SLOT as usize;
            slots[o..o + std::mem::size_of::<FrameUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        self.queue.write_buffer(&self.uniform, 0, &slots);
    }

    /// Encodes `count` ensembles (independent sample sets; seeds `first_seed..first_seed+count`).
    /// Each ensemble reads its own uniform slot, so all of them can be recorded in one
    /// submission without a later `write_buffer` overwriting an earlier ensemble's parameters.
    pub fn render_ensembles(&mut self, cam: &OracleCamera, params: &RenderParams, first_seed: u32, count: u32) -> Result<(), String> {
        if count == 0 {
            return Ok(());
        }
        if count > MAX_ENSEMBLES_PER_SUBMIT {
            return Err(format!("at most {MAX_ENSEMBLES_PER_SUBMIT} ensembles per submission"));
        }
        if params.spp_side.clamp(1, 4) != self.spp_side {
            return Err("params.spp_side differs from the renderer's; call resize() first".into());
        }
        let n_sub = self.width as u64 * self.height as u64 * (self.spp_side * self.spp_side) as u64;
        let (sx, sy, _) = split_dispatch(n_sub);
        let (px, py, _) = split_dispatch(self.width as u64 * self.height as u64);
        let (gx, gy, _) = split_dispatch(self.n_splats as u64);
        if sy > 65535 || py > 65535 || gy > 65535 {
            return Err("dispatch too large".into());
        }

        let method = params.method;
        self.upload_frame(cam, params, first_seed, count);

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gps ensembles") });
        let use_particles = (params.path == RenderPath::Particle || method != Method::Gps) && self.n_splats > 0;
        for e in 0..count as usize {
            // Every stage is its own compute pass: workgroup barriers are not cross-workgroup syncs,
            // and pass boundaries give the storage-buffer dependencies (depth -> colour -> resolve).
            let pass = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, x: u32, y: u32| {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, &self.bind_groups[e], &[]);
                cp.dispatch_workgroups(x, y, 1);
            };
            pass(&mut enc, &self.pl_clear, sx, sy);
            if use_particles {
                pass(&mut enc, &self.pl_prepare, gx, gy);
                // exclusive scan of the per-splat counts, in place: up-sweep then down-sweep
                let levels = self.scan_sizes.len();
                let up: Vec<usize> = (0..levels).collect();
                let down: Vec<usize> = (0..levels - 1).rev().collect();
                for (pipe, order) in [(&self.pl_scan, up), (&self.pl_scan_add, down)] {
                    for l in order {
                        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                        cp.set_pipeline(pipe);
                        cp.set_bind_group(0, &self.scan_bgs[l], &[]);
                        cp.dispatch_workgroups(self.scan_sizes[l].div_ceil(256), 1, 1);
                    }
                }
                pass(&mut enc, &self.pl_finalize, 1, 1);
                enc.copy_buffer_to_buffer(&self.info, 0, &self.args, 0, 12);
                for pipe in [&self.pl_pdepth, &self.pl_pcolor] {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    cp.set_pipeline(pipe);
                    cp.set_bind_group(0, &self.bind_groups[e], &[]);
                    cp.dispatch_workgroups_indirect(&self.args, 0);
                }
            } else {
                pass(&mut enc, &self.pl_depth, gx, gy);
                pass(&mut enc, &self.pl_color, gx, gy);
            }
            pass(&mut enc, &self.pl_resolve, px, py);
        }
        self.queue.submit([enc.finish()]);
        self.accumulated += count;
        Ok(())
    }

    /// Depth-sorted alpha rendering of the current view (no sampling noise). Writes the finished image into the
    /// accumulation buffer with count 1 and reports it as `as_ensembles` accumulated ensembles, so callers that
    /// wait for an accumulation target treat it as converged. Err when this device cannot hold the buffers.
    pub fn render_sorted(&mut self, cam: &OracleCamera, params: &RenderParams, as_ensembles: u32) -> Result<(), String> {
        self.sorted.ensure(&self.device, &self.queue, &self.scan_bgl, self.n_splats, self.width, self.height)?;
        if params.spp_side.clamp(1, 4) != self.spp_side {
            return Err("params.spp_side differs from the renderer's; call resize() first".into());
        }
        let params = RenderParams { method: Method::Gps, ..*params };
        self.upload_frame(cam, &params, 0, 1);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sorted frame") });
        self.sorted.encode_render(
            &self.queue, &self.stats, &mut enc, &self.bind_groups[0], self.n_splats, self.width, self.height, params.background, &self.pl_scan, &self.pl_scan_add,
        );
        self.queue.submit([enc.finish()]);
        self.accumulated = as_ensembles.max(1);
        Ok(())
    }

    /// Cheap GPS-demand probe: projects every splat and sums the expected particle counts of the on-screen ones
    /// into the stats (`GpuStats::probe_demand`, delivered by the async readback). Does not touch the image.
    pub fn probe_demand(&mut self, cam: &OracleCamera, params: &RenderParams) -> Result<(), String> {
        self.sorted.ensure(&self.device, &self.queue, &self.scan_bgl, self.n_splats, self.width, self.height)?;
        self.upload_frame(cam, params, 0, 1);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("demand probe") });
        self.sorted.encode_probe(&self.queue, &self.stats, &mut enc, &self.bind_groups[0], self.n_splats);
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Test hook: reads back the sorted keys and values of the last `render_sorted` (n entries each).
    pub async fn read_sorted(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        let (k, v) = self.sorted.sorted_buffers()?;
        let bytes = self.n_splats as u64 * 4;
        let (kr, vr) = (self.read_buffer(k, bytes).await, self.read_buffer(v, bytes).await);
        Some((bytemuck::cast_slice::<u8, u32>(&kr).to_vec(), bytemuck::cast_slice::<u8, u32>(&vr).to_vec()))
    }

    /// Records the composite pass into `view`, which must have this renderer's target format
    /// and the internal resolution.
    pub fn composite(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, params: &RenderParams) {
        let cu = CompositeUniform { width: self.width, height: self.height, exposure: params.exposure, gamma_on: 1 };
        self.queue.write_buffer(&self.composite_uniform, 0, bytemuck::bytes_of(&cu));
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gps composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&self.pl_composite);
        rp.set_bind_group(0, &self.composite_bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Reads the accumulation buffer (RGB sum, A = ensemble count per pixel).
    pub async fn read_accum(&self) -> Vec<[f32; 4]> {
        let bytes = self.width as u64 * self.height as u64 * 16;
        let raw = self.read_buffer(&self.accum, bytes).await;
        bytemuck::cast_slice::<u8, [f32; 4]>(&raw).to_vec()
    }

    /// Reads and returns the stats counters (cumulative since `reset_stats`).
    pub async fn read_stats(&self) -> GpuStats {
        let raw = self.read_buffer(&self.stats, (STATS_WORDS * 4) as u64).await;
        let w: &[u32] = bytemuck::cast_slice(&raw);
        GpuStats::from_words(w)
    }

    /// Starts a non-blocking stats readback (no-op while one is in flight). The result appears in
    /// [`latest_stats`](Self::latest_stats) once the GPU has finished. Never holds `&self` across
    /// an await, which matters for wasm-bindgen objects that are also mutated by input events.
    pub fn request_stats(&self) {
        if self.stats_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.stats, 0, &self.stats_staging, 0, (STATS_WORDS * 4) as u64);
        self.queue.submit([enc.finish()]);
        let issue = self.stats_issued.fetch_add(1, Ordering::AcqRel) + 1;
        let (staging, pending, latest) = (self.stats_staging.clone(), self.stats_pending.clone(), self.stats_latest.clone());
        self.stats_staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                if let Ok(view) = staging.slice(..).get_mapped_range() {
                    let w: &[u32] = bytemuck::cast_slice(&view);
                    if let Ok(mut g) = latest.lock() {
                        *g = GpuStats::from_words(w);
                        g.issue = issue;
                    }
                }
                staging.unmap();
            }
            pending.store(false, Ordering::Release);
        });
    }

    /// Number of stats readbacks issued so far. A snapshot with `issue` greater than the value read right after a
    /// GPU submission was copied after that submission completed.
    pub fn stats_issued(&self) -> u32 {
        self.stats_issued.load(Ordering::Acquire)
    }

    /// Most recent stats delivered by [`request_stats`](Self::request_stats).
    pub fn latest_stats(&self) -> GpuStats {
        self.stats_latest.lock().map(|g| *g).unwrap_or_default()
    }

    async fn read_buffer(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        map_read(&self.device, &staging).await
    }

    /// Blocks until all submitted work finished (native only; used by benchmarks).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Maps a MAP_READ buffer and copies it out (native polls; the browser fires the callback itself).
pub(crate) async fn map_read(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<u8> {
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    rx.await.expect("map channel dropped").expect("map failed");
    let data = slice.get_mapped_range().expect("mapped range");
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_dispatch_respects_group_limit() {
        let (x, y, stride) = split_dispatch(64 * 70_000);
        assert_eq!(x, 65535);
        assert_eq!(y, 2);
        assert_eq!(stride, 65535 * 64);
        assert!(x as u64 * y as u64 * 64 >= 64 * 70_000);
        let (x, y, _) = split_dispatch(0);
        assert_eq!((x, y), (1, 1));
    }

    #[test]
    fn layout_sizes() {
        assert_eq!(std::mem::size_of::<GpuGaussian>(), 64);
        assert_eq!(std::mem::size_of::<FrameUniform>(), 192);
        assert!(std::mem::size_of::<FrameUniform>() as u64 <= SLOT);
    }

    #[test]
    fn memory_estimate_matches_plan() {
        let m = estimate_memory(1920, 1080, 2, 0); // S = 4 subpixels
        assert_eq!(m.depth_winner_bytes + m.accum_bytes, (8 * 1920 * 1080 * 4 + 16 * 1920 * 1080) as u64);
        assert!(((m.depth_winner_bytes + m.accum_bytes) as f64 / 1048576.0 - 95.0).abs() < 1.0);
    }
}
