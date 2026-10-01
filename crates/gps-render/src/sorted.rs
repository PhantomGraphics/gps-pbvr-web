//! Depth-sorted alpha path for close views (see `sp_project` in `shaders/gps_splat.wgsl`).
//!
//! project (+ GPS particle demand probe) -> 3-pass GPU radix sort of 24-bit depth keys -> back-to-front
//! instanced quads into an rgba16float target -> resolve into the shared accumulation buffer (count 1).
//! All sizes are fixed by the splat count (invisible splats get the invalid key and sort to the end), so
//! no pass needs GPU-dependent dispatch sizes except the indirect draw.

use crate::gps::split_dispatch;

const SORT_WGSL: &str = include_str!("../../../shaders/gps_sort.wgsl");
const DRAW_WGSL: &str = include_str!("../../../shaders/gps_sorted_draw.wgsl");
const SLOT: u64 = 256;
const PASSES: usize = 3;
const MAX_LEVELS: usize = 6;
const TEX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn storage(binding: u32, read_only: bool, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// Buffers and bind groups that depend on (splat count, resolution).
struct Sized {
    key: (u32, u32, u32),
    keys_b: wgpu::Buffer,
    vals_b: wgpu::Buffer,
    counter: wgpu::Buffer,
    args: wgpu::Buffer,
    sp_bg: wgpu::BindGroup,
    sort_bgs: Vec<wgpu::BindGroup>,
    draw_bg: wgpu::BindGroup,
    scan_sizes: Vec<u32>,
    scan_bgs: Vec<wgpu::BindGroup>,
    tex_view: wgpu::TextureView,
    // kept alive for the bind groups
    _bufs: Vec<wgpu::Buffer>,
    _tex: wgpu::Texture,
}

pub(crate) struct SortedPath {
    sp_bgl: wgpu::BindGroupLayout,
    pl_project: wgpu::ComputePipeline,
    pl_resolve: wgpu::ComputePipeline,
    pl_hist: wgpu::ComputePipeline,
    pl_scatter: wgpu::ComputePipeline,
    pl_draw: wgpu::RenderPipeline,
    sort_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    sort_params: wgpu::Buffer,
    draw_params: wgpu::Buffer,
    scan_params: wgpu::Buffer,
    sized: Option<Sized>,
}

impl SortedPath {
    pub(crate) fn new(device: &wgpu::Device, splat_module: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let sp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorted bgl"),
            entries: &[
                storage(0, false, c),
                storage(1, false, c),
                storage(2, false, c),
                storage(3, false, c),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: c,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorted layout"),
            bind_group_layouts: &[Some(bgl), Some(&sp_bgl)],
            immediate_size: 0,
        });
        let mk = |module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (pl_project, pl_resolve) = (mk(splat_module, &layout, "sp_project"), mk(splat_module, &layout, "sp_resolve"));

        let sort_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gps_sort"), source: wgpu::ShaderSource::Wgsl(SORT_WGSL.into()) });
        let sort_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort bgl"),
            entries: &[uniform(0, c), storage(1, true, c), storage(2, true, c), storage(3, false, c), storage(4, false, c), storage(5, false, c)],
        });
        let sort_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sort layout"), bind_group_layouts: &[Some(&sort_bgl)], immediate_size: 0 });
        let (pl_hist, pl_scatter) = (mk(&sort_module, &sort_layout, "rs_hist"), mk(&sort_module, &sort_layout, "rs_scatter"));

        let draw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gps_sorted_draw"), source: wgpu::ShaderSource::Wgsl(DRAW_WGSL.into()) });
        let v = wgpu::ShaderStages::VERTEX;
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("draw bgl"), entries: &[uniform(0, v), storage(1, true, v), storage(2, true, v)] });
        let draw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("draw layout"), bind_group_layouts: &[Some(&draw_bgl)], immediate_size: 0 });
        let pl_draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sorted draw"),
            layout: Some(&draw_layout),
            vertex: wgpu::VertexState { module: &draw_module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEX_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let ub = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        Self {
            sp_bgl,
            pl_project,
            pl_resolve,
            pl_hist,
            pl_scatter,
            pl_draw,
            sort_bgl,
            draw_bgl,
            sort_params: ub("sort params", SLOT * PASSES as u64),
            draw_params: ub("draw params", 16),
            scan_params: ub("sorted scan params", SLOT * MAX_LEVELS as u64),
            sized: None,
        }
    }

    /// Allocates (or reuses) the buffers for `n` splats at `w` x `h`. Err when this device cannot hold them.
    pub(crate) fn ensure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scan_bgl: &wgpu::BindGroupLayout, n: u32, w: u32, h: u32) -> Result<(), String> {
        if n == 0 {
            return Err("empty scene".into());
        }
        if let Some(s) = &self.sized {
            if s.key == (n, w, h) {
                return Ok(());
            }
        }
        let lim = device.limits();
        let max = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
        let need = 48 * n as u64;
        if need > max {
            return Err(format!("sorted path needs a {:.0} MiB buffer, above this device's limit of {:.0} MiB", need as f64 / 1048576.0, max as f64 / 1048576.0));
        }
        if w > lim.max_texture_dimension_2d || h > lim.max_texture_dimension_2d {
            return Err("resolution exceeds the maximum texture size".into());
        }
        let buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(64), usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let proj2 = buf("sp proj", 48 * n as u64, st);
        let keys_a = buf("sp keys a", 4 * n as u64, st);
        let vals_a = buf("sp vals a", 4 * n as u64, st);
        let keys_b = buf("sp keys b", 4 * n as u64, st);
        let vals_b = buf("sp vals b", 4 * n as u64, st);
        let nblocks = n.div_ceil(256);
        let n_hist = 256 * nblocks;
        let hist = buf("sp hist", 4 * n_hist as u64, st);
        let counter = buf("sp counter", 16, st);
        let args = buf("sp draw args", 16, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST);
        let scan_dummy = buf("sp scan dummy", 16, st);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sorted target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEX_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tex_view = tex.create_view(&Default::default());

        let sp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorted group 1"),
            layout: &self.sp_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: proj2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: keys_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vals_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&tex_view) },
            ],
        });

        // 3 passes: a -> b, b -> a, a -> b (result in b)
        let mut params = vec![0u8; SLOT as usize * PASSES];
        for p in 0..PASSES {
            let o = p * SLOT as usize;
            params[o..o + 4].copy_from_slice(&n.to_le_bytes());
            params[o + 4..o + 8].copy_from_slice(&(8 * p as u32).to_le_bytes());
            params[o + 8..o + 12].copy_from_slice(&nblocks.to_le_bytes());
        }
        queue.write_buffer(&self.sort_params, 0, &params);
        let sort_bgs: Vec<wgpu::BindGroup> = (0..PASSES)
            .map(|p| {
                let (ki, vi, ko, vo) = if p % 2 == 0 { (&keys_a, &vals_a, &keys_b, &vals_b) } else { (&keys_b, &vals_b, &keys_a, &vals_a) };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sort pass"),
                    layout: &self.sort_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.sort_params, offset: p as u64 * SLOT, size: wgpu::BufferSize::new(16) }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: ki.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: vi.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: ko.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: vo.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: hist.as_entire_binding() },
                    ],
                })
            })
            .collect();

        queue.write_buffer(&self.draw_params, 0, bytemuck::cast_slice(&[w as f32, h as f32, 0.0, 0.0]));
        let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorted draw"),
            layout: &self.draw_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.draw_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: proj2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vals_b.as_entire_binding() },
            ],
        });

        // exclusive scan chain over the histogram (level 0 = hist)
        let mut sizes = vec![n_hist.max(1)];
        while *sizes.last().unwrap_or(&1) > 256 {
            let next = sizes.last().unwrap_or(&1).div_ceil(256);
            sizes.push(next);
        }
        if sizes.len() > MAX_LEVELS {
            return Err("too many scan levels".into());
        }
        let level_bufs: Vec<wgpu::Buffer> = sizes.iter().skip(1).map(|&m| buf("sp scan level", 4 * m as u64, st)).collect();
        let mut sp = vec![0u8; SLOT as usize * sizes.len()];
        for (l, &m) in sizes.iter().enumerate() {
            sp[l * SLOT as usize..l * SLOT as usize + 4].copy_from_slice(&m.to_le_bytes());
        }
        queue.write_buffer(&self.scan_params, 0, &sp);
        let scan_bgs = (0..sizes.len())
            .map(|l| {
                let data = if l == 0 { &hist } else { &level_bufs[l - 1] };
                let sums = if l + 1 < sizes.len() { &level_bufs[l] } else { &scan_dummy };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sorted scan level"),
                    layout: scan_bgl,
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

        let mut bufs = vec![proj2, keys_a, vals_a, hist, scan_dummy];
        bufs.extend(level_bufs);
        self.sized = Some(Sized {
            key: (n, w, h),
            keys_b,
            vals_b,
            counter,
            args,
            sp_bg,
            sort_bgs,
            draw_bg,
            scan_sizes: sizes,
            scan_bgs,
            tex_view,
            _bufs: bufs,
            _tex: tex,
        });
        Ok(())
    }

    fn reset_counters(&self, queue: &wgpu::Queue, stats: &wgpu::Buffer) {
        if let Some(s) = &self.sized {
            queue.write_buffer(&s.counter, 0, bytemuck::cast_slice(&[4u32, 0, 0, 0]));
        }
        // probe demand (stats words 11, 12)
        queue.write_buffer(stats, 11 * 4, &[0u8; 8]);
    }

    /// Project only: fills `stats[11..13]` with the GPS particle demand of the current view (no sort, no draw).
    pub(crate) fn encode_probe(&self, queue: &wgpu::Queue, stats: &wgpu::Buffer, enc: &mut wgpu::CommandEncoder, bg0: &wgpu::BindGroup, n: u32) {
        let Some(s) = &self.sized else { return };
        self.reset_counters(queue, stats);
        let (gx, gy, _) = split_dispatch(n as u64);
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sorted probe"), timestamp_writes: None });
        cp.set_pipeline(&self.pl_project);
        cp.set_bind_group(0, bg0, &[]);
        cp.set_bind_group(1, &s.sp_bg, &[]);
        cp.dispatch_workgroups(gx, gy, 1);
    }

    /// Full path: project, sort, draw, resolve into the accumulation buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_render(
        &self, queue: &wgpu::Queue, stats: &wgpu::Buffer, enc: &mut wgpu::CommandEncoder, bg0: &wgpu::BindGroup, n: u32, w: u32, h: u32, bg: [f32; 3],
        pl_scan: &wgpu::ComputePipeline, pl_scan_add: &wgpu::ComputePipeline,
    ) {
        let Some(s) = &self.sized else { return };
        self.reset_counters(queue, stats);
        let (gx, gy, _) = split_dispatch(n as u64);
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sorted project"), timestamp_writes: None });
            cp.set_pipeline(&self.pl_project);
            cp.set_bind_group(0, bg0, &[]);
            cp.set_bind_group(1, &s.sp_bg, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&s.counter, 0, &s.args, 0, 16);
        let nblocks = n.div_ceil(256);
        let levels = s.scan_sizes.len();
        for p in 0..PASSES {
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("radix hist"), timestamp_writes: None });
                cp.set_pipeline(&self.pl_hist);
                cp.set_bind_group(0, &s.sort_bgs[p], &[]);
                cp.dispatch_workgroups(nblocks, 1, 1);
            }
            let up: Vec<usize> = (0..levels).collect();
            let down: Vec<usize> = (0..levels - 1).rev().collect();
            for (pipe, order) in [(pl_scan, up), (pl_scan_add, down)] {
                for l in order {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("radix scan"), timestamp_writes: None });
                    cp.set_pipeline(pipe);
                    cp.set_bind_group(0, &s.scan_bgs[l], &[]);
                    cp.dispatch_workgroups(s.scan_sizes[l].div_ceil(256), 1, 1);
                }
            }
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("radix scatter"), timestamp_writes: None });
                cp.set_pipeline(&self.pl_scatter);
                cp.set_bind_group(0, &s.sort_bgs[p], &[]);
                cp.dispatch_workgroups(nblocks, 1, 1);
            }
        }
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sorted draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &s.tex_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0] as f64, g: bg[1] as f64, b: bg[2] as f64, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.pl_draw);
            rp.set_bind_group(0, &s.draw_bg, &[]);
            rp.draw_indirect(&s.args, 0);
        }
        let (px, py, _) = split_dispatch(w as u64 * h as u64);
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sorted resolve"), timestamp_writes: None });
        cp.set_pipeline(&self.pl_resolve);
        cp.set_bind_group(0, bg0, &[]);
        cp.set_bind_group(1, &s.sp_bg, &[]);
        cp.dispatch_workgroups(px, py, 1);
    }

    /// Test hooks: the sorted keys / values of the last render (valid after a GPU wait).
    pub(crate) fn sorted_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        self.sized.as_ref().map(|s| (&s.keys_b, &s.vals_b))
    }
}
