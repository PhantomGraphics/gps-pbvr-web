//! Phase 0 GPU verification (shared by native `gps-verify` and browser `gps-web`).
//!
//! Checks, against CPU references computed with `gps-core`:
//!   1. u32 atomicMin two-pass (depth, then colour of depth winners), several
//!      ensembles in ONE submission using separate uniform slots;
//!   2. hierarchical exclusive scan (multi-level, separate dispatches);
//! and prints adapter info + limits + memory estimates for the record.
//! Exit code 0 only if every check passes.

use gps_core::{ordered_depth_key, pcg_hash};
use wgpu::util::DeviceExt;

pub mod gps;
mod sorted;
pub mod verify;

const SAMPLE_WGSL: &str = include_str!("../../../shaders/p0_sample.wgsl");
const SCAN_WGSL: &str = include_str!("../../../shaders/gps_scan.wgsl");

const UNIFORM_ALIGN: u64 = 256; // >= min_uniform_buffer_offset_alignment on all targets

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Slot {
    seed: u32,
    pixel_base: u32,
    pixels: u32,
    samples: u32,
}

fn cpu_sample(slot: &Slot, i: u32) -> (u32, u32, u32) {
    let h0 = pcg_hash(i ^ pcg_hash(slot.seed));
    let h1 = pcg_hash(h0);
    let h2 = pcg_hash(h1);
    let pixel = slot.pixel_base + (h0 % slot.pixels);
    let bits = 0x3F80_0000u32 | (h1 & 0x007F_FFFF);
    let key = ordered_depth_key(f32::from_bits(bits)) & 0xFFFF_0000;
    (pixel, key, h2 & 0x00FF_FFFF)
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn readback(gpu: &Gpu, buf: &wgpu::Buffer, size: u64) -> Vec<u32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    gpu.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // Native needs an explicit poll; in the browser the map callback fires on its own.
    #[cfg(not(target_arch = "wasm32"))]
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    rx.await.expect("map channel dropped").expect("map failed");
    let data = slice.get_mapped_range().expect("mapped range");
    bytemuck::cast_slice::<u8, u32>(&data).to_vec()
}

fn compute_pipeline(gpu: &Gpu, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn storage(gpu: &Gpu, label: &str, bytes: u64, init: Option<&[u32]>) -> wgpu::Buffer {
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    match init {
        Some(d) => gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(d),
            usage,
        }),
        None => gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        }),
    }
}

// ------------------------------------------------------------ check 1: atomics

async fn check_atomic_two_pass(gpu: &Gpu, log: &mut dyn FnMut(String)) -> bool {
    const ENSEMBLES: u32 = 3;
    const PIXELS: u32 = 4096;
    const SAMPLES: u32 = 200_000;

    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("p0_sample"),
        source: wgpu::ShaderSource::Wgsl(SAMPLE_WGSL.into()),
    });
    let clear = compute_pipeline(gpu, &module, "clear_buffers");
    let dpass = compute_pipeline(gpu, &module, "depth_pass");
    let cpass = compute_pipeline(gpu, &module, "color_pass");

    let slots: Vec<Slot> = (0..ENSEMBLES)
        .map(|e| Slot { seed: 1000 + e * 7, pixel_base: e * PIXELS, pixels: PIXELS, samples: SAMPLES })
        .collect();
    // One uniform buffer, one aligned slot per ensemble (never a single overwritten region).
    let mut ub = vec![0u8; (UNIFORM_ALIGN * ENSEMBLES as u64) as usize];
    for (e, s) in slots.iter().enumerate() {
        let o = e * UNIFORM_ALIGN as usize;
        ub[o..o + 16].copy_from_slice(bytemuck::bytes_of(s));
    }
    let uniform = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("slots"),
        contents: &ub,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let total = (ENSEMBLES * PIXELS) as u64 * 4;
    let depth = storage(gpu, "depth", total, None);
    let color = storage(gpu, "color", total, None);

    // Bind groups per ensemble (slot offset) per pipeline layout.
    // Auto layouts only contain bindings the entry point statically uses
    // (depth_pass never touches `color`), so build entries per pipeline.
    let mk_bg = |pipe: &wgpu::ComputePipeline, e: u32, uses_color: bool| {
        let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: e as u64 * UNIFORM_ALIGN,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: depth.as_entire_binding() },
        ];
        if uses_color {
            entries.push(wgpu::BindGroupEntry { binding: 2, resource: color.as_entire_binding() });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        })
    };

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for e in 0..ENSEMBLES {
        // Separate compute passes => ordered, with storage-buffer dependencies made explicit.
        for (pipe, n, uc) in [(&clear, PIXELS, true), (&dpass, SAMPLES, false), (&cpass, SAMPLES, true)] {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &mk_bg(pipe, e, uc), &[]);
            cp.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
    }
    gpu.queue.submit([enc.finish()]);
    let gd = readback(gpu, &depth, total).await;
    let gc = readback(gpu, &color, total).await;

    // CPU reference.
    let mut rd = vec![0xFFFF_FFFFu32; (ENSEMBLES * PIXELS) as usize];
    let mut rc = rd.clone();
    for s in &slots {
        for i in 0..s.samples {
            let (p, k, _) = cpu_sample(s, i);
            rd[p as usize] = rd[p as usize].min(k);
        }
        for i in 0..s.samples {
            let (p, k, c) = cpu_sample(s, i);
            if rd[p as usize] == k {
                rc[p as usize] = rc[p as usize].min(c);
            }
        }
    }
    let bad_d = gd.iter().zip(&rd).filter(|(a, b)| a != b).count();
    let bad_c = gc.iter().zip(&rc).filter(|(a, b)| a != b).count();
    let ties = slots.iter().map(|s| {
        (0..s.samples).filter(|&i| { let (p, k, _) = cpu_sample(s, i); rd[p as usize] == k }).count()
    }).sum::<usize>();
    log(format!("[atomic two-pass] ensembles={ENSEMBLES} pixels/ens={PIXELS} samples/ens={SAMPLES} \
         depth-winner samples={ties} mismatches: depth={bad_d} colour={bad_c}"));
    bad_d == 0 && bad_c == 0
}

// ------------------------------------------------------------- check 2: scan

async fn check_scan(gpu: &Gpu, n: u32, log: &mut dyn FnMut(String)) -> bool {
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("p0_scan"),
        source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()),
    });
    let scan = compute_pipeline(gpu, &module, "scan_block");
    let add = compute_pipeline(gpu, &module, "add_offsets");

    let input: Vec<u32> = (0..n).map(|i| pcg_hash(i) % 7).collect();
    // Level sizes: n, ceil(n/256), ... until 1.
    let mut sizes = vec![n];
    // Stop at a level that fits one workgroup; that top level is scanned too.
    while *sizes.last().unwrap() > 256 {
        sizes.push(sizes.last().unwrap().div_ceil(256));
    }
    let bufs: Vec<wgpu::Buffer> = sizes
        .iter()
        .enumerate()
        .map(|(l, &s)| {
            let padded = (s.max(1) as u64) * 4;
            if l == 0 { storage(gpu, "lvl0", padded, Some(&input)) } else { storage(gpu, "lvl", padded, None) }
        })
        .collect();
    // The top level's block total has nowhere to go; give it its own scratch word
    // (aliasing it with the level's data buffer would clobber data[0]).
    let dummy = storage(gpu, "dummy_sums", 4, None);
    let params: Vec<wgpu::Buffer> = sizes
        .iter()
        .map(|&s| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sp"),
                contents: bytemuck::cast_slice(&[s, 0u32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect();
    let bg = |pipe: &wgpu::ComputePipeline, lvl: usize| {
        // sums buffer for the top level is a dummy (its own buffer, unused result)
        let sums = if lvl + 1 < bufs.len() { &bufs[lvl + 1] } else { &dummy };
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params[lvl].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs[lvl].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sums.as_entire_binding() },
            ],
        })
    };

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let levels = sizes.len();
    // Up-sweep: scan every level, including the top one (<= 256 items, one group).
    for l in 0..levels {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&scan);
        cp.set_bind_group(0, &bg(&scan, l), &[]);
        cp.dispatch_workgroups(sizes[l].div_ceil(256), 1, 1);
    }
    // Down-sweep: add scanned block offsets back.
    for l in (0..levels - 1).rev() {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&add);
        cp.set_bind_group(0, &bg(&add, l), &[]);
        cp.dispatch_workgroups(sizes[l].div_ceil(256), 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    let got = readback(gpu, &bufs[0], n as u64 * 4).await;

    let mut acc = 0u32;
    let mut bad = 0usize;
    for (i, &v) in input.iter().enumerate() {
        if got[i] != acc {
            bad += 1;
        }
        acc = acc.wrapping_add(v);
    }
    log(format!("[scan] n={n} levels={levels} total={acc} mismatches={bad}"));
    bad == 0
}

/// Runs every Phase 0 check on `adapter`; `log` receives report lines. True iff all pass.
pub async fn run_phase0(adapter: &wgpu::Adapter, log: &mut dyn FnMut(String)) -> bool {
    let info = adapter.get_info();
    log(format!("adapter: {} ({:?}, backend {:?}, driver '{}')", info.name, info.device_type, info.backend, info.driver_info));
    let lim = adapter.limits();
    log(format!(
        "limits: max_storage_buffer_binding_size={} max_buffer_size={} max_storage_buffers_per_stage={}          max_compute_workgroups_per_dimension={} max_compute_invocations_per_workgroup={}          min_uniform_buffer_offset_alignment={} max_texture_dimension_2d={}",
        lim.max_storage_buffer_binding_size,
        lim.max_buffer_size,
        lim.max_storage_buffers_per_shader_stage,
        lim.max_compute_workgroups_per_dimension,
        lim.max_compute_invocations_per_workgroup,
        lim.min_uniform_buffer_offset_alignment,
        lim.max_texture_dimension_2d
    ));
    if lim.min_uniform_buffer_offset_alignment as u64 > UNIFORM_ALIGN {
        log("FAIL: uniform offset alignment exceeds 256".into());
        return false;
    }
    for (w, h, s) in [(1280u64, 720u64, 1u64), (1920, 1080, 4), (1920, 1080, 16)] {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        log(format!(
            "memory {w}x{h} S={s}: depth+color={:.1} MiB, accum={:.1} MiB, depth alone={:.1} MiB",
            mib(8 * w * h * s), mib(16 * w * h), mib(4 * w * h * s)
        ));
    }
    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("gps-phase0"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
    {
        Ok(d) => d,
        Err(e) => {
            log(format!("request_device failed: {e}"));
            return false;
        }
    };
    let gpu = Gpu { device, queue };
    let mut ok = check_atomic_two_pass(&gpu, log).await;
    for n in [1u32, 255, 256, 257, 65_536, 300_000, 1_000_003] {
        ok &= check_scan(&gpu, n, log).await;
    }
    log(if ok { "PHASE0: PASS".into() } else { "PHASE0: FAIL".into() });
    ok
}
