// Phase 0 technology checks: u32 atomicMin two-pass depth/colour + hierarchical scan.
// Mirrors the gps_splat.comp structure (depth pass, then colour pass re-generating
// the same samples), without 64-bit atomics.

struct Slot {
    seed: u32,        // per-ensemble seed
    pixel_base: u32,  // offset of this ensemble's pixel block in depth/color
    pixels: u32,
    samples: u32,
};

@group(0) @binding(0) var<uniform> slot: Slot;
@group(0) @binding(1) var<storage, read_write> depth: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> color: array<atomic<u32>>;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Sample i of this ensemble: (pixel, depth key, colour payload). Regenerated
// identically by both passes.
struct Sample { pixel: u32, key: u32, rgb: u32 };

fn gen_sample(i: u32) -> Sample {
    let h0 = pcg(i ^ pcg(slot.seed));
    let h1 = pcg(h0);
    let h2 = pcg(h1);
    var s: Sample;
    s.pixel = slot.pixel_base + (h0 % slot.pixels);
    // depth in [1,2): exact float from mantissa bits; ordered key for positive floats.
    let bits = 0x3F800000u | (h1 & 0x007FFFFFu);
    s.key = bitcast<u32>(bitcast<f32>(bits)) | 0x80000000u;
    // deliberately few depth values -> many exact-depth ties
    s.key = s.key & 0xFFFF0000u;
    s.rgb = h2 & 0x00FFFFFFu;
    return s;
}

@compute @workgroup_size(64)
fn clear_buffers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = slot.pixel_base + gid.x;
    if (gid.x < slot.pixels) {
        atomicStore(&depth[p], 0xFFFFFFFFu);
        atomicStore(&color[p], 0xFFFFFFFFu);
    }
}

@compute @workgroup_size(64)
fn depth_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= slot.samples) { return; }
    let s = gen_sample(gid.x);
    atomicMin(&depth[s.pixel], s.key);
}

@compute @workgroup_size(64)
fn color_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= slot.samples) { return; }
    let s = gen_sample(gid.x);
    if (atomicLoad(&depth[s.pixel]) == s.key) {
        atomicMin(&color[s.pixel], s.rgb);   // tie rule: smallest colour wins
    }
}

