// GPS (Gaussian Point Splatting) MVP, SH0, primitive path: one thread per splat.
// Port of the GPU pipeline modelled by GaussianPointOracle::renderMonteCarlo
// (docs/todo/PLAN_gps_pbvr_rust_webgpu.md section 4):
//   clear -> depth pass (atomicMin depth key) -> colour pass (atomicMin splat id among the
//   depth winners) -> resolve (average subpixels, accumulate), one ensemble per uniform slot.
//
// Samples are regenerated identically by the depth and colour passes from
// (splat id, ensemble seed) only. 32-bit atomics only; no 64-bit, no subgroups, no f16.

struct Frame {
    rot0: vec4<f32>,   // rows of view_rot (xyz)
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    pos: vec4<f32>,    // xyz = camera position, w = low-pass (px^2)
    intr: vec4<f32>,   // fx, fy, cx, cy
    dims: vec4<u32>,   // width, height, spp_side, num_splats
    cfg: vec4<u32>,    // ensemble seed, max points per splat, stride_splat, stride_sub
    cfg2: vec4<u32>,   // stride_pix
    prm: vec4<f32>,    // density scale, near, _, _
    bg: vec4<f32>,
    pbvr: vec4<u32>,   // method (0 = GPS, 1 = Proportional, 2 = Extinction, 3 = ViewConditioned), calibration level 0..3,
                       // flags (bit0 radial C3+R, bit1 centre depth, bit2 low-pass jitter), _
    pbp: vec4<f32>,    // base_k, reference pixel length, object depth, _
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;      // 4 x vec4 per splat
@group(0) @binding(2) var<storage, read_write> depth: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> winner: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
// Particle-parallel path (prepare -> scan -> finalize -> particle_depth/particle_color):
@group(0) @binding(6) var<storage, read_write> offsets: array<u32>;      // counts, then exclusive scan
@group(0) @binding(7) var<storage, read_write> proj: array<vec4<f32>>;   // 2 x vec4 per splat
@group(0) @binding(8) var<storage, read_write> info: array<u32>;         // dispatch info, see finalize
@group(0) @binding(10) var<storage, read> radius_lut: array<f32>;         // r(o, w): 65 x 129, see gps_core::radius_lut
@group(0) @binding(9) var<storage, read> sh_rest: array<f32>;             // SH bands 1..3, channel-major, stride cfg2.y per channel
// stats: 0 = generated points, 1 = truncated splats, 2 = dropped points, 3 = visible splats,
//        4 = orphan subpixels (depth written but no colour winner: pass divergence, must be 0),
//        5 = candidate particles (PBVR, before keep tests), 6 = under-covered splats (ViewConditioned),
//        7 = ensembles skipped because the particle total would overflow u32 (cumulative),
//        16..22 = high words of counters 0..6 (stat_add carries into stats[16 + k]; see GpuStats::from_words),
//        8 = running particle total of the current ensemble, 9 = overflow flag of the current ensemble (both cleared by clear_buffers)

// 64-bit counter k = (stats[16 + k] << 32) | stats[k]: add to the low word, carry into the high word on wrap.
fn stat_add(k: u32, n: u32) {
    let before = atomicAdd(&stats[k], n);
    if (before + n < before) { atomicAdd(&stats[16u + k], 1u); }
}

const PI: f32 = 3.14159265358979;
const DILOG_ONE: f32 = 1.64493406684823;
const EMPTY: u32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------- RNG (bit-identical ints to C++)
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn particle_seed(splat_id: u32, particle_index: u32, frame_index: u32) -> u32 {
    var s = pcg(splat_id * 747796405u + particle_index + 1u);
    s = pcg(s ^ (frame_index * 2654435761u + 1u));   // SeedMode::FrameVarying
    if (s == 0u) { s = 1u; }
    return s;
}

var<private> rng_state: u32;

// U[0,1): top 24 bits, exact in f32 (the C++ stream uses state/2^32 in f64).
fn next() -> f32 {
    rng_state = pcg(rng_state);
    return f32(rng_state >> 8u) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------- Li2 and its inverse
fn dilog_series(x: f32) -> f32 {
    var term = x;
    var sum = x;
    for (var k = 2u; k < 32u; k = k + 1u) {
        term = term * x;
        sum = sum + term / f32(k * k);
    }
    return sum;
}

fn dilog(x: f32) -> f32 {
    if (x >= 1.0) { return DILOG_ONE; }
    if (x <= 0.0) { return 0.0; }
    if (x <= 0.5) { return dilog_series(x); }
    return DILOG_ONE - log(x) * log(1.0 - x) - dilog_series(1.0 - x);
}

// y in [0, hi] with dilog(y) == tgt (safeguarded Newton; dilog is increasing).
fn inv_dilog(tgt: f32, hi_in: f32) -> f32 {
    if (tgt <= 0.0) { return 0.0; }
    var lo = 0.0;
    var hi = hi_in;
    var y = hi * 0.5;
    for (var i = 0u; i < 24u; i = i + 1u) {
        let f = dilog(y) - tgt;
        if (f > 0.0) { hi = y; } else { lo = y; }
        var dfdy = 1.0;
        if (y > 1e-6 && y < 0.9999999) { dfdy = -log(1.0 - y) / y; }
        var ny = y - f / dfdy;
        if (!(ny > lo && ny < hi)) { ny = 0.5 * (lo + hi); }
        if (abs(ny - y) <= 1e-6 * (1.0 + y)) { return ny; }
        y = ny;
    }
    return y;
}

// Table version of corrected_radius: bilinear in (x, w) with x = 1 - sqrt(1 - o) and w = sqrt(-ln(1 - u)).
// Same law, ~1e-3 sigma accuracy (gps_core test), without the per-point Li2 inversion.
fn corrected_radius_lut(o: f32, u: f32) -> f32 {
    if (o <= 0.0) { return 0.0; }
    let v = -log(max(1.0 - u, 5.96e-8));
    let fo = (1.0 - sqrt(max(1.0 - min(o, 1.0), 0.0))) * 64.0;
    let fv = min(sqrt(v) * (128.0 / 4.123105625617661), 128.0);
    let i0 = min(u32(fo), 63u);
    let j0 = min(u32(fv), 127u);
    let a = fo - f32(i0);
    let b = fv - f32(j0);
    let k = i0 * 129u + j0;
    let r0 = mix(radius_lut[k], radius_lut[k + 1u], b);
    let r1 = mix(radius_lut[k + 129u], radius_lut[k + 130u], b);
    return mix(r0, r1, a);
}

// Whitened radius of the corrected law, F(r) = 1 - Li2(o e^{-r^2/2}) / Li2(o).
fn corrected_radius(o: f32, u: f32) -> f32 {
    if (o <= 0.0) { return 0.0; }
    let d = dilog(o);
    let tgt = (1.0 - u) * d;
    let y = min(inv_dilog(tgt, o), o);
    if (y <= 0.0) { return 40.0; }
    return sqrt(max(-2.0 * log(y / o), 0.0));
}

fn poisson(lambda: f32) -> u32 {
    if (lambda <= 0.0) { return 0u; }
    if (lambda < 30.0) {
        let l = exp(-lambda);
        var k = 0u;
        var p = 1.0;
        loop {
            k = k + 1u;
            p = p * next();
            if (p <= l) { break; }
        }
        return k - 1u;
    }
    // Box-Muller, one value (the C++ spare is never reused on this path).
    let u1 = max(next(), 5.96e-8);
    let u2 = next();
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return u32(max(round(lambda + sqrt(lambda) * z), 0.0));
}

// ---------------------------------------------------------------- projection
struct Proj { ok: bool, mean: vec2<f32>, a: f32, b: f32, c: f32, depth: f32 };

fn project(gid: u32) -> Proj {
    var pr: Proj;
    pr.ok = false;
    let g0 = splats[gid * 4u];
    let g1 = splats[gid * 4u + 1u];
    let g2 = splats[gid * 4u + 2u];
    let p = g0.xyz - frame.pos.xyz;
    let cam = vec3<f32>(dot(frame.rot0.xyz, p), dot(frame.rot1.xyz, p), dot(frame.rot2.xyz, p));
    if (cam.z <= max(frame.prm.y, 1e-4)) { return pr; }

    // 3DGS frustum clamp with the oracle's generous tan(fov/2) = 4.
    let lim = 1.3 * 4.0;
    let tx = clamp(cam.x / cam.z, -lim, lim) * cam.z;
    let ty = clamp(cam.y / cam.z, -lim, lim) * cam.z;
    let fx = frame.intr.x;
    let fy = frame.intr.y;
    let iz = 1.0 / cam.z;
    // M = J W (2x3): rows of J are (fx/z, 0, -fx tx/z^2) and (0, fy/z, -fy ty/z^2)
    let m0 = frame.rot0.xyz * (fx * iz) + frame.rot2.xyz * (-fx * tx * iz * iz);
    let m1 = frame.rot1.xyz * (fy * iz) + frame.rot2.xyz * (-fy * ty * iz * iz);

    // R columns from the (w,x,y,z) quaternion; Sigma = sum_k s_k^2 r_k r_k^T
    let q = normalize(g2);
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let c0 = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    let c1 = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    let c2 = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    let s2 = exp(2.0 * g1.xyz);
    let u = vec3<f32>(dot(m0, c0), dot(m0, c1), dot(m0, c2));
    let v = vec3<f32>(dot(m1, c0), dot(m1, c1), dot(m1, c2));
    let a = dot(s2, u * u) + frame.pos.w;
    let b = dot(s2, u * v);
    let c = dot(s2, v * v) + frame.pos.w;
    let det = a * c - b * b;
    if (!(det > 0.0)) { return pr; }

    pr.ok = true;
    pr.mean = vec2<f32>(fx * cam.x / cam.z + frame.intr.z, fy * cam.y / cam.z + frame.intr.w);
    pr.a = a; pr.b = b; pr.c = c;
    pr.depth = cam.z;
    return pr;
}

// View-dependent colour of splat `id`: the SH0 colour stored with the splat plus the active higher bands
// evaluated for the direction camera -> splat (identical to gps_core::eval_sh; the +0.5 is already in `base`).
fn splat_colour(id: u32, base: vec3<f32>) -> vec3<f32> {
    let deg = frame.pbvr.w;
    if (deg == 0u) { return max(base, vec3<f32>(0.0)); }
    let d = normalize(splats[id * 4u].xyz - frame.pos.xyz);
    let x = d.x; let y = d.y; let z = d.z;
    let r = frame.cfg2.y;
    let o = id * 3u * r;
    var res = base;
    // SH(k) = (sh_rest[o + k], sh_rest[o + r + k], sh_rest[o + 2r + k])
    let s0 = vec3<f32>(sh_rest[o], sh_rest[o + r], sh_rest[o + 2u * r]);
    let s1 = vec3<f32>(sh_rest[o + 1u], sh_rest[o + r + 1u], sh_rest[o + 2u * r + 1u]);
    let s2 = vec3<f32>(sh_rest[o + 2u], sh_rest[o + r + 2u], sh_rest[o + 2u * r + 2u]);
    res = res - 0.4886025119029199 * y * s0 + 0.4886025119029199 * z * s1 - 0.4886025119029199 * x * s2;
    if (deg >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        let xy = x * y; let yz = y * z; let xz = x * z;
        let s3 = vec3<f32>(sh_rest[o + 3u], sh_rest[o + r + 3u], sh_rest[o + 2u * r + 3u]);
        let s4 = vec3<f32>(sh_rest[o + 4u], sh_rest[o + r + 4u], sh_rest[o + 2u * r + 4u]);
        let s5 = vec3<f32>(sh_rest[o + 5u], sh_rest[o + r + 5u], sh_rest[o + 2u * r + 5u]);
        let s6 = vec3<f32>(sh_rest[o + 6u], sh_rest[o + r + 6u], sh_rest[o + 2u * r + 6u]);
        let s7 = vec3<f32>(sh_rest[o + 7u], sh_rest[o + r + 7u], sh_rest[o + 2u * r + 7u]);
        res = res + 1.0925484305920792 * xy * s3 - 1.0925484305920792 * yz * s4
            + 0.31539156525252005 * (2.0 * zz - xx - yy) * s5
            - 1.0925484305920792 * xz * s6 + 0.5462742152960396 * (xx - yy) * s7;
        if (deg >= 3u) {
            let s8 = vec3<f32>(sh_rest[o + 8u], sh_rest[o + r + 8u], sh_rest[o + 2u * r + 8u]);
            let s9 = vec3<f32>(sh_rest[o + 9u], sh_rest[o + r + 9u], sh_rest[o + 2u * r + 9u]);
            let s10 = vec3<f32>(sh_rest[o + 10u], sh_rest[o + r + 10u], sh_rest[o + 2u * r + 10u]);
            let s11 = vec3<f32>(sh_rest[o + 11u], sh_rest[o + r + 11u], sh_rest[o + 2u * r + 11u]);
            let s12 = vec3<f32>(sh_rest[o + 12u], sh_rest[o + r + 12u], sh_rest[o + 2u * r + 12u]);
            let s13 = vec3<f32>(sh_rest[o + 13u], sh_rest[o + r + 13u], sh_rest[o + 2u * r + 13u]);
            let s14 = vec3<f32>(sh_rest[o + 14u], sh_rest[o + r + 14u], sh_rest[o + 2u * r + 14u]);
            res = res - 0.5900435899266435 * y * (3.0 * xx - yy) * s8 + 2.890611442640554 * xy * z * s9
                - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * s10
                + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * s11
                - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * s12
                + 1.445305721320277 * z * (xx - yy) * s13
                - 0.5900435899266435 * x * (xx - 3.0 * yy) * s14;
        }
    }
    return max(res, vec3<f32>(0.0));
}

fn thread_index(gid: vec3<u32>, stride: u32) -> u32 {
    return gid.y * stride + gid.x;
}

// One particle: position from (radius, angle) draws, then the depth / colour write.
// mode: 0 = depth pass, 1 = colour pass. Uses the current rng stream.
fn emit_particle(gid: u32, mode: u32, mean: vec2<f32>, depth_z: f32, o: f32, l00: f32, l10: f32, l11: f32) {
    let spp_side = frame.dims.z;
    let spp = spp_side * spp_side;
    let W = frame.dims.x;
    let H = frame.dims.y;
    var r: f32;
    if (frame.cfg2.z == 0u) { r = corrected_radius_lut(o, next()); } else { r = corrected_radius(o, next()); }
    let th = 2.0 * PI * next();
    let wx = r * cos(th);
    let wy = r * sin(th);
    let pix = mean + vec2<f32>(l00 * wx, l10 * wx + l11 * wy);
    if (pix.x < 0.0 || pix.x >= f32(W) || pix.y < 0.0 || pix.y >= f32(H)) { return; }
    let px = u32(pix.x);
    let py = u32(pix.y);
    let sx = min(spp_side - 1u, u32((pix.x - f32(px)) * f32(spp_side)));
    let sy = min(spp_side - 1u, u32((pix.y - f32(py)) * f32(spp_side)));
    let idx = (py * W + px) * spp + sy * spp_side + sx;
    let key = bitcast<u32>(depth_z) | 0x80000000u;
    if (mode == 0u) {
        atomicMin(&depth[idx], key);
    } else if (atomicLoad(&depth[idx]) == key) {
        atomicMin(&winner[idx], gid);      // tie: smallest splat id (= C++ first-come)
    }
}

// Expected particle count of a projected splat (before Poisson).
fn expected_count(pr: Proj, o: f32) -> f32 {
    let spp = frame.dims.z * frame.dims.z;
    let det = pr.a * pr.c - pr.b * pr.b;
    return frame.prm.x * f32(spp) * 2.0 * PI * sqrt(det) * dilog(o);
}

// ---------------------------------------------------------------- splat-parallel path (reference)
// One thread per splat generates all of that splat's particles from one sequential RNG stream.
fn splat_pass(gid: u32, mode: u32) {
    let pr = project(gid);
    if (!pr.ok) { return; }
    let o = clamp(splats[gid * 4u].w, 0.0, 1.0);
    if (o <= 0.0) { return; }
    let en = expected_count(pr, o);
    if (!(en > 0.0)) { return; }

    rng_state = particle_seed(gid, 0u, frame.cfg.x);
    var n = poisson(en);
    if (mode == 0u) {
        stat_add(0u, n);
        stat_add(3u, 1u);
    }
    let cap = frame.cfg.y;
    if (n > cap) {
        if (mode == 0u) {
            stat_add(1u, 1u);
            stat_add(2u, n - cap);
        }
        n = cap;
    }
    let l00 = sqrt(max(pr.a, 1e-30));
    let l10 = pr.b / l00;
    let l11 = sqrt(max(pr.c - l10 * l10, 1e-30));
    for (var i = 0u; i < n; i = i + 1u) {
        emit_particle(gid, mode, pr.mean, pr.depth, o, l00, l10, l11);
    }
}

@compute @workgroup_size(64)
fn clear_buffers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid, frame.cfg.w);
    if (i == 0u) {
        atomicStore(&stats[8], 0u);
        atomicStore(&stats[9], 0u);
    }
    if (i < frame.dims.x * frame.dims.y * frame.dims.z * frame.dims.z) {
        atomicStore(&depth[i], EMPTY);
        atomicStore(&winner[i], EMPTY);
    }
}

@compute @workgroup_size(64)
fn depth_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid, frame.cfg.z);
    if (i < frame.dims.w) { splat_pass(i, 0u); }
}

@compute @workgroup_size(64)
fn color_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid, frame.cfg.z);
    if (i < frame.dims.w) { splat_pass(i, 1u); }
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    // An ensemble whose particle total overflowed u32 was skipped (zero particles): it must not add
    // background samples to the history or count towards convergence. stats[9] is stable here.
    if (atomicLoad(&stats[9]) != 0u) { return; }
    let i = thread_index(gid, frame.cfg2.x);
    let n_pix = frame.dims.x * frame.dims.y;
    if (i >= n_pix) { return; }
    let spp = frame.dims.z * frame.dims.z;
    var sum = vec3<f32>(0.0);
    for (var k = 0u; k < spp; k = k + 1u) {
        let d = atomicLoad(&depth[i * spp + k]);
        if (d == EMPTY) {
            sum = sum + frame.bg.xyz;
        } else {
            let id = atomicLoad(&winner[i * spp + k]);
            if (id == EMPTY) {
                stat_add(4u, 1u);
                sum = sum + frame.bg.xyz;
            } else {
                sum = sum + splat_colour(id, splats[id * 4u + 3u].xyz);
            }
        }
    }
    accum[i] = accum[i] + vec4<f32>(sum / f32(spp), 1.0);
}

// ---------------------------------------------------------------- PBVR (object-space 3D particles)
// Port of GaussianPointOracle::renderParticles3D / gps_pbvr3d.comp: particles are drawn from the 3D Gaussian
// itself, projected with the exact pinhole model, optionally jittered by the low-pass variance and thinned
// radially (C3+R). The colour is the splat's SH0 colour, so the winner buffer still holds a splat id.
const METHOD_PROPORTIONAL: u32 = 1u;
const METHOD_VIEW_CONDITIONED: u32 = 3u;
const FLAG_RADIAL: u32 = 1u;
const FLAG_CENTRE: u32 = 2u;
const FLAG_JITTER: u32 = 4u;

var<private> spare_normal: f32;
var<private> has_spare: bool;

// N(0,1) by Box-Muller, one value per call and the sine half kept for the next call (the C++ RandStream).
fn normal() -> f32 {
    if (has_spare) {
        has_spare = false;
        return spare_normal;
    }
    let u1 = max(next(), 5.96e-8);
    let u2 = next();
    let r = sqrt(-2.0 * log(u1));
    let th = 2.0 * PI * u2;
    spare_normal = r * sin(th);
    has_spare = true;
    return r * cos(th);
}

// -log(1 - o) without log1p.
fn extinction_term(o: f32) -> f32 {
    let c = min(o, 1.0 - 1e-6);
    if (c < 1e-3) { return c * (1.0 + c * (0.5 + c * 0.33333334)); }
    return -log(1.0 - c);
}

fn pixel_density_scale(depth: f32) -> f32 {
    if (!(depth > max(frame.prm.y, 0.0))) { return 0.0; }
    let ratio = frame.pbp.y / depth;
    return min(ratio * ratio * frame.intr.x * frame.intr.y, 1.0e12);
}

// h(x) = -log(1 - x) / x. The C3+R keep probability is h(o e^{-r^2/2}) / h(o).
fn extinction_over_linear(x: f32) -> f32 {
    if (x < 1e-4) { return 1.0 + 0.5 * x; }
    return -log(1.0 - min(x, 1.0 - 1e-6)) / x;
}
fn radial_keep(o: f32, r2: f32) -> f32 {
    let oo = min(o, 1.0 - 1e-6);
    let y = oo * exp(-0.5 * max(r2, 0.0));
    return clamp(extinction_over_linear(y) / extinction_over_linear(oo), 0.0, 1.0);
}

// Expected candidate count of one splat for the selected opacity rule and calibration level.
fn pbvr_lambda(pr: Proj, o: f32) -> f32 {
    let method = frame.pbvr.x;
    var g = extinction_term(o);
    if (method == METHOD_PROPORTIONAL) { g = min(o, 1.0 - 1e-6); }
    var scale = frame.pbp.x;
    if (method != METHOD_VIEW_CONDITIONED) {
        let level = frame.pbvr.y;
        if (level == 1u) {
            scale = scale * pixel_density_scale(frame.pbp.z);
        } else if (level == 2u) {
            scale = scale * pixel_density_scale(pr.depth);
        } else if (level == 3u) {
            let spp = f32(frame.dims.z * frame.dims.z);
            scale = spp * 2.0 * PI * sqrt(pr.a * pr.c - pr.b * pr.b);
        }
    }
    let n = frame.prm.x * g * scale;
    if (!(n > 0.0) || n > 3.0e38) { return 0.0; }
    return n;
}

// Lower-triangular Cholesky factor of the world covariance R S^2 R^T: (l00, l10, l11) and (l20, l21, l22).
struct Chol3 { a: vec3<f32>, b: vec3<f32> };
fn chol_of(gid: u32) -> Chol3 {
    let g1 = splats[gid * 4u + 1u];
    let q = normalize(splats[gid * 4u + 2u]);
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let c0 = vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    let c1 = vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    let c2 = vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    let s2 = exp(2.0 * g1.xyz);
    let a00 = s2.x * c0.x * c0.x + s2.y * c1.x * c1.x + s2.z * c2.x * c2.x;
    let a10 = s2.x * c0.y * c0.x + s2.y * c1.y * c1.x + s2.z * c2.y * c2.x;
    let a20 = s2.x * c0.z * c0.x + s2.y * c1.z * c1.x + s2.z * c2.z * c2.x;
    let a11 = s2.x * c0.y * c0.y + s2.y * c1.y * c1.y + s2.z * c2.y * c2.y;
    let a21 = s2.x * c0.z * c0.y + s2.y * c1.z * c1.y + s2.z * c2.z * c2.y;
    let a22 = s2.x * c0.z * c0.z + s2.y * c1.z * c1.z + s2.z * c2.z * c2.z;
    let l00 = sqrt(max(a00, 1e-30));
    let l10 = a10 / l00;
    let l20 = a20 / l00;
    let l11 = sqrt(max(a11 - l10 * l10, 1e-30));
    let l21 = (a21 - l20 * l10) / l11;
    let l22 = sqrt(max(a22 - l20 * l20 - l21 * l21, 1e-30));
    return Chol3(vec3<f32>(l00, l10, l11), vec3<f32>(l20, l21, l22));
}

// One PBVR particle. Draw order (identical to the CPU oracle): z, y, x normals, [view-conditioned keep],
// [jitter y, x], [radial keep]. a = (mean, depth, opacity), b.w = view-conditioned keep probability,
// c = (world position, inverse 2D covariance xx), d.xy = (inverse xy, inverse yy).
fn emit_particle_pbvr(gid: u32, mode: u32, a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, ch_a: vec3<f32>, ch_b: vec3<f32>) {
    has_spare = false;
    let uz = normal();
    let uy = normal();
    let ux = normal();
    if (frame.pbvr.x == METHOD_VIEW_CONDITIONED) {
        if (next() >= b.w) { return; }
    }
    let world = c.xyz + vec3<f32>(ch_a.x * ux, ch_a.y * ux + ch_a.z * uy, ch_b.x * ux + ch_b.y * uy + ch_b.z * uz);
    let p = world - frame.pos.xyz;
    let cam = vec3<f32>(dot(frame.rot0.xyz, p), dot(frame.rot1.xyz, p), dot(frame.rot2.xyz, p));
    if (cam.z <= frame.prm.y) { return; }
    var pix = vec2<f32>(frame.intr.x * cam.x / cam.z + frame.intr.z, frame.intr.y * cam.y / cam.z + frame.intr.w);
    let flags = frame.pbvr.z;
    if ((flags & FLAG_JITTER) != 0u) {
        let jy = normal();
        let jx = normal();
        pix = pix + sqrt(max(frame.pos.w, 0.0)) * vec2<f32>(jx, jy);
    }
    if ((flags & FLAG_RADIAL) != 0u) {
        let dv = pix - a.xy;
        let r2 = c.w * dv.x * dv.x + 2.0 * d.x * dv.x * dv.y + d.y * dv.y * dv.y;
        if (next() >= radial_keep(a.w, r2)) { return; }
    }
    var depth_z = cam.z;
    if ((flags & FLAG_CENTRE) != 0u) { depth_z = a.z; }
    if (mode == 0u) { stat_add(0u, 1u); }
    let W = frame.dims.x;
    let H = frame.dims.y;
    if (pix.x < 0.0 || pix.x >= f32(W) || pix.y < 0.0 || pix.y >= f32(H)) { return; }
    let spp_side = frame.dims.z;
    let px = u32(pix.x);
    let py = u32(pix.y);
    let sx = min(spp_side - 1u, u32((pix.x - f32(px)) * f32(spp_side)));
    let sy = min(spp_side - 1u, u32((pix.y - f32(py)) * f32(spp_side)));
    let idx = (py * W + px) * spp_side * spp_side + sy * spp_side + sx;
    let key = bitcast<u32>(depth_z) | 0x80000000u;
    if (mode == 0u) {
        atomicMin(&depth[idx], key);
    } else if (atomicLoad(&depth[idx]) == key) {
        atomicMin(&winner[idx], gid);
    }
}

// ---------------------------------------------------------------- particle-parallel path
fn prepare_gps(i: u32, pr: Proj, o: f32) -> u32 {
    let en = expected_count(pr, o);
    if (!(en > 0.0)) { return 0u; }
    rng_state = particle_seed(i, 0u, frame.cfg.x);
    var n = poisson(en);
    stat_add(0u, n);
    stat_add(3u, 1u);
    let cap = frame.cfg.y;
    if (n > cap) {
        stat_add(1u, 1u);
        stat_add(2u, n - cap);
        n = cap;
    }
    return n;
}

fn prepare_pbvr(i: u32, pr: Proj, o: f32) -> u32 {
    let lambda = pbvr_lambda(pr, o);
    if (!(lambda > 0.0)) { return 0u; }
    let det = pr.a * pr.c - pr.b * pr.b;
    var keep = 1.0;
    if (frame.pbvr.x == METHOD_VIEW_CONDITIONED) {
        // thin the extinction candidates so the on-screen count matches GPS: spp * 2 pi sqrt(det) * Li2(o)
        let spp = f32(frame.dims.z * frame.dims.z);
        let target_count = frame.prm.x * spp * 2.0 * PI * sqrt(det) * dilog(o);
        if (lambda > 0.5) { keep = clamp(target_count / lambda, 0.0, 1.0); } else { keep = 0.0; }
        if (target_count > lambda) { stat_add(6u, 1u); }
    }
    rng_state = particle_seed(i, 0u, frame.cfg.x);
    var n = poisson(lambda);
    stat_add(5u, n);
    stat_add(3u, 1u);
    let cap = frame.cfg.y;
    if (n > cap) {
        stat_add(1u, 1u);
        stat_add(2u, n - cap);
        n = cap;
    }
    let idet = 1.0 / max(det, 1e-30);
    let g0 = splats[i * 4u];
    proj[i * 6u + 1u].w = keep;
    proj[i * 6u + 2u] = vec4<f32>(g0.xyz, pr.c * idet);
    proj[i * 6u + 3u] = vec4<f32>(-pr.b * idet, pr.a * idet, 0.0, 0.0);
    let ch = chol_of(i);                    // cached: recomputing it per particle dominated the PBVR cost
    proj[i * 6u + 4u] = vec4<f32>(ch.a, 0.0);
    proj[i * 6u + 5u] = vec4<f32>(ch.b, 0.0);
    return n;
}

// prepare: per splat, Poisson count (same stream / same cap as the reference path) + projection.
// Every splat writes its count (0 when invisible), so the scan input is always fully defined.
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid, frame.cfg.z);
    if (i >= frame.dims.w) { return; }
    var cnt = 0u;
    let pr = project(i);
    let o = clamp(splats[i * 4u].w, 0.0, 1.0);
    if (pr.ok && o > 0.0) {
        let l00 = sqrt(max(pr.a, 1e-30));
        let l10 = pr.b / l00;
        let l11 = sqrt(max(pr.c - l10 * l10, 1e-30));
        proj[i * 6u] = vec4<f32>(pr.mean, pr.depth, o);
        proj[i * 6u + 1u] = vec4<f32>(l00, l10, l11, 1.0);
        if (frame.pbvr.x == 0u) {
            cnt = prepare_gps(i, pr, o);
        } else {
            cnt = prepare_pbvr(i, pr, o);
        }
    }
    if (cnt > 0u) {
        // exact overflow detection of the u32 scan total (every add sees the running sum before it)
        let before = atomicAdd(&stats[8], cnt);
        if (before > 0xffffffffu - cnt) { atomicStore(&stats[9], 1u); }
    }
    offsets[i] = cnt;                       // scanned in place by gps_scan.wgsl
    if (i == frame.dims.w - 1u) { info[5] = cnt; }
}

// After the scan: total = offsets[n-1] + last count. info = [gx, gy, 1, stride, total, last_count].
// The indirect-dispatch arguments are COPIED out of `info` (a buffer bound as storage cannot also
// be the indirect buffer of the same pass).
@compute @workgroup_size(1)
fn finalize_dispatch() {
    var total = offsets[frame.dims.w - 1u] + info[5];
    if (atomicLoad(&stats[9]) != 0u) {
        // the total does not fit in 32 bits: skip this ensemble (counted, reported by the UI) instead of corrupting it
        total = 0u;
        atomicAdd(&stats[7], 1u);
    }
    let groups = max((total + 63u) / 64u, 1u);
    let gx = min(groups, 65535u);
    let gy = (groups + gx - 1u) / gx;
    info[0] = gx;
    info[1] = gy;
    info[2] = 1u;
    info[3] = gx * 64u;
    info[4] = total;
}

// Owner of global particle p: the last splat whose exclusive offset is <= p. Zero-count splats share
// their successor's offset, so the last such splat always has a positive count.
fn owner_of(p: u32) -> u32 {
    var lo = 0u;
    var hi = frame.dims.w - 1u;
    while (lo < hi) {
        let mid = (lo + hi + 1u) / 2u;
        if (offsets[mid] <= p) { lo = mid; } else { hi = mid - 1u; }
    }
    return lo;
}

fn particle_pass(gid: vec3<u32>, mode: u32) {
    let p = gid.y * info[3] + gid.x;
    if (p >= info[4]) { return; }
    let i = owner_of(p);
    let k = p - offsets[i];
    let a = proj[i * 6u];
    let b = proj[i * 6u + 1u];
    // Random access: every particle has its own stream, so no thread depends on another's draws.
    rng_state = particle_seed(i, k + 1u, frame.cfg.x);
    if (frame.pbvr.x == 0u) {
        emit_particle(i, mode, a.xy, a.z, a.w, b.x, b.y, b.z);
    } else {
        emit_particle_pbvr(i, mode, a, b, proj[i * 6u + 2u], proj[i * 6u + 3u], proj[i * 6u + 4u].xyz, proj[i * 6u + 5u].xyz);
    }
}

@compute @workgroup_size(64)
fn particle_depth(@builtin(global_invocation_id) gid: vec3<u32>) { particle_pass(gid, 0u); }

@compute @workgroup_size(64)
fn particle_color(@builtin(global_invocation_id) gid: vec3<u32>) { particle_pass(gid, 1u); }
