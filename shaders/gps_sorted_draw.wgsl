// Back-to-front instanced quads of the depth-sorted splats (see sp_project in gps_splat.wgsl).
struct DrawParams { size: vec2<f32>, pad: vec2<f32> };
@group(0) @binding(0) var<uniform> dp: DrawParams;
@group(0) @binding(1) var<storage, read> sp_proj: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sp_vals: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) col: vec3<f32>,
    @location(2) ok: vec2<f32>,   // opacity, K
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let id = sp_vals[ii];
    let a = sp_proj[id * 3u];
    let b = sp_proj[id * 3u + 1u];
    let c = sp_proj[id * 3u + 2u];
    let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let uv = corner * a.w;                                   // whitened coordinates in [-K, K]^2
    let p = a.xy + b.xy * uv.x + b.zw * uv.y;                // pixels, y down
    var o: VsOut;
    o.pos = vec4<f32>(2.0 * p.x / dp.size.x - 1.0, 1.0 - 2.0 * p.y / dp.size.y, 0.0, 1.0);
    o.uv = uv;
    o.col = c.xyz;
    o.ok = vec2<f32>(a.z, a.w);
    return o;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(i.uv, i.uv);
    if (r2 > i.ok.y * i.ok.y) { discard; }
    let alpha = min(i.ok.x * exp(-0.5 * r2), 0.99);
    if (alpha < 0.00392156862745098) { discard; }
    return vec4<f32>(i.col, alpha);
}
