// Composite: accum (linear RGB sum, count in .w) -> display. Manual sRGB OETF; the target must be
// a NON-sRGB format (bgra8unorm / rgba8unorm) so gamma is applied exactly once.

struct Params {
    width: u32,
    height: u32,
    exposure: f32,
    gamma_on: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // one big triangle
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn oetf(c: f32) -> f32 {
    if (c <= 0.0031308) { return 12.92 * c; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // The accumulation buffer has the internal resolution; the target may differ in size
    // (nearest mapping keeps a 1:1 pixel mapping when they match).
    let x = min(u32(frag.x), params.width - 1u);
    let y = min(u32(frag.y), params.height - 1u);
    let a = accum[y * params.width + x];
    var c = a.xyz / max(a.w, 1.0) * params.exposure;
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.gamma_on != 0u) {
        c = vec3<f32>(oetf(c.x), oetf(c.y), oetf(c.z));
    }
    return vec4<f32>(c, 1.0);
}
