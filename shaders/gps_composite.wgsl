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

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOut {
    // one big triangle
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

fn oetf(c: f32) -> f32 {
    if (c <= 0.0031308) { return 12.92 * c; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_pixel(p: vec2<i32>) -> vec3<f32> {
    let q = vec2<u32>(clamp(p, vec2<i32>(0), vec2<i32>(i32(params.width) - 1, i32(params.height) - 1)));
    let a = accum[q.y * params.width + q.x];
    return a.xyz / max(a.w, 1.0);
}

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Pixel-centred interpolation in linear space, before exposure and OETF.
    // Normalised UVs let a full-size canvas present a half-size accumulation.
    let q = uv * vec2<f32>(f32(params.width), f32(params.height)) - 0.5;
    let p = vec2<i32>(floor(q));
    let f = fract(q);
    let top = mix(linear_pixel(p), linear_pixel(p + vec2<i32>(1, 0)), f.x);
    let bottom = mix(linear_pixel(p + vec2<i32>(0, 1)), linear_pixel(p + vec2<i32>(1, 1)), f.x);
    var c = mix(top, bottom, f.y) * params.exposure;
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.gamma_on != 0u) {
        c = vec3<f32>(oetf(c.x), oetf(c.y), oetf(c.z));
    }
    return vec4<f32>(c, 1.0);
}
