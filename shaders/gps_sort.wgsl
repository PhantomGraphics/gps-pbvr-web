// LSD radix sort of (key, value) pairs, 8 bits per pass, keys < 2^24 (3 passes). Stable.
// hist is digit-major: hist[digit * nblocks + block]; an exclusive scan over it (gps_scan.wgsl) gives every
// (digit, block) its global start. A block is 256 consecutive items = one workgroup.
struct SortParams { n: u32, shift: u32, nblocks: u32, pad: u32 };
@group(0) @binding(0) var<uniform> sp: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> hist: array<u32>;

var<workgroup> loc_hist: array<atomic<u32>, 256>;
var<workgroup> digs: array<u32, 256>;

@compute @workgroup_size(256)
fn rs_hist(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    atomicStore(&loc_hist[lid.x], 0u);
    workgroupBarrier();
    let i = wid.x * 256u + lid.x;
    if (i < sp.n) {
        atomicAdd(&loc_hist[(keys_in[i] >> sp.shift) & 255u], 1u);
    }
    workgroupBarrier();
    hist[lid.x * sp.nblocks + wid.x] = atomicLoad(&loc_hist[lid.x]);
}

@compute @workgroup_size(256)
fn rs_scatter(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    let valid = i < sp.n;
    var d = 256u;
    var key = 0u;
    var val = 0u;
    if (valid) {
        key = keys_in[i];
        val = vals_in[i];
        d = (key >> sp.shift) & 255u;
    }
    digs[lid.x] = d;
    workgroupBarrier();
    if (valid) {
        var rank = 0u;
        for (var j = 0u; j < lid.x; j = j + 1u) {
            if (digs[j] == d) { rank = rank + 1u; }
        }
        let pos = hist[d * sp.nblocks + wid.x] + rank;
        keys_out[pos] = key;
        vals_out[pos] = val;
    }
}
