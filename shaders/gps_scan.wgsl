// Hierarchical exclusive scan (workgroup barriers are NOT cross-workgroup syncs:
// every level is a separate dispatch).
struct ScanParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

var<workgroup> tmp: array<u32, 256>;

// Exclusive scan of one 256-element block, written in place; block total -> sums[group].
@compute @workgroup_size(256)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    var v = 0u;
    if (i < sp.n) { v = data[i]; }
    tmp[lid.x] = v;
    workgroupBarrier();
    // Hillis-Steele inclusive scan
    for (var off = 1u; off < 256u; off = off << 1u) {
        var add = 0u;
        if (lid.x >= off) { add = tmp[lid.x - off]; }
        workgroupBarrier();
        tmp[lid.x] = tmp[lid.x] + add;
        workgroupBarrier();
    }
    if (i < sp.n) { data[i] = tmp[lid.x] - v; }
    if (lid.x == 255u) { sums[wid.x] = tmp[255]; }
}

// data[i] += sums[group]   (sums already exclusive-scanned)
@compute @workgroup_size(256)
fn add_offsets(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    if (gid.x < sp.n) { data[gid.x] = data[gid.x] + sums[wid.x]; }
}
