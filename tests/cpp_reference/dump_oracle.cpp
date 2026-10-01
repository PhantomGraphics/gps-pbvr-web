// Dumps C++ oracle images (Phantom/PointCloud/GSView) as raw little-endian f64 RGB so the Rust
// port (crates/gps-core/tests/cpp_crosscheck.rs) can be compared against the original
// implementation on identical scenes. Build+run: tests/cpp_reference/run.ps1
#include "GaussianPointOracle.h"
#include <cstdio>
#include <string>
#include <glm/gtc/quaternion.hpp>
using namespace GSView::oracle;

static constexpr int W = 48, H = 48, SPP = 2, SETS = 20;
static const glm::dvec3 BG(0.05, 0.05, 0.08);

static OracleCamera camera() {
    OracleCamera c; c.focalX = c.focalY = 100.0;   // viewRot identity, at origin looking +Z
    return c;
}
static Gaussian3D iso(double z, double o, glm::dvec3 col, double x = 0.0) {
    Gaussian3D g; g.pos = {x, 0.0, z}; g.logScale = glm::dvec3(std::log(0.15)); g.opacity = o; g.color = col; return g;
}
static std::vector<Gaussian3D> sceneAniso() {
    Gaussian3D g; g.pos = {0.1, -0.05, 6.0};
    g.logScale = {std::log(0.28), std::log(0.10), std::log(0.06)};
    g.rot = glm::normalize(glm::dquat(0.82, 0.20, -0.45, 0.30));
    g.opacity = 0.7; g.color = {0.3, 0.5, 0.95};
    return {g};
}
static std::vector<Gaussian3D> sceneLayers() {
    return { iso(9.0, 0.8, {0.15, 0.7, 0.25}, 0.1), iso(5.0, 0.6, {0.9, 0.2, 0.2}, -0.1), iso(7.0, 0.4, {0.2, 0.3, 0.9}, 0.0) };
}
static void dump(const char* name, const Image& img) {
    std::string path = std::string(OUT_DIR) + "/cpp_" + name + ".f64";
    FILE* f = std::fopen(path.c_str(), "wb");
    for (const auto& p : img) { double v[3] = {p.x, p.y, p.z}; std::fwrite(v, sizeof(double), 3, f); }
    std::fclose(f);
    std::printf("wrote %s (%zu px)\n", path.c_str(), img.size());
}
int main() {
    dump("analytic_layers", renderAnalytic(sceneLayers(), camera(), W, H, BG));
    dump("mc_aniso", renderMonteCarlo(sceneAniso(), camera(), W, H, SPP, SETS, BG, 11u));
    dump("mc_layers", renderMonteCarlo(sceneLayers(), camera(), W, H, SPP, SETS, BG, 5u));
    MonteCarloOptions mo; mo.footprintSubpixels = 2; mo.compensateBlur = true;
    dump("mc_footprint2", renderMonteCarlo(sceneLayers(), camera(), W, H, SPP, SETS, BG, 5u, mo));
    ParticleOptions po; po.radialCorrection = true;
    dump("particles_c3r", renderParticles3D(sceneLayers(), camera(), W, H, SPP, SETS, BG, 3u, po));
    ParticleOptions p1; p1.level = GSView::gpm::CalibrationLevel::ObjectZoom; p1.rule = GSView::gpm::OpacityRule::Proportional;
    p1.linearizedProjection = true; p1.centreDepth = true;
    dump("particles_c1_lin", renderParticles3D(sceneAniso(), camera(), W, H, SPP, SETS, BG, 3u, p1));
    return 0;
}
