//! Shared test vectors, copied verbatim from
//! `Phantom/PointCloud/GSView/GaussianPointTestVectors.h`. Values were derived
//! analytically (not from the implementation), so tests against them are real checks.

pub struct DilogCase { pub x: f64, pub li2: f64 }
pub const DILOG: &[DilogCase] = &[
    DilogCase { x: 0.0, li2: 0.0 },
    DilogCase { x: 0.5, li2: 0.582_240_526_465_012_5 },
    DilogCase { x: 1.0, li2: 1.644_934_066_848_226_4 },
    DilogCase { x: -1.0, li2: -0.822_467_033_424_113_2 },
    DilogCase { x: 0.25, li2: 0.267_652_639_082_732_5 },
    DilogCase { x: 0.9, li2: 1.299_714_723_004_958_8 },
];

pub struct SigmoidCase { pub x: f64, pub s: f64 }
pub const SIGMOID: &[SigmoidCase] = &[
    SigmoidCase { x: 0.0, s: 0.5 },
    SigmoidCase { x: 1.0, s: 0.731_058_578_630_004_9 },
    SigmoidCase { x: -2.0, s: 0.119_202_922_022_117_55 },
    SigmoidCase { x: 40.0, s: 1.0 },
];

/// E[N] = 2*pi*s2*Li2(o) for isotropic Sigma2d = s2*I.
pub struct ExpectedCountCase { pub cov2d_isotropic: f64, pub opacity: f64, pub expected_n: f64 }
pub const EXPECTED_COUNT: &[ExpectedCountCase] = &[
    ExpectedCountCase { cov2d_isotropic: 1.0, opacity: 1.0, expected_n: 10.335_425_560_099_939 },
    ExpectedCountCase { cov2d_isotropic: 4.0, opacity: 0.5, expected_n: 14.633_300_484_517_893 },
    ExpectedCountCase { cov2d_isotropic: 2.5, opacity: 0.9, expected_n: 20.415_871_127_774_36 },
];

pub struct DepthKeyCase { pub depth: f32, pub key: u32 }
pub const DEPTH_KEY: &[DepthKeyCase] = &[
    DepthKeyCase { depth: 0.0, key: 0x8000_0000 },
    DepthKeyCase { depth: 1.0, key: 0xBF80_0000 },
    DepthKeyCase { depth: 2.0, key: 0xC000_0000 },
    DepthKeyCase { depth: 100.0, key: 0xC2C8_0000 },
];
