//! Synthetic GFX10 mesh fixture and an independent HLSL raster reference.

use super::container::SigElement;
use super::host::{Draw, Host, adapters};
use super::{HostShader, Target};
use crate::dxbc::{Container, FourCC, Part};
use crate::native::{MeshIo, NativeProgram, StageIo};
use crate::{HwStage, ShaderError, io, ir, resource, structure};

const WAVE32: Target = Target {
    lanes_min: 32,
    lanes_max: 32,
};

fn mesh(threads: u32, cull: bool) -> NativeProgram {
    let words: &[u32] = &[
        0xD765_0000,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD766_0000,
        0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // Only the lower half has EXEC bits here. Read an inactive lane in
        // the upper half, and fetch that half's earlier LDS write.
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE88_0380, // s_mov_b32 s8, 0
        0xBE87_03BC, // s_mov_b32 s7, 60
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E0C_0300, // v_mov_b32 v6, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0007, // s_cbranch_execz +7
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0xD761_0006,
        0x0001_5007, // v_writelane_b32 v6, s7, 40
        0xD8D8_0080,
        0x0300_0004, // ds_read_b32 v3, v4 offset:128
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        // Only the upper half has EXEC bits here. Both halves must see
        // the same scalar result of v_readfirstlane.
        0x7D82_00A7, // v_cmp_lt_u32 vcc, 39, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0001, // s_cbranch_execz +1
        0x7E10_0506, // v_readfirstlane_b32 s8, v6
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A06_0609, // v_add_nc_u32 v3, s9, v3
        0x4A06_0608, // v_add_nc_u32 v3, s8, v3
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1006_06FF,
        0x3B80_0000, // v_mul_f32 v3, 1/256, v3
        0x807C_FF09,
        0x0FDB,      // s_add_u32 m0, s9, ((1 << 12) | 3) - 40
        0xBF90_0009, // s_sendmsg GS_ALLOC_REQ
        0x7E02_02FF,
        0xBF40_0000, // v_mov_b32 v1, -0.75
        0x7E04_02FF,
        0xBF40_0000, // v_mov_b32 v2, -0.75
        0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0x7E02_02FF,
        0x3F40_0000, // v_mov_b32 v1, 0.75
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7D84_0082, // v_cmp_eq_u32 vcc, 2, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0x7E04_02FF,
        0x3F40_0000, // v_mov_b32 v2, 0.75
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7E08_02F2, // v_mov_b32 v4, 1.0
        0x7E0A_02FF,
        0x0020_0400 | (u32::from(cull) << 31), // v_mov_b32 v5, packed indices and cull
        0xF800_00CF,
        0x0403_0201, // exp pos0, v1, v2, v3, v4
        0xF800_0941,
        0x0000_0005, // exp prim, v5, off, off, off done
        0xBF81_0000, // s_endpgm
    ];
    NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads,
            max_vertices: 4,
            max_primitives: 2,
            topology: 2,
            lds_bytes: 256,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn emit_mesh(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let signature = super::container::signature(
        &[SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }],
        false,
    );
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![Part {
            fourcc: FourCC::OSG1,
            offset: 0,
            data: &signature,
        }],
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    super::emit(&c, native, &p, &st, &io, &res, WAVE32)
}

#[test]
fn mesh_wave64_refuses_a_group_of_two_console_waves() {
    assert!(matches!(
        emit_mesh(&mesh(128, false)),
        Err(ShaderError::Unsupported {
            what: "wave64 mesh emulation needs one console wave per group",
            ..
        })
    ));
}

#[test]
fn mesh_wave64_matches_reference_geometry_and_depth() {
    let translate = |cull| {
        let mut shader = emit_mesh(&mesh(64, cull)).expect("translate synthetic Wave64 mesh");
        if let Some(path) = super::validator::Validator::default_path() {
            let mut validator =
                super::validator::Validator::load(Some(&path)).expect("load DXIL validator");
            validator
                .validate(&mut shader.dxil)
                .expect("validate synthetic Wave64 mesh DXIL");
        }
        shader
    };
    let translated = translate(false);
    let culled = translate(true);
    let Some(dxc) = std::env::var_os("VICTORY_PC_DXC") else {
        eprintln!("no VICTORY_PC_DXC; skipped GPU comparison");
        return;
    };
    let Some(host) = adapters()
        .unwrap_or_default()
        .iter()
        .filter_map(|name| Host::on(name).ok())
        .find(|host| host.target() == WAVE32)
    else {
        eprintln!("no Wave32-only SM 6.6 hardware; skipped GPU comparison");
        return;
    };
    let dir = std::env::temp_dir().join(format!("victory-mesh-wave64-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let result = std::panic::catch_unwind(|| {
        let source = dir.join("reference.hlsl");
        std::fs::write(&source, REFERENCE).unwrap();
        let compile = |entry: &str, profile: &str| {
            let path = dir.join(format!("{entry}.dxil"));
            let output = std::process::Command::new(&dxc)
                .args(["-T", profile, "-E", entry, "-Fo"])
                .arg(&path)
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            std::fs::read(path).unwrap()
        };
        let reference = compile("ms", "ms_6_6");
        let pixel = compile("ps", "ps_6_6");
        let root = host.root_signature_for(&[]).unwrap();
        let draw = |mesh: &[u8]| {
            let pso = host.mesh_pipeline_with(&root, mesh, &pixel).unwrap();
            host.draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![],
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (32, 32),
                    mesh_groups: 1,
                },
            )
            .unwrap()
        };
        let expected = draw(&reference);
        let actual = draw(&translated.dxil);
        let covered = actual.iter().filter(|p| p[3] == 1.0).count();
        assert!(covered > 200, "covered pixels: {covered}");
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a[3], b[3], "coverage at pixel {i}");
            assert!(
                (a[0] - b[0]).abs() < 1e-6,
                "depth at pixel {i}: {a:?}, {b:?}"
            );
        }
        assert!(draw(&culled.dxil).iter().all(|pixel| *pixel == [0.0; 4]));
        eprintln!(
            "Wave32 mesh matched all 1024 coverage pixels and {covered} interpolated depth samples. The culled primitive covered zero pixels."
        );
    });
    std::fs::remove_dir_all(dir).unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

const REFERENCE: &str = r#"
struct Vertex { float4 position : SV_Position; };
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void ms(uint lane : SV_GroupIndex, out vertices Vertex v[4], out indices uint3 p[2]) {
    SetMeshOutputCounts(3, 1);
    if (lane < 3) {
        v[lane].position = float4(lane == 1 ? .75 : -.75, lane == 2 ? .75 : -.75, (132.0 + lane) / 256.0, 1);
    }
    if (lane == 0) p[0] = uint3(0, 1, 2);
}
float4 ps(float4 position : SV_Position) : SV_Target { return float4(position.z, 0, 0, 1); }
"#;
