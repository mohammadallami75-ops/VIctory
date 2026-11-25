//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

// The title's shadow buffers stay UAVs, including when arguments and count
// are in the same buffer. Only the packed destination is written here.
const DISPATCH: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Dispatch : register(b0) {
    uint stride; uint records; uint waves; uint has_count; uint first; uint phase;
};
RWByteAddressBuffer source : register(u0);
RWByteAddressBuffer count : register(u1);
RWByteAddressBuffer args : register(u2);
RWByteAddressBuffer budget : register(u3);
static const uint BUDGET = 1u << 21; // in groups of 64 threads
static const uint GROUPS = 1u << 20;

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // The legacy clamp charges the first word in threads, including draws
    // whose charges need not be multiples of 64. Keep it intact and use a
    // second word in groups so even thousands of pending charges fit.
    if (phase != 1) {
        if (id.x != 0) return;
        uint threads = budget.Load(0);
        uint initial = (min(threads, BUDGET * 64) + 63) / 64;
        if (phase == 0) {
            budget.Store(4, initial);
        } else if (threads < BUDGET * 64) {
            budget.Store(0, threads + (budget.Load(4) - initial) * 64);
        }
        return;
    }
    uint record = first + id.x;
    if (record >= records) return;
    uint active = has_count == 0 ? records : min(records, count.Load(0));
    if (record == 0) args.Store(records * 12, active);
    uint3 g = 0;
    if (record < active) {
        g = min(source.Load3(record * stride), uint3(65535, 65535, 65535));
        g.x = g.x <= 65535u / waves ? g.x * waves : 0;
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = 0;
        }
        uint cost = g.x * g.y * g.z;
        if (cost != 0) {
            uint used;
            budget.InterlockedAdd(4, cost, used);
            // Pending refunds can cause conservative refusals. The counter
            // includes all accepted work and cannot wrap within a batch.
            if (used + cost > BUDGET) {
                budget.InterlockedAdd(4, 0u - cost);
                g = 0;
            }
        }
    }
    args.Store3(record * 12, g);
}
"#;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// Bounds and scales dispatches into a list-owned buffer of packed records,
/// followed by one word containing the effective count.
/// The source and count remain usable by later GPU writers.
pub(crate) fn dispatch(
    list: &HostList,
    source: u64,
    count: Option<u64>,
    (stride, records, waves): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let groups = records.div_ceil(64);
    if groups == 0
        || groups > 65535
        || waves < 2
        || !source.is_multiple_of(4)
        || count.is_some_and(|at| !at.is_multiple_of(4))
        || stride < 12
        || !stride.is_multiple_of(4)
        || u64::from(records - 1) * u64::from(stride) + 11 > u64::from(u32::MAX)
    {
        return None;
    }
    static REWRITE: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    let rewrite = REWRITE
        .get_or_init(|| {
            make(DISPATCH)
                .inspect_err(|e| tracing::warn!(error = %e, "no dispatch argument rewrite"))
                .ok()
        })
        .as_ref()?;
    let budget = list.budget()?;
    let output = list.dispatch_args(u64::from(records) * 12 + 4)?;
    let mut constants = [stride, records, waves, u32::from(count.is_some()), 0, 0];
    // At most 2^21 groups carried in and 2^20 per record, even if every
    // refusal is still awaiting its refund. Each batch stays below 2^32.
    const BATCH_GROUPS: u32 = 63;
    // SAFETY: the caller kept the source and count alive and checked their
    // extents. The scratch buffer has one full record per thread.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&rewrite.root);
        l.SetPipelineState(&rewrite.pso);
        l.SetComputeRootUnorderedAccessView(1, source);
        l.SetComputeRootUnorderedAccessView(2, count.unwrap_or(source));
        l.SetComputeRootUnorderedAccessView(3, output.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, budget);
        let launch = |constants: &[u32; 6], groups| {
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.Dispatch(groups, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);
        };
        launch(&constants, 1);
        constants[5] = 1;
        for group in (0..groups).step_by(BATCH_GROUPS as usize) {
            constants[4] = group * 64;
            launch(&constants, (groups - group).min(BATCH_GROUPS));
        }
        constants[5] = 2;
        launch(&constants, 1);
        l.ResourceBarrier(&[transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    Some(output)
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
#[path = "indirect_tests.rs"]
mod regressions;

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}
