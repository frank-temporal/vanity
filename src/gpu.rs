use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, IntoResult, LaunchConfig};
use cuda_host::cuda_launch;
use cuda_bindings::{
    cuDeviceGetAttribute,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
};

use std::sync::Arc;

use crate::kernels::vanity::__vanity_search_CudaKernel;

fn dev_attr(device: cuda_bindings::CUdevice, attr: u32) -> i32 {
    let mut val = std::mem::MaybeUninit::uninit();
    unsafe {
        cuDeviceGetAttribute(val.as_mut_ptr(), attr, device)
            .result()
            .expect("attr query");
        val.assume_init()
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

pub struct GpuGrindCtx {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    d_base: DeviceBuffer<u8>,
    d_owner: DeviceBuffer<u8>,
    d_target: DeviceBuffer<u8>,
    d_suffix: DeviceBuffer<u8>,

    d_seed: DeviceBuffer<u8>,
    d_out: DeviceBuffer<u8>,
    d_done: DeviceBuffer<i32>,
    d_count: DeviceBuffer<u64>,

    num_blocks: u32,
    num_threads: u32,
    target_len: u64,
    suffix_len: u64,
    case_insensitive: u32,
    target_cycles: u64,
}

impl GpuGrindCtx {
    pub fn new(
        device_id: usize,
        base: &[u8; 32],
        owner: &[u8; 32],
        target: &[u8],
        suffix: &[u8],
        case_insensitive: bool,
    ) -> Self {
        let ctx = CudaContext::new(device_id).expect("CudaContext::new");
        let stream = ctx.default_stream();
        let module = cuda_host::load_kernel_module(&ctx, "vanity").expect("load_kernel_module");

        let max_tpb = dev_attr(device_id as i32, CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) as u32;
        let max_tpm = dev_attr(device_id as i32, CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) as u32;
        let mpc_count = dev_attr(device_id as i32, CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as u32;
        let block_size = max_tpm / gcd(max_tpm, max_tpb);

        let num_threads: u32 = 256;
        let num_blocks: u32 = block_size * mpc_count;

        let target_len = target.len() as u64;
        let suffix_len = suffix.len() as u64;

        let d_base   = DeviceBuffer::from_host(&stream, base).expect("d_base");
        let d_owner  = DeviceBuffer::from_host(&stream, owner).expect("d_owner");
        let d_target = if target.is_empty() {
            DeviceBuffer::<u8>::zeroed(&stream, 1).expect("d_target")
        } else {
            DeviceBuffer::from_host(&stream, target).expect("d_target")
        };
        let d_suffix = if suffix.is_empty() {
            DeviceBuffer::<u8>::zeroed(&stream, 1).expect("d_suffix")
        } else {
            DeviceBuffer::from_host(&stream, suffix).expect("d_suffix")
        };

        let d_seed  = DeviceBuffer::<u8>::zeroed(&stream, 32).expect("d_seed");
        let d_out   = DeviceBuffer::<u8>::zeroed(&stream, 16).expect("d_out");
        let d_done  = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("d_done");
        let d_count = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("d_count");

        let clock_khz = dev_attr(device_id as i32, CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE) as u64;
        let target_cycles = clock_khz as u64 * 1000 * 5;

        Self {
            _ctx: ctx,
            stream,
            module,
            d_base,
            d_owner,
            d_target,
            d_suffix,
            d_seed,
            d_out,
            d_done,
            d_count,
            num_blocks,
            num_threads,
            target_len,
            suffix_len,
            case_insensitive: if case_insensitive { 1 } else { 0 },
            target_cycles,
        }
    }

    pub fn launch(&mut self, seed: &[u8; 32]) {
        // reset per-launch state by rebuilding (only API available)
        self.d_seed  = DeviceBuffer::from_host(&self.stream, seed).expect("d_seed upload");
        self.d_out   = DeviceBuffer::<u8>::zeroed(&self.stream, 16).expect("d_out reset");
        self.d_done  = DeviceBuffer::<i32>::zeroed(&self.stream, 1).expect("d_done reset");
        self.d_count = DeviceBuffer::<u64>::zeroed(&self.stream, 1).expect("d_count reset");

        cuda_launch! {
            kernel: vanity_search,
            stream: self.stream,
            module: self.module,
            config: LaunchConfig {
                grid_dim: (self.num_blocks, 1, 1),
                block_dim: (self.num_threads, 1, 1),
                shared_mem_bytes: 0,
            },
            args: [
                self.d_seed.cu_deviceptr()   as *mut u8,
                self.d_base.cu_deviceptr()   as *mut u8,
                self.d_owner.cu_deviceptr()  as *mut u8,
                self.d_target.cu_deviceptr() as *mut u8,
                self.target_len,
                self.d_suffix.cu_deviceptr() as *mut u8,
                self.suffix_len,
                self.d_out.cu_deviceptr()    as *mut u8,
                self.case_insensitive,
                self.target_cycles,
                self.d_done.cu_deviceptr()   as *mut i32,
                self.d_count.cu_deviceptr()  as *mut u64,
            ]
        }
        .expect("kernel launch");
    }

    pub fn sync(&self) {
        self.stream.synchronize().expect("sync");
    }

    pub fn read(&self) -> ([u8; 16], u64, bool) {
        let mut seed_out = [0u8; 16];
        self.d_out.copy_to_host(&self.stream, &mut seed_out).expect("read out");

        let mut count = [0u64; 1];
        self.d_count.copy_to_host(&self.stream, &mut count).expect("read count");

        let mut done = [0i32; 1];
        self.d_done.copy_to_host(&self.stream, &mut done).expect("read done");

        (seed_out, count[0], done[0] != 0)
    }
}