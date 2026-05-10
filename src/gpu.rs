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

use crate::kernels::vanity::{__vanity_search_CudaKernel, MAX_PREFIX, MAX_SUFFIX};

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

/// Numeric base58 alphabets in **digit order** (index 0..57). The kernel
/// matches in this numeric space, so the host has to convert prefix/suffix
/// chars to a bitmask of valid digit values.
///
/// Both alphabets mirror `kernels::base58::alphabet_at`. The CI alphabet has
/// duplicate entries so a lowercased prefix char can match the digit slot
/// originally assigned to either case (e.g. 'a' is at indices 9 and 32).
const ALPHABET_NO_CI: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
// alphabet_at(_, true) — entries 0..57 in order. Index 57 falls back to 'z' too.
const ALPHABET_CI: &[u8; 58] = b"123456789abcdefghjLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyzz";

/// For one prefix/suffix character, return a u64 mask whose bit i is set
/// iff numeric base58 digit i would print as that character (under the chosen
/// case-sensitivity mode).
fn char_to_digit_mask(c: u8, case_insensitive: bool) -> u64 {
    let alphabet: &[u8; 58] = if case_insensitive { ALPHABET_CI } else { ALPHABET_NO_CI };
    let mut mask: u64 = 0;
    for (i, &b) in alphabet.iter().enumerate() {
        if b == c {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// Build the prefix mask buffer (MAX_PREFIX × u64). Slot i is the mask for
/// `prefix[i]`; trailing slots stay 0 (unused — gated on prefix_len in the kernel).
fn build_prefix_masks(prefix: &[u8], ci: bool) -> [u64; MAX_PREFIX] {
    assert!(prefix.len() <= MAX_PREFIX, "prefix too long: max {MAX_PREFIX}");
    let mut out = [0u64; MAX_PREFIX];
    for (i, &c) in prefix.iter().enumerate() {
        let m = char_to_digit_mask(c, ci);
        assert!(m != 0, "prefix char {:?} not in base58 alphabet", c as char);
        out[i] = m;
    }
    out
}

/// Build the suffix mask buffer in **reversed** order (slot i = (i+1)-th-from-last
/// character of the suffix). Matches the kernel's per-iter suffix check, which
/// reads sm[0] for the last digit, sm[1] for the second-to-last, etc.
fn build_suffix_masks(suffix: &[u8], ci: bool) -> [u64; MAX_SUFFIX] {
    assert!(suffix.len() <= MAX_SUFFIX, "suffix too long: max {MAX_SUFFIX}");
    let mut out = [0u64; MAX_SUFFIX];
    for (i, &c) in suffix.iter().rev().enumerate() {
        let m = char_to_digit_mask(c, ci);
        assert!(m != 0, "suffix char {:?} not in base58 alphabet", c as char);
        out[i] = m;
    }
    out
}

pub struct GpuGrindCtx {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    d_base: DeviceBuffer<u8>,
    d_owner: DeviceBuffer<u8>,
    d_prefix_masks: DeviceBuffer<u64>,
    d_suffix_masks: DeviceBuffer<u64>,

    d_seed: DeviceBuffer<u8>,
    d_out: DeviceBuffer<u8>,
    d_done: DeviceBuffer<i32>,
    d_count: DeviceBuffer<u64>,

    num_blocks: u32,
    num_threads: u32,
    prefix_len: u64,
    suffix_len: u64,
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

        let prefix_len = target.len() as u64;
        let suffix_len = suffix.len() as u64;

        let prefix_masks = build_prefix_masks(target, case_insensitive);
        let suffix_masks = build_suffix_masks(suffix, case_insensitive);

        let d_base   = DeviceBuffer::from_host(&stream, base).expect("d_base");
        let d_owner  = DeviceBuffer::from_host(&stream, owner).expect("d_owner");
        let d_prefix_masks = DeviceBuffer::from_host(&stream, &prefix_masks).expect("d_prefix_masks");
        let d_suffix_masks = DeviceBuffer::from_host(&stream, &suffix_masks).expect("d_suffix_masks");

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
            d_prefix_masks,
            d_suffix_masks,
            d_seed,
            d_out,
            d_done,
            d_count,
            num_blocks,
            num_threads,
            prefix_len,
            suffix_len,
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
                self.d_seed.cu_deviceptr()         as *mut u8,
                self.d_base.cu_deviceptr()         as *mut u8,
                self.d_owner.cu_deviceptr()        as *mut u8,
                self.d_prefix_masks.cu_deviceptr() as *mut u64,
                self.prefix_len,
                self.d_suffix_masks.cu_deviceptr() as *mut u64,
                self.suffix_len,
                self.d_out.cu_deviceptr()          as *mut u8,
                self.target_cycles,
                self.d_done.cu_deviceptr()         as *mut i32,
                self.d_count.cu_deviceptr()        as *mut u64,
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
