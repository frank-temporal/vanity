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

/// Pack 32 BE bytes into 8 u32 words (b0 holds bytes 0..4, MSB first).
fn pack_be_words(bytes: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = (bytes[4*i+0] as u32) << 24
               | (bytes[4*i+1] as u32) << 16
               | (bytes[4*i+2] as u32) << 8
               | (bytes[4*i+3] as u32);
    }
    out
}

/// SHA-256 K constants (rounds 0..63).
const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
/// SHA-256 IV.
const SHA_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Compute K[i] + W[i] for all 64 rounds of block 2. Block 2's message schedule
/// is fully loop-invariant per launch (W[0..3] = owner_w[4..7], W[4..15] are
/// fixed padding constants), so the entire schedule + K-add reduces to a 256-byte
/// table the kernel reads once into shared memory. Avoids ~250 ops per iter
/// (the 48 W[16..63] sig0/sig1 evaluations + 64 K-adds).
fn sha256_block2_kw(owner_w: &[u32; 8]) -> [u32; 64] {
    use crate::kernels::sha256::{sig0, sig1};

    // Block 2 layout: owner[16..32] || 0x80 || zeros... || bitlen_be(640)
    let mut w = [0u32; 64];
    w[0] = owner_w[4]; w[1] = owner_w[5]; w[2] = owner_w[6]; w[3] = owner_w[7];
    w[4] = 0x80000000;
    // w[5..14] = 0
    w[15] = 0x00000280; // bit length (640)
    for i in 16..64 {
        w[i] = sig1(w[i-2]).wrapping_add(w[i-7]).wrapping_add(sig0(w[i-15])).wrapping_add(w[i-16]);
    }
    let mut kw = [0u32; 64];
    for i in 0..64 {
        kw[i] = SHA_K[i].wrapping_add(w[i]);
    }
    kw
}

/// Run rounds 0..7 of SHA-256's compression with W[0..7] = base words, starting
/// from the IV. Returns the (a,b,c,d,e,f,g,h) state at the start of round 8.
/// The kernel's `sha256_80!` macro receives this and skips rounds 0..7.
fn sha256_midstate_after_round_7(base_w: &[u32; 8]) -> [u32; 8] {
    use crate::kernels::sha256::{ep0, ep1, ch, maj};

    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) =
        (SHA_IV[0], SHA_IV[1], SHA_IV[2], SHA_IV[3],
         SHA_IV[4], SHA_IV[5], SHA_IV[6], SHA_IV[7]);
    for i in 0..8 {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g))
                  .wrapping_add(SHA_K[i]).wrapping_add(base_w[i]);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    [a, b, c, d, e, f, g, h]
}

pub struct GpuGrindCtx {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    midstate: [u32; 8],
    base_w: [u32; 8],
    owner_w: [u32; 8],
    d_kw2: DeviceBuffer<u32>,
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

        let base_w = pack_be_words(base);
        let owner_w = pack_be_words(owner);
        let midstate = sha256_midstate_after_round_7(&base_w);
        let kw2 = sha256_block2_kw(&owner_w);
        let d_kw2 = DeviceBuffer::from_host(&stream, &kw2).expect("d_kw2");
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
            midstate,
            base_w,
            owner_w,
            d_kw2,
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
                self.midstate[0], self.midstate[1], self.midstate[2], self.midstate[3],
                self.midstate[4], self.midstate[5], self.midstate[6], self.midstate[7],
                self.base_w[0], self.base_w[1], self.base_w[2], self.base_w[3],
                self.base_w[4], self.base_w[5], self.base_w[6], self.base_w[7],
                self.owner_w[0], self.owner_w[1], self.owner_w[2], self.owner_w[3],
                self.owner_w[4], self.owner_w[5], self.owner_w[6], self.owner_w[7],
                self.d_kw2.cu_deviceptr()          as *mut u32,
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
