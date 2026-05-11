use cuda_device::{kernel, launch_bounds, thread, SharedArray, debug::clock64};
use cuda_device::atomic::{DeviceAtomicI32, DeviceAtomicU64, AtomicOrdering};
use crate::{init_xorshift, xorshift128p, sha256_80, base58_chunks};

/// Maximum supported prefix length (44-char encoded case can use up to all 12;
/// 43-char case can use up to 11, since it consumes one extra leading digit).
pub const MAX_PREFIX: usize = 12;
/// Maximum supported suffix length.
pub const MAX_SUFFIX: usize = 9;

/// Test bit `dx` (0..58) of mask. Branchless.
#[inline(always)]
fn mask_has(mask: u64, dx: u8) -> bool {
    ((mask >> (dx as u32)) & 1) != 0
}

#[kernel]
#[launch_bounds(256, 2)]
pub fn vanity_search(
    seed_ptr: *const u8,             // 32 — xorshift seed
    // Host-precomputed state at round 8 of block 1 — function of base only,
    // so we run rounds 0..7 once on the CPU per launch.
    mid0: u32, mid1: u32, mid2: u32, mid3: u32,
    mid4: u32, mid5: u32, mid6: u32, mid7: u32,
    // base and owner come in as 8 BE-packed u32s each — direct kernel
    // params instead of pointers, so they end up as ld.param values that
    // are trivially loop-invariant. With *const u8 they were being
    // re-loaded byte-by-byte inside the loop because LLVM couldn't prove
    // no-alias against the per-iter atomic ops. base_w is still passed
    // because the SHA message schedule W[16..22] references W[0..7].
    base_w0: u32, base_w1: u32, base_w2: u32, base_w3: u32,
    base_w4: u32, base_w5: u32, base_w6: u32, base_w7: u32,
    owner_w0: u32, owner_w1: u32, owner_w2: u32, owner_w3: u32,
    owner_w4: u32, owner_w5: u32, owner_w6: u32, owner_w7: u32,
    // 64 u32s — host-precomputed K[i] + W[i] for block 2's compression.
    // Block 2's entire message schedule is loop-invariant per launch, so we
    // skip running it on-device. The kernel stages this into shared memory
    // once, then the SHA macro reads kw2[i] per round.
    kw2_ptr: *const u32,
    prefix_masks_ptr: *const u64,    // MAX_PREFIX × u64; mask[i] = valid numeric digits for prefix[i]
    prefix_len: u64,
    suffix_masks_ptr: *const u64,    // MAX_SUFFIX × u64; mask[i] = valid numeric digits for the (i+1)-th-from-last suffix char
    suffix_len: u64,
    out_ptr: *mut u8,                // 16 — found seed bytes
    target_cycles: u64,
    done_ptr: *mut i32,
    count_ptr: *mut u64,
) {
    let plen = prefix_len as usize;
    let slen = suffix_len as usize;

    let done  = unsafe { DeviceAtomicI32::from_ptr(done_ptr) };
    let count = unsafe { DeviceAtomicU64::from_ptr(count_ptr) };

    let idx = thread::index_1d().get() as u64;

    let mut st = init_xorshift!(seed_ptr, idx);

    // Stage prefix/suffix masks into shared memory once per block. The masks
    // are loop-invariant and small (≤ 21×8B), but reading them straight from
    // .global every iter goes through L1 — ~4-cycle hit best case, and L1
    // gets evicted by the SHA work. Shared memory broadcast is 1 cycle and
    // doesn't compete with L1.
    static mut SH_PM: SharedArray<u64, MAX_PREFIX> = SharedArray::UNINIT;
    static mut SH_SM: SharedArray<u64, MAX_SUFFIX> = SharedArray::UNINIT;
    static mut SH_KW2: SharedArray<u32, 64> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    if tid < MAX_PREFIX {
        unsafe { SH_PM[tid] = *prefix_masks_ptr.add(tid); }
    }
    if tid < MAX_SUFFIX {
        unsafe { SH_SM[tid] = *suffix_masks_ptr.add(tid); }
    }
    if tid < 64 {
        unsafe { SH_KW2[tid] = *kw2_ptr.add(tid); }
    }
    thread::sync_threads();

    macro_rules! pm {
        ($i:expr) => { unsafe { SH_PM[$i] } };
    }
    macro_rules! sm {
        ($i:expr) => { unsafe { SH_SM[$i] } };
    }

    let start_clock = clock64();
    let mut iter: u64 = 0;
    while iter < 1000*1000*1000*1000 {
        // poll for global stop every 100 iters
        if iter % 100 == 0 {
            if done.load(AtomicOrdering::Relaxed) != 0 {
                count.fetch_add(iter, AtomicOrdering::Relaxed);
                return;
            }
            if clock64() - start_clock > target_cycles {
                count.fetch_add(iter, AtomicOrdering::Relaxed);
                return;
            }
        }

        // generate 16-byte seed: 2 prng draws → 16 alphanumeric bytes,
        // each kept as a named u8 local (no array, no local-mem traffic).
        let rnd0 = xorshift128p!(&mut st);
        let rnd1 = xorshift128p!(&mut st);
        macro_rules! seed_byte {
            ($rnd:ident, $shift:expr) => {{
                let idx8 = (($rnd >> $shift) & 0xFF) as u8;
                let k = (idx8 % 62) as usize;
                // alphanumeric: 0..9 → '0'..'9', 10..35 → 'A'..'Z', 36..61 → 'a'..'z'.
                b'0' + (k as u8)
                     + ((k >= 10) as u8) * 7
                     + ((k >= 36) as u8) * 6
            }};
        }
        let sb00 = seed_byte!(rnd0,  0);
        let sb01 = seed_byte!(rnd0,  8);
        let sb02 = seed_byte!(rnd0, 16);
        let sb03 = seed_byte!(rnd0, 24);
        let sb04 = seed_byte!(rnd0, 32);
        let sb05 = seed_byte!(rnd0, 40);
        let sb06 = seed_byte!(rnd0, 48);
        let sb07 = seed_byte!(rnd0, 56);
        let sb08 = seed_byte!(rnd1,  0);
        let sb09 = seed_byte!(rnd1,  8);
        let sb10 = seed_byte!(rnd1, 16);
        let sb11 = seed_byte!(rnd1, 24);
        let sb12 = seed_byte!(rnd1, 32);
        let sb13 = seed_byte!(rnd1, 40);
        let sb14 = seed_byte!(rnd1, 48);
        let sb15 = seed_byte!(rnd1, 56);

        // pack seed bytes into 4 BE u32s = the SHA message words ma08..ma11
        let s0 = (sb00 as u32) << 24 | (sb01 as u32) << 16 | (sb02 as u32) << 8 | (sb03 as u32);
        let s1 = (sb04 as u32) << 24 | (sb05 as u32) << 16 | (sb06 as u32) << 8 | (sb07 as u32);
        let s2 = (sb08 as u32) << 24 | (sb09 as u32) << 16 | (sb10 as u32) << 8 | (sb11 as u32);
        let s3 = (sb12 as u32) << 24 | (sb13 as u32) << 16 | (sb14 as u32) << 8 | (sb15 as u32);

        // sha256(base || seed || owner) → 8 u32s in registers (no [u8;32] buffer)
        // - rounds 0..7 of block 1: skipped via host-precomputed midstate
        // - block 2 message schedule: skipped via host-precomputed kw2 in shared mem
        sha256_80!(
            mid0, mid1, mid2, mid3, mid4, mid5, mid6, mid7,
            base_w0, base_w1, base_w2, base_w3, base_w4, base_w5, base_w6, base_w7,
            s0, s1, s2, s3,
            owner_w0, owner_w1, owner_w2, owner_w3, owner_w4, owner_w5, owner_w6, owner_w7,
            SH_KW2,
            h0, h1, h2, h3, h4, h5, h6, h7);

        // base58 → 9 carry-propagated chunks in registers (no [u8;45] buffer)
        base58_chunks!(h0, h1, h2, h3, h4, h5, h6, h7,
                       c7_0, c7_1, c6_2, c5_3, c4_4, c3_5, c2_6, c1_7, c0_8);
        // silence unused-variable warnings for the mid chunks we don't decompose
        let _ = (c5_3, c4_4, c3_5, c2_6);

        // Decompose only the digits the prefix/suffix can touch.
        // Top window (digits 0..14): c7_0 → 0..5, c7_1 → 5..10, c6_2 → 10..15.
        // Digit 0 is always 0 for a 32-byte input; skip it.
        let d01 = ((c7_0 / 195112) % 58) as u8;
        let d02 = ((c7_0 / 3364) % 58) as u8;
        let d03 = ((c7_0 / 58) % 58) as u8;
        let d04 = (c7_0 % 58) as u8;
        let d05 = (c7_1 / 11316496) as u8;
        let d06 = ((c7_1 / 195112) % 58) as u8;
        let d07 = ((c7_1 / 3364) % 58) as u8;
        let d08 = ((c7_1 / 58) % 58) as u8;
        let d09 = (c7_1 % 58) as u8;
        let d10 = (c6_2 / 11316496) as u8;
        let d11 = ((c6_2 / 195112) % 58) as u8;
        let d12 = ((c6_2 / 3364) % 58) as u8;
        let d13 = ((c6_2 / 58) % 58) as u8;
        // d14 not needed (43-char case prefix max is 12, top digit at index 13)

        // Bottom window (digits 36..45): c1_7 → 35..40, c0_8 → 40..45.
        let d36 = ((c1_7 / 195112) % 58) as u8;
        let d37 = ((c1_7 / 3364) % 58) as u8;
        let d38 = ((c1_7 / 58) % 58) as u8;
        let d39 = (c1_7 % 58) as u8;
        let d40 = (c0_8 / 11316496) as u8;
        let d41 = ((c0_8 / 195112) % 58) as u8;
        let d42 = ((c0_8 / 3364) % 58) as u8;
        let d43 = ((c0_8 / 58) % 58) as u8;
        let d44 = (c0_8 % 58) as u8;

        // 44-char encoded case detection: digit 1 != 0 (occurs ~84% of the time).
        // 43-char case: digit 1 == 0 (digits shifted by 1 to the right).
        let is_44 = d01 != 0;

        // Prefix match: in the 44-char case, address[i] == raw_p[1+i]; in the
        // 43-char case, address[i] == raw_p[2+i]. We OR the two chains so we
        // don't lose the 16% of 43-char hashes. Masks are lazily loaded from
        // global memory (L1-cached) — see comment at top of fn.
        // Load each prefix mask exactly once per iter and use it twice (once for
        // the 44-char layout, once for 43-char). Without this, the compiler can't
        // CSE the two pm!(i) calls and we get 24 global loads instead of 12.
        // The masks live as short-lived locals (one mask_has check, then dead).
        macro_rules! check_prefix_both {
            ( $( ($mask_idx:expr, $d44:ident, $d43:ident, $i:expr) ),+ $(,)? ) => {{
                let mut ok44 = true;
                let mut ok43 = true;
                $(
                    if plen > $i {
                        let m = pm!($mask_idx);
                        if !mask_has(m, $d44) { ok44 = false; }
                        if !mask_has(m, $d43) { ok43 = false; }
                    }
                )+
                (ok44, ok43)
            }};
        }
        let (p_match_44, p_match_43) = check_prefix_both!(
            (0,  d01, d02,  0), (1,  d02, d03,  1), (2,  d03, d04,  2),
            (3,  d04, d05,  3), (4,  d05, d06,  4), (5,  d06, d07,  5),
            (6,  d07, d08,  6), (7,  d08, d09,  7), (8,  d09, d10,  8),
            (9,  d10, d11,  9), (10, d11, d12, 10), (11, d12, d13, 11),
        );
        let p_match = if is_44 { p_match_44 } else { p_match_43 };

        // Suffix is anchored at the end → same digit indices regardless of
        // encoded length. sm[i] is the (i+1)-th-from-last digit.
        let s_match = {
            let mut ok = true;
            macro_rules! check_suffix {
                ( $( ($mask_idx:expr, $digit:ident, $i:expr) ),+ $(,)? ) => {{
                    $(
                        if slen > $i {
                            if !mask_has(sm!($mask_idx), $digit) { ok = false; }
                        }
                    )+
                }};
            }
            check_suffix!(
                (0, d44, 0), (1, d43, 1), (2, d42, 2),
                (3, d41, 3), (4, d40, 4), (5, d39, 5),
                (6, d38, 6), (7, d37, 7), (8, d36, 8),
            );
            ok
        };

        if p_match & s_match {
            if done.compare_exchange(0, 1, AtomicOrdering::AcqRel, AtomicOrdering::Relaxed).is_ok() {
                // 16 explicit constant-index writes — keeps the seed bytes in
                // registers up through the unique match path.
                unsafe {
                    *out_ptr.add(0)  = sb00; *out_ptr.add(1)  = sb01;
                    *out_ptr.add(2)  = sb02; *out_ptr.add(3)  = sb03;
                    *out_ptr.add(4)  = sb04; *out_ptr.add(5)  = sb05;
                    *out_ptr.add(6)  = sb06; *out_ptr.add(7)  = sb07;
                    *out_ptr.add(8)  = sb08; *out_ptr.add(9)  = sb09;
                    *out_ptr.add(10) = sb10; *out_ptr.add(11) = sb11;
                    *out_ptr.add(12) = sb12; *out_ptr.add(13) = sb13;
                    *out_ptr.add(14) = sb14; *out_ptr.add(15) = sb15;
                }
            }
            count.fetch_add(iter + 1, AtomicOrdering::Relaxed);
            return;
        }

        iter += 1;
    }

    count.fetch_add(iter, AtomicOrdering::Relaxed);
}
