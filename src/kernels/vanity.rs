use cuda_device::{kernel, thread, debug::clock64};
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
pub fn vanity_search(
    seed_ptr: *const u8,             // 32 — xorshift seed
    base_ptr: *const u8,             // 32
    owner_ptr: *const u8,            // 32
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

    let base_arr  = unsafe { &*(base_ptr  as *const [u8; 32]) };
    let owner_arr = unsafe { &*(owner_ptr as *const [u8; 32]) };

    // Hoist masks into registers — loop-invariant, read once.
    let pm0  = unsafe { *prefix_masks_ptr.add(0) };
    let pm1  = unsafe { *prefix_masks_ptr.add(1) };
    let pm2  = unsafe { *prefix_masks_ptr.add(2) };
    let pm3  = unsafe { *prefix_masks_ptr.add(3) };
    let pm4  = unsafe { *prefix_masks_ptr.add(4) };
    let pm5  = unsafe { *prefix_masks_ptr.add(5) };
    let pm6  = unsafe { *prefix_masks_ptr.add(6) };
    let pm7  = unsafe { *prefix_masks_ptr.add(7) };
    let pm8  = unsafe { *prefix_masks_ptr.add(8) };
    let pm9  = unsafe { *prefix_masks_ptr.add(9) };
    let pm10 = unsafe { *prefix_masks_ptr.add(10) };
    let pm11 = unsafe { *prefix_masks_ptr.add(11) };

    let sm0 = unsafe { *suffix_masks_ptr.add(0) };
    let sm1 = unsafe { *suffix_masks_ptr.add(1) };
    let sm2 = unsafe { *suffix_masks_ptr.add(2) };
    let sm3 = unsafe { *suffix_masks_ptr.add(3) };
    let sm4 = unsafe { *suffix_masks_ptr.add(4) };
    let sm5 = unsafe { *suffix_masks_ptr.add(5) };
    let sm6 = unsafe { *suffix_masks_ptr.add(6) };
    let sm7 = unsafe { *suffix_masks_ptr.add(7) };
    let sm8 = unsafe { *suffix_masks_ptr.add(8) };

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
        sha256_80!(base_arr, s0, s1, s2, s3, owner_arr,
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
        // don't lose the 16% of 43-char hashes.
        macro_rules! check_prefix_at {
            // `$( ($mask:ident, $digit:ident) ),+` — checked in order, gated on plen.
            ( $( ($mask:ident, $digit:ident) ),+ $(,)? ) => {{
                let mut ok = true;
                let mut i = 0usize;
                $(
                    if plen > i {
                        if !mask_has($mask, $digit) { ok = false; }
                    }
                    let _ = i; i += 1;
                )+
                ok
            }};
        }
        let p_match_44 = check_prefix_at!(
            (pm0,  d01), (pm1,  d02), (pm2,  d03), (pm3,  d04),
            (pm4,  d05), (pm5,  d06), (pm6,  d07), (pm7,  d08),
            (pm8,  d09), (pm9,  d10), (pm10, d11), (pm11, d12),
        );
        let p_match_43 = check_prefix_at!(
            (pm0,  d02), (pm1,  d03), (pm2,  d04), (pm3,  d05),
            (pm4,  d06), (pm5,  d07), (pm6,  d08), (pm7,  d09),
            (pm8,  d10), (pm9,  d11), (pm10, d12), (pm11, d13),
        );
        let p_match = if is_44 { p_match_44 } else { p_match_43 };

        // Suffix is anchored at the end → same digit indices regardless of
        // encoded length. sm[i] corresponds to the i-th-from-last digit.
        let s_match = {
            let mut ok = true;
            macro_rules! check_suffix {
                ( $( ($mask:ident, $digit:ident, $i:expr) ),+ $(,)? ) => {{
                    $(
                        if slen > $i {
                            if !mask_has($mask, $digit) { ok = false; }
                        }
                    )+
                }};
            }
            check_suffix!(
                (sm0, d44, 0), (sm1, d43, 1), (sm2, d42, 2),
                (sm3, d41, 3), (sm4, d40, 4), (sm5, d39, 5),
                (sm6, d38, 6), (sm7, d37, 7), (sm8, d36, 8),
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
