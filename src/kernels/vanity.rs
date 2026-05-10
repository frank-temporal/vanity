use cuda_device::{kernel, thread, device, debug::clock64};
use cuda_device::atomic::{DeviceAtomicI32, DeviceAtomicU64, AtomicOrdering};
use super::sha256::sha256_80;
use super::base58::base58_encode_32;
use super::xorshift::{init_xorshift, xorshift128p};

#[device]
fn alphanumeric_at(i: usize) -> u8 {
    match i {
         0 => b'0',  1 => b'1',  2 => b'2',  3 => b'3',  4 => b'4',
         5 => b'5',  6 => b'6',  7 => b'7',  8 => b'8',  9 => b'9',
        10 => b'A', 11 => b'B', 12 => b'C', 13 => b'D', 14 => b'E',
        15 => b'F', 16 => b'G', 17 => b'H', 18 => b'I', 19 => b'J',
        20 => b'K', 21 => b'L', 22 => b'M', 23 => b'N', 24 => b'O',
        25 => b'P', 26 => b'Q', 27 => b'R', 28 => b'S', 29 => b'T',
        30 => b'U', 31 => b'V', 32 => b'W', 33 => b'X', 34 => b'Y',
        35 => b'Z',
        36 => b'a', 37 => b'b', 38 => b'c', 39 => b'd', 40 => b'e',
        41 => b'f', 42 => b'g', 43 => b'h', 44 => b'i', 45 => b'j',
        46 => b'k', 47 => b'l', 48 => b'm', 49 => b'n', 50 => b'o',
        51 => b'p', 52 => b'q', 53 => b'r', 54 => b's', 55 => b't',
        56 => b'u', 57 => b'v', 58 => b'w', 59 => b'x', 60 => b'y',
        61 => b'z',
        _ => 0,
    }
}

#[kernel]
pub fn vanity_search(
    seed_ptr: *const u8,        // 32
    base_ptr: *const u8,        // 32
    owner_ptr: *const u8,       // 32
    target_ptr: *const u8,      // target_len bytes
    target_len: u64,
    suffix_ptr: *const u8,      // suffix_len bytes
    suffix_len: u64,
    out_ptr: *mut u8,           // 16
    case_insensitive: u32,
    target_cycles: u64,
    done_ptr: *mut i32,
    count_ptr: *mut u64,
) {
    let target_len_u = target_len as usize;
    let suffix_len_u = suffix_len as usize;
    let ci = case_insensitive != 0;

    let done  = unsafe { DeviceAtomicI32::from_ptr(done_ptr) };
    let count = unsafe { DeviceAtomicU64::from_ptr(count_ptr) };

    let idx = thread::index_1d().get() as u64;

    let mut st = init_xorshift(seed_ptr, idx);

    let base_arr  = unsafe { &*(base_ptr  as *const [u8; 32]) };
    let owner_arr = unsafe { &*(owner_ptr as *const [u8; 32]) };

    let mut create_account_seed = [0u8; 16];
    let mut local_hash = [0u8; 32];
    let mut local_encoded = [0u8; 44];
    let mut encoded_len: u32 = 0;

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

        // generate 16-byte seed via two 64-bit prng draws
        let csp = create_account_seed.as_mut_ptr();
        for i in 0..2 {
            let rnd = xorshift128p(&mut st);
            for b in 0..8 {
                let idx8 = ((rnd >> (b * 8)) & 0xFF) as u8;
                let ch = alphanumeric_at((idx8 % 62) as usize);
                unsafe { *csp.add(i * 8 + b) = ch; }
            }
        }

        // sha256(base || seed || owner)
        sha256_80(base_arr, &create_account_seed, owner_arr, &mut local_hash);

        // base58 encode the 32-byte hash
        base58_encode_32(&local_hash, &mut local_encoded, &mut encoded_len, ci);

        // check prefix + suffix
        if matches_target(
            local_encoded.as_ptr(),
            target_ptr,
            target_len_u,
            suffix_ptr,
            suffix_len_u,
            encoded_len as usize,
        ) {
            if done.compare_exchange(0, 1, AtomicOrdering::AcqRel, AtomicOrdering::Relaxed).is_ok() {
                let cs_p = create_account_seed.as_ptr();
                for i in 0..16 {
                    unsafe { *out_ptr.add(i) = *cs_p.add(i); }
                }
            }
            count.fetch_add(iter + 1, AtomicOrdering::Relaxed);
            return;
        }

        iter += 1;
    }

    count.fetch_add(iter, AtomicOrdering::Relaxed);
}

#[device]
pub fn matches_target(
    a: *const u8,
    target: *const u8,
    n: usize,
    suffix: *const u8,
    suffix_len: usize,
    encoded_len: usize,
) -> bool {
    let mut i = 0;
    while i < n {
        if unsafe { *a.add(i) != *target.add(i) } {
            return false;
        }
        i += 1;
    }
    let base = encoded_len - suffix_len;
    let mut i = 0;
    while i < suffix_len {
        if unsafe { *a.add(base + i) != *suffix.add(i) } {
            return false;
        }
        i += 1;
    }
    true
}