use cuda_device::device;

#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (!x & z) }
#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (x & z) ^ (y & z) }
#[inline(always)]
fn ep0(x: u32) -> u32 { x.rotate_right(2)  ^ x.rotate_right(13) ^ x.rotate_right(22) }
#[inline(always)]
fn ep1(x: u32) -> u32 { x.rotate_right(6)  ^ x.rotate_right(11) ^ x.rotate_right(25) }
#[inline(always)]
fn sig0(x: u32) -> u32 { x.rotate_right(7)  ^ x.rotate_right(18) ^ (x >> 3) }
#[inline(always)]
fn sig1(x: u32) -> u32 { x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10) }

#[device]
fn k_at(i: usize) -> u32 {
    match i {
        0 => 0x428a2f98, 1 => 0x71374491, 2 => 0xb5c0fbcf, 3 => 0xe9b5dba5,
        4 => 0x3956c25b, 5 => 0x59f111f1, 6 => 0x923f82a4, 7 => 0xab1c5ed5,
        8 => 0xd807aa98, 9 => 0x12835b01, 10 => 0x243185be, 11 => 0x550c7dc3,
        12 => 0x72be5d74, 13 => 0x80deb1fe, 14 => 0x9bdc06a7, 15 => 0xc19bf174,
        16 => 0xe49b69c1, 17 => 0xefbe4786, 18 => 0x0fc19dc6, 19 => 0x240ca1cc,
        20 => 0x2de92c6f, 21 => 0x4a7484aa, 22 => 0x5cb0a9dc, 23 => 0x76f988da,
        24 => 0x983e5152, 25 => 0xa831c66d, 26 => 0xb00327c8, 27 => 0xbf597fc7,
        28 => 0xc6e00bf3, 29 => 0xd5a79147, 30 => 0x06ca6351, 31 => 0x14292967,
        32 => 0x27b70a85, 33 => 0x2e1b2138, 34 => 0x4d2c6dfc, 35 => 0x53380d13,
        36 => 0x650a7354, 37 => 0x766a0abb, 38 => 0x81c2c92e, 39 => 0x92722c85,
        40 => 0xa2bfe8a1, 41 => 0xa81a664b, 42 => 0xc24b8b70, 43 => 0xc76c51a3,
        44 => 0xd192e819, 45 => 0xd6990624, 46 => 0xf40e3585, 47 => 0x106aa070,
        48 => 0x19a4c116, 49 => 0x1e376c08, 50 => 0x2748774c, 51 => 0x34b0bcb5,
        52 => 0x391c0cb3, 53 => 0x4ed8aa4a, 54 => 0x5b9cca4f, 55 => 0x682e6ff3,
        56 => 0x748f82ee, 57 => 0x78a5636f, 58 => 0x84c87814, 59 => 0x8cc70208,
        60 => 0x90befffa, 61 => 0xa4506ceb, 62 => 0xbef9a3f7, 63 => 0xc67178f2,
        _ => 0,
    }
}

#[device]
fn sha256_transform(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut m = [0u32; 64];
    for i in 0..16 {
        m[i] =  (block[i*4]   as u32) << 24
              | (block[i*4+1] as u32) << 16
              | (block[i*4+2] as u32) << 8
              | (block[i*4+3] as u32);
    }
    for i in 16..64 {
        m[i] = sig1(m[i-2])
            .wrapping_add(m[i-7])
            .wrapping_add(sig0(m[i-15]))
            .wrapping_add(m[i-16]);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for i in 0..64 {
        let t1 = h
            .wrapping_add(ep1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(k_at(i))
            .wrapping_add(m[i]);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e;
        e = d.wrapping_add(t1);
        d = c; c = b; b = a;
        a = t1.wrapping_add(t2);
    }
    let p = state.as_mut_ptr();
    unsafe {
        *p.add(0) = (*p.add(0)).wrapping_add(a);
        *p.add(1) = (*p.add(1)).wrapping_add(b);
        *p.add(2) = (*p.add(2)).wrapping_add(c);
        *p.add(3) = (*p.add(3)).wrapping_add(d);
        *p.add(4) = (*p.add(4)).wrapping_add(e);
        *p.add(5) = (*p.add(5)).wrapping_add(f);
        *p.add(6) = (*p.add(6)).wrapping_add(g);
        *p.add(7) = (*p.add(7)).wrapping_add(h);
    }
}

#[device]
pub fn sha256_80(
    base: &[u8; 32],
    seed: &[u8; 16],
    owner: &[u8; 32],
    hash: &mut [u8; 32],
) {
    let bp = base.as_ptr();
    let sp = seed.as_ptr();
    let op = owner.as_ptr();

    // block 0 = base[0..32] || seed[0..16] || owner[0..16]
    let mut block0 = [0u8; 64];
    let b0 = block0.as_mut_ptr();
    unsafe {
        for i in 0..32 { *b0.add(i)      = *bp.add(i); }
        for i in 0..16 { *b0.add(32 + i) = *sp.add(i); }
        for i in 0..16 { *b0.add(48 + i) = *op.add(i); }
    }

    // block 1 = owner[16..32] || 0x80 || zeros[17..56] || bitlen_be(8)
    let mut block1 = [0u8; 64];
    let b1 = block1.as_mut_ptr();
    unsafe {
        for i in 0..16 { *b1.add(i) = *op.add(16 + i); }
        *b1.add(16) = 0x80;
        // bytes 17..56 stay zero from the array initializer
        let bitlen: u64 = 80 * 8; // 640
        *b1.add(56) = (bitlen >> 56) as u8;
        *b1.add(57) = (bitlen >> 48) as u8;
        *b1.add(58) = (bitlen >> 40) as u8;
        *b1.add(59) = (bitlen >> 32) as u8;
        *b1.add(60) = (bitlen >> 24) as u8;
        *b1.add(61) = (bitlen >> 16) as u8;
        *b1.add(62) = (bitlen >>  8) as u8;
        *b1.add(63) =  bitlen        as u8;
    }

    // run two transforms over the same state
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    sha256_transform(&mut state, &block0);
    sha256_transform(&mut state, &block1);

    // big-endian state -> hash
    let hp = hash.as_mut_ptr();
    unsafe {
        for i in 0..4 {
            *hp.add(i)      = (state[0] >> (24 - i*8)) as u8;
            *hp.add(i + 4)  = (state[1] >> (24 - i*8)) as u8;
            *hp.add(i + 8)  = (state[2] >> (24 - i*8)) as u8;
            *hp.add(i + 12) = (state[3] >> (24 - i*8)) as u8;
            *hp.add(i + 16) = (state[4] >> (24 - i*8)) as u8;
            *hp.add(i + 20) = (state[5] >> (24 - i*8)) as u8;
            *hp.add(i + 24) = (state[6] >> (24 - i*8)) as u8;
            *hp.add(i + 28) = (state[7] >> (24 - i*8)) as u8;
        }
    }
}