use cuda_device::device;

#[inline(always)]
fn ep0(x: u32) -> u32 { x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22) }
#[inline(always)]
fn ep1(x: u32) -> u32 { x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25) }
#[inline(always)]
fn sig0(x: u32) -> u32 { x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3) }
#[inline(always)]
fn sig1(x: u32) -> u32 { x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10) }
#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (!x & z) }
#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (x & z) ^ (y & z) }

/// Standalone unrolled transform — kept for tests against generic sha256.
#[device]
pub fn sha256_transform(state: &mut [u32; 8], block: &[u8; 64]) {
    let bp = block.as_ptr();

    let m00: u32 =
          ((unsafe { *bp.add(0) }) as u32) << 24
        | ((unsafe { *bp.add(1) }) as u32) << 16
        | ((unsafe { *bp.add(2) }) as u32) << 8
        | ((unsafe { *bp.add(3) }) as u32);
    let m01: u32 =
          ((unsafe { *bp.add(4) }) as u32) << 24
        | ((unsafe { *bp.add(5) }) as u32) << 16
        | ((unsafe { *bp.add(6) }) as u32) << 8
        | ((unsafe { *bp.add(7) }) as u32);
    let m02: u32 =
          ((unsafe { *bp.add(8) }) as u32) << 24
        | ((unsafe { *bp.add(9) }) as u32) << 16
        | ((unsafe { *bp.add(10) }) as u32) << 8
        | ((unsafe { *bp.add(11) }) as u32);
    let m03: u32 =
          ((unsafe { *bp.add(12) }) as u32) << 24
        | ((unsafe { *bp.add(13) }) as u32) << 16
        | ((unsafe { *bp.add(14) }) as u32) << 8
        | ((unsafe { *bp.add(15) }) as u32);
    let m04: u32 =
          ((unsafe { *bp.add(16) }) as u32) << 24
        | ((unsafe { *bp.add(17) }) as u32) << 16
        | ((unsafe { *bp.add(18) }) as u32) << 8
        | ((unsafe { *bp.add(19) }) as u32);
    let m05: u32 =
          ((unsafe { *bp.add(20) }) as u32) << 24
        | ((unsafe { *bp.add(21) }) as u32) << 16
        | ((unsafe { *bp.add(22) }) as u32) << 8
        | ((unsafe { *bp.add(23) }) as u32);
    let m06: u32 =
          ((unsafe { *bp.add(24) }) as u32) << 24
        | ((unsafe { *bp.add(25) }) as u32) << 16
        | ((unsafe { *bp.add(26) }) as u32) << 8
        | ((unsafe { *bp.add(27) }) as u32);
    let m07: u32 =
          ((unsafe { *bp.add(28) }) as u32) << 24
        | ((unsafe { *bp.add(29) }) as u32) << 16
        | ((unsafe { *bp.add(30) }) as u32) << 8
        | ((unsafe { *bp.add(31) }) as u32);
    let m08: u32 =
          ((unsafe { *bp.add(32) }) as u32) << 24
        | ((unsafe { *bp.add(33) }) as u32) << 16
        | ((unsafe { *bp.add(34) }) as u32) << 8
        | ((unsafe { *bp.add(35) }) as u32);
    let m09: u32 =
          ((unsafe { *bp.add(36) }) as u32) << 24
        | ((unsafe { *bp.add(37) }) as u32) << 16
        | ((unsafe { *bp.add(38) }) as u32) << 8
        | ((unsafe { *bp.add(39) }) as u32);
    let m10: u32 =
          ((unsafe { *bp.add(40) }) as u32) << 24
        | ((unsafe { *bp.add(41) }) as u32) << 16
        | ((unsafe { *bp.add(42) }) as u32) << 8
        | ((unsafe { *bp.add(43) }) as u32);
    let m11: u32 =
          ((unsafe { *bp.add(44) }) as u32) << 24
        | ((unsafe { *bp.add(45) }) as u32) << 16
        | ((unsafe { *bp.add(46) }) as u32) << 8
        | ((unsafe { *bp.add(47) }) as u32);
    let m12: u32 =
          ((unsafe { *bp.add(48) }) as u32) << 24
        | ((unsafe { *bp.add(49) }) as u32) << 16
        | ((unsafe { *bp.add(50) }) as u32) << 8
        | ((unsafe { *bp.add(51) }) as u32);
    let m13: u32 =
          ((unsafe { *bp.add(52) }) as u32) << 24
        | ((unsafe { *bp.add(53) }) as u32) << 16
        | ((unsafe { *bp.add(54) }) as u32) << 8
        | ((unsafe { *bp.add(55) }) as u32);
    let m14: u32 =
          ((unsafe { *bp.add(56) }) as u32) << 24
        | ((unsafe { *bp.add(57) }) as u32) << 16
        | ((unsafe { *bp.add(58) }) as u32) << 8
        | ((unsafe { *bp.add(59) }) as u32);
    let m15: u32 =
          ((unsafe { *bp.add(60) }) as u32) << 24
        | ((unsafe { *bp.add(61) }) as u32) << 16
        | ((unsafe { *bp.add(62) }) as u32) << 8
        | ((unsafe { *bp.add(63) }) as u32);
    let m16: u32 = sig1(m14).wrapping_add(m09).wrapping_add(sig0(m01)).wrapping_add(m00);
    let m17: u32 = sig1(m15).wrapping_add(m10).wrapping_add(sig0(m02)).wrapping_add(m01);
    let m18: u32 = sig1(m16).wrapping_add(m11).wrapping_add(sig0(m03)).wrapping_add(m02);
    let m19: u32 = sig1(m17).wrapping_add(m12).wrapping_add(sig0(m04)).wrapping_add(m03);
    let m20: u32 = sig1(m18).wrapping_add(m13).wrapping_add(sig0(m05)).wrapping_add(m04);
    let m21: u32 = sig1(m19).wrapping_add(m14).wrapping_add(sig0(m06)).wrapping_add(m05);
    let m22: u32 = sig1(m20).wrapping_add(m15).wrapping_add(sig0(m07)).wrapping_add(m06);
    let m23: u32 = sig1(m21).wrapping_add(m16).wrapping_add(sig0(m08)).wrapping_add(m07);
    let m24: u32 = sig1(m22).wrapping_add(m17).wrapping_add(sig0(m09)).wrapping_add(m08);
    let m25: u32 = sig1(m23).wrapping_add(m18).wrapping_add(sig0(m10)).wrapping_add(m09);
    let m26: u32 = sig1(m24).wrapping_add(m19).wrapping_add(sig0(m11)).wrapping_add(m10);
    let m27: u32 = sig1(m25).wrapping_add(m20).wrapping_add(sig0(m12)).wrapping_add(m11);
    let m28: u32 = sig1(m26).wrapping_add(m21).wrapping_add(sig0(m13)).wrapping_add(m12);
    let m29: u32 = sig1(m27).wrapping_add(m22).wrapping_add(sig0(m14)).wrapping_add(m13);
    let m30: u32 = sig1(m28).wrapping_add(m23).wrapping_add(sig0(m15)).wrapping_add(m14);
    let m31: u32 = sig1(m29).wrapping_add(m24).wrapping_add(sig0(m16)).wrapping_add(m15);
    let m32: u32 = sig1(m30).wrapping_add(m25).wrapping_add(sig0(m17)).wrapping_add(m16);
    let m33: u32 = sig1(m31).wrapping_add(m26).wrapping_add(sig0(m18)).wrapping_add(m17);
    let m34: u32 = sig1(m32).wrapping_add(m27).wrapping_add(sig0(m19)).wrapping_add(m18);
    let m35: u32 = sig1(m33).wrapping_add(m28).wrapping_add(sig0(m20)).wrapping_add(m19);
    let m36: u32 = sig1(m34).wrapping_add(m29).wrapping_add(sig0(m21)).wrapping_add(m20);
    let m37: u32 = sig1(m35).wrapping_add(m30).wrapping_add(sig0(m22)).wrapping_add(m21);
    let m38: u32 = sig1(m36).wrapping_add(m31).wrapping_add(sig0(m23)).wrapping_add(m22);
    let m39: u32 = sig1(m37).wrapping_add(m32).wrapping_add(sig0(m24)).wrapping_add(m23);
    let m40: u32 = sig1(m38).wrapping_add(m33).wrapping_add(sig0(m25)).wrapping_add(m24);
    let m41: u32 = sig1(m39).wrapping_add(m34).wrapping_add(sig0(m26)).wrapping_add(m25);
    let m42: u32 = sig1(m40).wrapping_add(m35).wrapping_add(sig0(m27)).wrapping_add(m26);
    let m43: u32 = sig1(m41).wrapping_add(m36).wrapping_add(sig0(m28)).wrapping_add(m27);
    let m44: u32 = sig1(m42).wrapping_add(m37).wrapping_add(sig0(m29)).wrapping_add(m28);
    let m45: u32 = sig1(m43).wrapping_add(m38).wrapping_add(sig0(m30)).wrapping_add(m29);
    let m46: u32 = sig1(m44).wrapping_add(m39).wrapping_add(sig0(m31)).wrapping_add(m30);
    let m47: u32 = sig1(m45).wrapping_add(m40).wrapping_add(sig0(m32)).wrapping_add(m31);
    let m48: u32 = sig1(m46).wrapping_add(m41).wrapping_add(sig0(m33)).wrapping_add(m32);
    let m49: u32 = sig1(m47).wrapping_add(m42).wrapping_add(sig0(m34)).wrapping_add(m33);
    let m50: u32 = sig1(m48).wrapping_add(m43).wrapping_add(sig0(m35)).wrapping_add(m34);
    let m51: u32 = sig1(m49).wrapping_add(m44).wrapping_add(sig0(m36)).wrapping_add(m35);
    let m52: u32 = sig1(m50).wrapping_add(m45).wrapping_add(sig0(m37)).wrapping_add(m36);
    let m53: u32 = sig1(m51).wrapping_add(m46).wrapping_add(sig0(m38)).wrapping_add(m37);
    let m54: u32 = sig1(m52).wrapping_add(m47).wrapping_add(sig0(m39)).wrapping_add(m38);
    let m55: u32 = sig1(m53).wrapping_add(m48).wrapping_add(sig0(m40)).wrapping_add(m39);
    let m56: u32 = sig1(m54).wrapping_add(m49).wrapping_add(sig0(m41)).wrapping_add(m40);
    let m57: u32 = sig1(m55).wrapping_add(m50).wrapping_add(sig0(m42)).wrapping_add(m41);
    let m58: u32 = sig1(m56).wrapping_add(m51).wrapping_add(sig0(m43)).wrapping_add(m42);
    let m59: u32 = sig1(m57).wrapping_add(m52).wrapping_add(sig0(m44)).wrapping_add(m43);
    let m60: u32 = sig1(m58).wrapping_add(m53).wrapping_add(sig0(m45)).wrapping_add(m44);
    let m61: u32 = sig1(m59).wrapping_add(m54).wrapping_add(sig0(m46)).wrapping_add(m45);
    let m62: u32 = sig1(m60).wrapping_add(m55).wrapping_add(sig0(m47)).wrapping_add(m46);
    let m63: u32 = sig1(m61).wrapping_add(m56).wrapping_add(sig0(m48)).wrapping_add(m47);
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    // round 0
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x428a2f98u32).wrapping_add(m00);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 1
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x71374491u32).wrapping_add(m01);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 2
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb5c0fbcfu32).wrapping_add(m02);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 3
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe9b5dba5u32).wrapping_add(m03);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 4
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x3956c25bu32).wrapping_add(m04);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 5
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x59f111f1u32).wrapping_add(m05);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 6
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x923f82a4u32).wrapping_add(m06);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 7
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xab1c5ed5u32).wrapping_add(m07);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 8
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd807aa98u32).wrapping_add(m08);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 9
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x12835b01u32).wrapping_add(m09);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 10
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x243185beu32).wrapping_add(m10);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 11
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x550c7dc3u32).wrapping_add(m11);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 12
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x72be5d74u32).wrapping_add(m12);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 13
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x80deb1feu32).wrapping_add(m13);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 14
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x9bdc06a7u32).wrapping_add(m14);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 15
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc19bf174u32).wrapping_add(m15);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 16
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe49b69c1u32).wrapping_add(m16);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 17
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xefbe4786u32).wrapping_add(m17);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 18
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x0fc19dc6u32).wrapping_add(m18);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 19
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x240ca1ccu32).wrapping_add(m19);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 20
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2de92c6fu32).wrapping_add(m20);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 21
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4a7484aau32).wrapping_add(m21);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 22
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5cb0a9dcu32).wrapping_add(m22);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 23
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x76f988dau32).wrapping_add(m23);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 24
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x983e5152u32).wrapping_add(m24);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 25
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa831c66du32).wrapping_add(m25);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 26
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb00327c8u32).wrapping_add(m26);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 27
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbf597fc7u32).wrapping_add(m27);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 28
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc6e00bf3u32).wrapping_add(m28);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 29
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd5a79147u32).wrapping_add(m29);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 30
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x06ca6351u32).wrapping_add(m30);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 31
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x14292967u32).wrapping_add(m31);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 32
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x27b70a85u32).wrapping_add(m32);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 33
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2e1b2138u32).wrapping_add(m33);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 34
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4d2c6dfcu32).wrapping_add(m34);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 35
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x53380d13u32).wrapping_add(m35);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 36
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x650a7354u32).wrapping_add(m36);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 37
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x766a0abbu32).wrapping_add(m37);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 38
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x81c2c92eu32).wrapping_add(m38);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 39
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x92722c85u32).wrapping_add(m39);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 40
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa2bfe8a1u32).wrapping_add(m40);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 41
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa81a664bu32).wrapping_add(m41);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 42
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc24b8b70u32).wrapping_add(m42);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 43
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc76c51a3u32).wrapping_add(m43);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 44
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd192e819u32).wrapping_add(m44);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 45
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd6990624u32).wrapping_add(m45);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 46
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xf40e3585u32).wrapping_add(m46);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 47
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x106aa070u32).wrapping_add(m47);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 48
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x19a4c116u32).wrapping_add(m48);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 49
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x1e376c08u32).wrapping_add(m49);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 50
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2748774cu32).wrapping_add(m50);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 51
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x34b0bcb5u32).wrapping_add(m51);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 52
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x391c0cb3u32).wrapping_add(m52);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 53
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4ed8aa4au32).wrapping_add(m53);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 54
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5b9cca4fu32).wrapping_add(m54);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 55
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x682e6ff3u32).wrapping_add(m55);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 56
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x748f82eeu32).wrapping_add(m56);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 57
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x78a5636fu32).wrapping_add(m57);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 58
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x84c87814u32).wrapping_add(m58);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 59
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x8cc70208u32).wrapping_add(m59);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 60
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x90befffau32).wrapping_add(m60);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 61
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa4506cebu32).wrapping_add(m61);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 62
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbef9a3f7u32).wrapping_add(m62);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 63
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc67178f2u32).wrapping_add(m63);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    let ns0 = state[0].wrapping_add(a);
    let ns1 = state[1].wrapping_add(b);
    let ns2 = state[2].wrapping_add(c);
    let ns3 = state[3].wrapping_add(d);
    let ns4 = state[4].wrapping_add(e);
    let ns5 = state[5].wrapping_add(f);
    let ns6 = state[6].wrapping_add(g);
    let ns7 = state[7].wrapping_add(h);

    let sp = state.as_mut_ptr();
    unsafe {
        *sp.add(0) = ns0;
        *sp.add(1) = ns1;
        *sp.add(2) = ns2;
        *sp.add(3) = ns3;
        *sp.add(4) = ns4;
        *sp.add(5) = ns5;
        *sp.add(6) = ns6;
        *sp.add(7) = ns7;
    }
}


/// Fully-unrolled sha256 of base(32) || seed(16) || owner(32) — 80 bytes total.
#[device]
pub fn sha256_80(base: &[u8; 32], seed: &[u8; 16], owner: &[u8; 32], hash: &mut [u8; 32]) {
    let bp = base.as_ptr();
    let sp = seed.as_ptr();
    let op = owner.as_ptr();

    let ma00: u32 =
          ((unsafe { *bp.add(0) }) as u32) << 24
        | ((unsafe { *bp.add(1) }) as u32) << 16
        | ((unsafe { *bp.add(2) }) as u32) << 8
        | ((unsafe { *bp.add(3) }) as u32);
    let ma01: u32 =
          ((unsafe { *bp.add(4) }) as u32) << 24
        | ((unsafe { *bp.add(5) }) as u32) << 16
        | ((unsafe { *bp.add(6) }) as u32) << 8
        | ((unsafe { *bp.add(7) }) as u32);
    let ma02: u32 =
          ((unsafe { *bp.add(8) }) as u32) << 24
        | ((unsafe { *bp.add(9) }) as u32) << 16
        | ((unsafe { *bp.add(10) }) as u32) << 8
        | ((unsafe { *bp.add(11) }) as u32);
    let ma03: u32 =
          ((unsafe { *bp.add(12) }) as u32) << 24
        | ((unsafe { *bp.add(13) }) as u32) << 16
        | ((unsafe { *bp.add(14) }) as u32) << 8
        | ((unsafe { *bp.add(15) }) as u32);
    let ma04: u32 =
          ((unsafe { *bp.add(16) }) as u32) << 24
        | ((unsafe { *bp.add(17) }) as u32) << 16
        | ((unsafe { *bp.add(18) }) as u32) << 8
        | ((unsafe { *bp.add(19) }) as u32);
    let ma05: u32 =
          ((unsafe { *bp.add(20) }) as u32) << 24
        | ((unsafe { *bp.add(21) }) as u32) << 16
        | ((unsafe { *bp.add(22) }) as u32) << 8
        | ((unsafe { *bp.add(23) }) as u32);
    let ma06: u32 =
          ((unsafe { *bp.add(24) }) as u32) << 24
        | ((unsafe { *bp.add(25) }) as u32) << 16
        | ((unsafe { *bp.add(26) }) as u32) << 8
        | ((unsafe { *bp.add(27) }) as u32);
    let ma07: u32 =
          ((unsafe { *bp.add(28) }) as u32) << 24
        | ((unsafe { *bp.add(29) }) as u32) << 16
        | ((unsafe { *bp.add(30) }) as u32) << 8
        | ((unsafe { *bp.add(31) }) as u32);
    let ma08: u32 =
          ((unsafe { *sp.add(0) }) as u32) << 24
        | ((unsafe { *sp.add(1) }) as u32) << 16
        | ((unsafe { *sp.add(2) }) as u32) << 8
        | ((unsafe { *sp.add(3) }) as u32);
    let ma09: u32 =
          ((unsafe { *sp.add(4) }) as u32) << 24
        | ((unsafe { *sp.add(5) }) as u32) << 16
        | ((unsafe { *sp.add(6) }) as u32) << 8
        | ((unsafe { *sp.add(7) }) as u32);
    let ma10: u32 =
          ((unsafe { *sp.add(8) }) as u32) << 24
        | ((unsafe { *sp.add(9) }) as u32) << 16
        | ((unsafe { *sp.add(10) }) as u32) << 8
        | ((unsafe { *sp.add(11) }) as u32);
    let ma11: u32 =
          ((unsafe { *sp.add(12) }) as u32) << 24
        | ((unsafe { *sp.add(13) }) as u32) << 16
        | ((unsafe { *sp.add(14) }) as u32) << 8
        | ((unsafe { *sp.add(15) }) as u32);
    let ma12: u32 =
          ((unsafe { *op.add(0) }) as u32) << 24
        | ((unsafe { *op.add(1) }) as u32) << 16
        | ((unsafe { *op.add(2) }) as u32) << 8
        | ((unsafe { *op.add(3) }) as u32);
    let ma13: u32 =
          ((unsafe { *op.add(4) }) as u32) << 24
        | ((unsafe { *op.add(5) }) as u32) << 16
        | ((unsafe { *op.add(6) }) as u32) << 8
        | ((unsafe { *op.add(7) }) as u32);
    let ma14: u32 =
          ((unsafe { *op.add(8) }) as u32) << 24
        | ((unsafe { *op.add(9) }) as u32) << 16
        | ((unsafe { *op.add(10) }) as u32) << 8
        | ((unsafe { *op.add(11) }) as u32);
    let ma15: u32 =
          ((unsafe { *op.add(12) }) as u32) << 24
        | ((unsafe { *op.add(13) }) as u32) << 16
        | ((unsafe { *op.add(14) }) as u32) << 8
        | ((unsafe { *op.add(15) }) as u32);
    let ma16: u32 = sig1(ma14).wrapping_add(ma09).wrapping_add(sig0(ma01)).wrapping_add(ma00);
    let ma17: u32 = sig1(ma15).wrapping_add(ma10).wrapping_add(sig0(ma02)).wrapping_add(ma01);
    let ma18: u32 = sig1(ma16).wrapping_add(ma11).wrapping_add(sig0(ma03)).wrapping_add(ma02);
    let ma19: u32 = sig1(ma17).wrapping_add(ma12).wrapping_add(sig0(ma04)).wrapping_add(ma03);
    let ma20: u32 = sig1(ma18).wrapping_add(ma13).wrapping_add(sig0(ma05)).wrapping_add(ma04);
    let ma21: u32 = sig1(ma19).wrapping_add(ma14).wrapping_add(sig0(ma06)).wrapping_add(ma05);
    let ma22: u32 = sig1(ma20).wrapping_add(ma15).wrapping_add(sig0(ma07)).wrapping_add(ma06);
    let ma23: u32 = sig1(ma21).wrapping_add(ma16).wrapping_add(sig0(ma08)).wrapping_add(ma07);
    let ma24: u32 = sig1(ma22).wrapping_add(ma17).wrapping_add(sig0(ma09)).wrapping_add(ma08);
    let ma25: u32 = sig1(ma23).wrapping_add(ma18).wrapping_add(sig0(ma10)).wrapping_add(ma09);
    let ma26: u32 = sig1(ma24).wrapping_add(ma19).wrapping_add(sig0(ma11)).wrapping_add(ma10);
    let ma27: u32 = sig1(ma25).wrapping_add(ma20).wrapping_add(sig0(ma12)).wrapping_add(ma11);
    let ma28: u32 = sig1(ma26).wrapping_add(ma21).wrapping_add(sig0(ma13)).wrapping_add(ma12);
    let ma29: u32 = sig1(ma27).wrapping_add(ma22).wrapping_add(sig0(ma14)).wrapping_add(ma13);
    let ma30: u32 = sig1(ma28).wrapping_add(ma23).wrapping_add(sig0(ma15)).wrapping_add(ma14);
    let ma31: u32 = sig1(ma29).wrapping_add(ma24).wrapping_add(sig0(ma16)).wrapping_add(ma15);
    let ma32: u32 = sig1(ma30).wrapping_add(ma25).wrapping_add(sig0(ma17)).wrapping_add(ma16);
    let ma33: u32 = sig1(ma31).wrapping_add(ma26).wrapping_add(sig0(ma18)).wrapping_add(ma17);
    let ma34: u32 = sig1(ma32).wrapping_add(ma27).wrapping_add(sig0(ma19)).wrapping_add(ma18);
    let ma35: u32 = sig1(ma33).wrapping_add(ma28).wrapping_add(sig0(ma20)).wrapping_add(ma19);
    let ma36: u32 = sig1(ma34).wrapping_add(ma29).wrapping_add(sig0(ma21)).wrapping_add(ma20);
    let ma37: u32 = sig1(ma35).wrapping_add(ma30).wrapping_add(sig0(ma22)).wrapping_add(ma21);
    let ma38: u32 = sig1(ma36).wrapping_add(ma31).wrapping_add(sig0(ma23)).wrapping_add(ma22);
    let ma39: u32 = sig1(ma37).wrapping_add(ma32).wrapping_add(sig0(ma24)).wrapping_add(ma23);
    let ma40: u32 = sig1(ma38).wrapping_add(ma33).wrapping_add(sig0(ma25)).wrapping_add(ma24);
    let ma41: u32 = sig1(ma39).wrapping_add(ma34).wrapping_add(sig0(ma26)).wrapping_add(ma25);
    let ma42: u32 = sig1(ma40).wrapping_add(ma35).wrapping_add(sig0(ma27)).wrapping_add(ma26);
    let ma43: u32 = sig1(ma41).wrapping_add(ma36).wrapping_add(sig0(ma28)).wrapping_add(ma27);
    let ma44: u32 = sig1(ma42).wrapping_add(ma37).wrapping_add(sig0(ma29)).wrapping_add(ma28);
    let ma45: u32 = sig1(ma43).wrapping_add(ma38).wrapping_add(sig0(ma30)).wrapping_add(ma29);
    let ma46: u32 = sig1(ma44).wrapping_add(ma39).wrapping_add(sig0(ma31)).wrapping_add(ma30);
    let ma47: u32 = sig1(ma45).wrapping_add(ma40).wrapping_add(sig0(ma32)).wrapping_add(ma31);
    let ma48: u32 = sig1(ma46).wrapping_add(ma41).wrapping_add(sig0(ma33)).wrapping_add(ma32);
    let ma49: u32 = sig1(ma47).wrapping_add(ma42).wrapping_add(sig0(ma34)).wrapping_add(ma33);
    let ma50: u32 = sig1(ma48).wrapping_add(ma43).wrapping_add(sig0(ma35)).wrapping_add(ma34);
    let ma51: u32 = sig1(ma49).wrapping_add(ma44).wrapping_add(sig0(ma36)).wrapping_add(ma35);
    let ma52: u32 = sig1(ma50).wrapping_add(ma45).wrapping_add(sig0(ma37)).wrapping_add(ma36);
    let ma53: u32 = sig1(ma51).wrapping_add(ma46).wrapping_add(sig0(ma38)).wrapping_add(ma37);
    let ma54: u32 = sig1(ma52).wrapping_add(ma47).wrapping_add(sig0(ma39)).wrapping_add(ma38);
    let ma55: u32 = sig1(ma53).wrapping_add(ma48).wrapping_add(sig0(ma40)).wrapping_add(ma39);
    let ma56: u32 = sig1(ma54).wrapping_add(ma49).wrapping_add(sig0(ma41)).wrapping_add(ma40);
    let ma57: u32 = sig1(ma55).wrapping_add(ma50).wrapping_add(sig0(ma42)).wrapping_add(ma41);
    let ma58: u32 = sig1(ma56).wrapping_add(ma51).wrapping_add(sig0(ma43)).wrapping_add(ma42);
    let ma59: u32 = sig1(ma57).wrapping_add(ma52).wrapping_add(sig0(ma44)).wrapping_add(ma43);
    let ma60: u32 = sig1(ma58).wrapping_add(ma53).wrapping_add(sig0(ma45)).wrapping_add(ma44);
    let ma61: u32 = sig1(ma59).wrapping_add(ma54).wrapping_add(sig0(ma46)).wrapping_add(ma45);
    let ma62: u32 = sig1(ma60).wrapping_add(ma55).wrapping_add(sig0(ma47)).wrapping_add(ma46);
    let ma63: u32 = sig1(ma61).wrapping_add(ma56).wrapping_add(sig0(ma48)).wrapping_add(ma47);
    let mut a = 0x6a09e667u32;
    let mut b = 0xbb67ae85u32;
    let mut c = 0x3c6ef372u32;
    let mut d = 0xa54ff53au32;
    let mut e = 0x510e527fu32;
    let mut f = 0x9b05688cu32;
    let mut g = 0x1f83d9abu32;
    let mut h = 0x5be0cd19u32;
    // round 0
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x428a2f98u32).wrapping_add(ma00);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 1
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x71374491u32).wrapping_add(ma01);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 2
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb5c0fbcfu32).wrapping_add(ma02);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 3
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe9b5dba5u32).wrapping_add(ma03);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 4
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x3956c25bu32).wrapping_add(ma04);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 5
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x59f111f1u32).wrapping_add(ma05);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 6
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x923f82a4u32).wrapping_add(ma06);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 7
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xab1c5ed5u32).wrapping_add(ma07);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 8
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd807aa98u32).wrapping_add(ma08);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 9
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x12835b01u32).wrapping_add(ma09);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 10
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x243185beu32).wrapping_add(ma10);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 11
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x550c7dc3u32).wrapping_add(ma11);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 12
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x72be5d74u32).wrapping_add(ma12);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 13
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x80deb1feu32).wrapping_add(ma13);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 14
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x9bdc06a7u32).wrapping_add(ma14);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 15
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc19bf174u32).wrapping_add(ma15);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 16
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe49b69c1u32).wrapping_add(ma16);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 17
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xefbe4786u32).wrapping_add(ma17);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 18
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x0fc19dc6u32).wrapping_add(ma18);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 19
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x240ca1ccu32).wrapping_add(ma19);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 20
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2de92c6fu32).wrapping_add(ma20);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 21
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4a7484aau32).wrapping_add(ma21);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 22
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5cb0a9dcu32).wrapping_add(ma22);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 23
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x76f988dau32).wrapping_add(ma23);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 24
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x983e5152u32).wrapping_add(ma24);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 25
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa831c66du32).wrapping_add(ma25);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 26
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb00327c8u32).wrapping_add(ma26);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 27
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbf597fc7u32).wrapping_add(ma27);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 28
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc6e00bf3u32).wrapping_add(ma28);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 29
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd5a79147u32).wrapping_add(ma29);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 30
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x06ca6351u32).wrapping_add(ma30);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 31
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x14292967u32).wrapping_add(ma31);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 32
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x27b70a85u32).wrapping_add(ma32);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 33
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2e1b2138u32).wrapping_add(ma33);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 34
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4d2c6dfcu32).wrapping_add(ma34);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 35
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x53380d13u32).wrapping_add(ma35);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 36
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x650a7354u32).wrapping_add(ma36);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 37
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x766a0abbu32).wrapping_add(ma37);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 38
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x81c2c92eu32).wrapping_add(ma38);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 39
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x92722c85u32).wrapping_add(ma39);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 40
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa2bfe8a1u32).wrapping_add(ma40);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 41
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa81a664bu32).wrapping_add(ma41);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 42
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc24b8b70u32).wrapping_add(ma42);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 43
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc76c51a3u32).wrapping_add(ma43);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 44
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd192e819u32).wrapping_add(ma44);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 45
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd6990624u32).wrapping_add(ma45);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 46
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xf40e3585u32).wrapping_add(ma46);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 47
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x106aa070u32).wrapping_add(ma47);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 48
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x19a4c116u32).wrapping_add(ma48);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 49
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x1e376c08u32).wrapping_add(ma49);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 50
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2748774cu32).wrapping_add(ma50);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 51
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x34b0bcb5u32).wrapping_add(ma51);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 52
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x391c0cb3u32).wrapping_add(ma52);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 53
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4ed8aa4au32).wrapping_add(ma53);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 54
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5b9cca4fu32).wrapping_add(ma54);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 55
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x682e6ff3u32).wrapping_add(ma55);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 56
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x748f82eeu32).wrapping_add(ma56);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 57
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x78a5636fu32).wrapping_add(ma57);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 58
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x84c87814u32).wrapping_add(ma58);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 59
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x8cc70208u32).wrapping_add(ma59);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 60
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x90befffau32).wrapping_add(ma60);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 61
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa4506cebu32).wrapping_add(ma61);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 62
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbef9a3f7u32).wrapping_add(ma62);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 63
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc67178f2u32).wrapping_add(ma63);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    let a0 = 0x6a09e667u32.wrapping_add(a);
    let a1 = 0xbb67ae85u32.wrapping_add(b);
    let a2 = 0x3c6ef372u32.wrapping_add(c);
    let a3 = 0xa54ff53au32.wrapping_add(d);
    let a4 = 0x510e527fu32.wrapping_add(e);
    let a5 = 0x9b05688cu32.wrapping_add(f);
    let a6 = 0x1f83d9abu32.wrapping_add(g);
    let a7 = 0x5be0cd19u32.wrapping_add(h);
    let mb00: u32 =
          ((unsafe { *op.add(16) }) as u32) << 24
        | ((unsafe { *op.add(17) }) as u32) << 16
        | ((unsafe { *op.add(18) }) as u32) << 8
        | ((unsafe { *op.add(19) }) as u32);
    let mb01: u32 =
          ((unsafe { *op.add(20) }) as u32) << 24
        | ((unsafe { *op.add(21) }) as u32) << 16
        | ((unsafe { *op.add(22) }) as u32) << 8
        | ((unsafe { *op.add(23) }) as u32);
    let mb02: u32 =
          ((unsafe { *op.add(24) }) as u32) << 24
        | ((unsafe { *op.add(25) }) as u32) << 16
        | ((unsafe { *op.add(26) }) as u32) << 8
        | ((unsafe { *op.add(27) }) as u32);
    let mb03: u32 =
          ((unsafe { *op.add(28) }) as u32) << 24
        | ((unsafe { *op.add(29) }) as u32) << 16
        | ((unsafe { *op.add(30) }) as u32) << 8
        | ((unsafe { *op.add(31) }) as u32);
    let mb04: u32 =
          ((0x80u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb05: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb06: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb07: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb08: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb09: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb10: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb11: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb12: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb13: u32 =
          ((0u8) as u32) << 24
        | ((0u8) as u32) << 16
        | ((0u8) as u32) << 8
        | ((0u8) as u32);
    let mb14: u32 =
          ((0x00u8) as u32) << 24
        | ((0x00u8) as u32) << 16
        | ((0x00u8) as u32) << 8
        | ((0x00u8) as u32);
    let mb15: u32 =
          ((0x00u8) as u32) << 24
        | ((0x00u8) as u32) << 16
        | ((0x02u8) as u32) << 8
        | ((0x80u8) as u32);
    let mb16: u32 = sig1(mb14).wrapping_add(mb09).wrapping_add(sig0(mb01)).wrapping_add(mb00);
    let mb17: u32 = sig1(mb15).wrapping_add(mb10).wrapping_add(sig0(mb02)).wrapping_add(mb01);
    let mb18: u32 = sig1(mb16).wrapping_add(mb11).wrapping_add(sig0(mb03)).wrapping_add(mb02);
    let mb19: u32 = sig1(mb17).wrapping_add(mb12).wrapping_add(sig0(mb04)).wrapping_add(mb03);
    let mb20: u32 = sig1(mb18).wrapping_add(mb13).wrapping_add(sig0(mb05)).wrapping_add(mb04);
    let mb21: u32 = sig1(mb19).wrapping_add(mb14).wrapping_add(sig0(mb06)).wrapping_add(mb05);
    let mb22: u32 = sig1(mb20).wrapping_add(mb15).wrapping_add(sig0(mb07)).wrapping_add(mb06);
    let mb23: u32 = sig1(mb21).wrapping_add(mb16).wrapping_add(sig0(mb08)).wrapping_add(mb07);
    let mb24: u32 = sig1(mb22).wrapping_add(mb17).wrapping_add(sig0(mb09)).wrapping_add(mb08);
    let mb25: u32 = sig1(mb23).wrapping_add(mb18).wrapping_add(sig0(mb10)).wrapping_add(mb09);
    let mb26: u32 = sig1(mb24).wrapping_add(mb19).wrapping_add(sig0(mb11)).wrapping_add(mb10);
    let mb27: u32 = sig1(mb25).wrapping_add(mb20).wrapping_add(sig0(mb12)).wrapping_add(mb11);
    let mb28: u32 = sig1(mb26).wrapping_add(mb21).wrapping_add(sig0(mb13)).wrapping_add(mb12);
    let mb29: u32 = sig1(mb27).wrapping_add(mb22).wrapping_add(sig0(mb14)).wrapping_add(mb13);
    let mb30: u32 = sig1(mb28).wrapping_add(mb23).wrapping_add(sig0(mb15)).wrapping_add(mb14);
    let mb31: u32 = sig1(mb29).wrapping_add(mb24).wrapping_add(sig0(mb16)).wrapping_add(mb15);
    let mb32: u32 = sig1(mb30).wrapping_add(mb25).wrapping_add(sig0(mb17)).wrapping_add(mb16);
    let mb33: u32 = sig1(mb31).wrapping_add(mb26).wrapping_add(sig0(mb18)).wrapping_add(mb17);
    let mb34: u32 = sig1(mb32).wrapping_add(mb27).wrapping_add(sig0(mb19)).wrapping_add(mb18);
    let mb35: u32 = sig1(mb33).wrapping_add(mb28).wrapping_add(sig0(mb20)).wrapping_add(mb19);
    let mb36: u32 = sig1(mb34).wrapping_add(mb29).wrapping_add(sig0(mb21)).wrapping_add(mb20);
    let mb37: u32 = sig1(mb35).wrapping_add(mb30).wrapping_add(sig0(mb22)).wrapping_add(mb21);
    let mb38: u32 = sig1(mb36).wrapping_add(mb31).wrapping_add(sig0(mb23)).wrapping_add(mb22);
    let mb39: u32 = sig1(mb37).wrapping_add(mb32).wrapping_add(sig0(mb24)).wrapping_add(mb23);
    let mb40: u32 = sig1(mb38).wrapping_add(mb33).wrapping_add(sig0(mb25)).wrapping_add(mb24);
    let mb41: u32 = sig1(mb39).wrapping_add(mb34).wrapping_add(sig0(mb26)).wrapping_add(mb25);
    let mb42: u32 = sig1(mb40).wrapping_add(mb35).wrapping_add(sig0(mb27)).wrapping_add(mb26);
    let mb43: u32 = sig1(mb41).wrapping_add(mb36).wrapping_add(sig0(mb28)).wrapping_add(mb27);
    let mb44: u32 = sig1(mb42).wrapping_add(mb37).wrapping_add(sig0(mb29)).wrapping_add(mb28);
    let mb45: u32 = sig1(mb43).wrapping_add(mb38).wrapping_add(sig0(mb30)).wrapping_add(mb29);
    let mb46: u32 = sig1(mb44).wrapping_add(mb39).wrapping_add(sig0(mb31)).wrapping_add(mb30);
    let mb47: u32 = sig1(mb45).wrapping_add(mb40).wrapping_add(sig0(mb32)).wrapping_add(mb31);
    let mb48: u32 = sig1(mb46).wrapping_add(mb41).wrapping_add(sig0(mb33)).wrapping_add(mb32);
    let mb49: u32 = sig1(mb47).wrapping_add(mb42).wrapping_add(sig0(mb34)).wrapping_add(mb33);
    let mb50: u32 = sig1(mb48).wrapping_add(mb43).wrapping_add(sig0(mb35)).wrapping_add(mb34);
    let mb51: u32 = sig1(mb49).wrapping_add(mb44).wrapping_add(sig0(mb36)).wrapping_add(mb35);
    let mb52: u32 = sig1(mb50).wrapping_add(mb45).wrapping_add(sig0(mb37)).wrapping_add(mb36);
    let mb53: u32 = sig1(mb51).wrapping_add(mb46).wrapping_add(sig0(mb38)).wrapping_add(mb37);
    let mb54: u32 = sig1(mb52).wrapping_add(mb47).wrapping_add(sig0(mb39)).wrapping_add(mb38);
    let mb55: u32 = sig1(mb53).wrapping_add(mb48).wrapping_add(sig0(mb40)).wrapping_add(mb39);
    let mb56: u32 = sig1(mb54).wrapping_add(mb49).wrapping_add(sig0(mb41)).wrapping_add(mb40);
    let mb57: u32 = sig1(mb55).wrapping_add(mb50).wrapping_add(sig0(mb42)).wrapping_add(mb41);
    let mb58: u32 = sig1(mb56).wrapping_add(mb51).wrapping_add(sig0(mb43)).wrapping_add(mb42);
    let mb59: u32 = sig1(mb57).wrapping_add(mb52).wrapping_add(sig0(mb44)).wrapping_add(mb43);
    let mb60: u32 = sig1(mb58).wrapping_add(mb53).wrapping_add(sig0(mb45)).wrapping_add(mb44);
    let mb61: u32 = sig1(mb59).wrapping_add(mb54).wrapping_add(sig0(mb46)).wrapping_add(mb45);
    let mb62: u32 = sig1(mb60).wrapping_add(mb55).wrapping_add(sig0(mb47)).wrapping_add(mb46);
    let mb63: u32 = sig1(mb61).wrapping_add(mb56).wrapping_add(sig0(mb48)).wrapping_add(mb47);
    let mut a = a0;
    let mut b = a1;
    let mut c = a2;
    let mut d = a3;
    let mut e = a4;
    let mut f = a5;
    let mut g = a6;
    let mut h = a7;
    // round 0
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x428a2f98u32).wrapping_add(mb00);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 1
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x71374491u32).wrapping_add(mb01);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 2
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb5c0fbcfu32).wrapping_add(mb02);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 3
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe9b5dba5u32).wrapping_add(mb03);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 4
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x3956c25bu32).wrapping_add(mb04);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 5
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x59f111f1u32).wrapping_add(mb05);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 6
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x923f82a4u32).wrapping_add(mb06);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 7
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xab1c5ed5u32).wrapping_add(mb07);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 8
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd807aa98u32).wrapping_add(mb08);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 9
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x12835b01u32).wrapping_add(mb09);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 10
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x243185beu32).wrapping_add(mb10);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 11
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x550c7dc3u32).wrapping_add(mb11);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 12
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x72be5d74u32).wrapping_add(mb12);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 13
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x80deb1feu32).wrapping_add(mb13);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 14
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x9bdc06a7u32).wrapping_add(mb14);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 15
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc19bf174u32).wrapping_add(mb15);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 16
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xe49b69c1u32).wrapping_add(mb16);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 17
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xefbe4786u32).wrapping_add(mb17);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 18
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x0fc19dc6u32).wrapping_add(mb18);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 19
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x240ca1ccu32).wrapping_add(mb19);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 20
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2de92c6fu32).wrapping_add(mb20);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 21
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4a7484aau32).wrapping_add(mb21);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 22
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5cb0a9dcu32).wrapping_add(mb22);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 23
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x76f988dau32).wrapping_add(mb23);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 24
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x983e5152u32).wrapping_add(mb24);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 25
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa831c66du32).wrapping_add(mb25);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 26
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xb00327c8u32).wrapping_add(mb26);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 27
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbf597fc7u32).wrapping_add(mb27);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 28
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc6e00bf3u32).wrapping_add(mb28);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 29
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd5a79147u32).wrapping_add(mb29);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 30
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x06ca6351u32).wrapping_add(mb30);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 31
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x14292967u32).wrapping_add(mb31);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 32
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x27b70a85u32).wrapping_add(mb32);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 33
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2e1b2138u32).wrapping_add(mb33);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 34
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4d2c6dfcu32).wrapping_add(mb34);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 35
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x53380d13u32).wrapping_add(mb35);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 36
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x650a7354u32).wrapping_add(mb36);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 37
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x766a0abbu32).wrapping_add(mb37);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 38
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x81c2c92eu32).wrapping_add(mb38);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 39
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x92722c85u32).wrapping_add(mb39);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 40
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa2bfe8a1u32).wrapping_add(mb40);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 41
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa81a664bu32).wrapping_add(mb41);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 42
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc24b8b70u32).wrapping_add(mb42);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 43
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc76c51a3u32).wrapping_add(mb43);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 44
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd192e819u32).wrapping_add(mb44);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 45
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xd6990624u32).wrapping_add(mb45);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 46
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xf40e3585u32).wrapping_add(mb46);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 47
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x106aa070u32).wrapping_add(mb47);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 48
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x19a4c116u32).wrapping_add(mb48);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 49
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x1e376c08u32).wrapping_add(mb49);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 50
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x2748774cu32).wrapping_add(mb50);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 51
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x34b0bcb5u32).wrapping_add(mb51);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 52
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x391c0cb3u32).wrapping_add(mb52);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 53
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x4ed8aa4au32).wrapping_add(mb53);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 54
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x5b9cca4fu32).wrapping_add(mb54);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 55
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x682e6ff3u32).wrapping_add(mb55);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 56
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x748f82eeu32).wrapping_add(mb56);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 57
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x78a5636fu32).wrapping_add(mb57);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 58
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x84c87814u32).wrapping_add(mb58);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 59
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x8cc70208u32).wrapping_add(mb59);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 60
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0x90befffau32).wrapping_add(mb60);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 61
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xa4506cebu32).wrapping_add(mb61);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 62
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xbef9a3f7u32).wrapping_add(mb62);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    // round 63
    {
        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g)).wrapping_add(0xc67178f2u32).wrapping_add(mb63);
        let t2 = ep0(a).wrapping_add(maj(a, b, c));
        h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    let b0 = a0.wrapping_add(a);
    let b1 = a1.wrapping_add(b);
    let b2 = a2.wrapping_add(c);
    let b3 = a3.wrapping_add(d);
    let b4 = a4.wrapping_add(e);
    let b5 = a5.wrapping_add(f);
    let b6 = a6.wrapping_add(g);
    let b7 = a7.wrapping_add(h);

    let hp = hash.as_mut_ptr();
    unsafe {
        *hp.add(0) = (b0 >> 24) as u8;
        *hp.add(1) = (b0 >> 16) as u8;
        *hp.add(2) = (b0 >> 8) as u8;
        *hp.add(3) = (b0 >> 0) as u8;
        *hp.add(4) = (b1 >> 24) as u8;
        *hp.add(5) = (b1 >> 16) as u8;
        *hp.add(6) = (b1 >> 8) as u8;
        *hp.add(7) = (b1 >> 0) as u8;
        *hp.add(8) = (b2 >> 24) as u8;
        *hp.add(9) = (b2 >> 16) as u8;
        *hp.add(10) = (b2 >> 8) as u8;
        *hp.add(11) = (b2 >> 0) as u8;
        *hp.add(12) = (b3 >> 24) as u8;
        *hp.add(13) = (b3 >> 16) as u8;
        *hp.add(14) = (b3 >> 8) as u8;
        *hp.add(15) = (b3 >> 0) as u8;
        *hp.add(16) = (b4 >> 24) as u8;
        *hp.add(17) = (b4 >> 16) as u8;
        *hp.add(18) = (b4 >> 8) as u8;
        *hp.add(19) = (b4 >> 0) as u8;
        *hp.add(20) = (b5 >> 24) as u8;
        *hp.add(21) = (b5 >> 16) as u8;
        *hp.add(22) = (b5 >> 8) as u8;
        *hp.add(23) = (b5 >> 0) as u8;
        *hp.add(24) = (b6 >> 24) as u8;
        *hp.add(25) = (b6 >> 16) as u8;
        *hp.add(26) = (b6 >> 8) as u8;
        *hp.add(27) = (b6 >> 0) as u8;
        *hp.add(28) = (b7 >> 24) as u8;
        *hp.add(29) = (b7 >> 16) as u8;
        *hp.add(30) = (b7 >> 8) as u8;
        *hp.add(31) = (b7 >> 0) as u8;
    }
}

