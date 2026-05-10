use cuda_device::device;

#[device]
pub fn base58_encode_32(
    input: &[u8; 32],
    output: &mut [u8; 44],
    len: &mut u32,
    case_insensitive: bool,
) {
    let in_p = input.as_ptr();

    // count leading zero bytes
    let mut in_leading_0s: usize = 0;
    while in_leading_0s < 32 && unsafe { *in_p.add(in_leading_0s) } == 0 {
        in_leading_0s += 1;
    }

    // load 8 big-endian u32 limbs

    let bin0: u32 =
          (unsafe { *in_p.add(0) } as u32) << 24
        | (unsafe { *in_p.add(1) } as u32) << 16
        | (unsafe { *in_p.add(2) } as u32) << 8
        | (unsafe { *in_p.add(3) } as u32);
    let bin1: u32 =
          (unsafe { *in_p.add(4) } as u32) << 24
        | (unsafe { *in_p.add(5) } as u32) << 16
        | (unsafe { *in_p.add(6) } as u32) << 8
        | (unsafe { *in_p.add(7) } as u32);
    let bin2: u32 =
          (unsafe { *in_p.add(8) } as u32) << 24
        | (unsafe { *in_p.add(9) } as u32) << 16
        | (unsafe { *in_p.add(10) } as u32) << 8
        | (unsafe { *in_p.add(11) } as u32);
    let bin3: u32 =
          (unsafe { *in_p.add(12) } as u32) << 24
        | (unsafe { *in_p.add(13) } as u32) << 16
        | (unsafe { *in_p.add(14) } as u32) << 8
        | (unsafe { *in_p.add(15) } as u32);
    let bin4: u32 =
          (unsafe { *in_p.add(16) } as u32) << 24
        | (unsafe { *in_p.add(17) } as u32) << 16
        | (unsafe { *in_p.add(18) } as u32) << 8
        | (unsafe { *in_p.add(19) } as u32);
    let bin5: u32 =
          (unsafe { *in_p.add(20) } as u32) << 24
        | (unsafe { *in_p.add(21) } as u32) << 16
        | (unsafe { *in_p.add(22) } as u32) << 8
        | (unsafe { *in_p.add(23) } as u32);
    let bin6: u32 =
          (unsafe { *in_p.add(24) } as u32) << 24
        | (unsafe { *in_p.add(25) } as u32) << 16
        | (unsafe { *in_p.add(26) } as u32) << 8
        | (unsafe { *in_p.add(27) } as u32);
    let bin7: u32 =
          (unsafe { *in_p.add(28) } as u32) << 24
        | (unsafe { *in_p.add(29) } as u32) << 16
        | (unsafe { *in_p.add(30) } as u32) << 8
        | (unsafe { *in_p.add(31) } as u32);

    // build intermediate[0..9] via inlined enc_table multiply-accumulates
    // intermediate[0] is always 0 since enc_table only writes j+1 starting from j=0
    let im0: u64 = 0;
    let im1: u64 = bin0 as u64 * 513735u64;
    let im2: u64 = bin0 as u64 * 77223048u64 + bin1 as u64 * 78508u64;
    let im3: u64 = bin0 as u64 * 437087610u64 + bin1 as u64 * 646269101u64 + bin2 as u64 * 11997u64;
    let im4: u64 = bin0 as u64 * 300156666u64 + bin1 as u64 * 118408823u64 + bin2 as u64 * 486083817u64 + bin3 as u64 * 1833u64;
    let im5: u64 = bin0 as u64 * 605448490u64 + bin1 as u64 * 91512303u64 + bin2 as u64 * 3737691u64 + bin3 as u64 * 324463681u64 + bin4 as u64 * 280u64;
    let im6: u64 = bin0 as u64 * 214625350u64 + bin1 as u64 * 209184527u64 + bin2 as u64 * 294005210u64 + bin3 as u64 * 385795061u64 + bin4 as u64 * 127692781u64 + bin5 as u64 * 42u64;
    let im7: u64 = bin0 as u64 * 141436834u64 + bin1 as u64 * 413102373u64 + bin2 as u64 * 247894721u64 + bin3 as u64 * 551597588u64 + bin4 as u64 * 389432875u64 + bin5 as u64 * 537767569u64 + bin6 as u64 * 6u64;
    let im8: u64 = bin0 as u64 * 379377856u64 + bin1 as u64 * 153715680u64 + bin2 as u64 * 289024608u64 + bin3 as u64 * 21339008u64 + bin4 as u64 * 357132832u64 + bin5 as u64 * 410450016u64 + bin6 as u64 * 356826688u64 + bin7 as u64 * 1u64;

    // carry propagation: i in 8..1, intermediate[i-1] += intermediate[i] / R, intermediate[i] %= R
    const R1: u64 = 656356768; // 58^5

    let c0_7: u64 = im7 + (im8 / R1);
    let c0_8: u64 = im8 % R1;
    let c1_6: u64 = im6 + (c0_7 / R1);
    let c1_7: u64 = c0_7 % R1;
    let c2_5: u64 = im5 + (c1_6 / R1);
    let c2_6: u64 = c1_6 % R1;
    let c3_4: u64 = im4 + (c2_5 / R1);
    let c3_5: u64 = c2_5 % R1;
    let c4_3: u64 = im3 + (c3_4 / R1);
    let c4_4: u64 = c3_4 % R1;
    let c5_2: u64 = im2 + (c4_3 / R1);
    let c5_3: u64 = c4_3 % R1;
    let c6_1: u64 = im1 + (c5_2 / R1);
    let c6_2: u64 = c5_2 % R1;
    let c7_0: u64 = im0 + (c6_1 / R1);
    let c7_1: u64 = c6_1 % R1;

    // extract base58 digits (raw_base58[0..45])
    let mut raw_base58: [u8; 45] = [0; 45];
    let raw_p = raw_base58.as_mut_ptr();
    {
        let v: u32 = c7_0 as u32;
        unsafe {
            *raw_p.add(4) = (v % 58) as u8;
            *raw_p.add(3) = ((v / 58) % 58) as u8;
            *raw_p.add(2) = ((v / 3364) % 58) as u8;
            *raw_p.add(1) = ((v / 195112) % 58) as u8;
            *raw_p.add(0) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c7_1 as u32;
        unsafe {
            *raw_p.add(9) = (v % 58) as u8;
            *raw_p.add(8) = ((v / 58) % 58) as u8;
            *raw_p.add(7) = ((v / 3364) % 58) as u8;
            *raw_p.add(6) = ((v / 195112) % 58) as u8;
            *raw_p.add(5) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c6_2 as u32;
        unsafe {
            *raw_p.add(14) = (v % 58) as u8;
            *raw_p.add(13) = ((v / 58) % 58) as u8;
            *raw_p.add(12) = ((v / 3364) % 58) as u8;
            *raw_p.add(11) = ((v / 195112) % 58) as u8;
            *raw_p.add(10) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c5_3 as u32;
        unsafe {
            *raw_p.add(19) = (v % 58) as u8;
            *raw_p.add(18) = ((v / 58) % 58) as u8;
            *raw_p.add(17) = ((v / 3364) % 58) as u8;
            *raw_p.add(16) = ((v / 195112) % 58) as u8;
            *raw_p.add(15) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c4_4 as u32;
        unsafe {
            *raw_p.add(24) = (v % 58) as u8;
            *raw_p.add(23) = ((v / 58) % 58) as u8;
            *raw_p.add(22) = ((v / 3364) % 58) as u8;
            *raw_p.add(21) = ((v / 195112) % 58) as u8;
            *raw_p.add(20) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c3_5 as u32;
        unsafe {
            *raw_p.add(29) = (v % 58) as u8;
            *raw_p.add(28) = ((v / 58) % 58) as u8;
            *raw_p.add(27) = ((v / 3364) % 58) as u8;
            *raw_p.add(26) = ((v / 195112) % 58) as u8;
            *raw_p.add(25) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c2_6 as u32;
        unsafe {
            *raw_p.add(34) = (v % 58) as u8;
            *raw_p.add(33) = ((v / 58) % 58) as u8;
            *raw_p.add(32) = ((v / 3364) % 58) as u8;
            *raw_p.add(31) = ((v / 195112) % 58) as u8;
            *raw_p.add(30) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c1_7 as u32;
        unsafe {
            *raw_p.add(39) = (v % 58) as u8;
            *raw_p.add(38) = ((v / 58) % 58) as u8;
            *raw_p.add(37) = ((v / 3364) % 58) as u8;
            *raw_p.add(36) = ((v / 195112) % 58) as u8;
            *raw_p.add(35) = (v / 11316496) as u8;
        }
    }
    {
        let v: u32 = c0_8 as u32;
        unsafe {
            *raw_p.add(44) = (v % 58) as u8;
            *raw_p.add(43) = ((v / 58) % 58) as u8;
            *raw_p.add(42) = ((v / 3364) % 58) as u8;
            *raw_p.add(41) = ((v / 195112) % 58) as u8;
            *raw_p.add(40) = (v / 11316496) as u8;
        }
    }

    // count leading zero raw_base58 bytes
    let mut raw_leading_0s: usize = 0;
    while raw_leading_0s < 45 && raw_base58[raw_leading_0s] == 0 {
        raw_leading_0s += 1;
    }

    let skip = raw_leading_0s - in_leading_0s;
    let encoded_length = 45 - skip;

    let out_p = output.as_mut_ptr();
    let mut i = 0;
    while i < encoded_length {
        let idx = raw_base58[skip + i] as usize;
        let ch = alphabet_at(idx, case_insensitive);
        unsafe { *out_p.add(i) = ch; }
        i += 1;
    }
    *len = encoded_length as u32;
}

#[inline(always)]
fn alphabet_at(idx: usize, ci: bool) -> u8 {
    if ci {
        match idx {
             0 => b'1',  1 => b'2',  2 => b'3',  3 => b'4',  4 => b'5',
             5 => b'6',  6 => b'7',  7 => b'8',  8 => b'9',
             9 => b'a', 10 => b'b', 11 => b'c', 12 => b'd', 13 => b'e',
            14 => b'f', 15 => b'g', 16 => b'h', 17 => b'j', 18 => b'L',
            19 => b'm', 20 => b'n', 21 => b'p', 22 => b'q', 23 => b'r',
            24 => b's', 25 => b't', 26 => b'u', 27 => b'v', 28 => b'w',
            29 => b'x', 30 => b'y', 31 => b'z',
            32 => b'a', 33 => b'b', 34 => b'c', 35 => b'd', 36 => b'e',
            37 => b'f', 38 => b'g', 39 => b'h', 40 => b'i', 41 => b'j',
            42 => b'k', 43 => b'm', 44 => b'n', 45 => b'o', 46 => b'p',
            47 => b'q', 48 => b'r', 49 => b's', 50 => b't', 51 => b'u',
            52 => b'v', 53 => b'w', 54 => b'x', 55 => b'y', 56 => b'z',
            57 => b'z',  // technically 57 is the last char, fits z
            _ => 0,
        }
    } else {
        match idx {
             0 => b'1',  1 => b'2',  2 => b'3',  3 => b'4',  4 => b'5',
             5 => b'6',  6 => b'7',  7 => b'8',  8 => b'9',
             9 => b'A', 10 => b'B', 11 => b'C', 12 => b'D', 13 => b'E',
            14 => b'F', 15 => b'G', 16 => b'H', 17 => b'J', 18 => b'K',
            19 => b'L', 20 => b'M', 21 => b'N', 22 => b'P', 23 => b'Q',
            24 => b'R', 25 => b'S', 26 => b'T', 27 => b'U', 28 => b'V',
            29 => b'W', 30 => b'X', 31 => b'Y', 32 => b'Z',
            33 => b'a', 34 => b'b', 35 => b'c', 36 => b'd', 37 => b'e',
            38 => b'f', 39 => b'g', 40 => b'h', 41 => b'i', 42 => b'j',
            43 => b'k', 44 => b'm', 45 => b'n', 46 => b'o', 47 => b'p',
            48 => b'q', 49 => b'r', 50 => b's', 51 => b't', 52 => b'u',
            53 => b'v', 54 => b'w', 55 => b'x', 56 => b'y', 57 => b'z',
            _ => 0,
        }
    }
}

