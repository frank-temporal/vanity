use cuda_device::device;

#[device]
fn enc_table(row: usize, col: usize) -> u32 {
    // row in 0..8, col in 0..8 → returns enc_table_32[row][col]
    match row * 8 + col {
        // row 0
        0 => 513735, 1 => 77223048, 2 => 437087610, 3 => 300156666,
        4 => 605448490, 5 => 214625350, 6 => 141436834, 7 => 379377856,
        // row 1
        8 => 0, 9 => 78508, 10 => 646269101, 11 => 118408823,
        12 => 91512303, 13 => 209184527, 14 => 413102373, 15 => 153715680,
        // row 2
        16 => 0, 17 => 0, 18 => 11997, 19 => 486083817,
        20 => 3737691, 21 => 294005210, 22 => 247894721, 23 => 289024608,
        // row 3
        24 => 0, 25 => 0, 26 => 0, 27 => 1833,
        28 => 324463681, 29 => 385795061, 30 => 551597588, 31 => 21339008,
        // row 4
        32 => 0, 33 => 0, 34 => 0, 35 => 0,
        36 => 280, 37 => 127692781, 38 => 389432875, 39 => 357132832,
        // row 5
        40 => 0, 41 => 0, 42 => 0, 43 => 0,
        44 => 0, 45 => 42, 46 => 537767569, 47 => 410450016,
        // row 6
        48 => 0, 49 => 0, 50 => 0, 51 => 0,
        52 => 0, 53 => 0, 54 => 6, 55 => 356826688,
        // row 7
        56 => 0, 57 => 0, 58 => 0, 59 => 0,
        60 => 0, 61 => 0, 62 => 0, 63 => 1,
        _ => 0,
    }
}

#[device]
pub fn base58_encode_32(input: &[u8; 32], output: &mut [u8; 44], len: &mut u32, case_insensitive: bool) {
    let alphabet: &[u8; 58] = if case_insensitive {
        b"123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz"
    } else {
        b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    };

    let mut in_leading_0s = 0usize;
    while in_leading_0s < 32 && input[in_leading_0s] == 0 {
        in_leading_0s += 1;
    }

    let mut binary = [0u32; 8];
    for i in 0..8 {
        binary[i] = (input[i*4]   as u32) << 24
                  | (input[i*4+1] as u32) << 16
                  | (input[i*4+2] as u32) << 8
                  | (input[i*4+3] as u32);
    }

    let r1div: u64 = 656356768; // 58^5

    let mut intermediate = [0u64; 9];
    for i in 0..8 {
        for j in 0..8 {
            intermediate[j + 1] += binary[i] as u64 * enc_table(i, j) as u64;
        }
    }

    for i in (1..9).rev() {
        intermediate[i - 1] += intermediate[i] / r1div;
        intermediate[i] %= r1div;
    }

    let mut raw_base58 = [0u8; 45];
    let raw_p = raw_base58.as_mut_ptr();
    for i in 0..9 {
        let v = intermediate[i] as u32;
        unsafe {
            *raw_p.add(5*i + 4) = (v % 58) as u8;
            *raw_p.add(5*i + 3) = ((v / 58) % 58) as u8;
            *raw_p.add(5*i + 2) = ((v / 3364) % 58) as u8;
            *raw_p.add(5*i + 1) = ((v / 195112) % 58) as u8;
            *raw_p.add(5*i + 0) = (v / 11316496) as u8;
        }
    }

    let mut raw_leading_0s = 0usize;
    while raw_leading_0s < 45 && raw_base58[raw_leading_0s] == 0 {
        raw_leading_0s += 1;
    }

    let skip = raw_leading_0s - in_leading_0s;
    let encoded_length = 45 - skip;

    let out_p = output.as_mut_ptr();
    for i in 0..encoded_length {
        unsafe { *out_p.add(i) = alphabet[raw_base58[skip + i] as usize]; }
    }
    *len = encoded_length as u32;
}