#!/usr/bin/env python3
"""Generates a partially-unrolled base58_encode_32 with the enc_table multiply-accumulates
inlined as constants. Output to stdout — redirect to src/kernels/base58.rs."""

ENC_TABLE = [
    [513735,    77223048,  437087610, 300156666, 605448490, 214625350, 141436834, 379377856],
    [0,         78508,     646269101, 118408823, 91512303,  209184527, 413102373, 153715680],
    [0,         0,         11997,     486083817, 3737691,   294005210, 247894721, 289024608],
    [0,         0,         0,         1833,      324463681, 385795061, 551597588, 21339008],
    [0,         0,         0,         0,         280,       127692781, 389432875, 357132832],
    [0,         0,         0,         0,         0,         42,        537767569, 410450016],
    [0,         0,         0,         0,         0,         0,         6,         356826688],
    [0,         0,         0,         0,         0,         0,         0,         1],
]

print("""use cuda_device::device;

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
""")

# load binary[0..8]
for i in range(8):
    print(f"    let bin{i}: u32 =")
    print(f"          (unsafe {{ *in_p.add({i*4}) }} as u32) << 24")
    print(f"        | (unsafe {{ *in_p.add({i*4+1}) }} as u32) << 16")
    print(f"        | (unsafe {{ *in_p.add({i*4+2}) }} as u32) << 8")
    print(f"        | (unsafe {{ *in_p.add({i*4+3}) }} as u32);")

print()
print("    // build intermediate[0..9] via inlined enc_table multiply-accumulates")
print("    // intermediate[0] is always 0 since enc_table only writes j+1 starting from j=0")
print("    let im0: u64 = 0;")

# for each output index i (1..9), sum binary[k] * enc_table[k][i-1] for k in 0..8
# intermediate[j+1] += binary[i] * enc_table[i][j]
# so intermediate[k] (k in 1..9) = sum over i of binary[i] * enc_table[i][k-1]
for k in range(1, 9):
    j = k - 1  # column in enc_table
    terms = []
    for i_row in range(8):
        coeff = ENC_TABLE[i_row][j]
        if coeff != 0:
            terms.append(f"bin{i_row} as u64 * {coeff}u64")
    if terms:
        expr = " + ".join(terms)
        print(f"    let im{k}: u64 = {expr};")
    else:
        print(f"    let im{k}: u64 = 0;")

print("""
    // carry propagation: i in 8..1, intermediate[i-1] += intermediate[i] / R, intermediate[i] %= R
    const R1: u64 = 656356768; // 58^5
""")

# unroll carry: (1..9).rev() gives i = 8, 7, 6, 5, 4, 3, 2, 1
# we need new values for each step; do it explicitly
prev = [f"im{k}" for k in range(9)]
cur = list(prev)
for step, i in enumerate(reversed(range(1, 9))):
    cur[i-1] = f"c{step}_{i-1}"
    cur[i] = f"c{step}_{i}"
    print(f"    let {cur[i-1]}: u64 = {prev[i-1]} + ({prev[i]} / R1);")
    print(f"    let {cur[i]}: u64 = {prev[i]} % R1;")
    prev = list(cur)

# now prev[k] for k in 0..9 are the final intermediate values
print()
print("    // extract base58 digits (raw_base58[0..45])")
print("    let mut raw_base58: [u8; 45] = [0; 45];")
print("    let raw_p = raw_base58.as_mut_ptr();")
for k in range(9):
    src = prev[k]
    print(f"    {{")
    print(f"        let v: u32 = {src} as u32;")
    print(f"        unsafe {{")
    print(f"            *raw_p.add({5*k + 4}) = (v % 58) as u8;")
    print(f"            *raw_p.add({5*k + 3}) = ((v / 58) % 58) as u8;")
    print(f"            *raw_p.add({5*k + 2}) = ((v / 3364) % 58) as u8;")
    print(f"            *raw_p.add({5*k + 1}) = ((v / 195112) % 58) as u8;")
    print(f"            *raw_p.add({5*k + 0}) = (v / 11316496) as u8;")
    print(f"        }}")
    print(f"    }}")

print("""
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
""")