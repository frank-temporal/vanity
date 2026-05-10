#!/usr/bin/env python3
"""Generates fully-unrolled sha256_transform AND sha256_80 with constants inlined.
Outputs to stdout — redirect to src/kernels/sha256.rs"""

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

INIT_STATE = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def emit_load_m(prefix, byte_expr_for_idx):
    """emit `let m{prefix}{i:02} = ...` for i in 0..16, loading u32s big-endian.
    byte_expr_for_idx(byte_offset) returns a string for that byte expression."""
    for i in range(16):
        b0, b1, b2, b3 = i*4, i*4+1, i*4+2, i*4+3
        print(f"    let m{prefix}{i:02}: u32 =")
        print(f"          (({byte_expr_for_idx(b0)}) as u32) << 24")
        print(f"        | (({byte_expr_for_idx(b1)}) as u32) << 16")
        print(f"        | (({byte_expr_for_idx(b2)}) as u32) << 8")
        print(f"        | (({byte_expr_for_idx(b3)}) as u32);")


def emit_extend_m(prefix):
    """emit message-schedule extension m{prefix}16 .. m{prefix}63"""
    for i in range(16, 64):
        print(f"    let m{prefix}{i:02}: u32 = "
              f"sig1(m{prefix}{i-2:02})"
              f".wrapping_add(m{prefix}{i-7:02})"
              f".wrapping_add(sig0(m{prefix}{i-15:02}))"
              f".wrapping_add(m{prefix}{i-16:02});")


def emit_compress(prefix, in_state_vars, out_state_vars):
    """emit 64 compression rounds reading m{prefix}NN.
    in_state_vars/out_state_vars are 8-tuples of var names for state in/out."""
    a, b, c, d, e, f, g, h = in_state_vars
    print(f"    let mut a = {a};")
    print(f"    let mut b = {b};")
    print(f"    let mut c = {c};")
    print(f"    let mut d = {d};")
    print(f"    let mut e = {e};")
    print(f"    let mut f = {f};")
    print(f"    let mut g = {g};")
    print(f"    let mut h = {h};")
    for i in range(64):
        print(f"    // round {i}")
        print(f"    {{")
        print(f"        let t1 = h.wrapping_add(ep1(e)).wrapping_add(ch(e, f, g))"
              f".wrapping_add(0x{K[i]:08x}u32).wrapping_add(m{prefix}{i:02});")
        print(f"        let t2 = ep0(a).wrapping_add(maj(a, b, c));")
        print(f"        h = g; g = f; f = e; e = d.wrapping_add(t1); "
              f"d = c; c = b; b = a; a = t1.wrapping_add(t2);")
        print(f"    }}")
    oa, ob, oc, od, oe, of, og, oh = out_state_vars
    print(f"    let {oa} = {a}.wrapping_add(a);")
    print(f"    let {ob} = {b}.wrapping_add(b);")
    print(f"    let {oc} = {c}.wrapping_add(c);")
    print(f"    let {od} = {d}.wrapping_add(d);")
    print(f"    let {oe} = {e}.wrapping_add(e);")
    print(f"    let {of} = {f}.wrapping_add(f);")
    print(f"    let {og} = {g}.wrapping_add(g);")
    print(f"    let {oh} = {h}.wrapping_add(h);")


print("""use cuda_device::device;

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
""")
emit_load_m("", lambda b: f"unsafe {{ *bp.add({b}) }}")
emit_extend_m("")
in_state = ("state[0]", "state[1]", "state[2]", "state[3]",
            "state[4]", "state[5]", "state[6]", "state[7]")
out_state = ("ns0", "ns1", "ns2", "ns3", "ns4", "ns5", "ns6", "ns7")
emit_compress("", in_state, out_state)
print("""
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
""")

# === sha256_80: hashes base(32) || seed(16) || owner(32) = 80 bytes
# block0 = base[0..32] || seed[0..16] || owner[0..16]   bytes 0..63
# block1 = owner[16..32] || 0x80 || zeros... || bitlen_be(640)
print(f"""
/// Fully-unrolled sha256 of base(32) || seed(16) || owner(32) — 80 bytes total.
#[device]
pub fn sha256_80(base: &[u8; 32], seed: &[u8; 16], owner: &[u8; 32], hash: &mut [u8; 32]) {{
    let bp = base.as_ptr();
    let sp = seed.as_ptr();
    let op = owner.as_ptr();
""")

# block 0 byte expressions:
#   bytes 0..32   = base[i]
#   bytes 32..48  = seed[i-32]
#   bytes 48..64  = owner[i-48]
def block0_byte(b):
    if b < 32:
        return f"unsafe {{ *bp.add({b}) }}"
    elif b < 48:
        return f"unsafe {{ *sp.add({b - 32}) }}"
    else:
        return f"unsafe {{ *op.add({b - 48}) }}"

emit_load_m("a", block0_byte)
emit_extend_m("a")

in_state_a = (f"0x{INIT_STATE[i]:08x}u32" for i in range(8))
in_state_a = tuple(in_state_a)
out_state_a = ("a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7")
emit_compress("a", in_state_a, out_state_a)

# block 1 byte expressions:
#   bytes 0..16   = owner[16 + i]
#   byte 16       = 0x80
#   bytes 17..56  = 0
#   bytes 56..64  = bitlen_be(640) = 0x00 00 00 00 00 00 02 80
#                                  byte: 56=0x00 57=0x00 58=0x00 59=0x00
#                                        60=0x00 61=0x00 62=0x02 63=0x80
BITLEN_BYTES = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x80]
def block1_byte(b):
    if b < 16:
        return f"unsafe {{ *op.add({b + 16}) }}"
    elif b == 16:
        return "0x80u8"
    elif b < 56:
        return "0u8"
    else:
        return f"0x{BITLEN_BYTES[b - 56]:02x}u8"

emit_load_m("b", block1_byte)
emit_extend_m("b")

in_state_b = ("a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7")
out_state_b = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")
emit_compress("b", in_state_b, out_state_b)

# emit final hash bytes from b0..b7 (big-endian)
print("""
    let hp = hash.as_mut_ptr();
    unsafe {""")
for word_idx, var in enumerate(("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")):
    for byte_in_word in range(4):
        out_idx = word_idx * 4 + byte_in_word
        shift = 24 - byte_in_word * 8
        print(f"        *hp.add({out_idx}) = ({var} >> {shift}) as u8;")
print("""    }
}
""")