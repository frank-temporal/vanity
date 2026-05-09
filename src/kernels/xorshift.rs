use cuda_device::device;

pub struct Xor {
    a: u64,
    b: u64,
}

#[device]
pub fn init_xorshift(seed: *const u8, idx: u64) -> Xor {
    let k0 = unsafe { *(seed.add(0) as *const u64) };
    let k1 = unsafe { *(seed.add(8) as *const u64) };
    let k2 = unsafe { *(seed.add(16) as *const u64) };
    let k3 = unsafe { *(seed.add(24) as *const u64) };

    let mut z0 = k0 ^ k2;
    z0 = z0 + idx;
    z0 = (z0 ^ (z0 >> 30)) * 0xbf58476d1ce4e5b9;
    z0 = (z0 ^ (z0 >> 27)) * 0x94d049bb133111eb;

    let a = z0 ^ (z0 >> 31);

    let mut z1 = k1 ^ k3;
    z1 = z1 + idx + 0x9e3779b97f4a7c15;
    z1 = (z1 ^ (z1 >> 30)) * 0xbf58476d1ce4e5b9;
    z1 = (z1 ^ (z1 >> 27)) * 0x94d049bb133111eb;

    let b = z1 ^ (z1 >> 31);

    Xor { a, b }
}

#[device]
pub fn xorshift128p(x: &mut Xor) -> u64 {
    let mut s1 = x.a;
    let s0 = x.b;
    x.a = s0;
    s1 ^= s1 << 23;
    x.b = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    x.b.wrapping_add(s0)
}
