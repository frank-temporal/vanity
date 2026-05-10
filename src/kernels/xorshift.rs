// xorshift128+ helpers as macros so they expand at parse time and become part of
// the kernel's MIR body. The cuda-oxide codegen does not inline `#[inline(always)]`
// fns across module boundaries (its collector emits them as device-function calls
// in PTX), so we can't rely on rustc to inline them — macros are the workaround.

pub struct Xor {
    pub a: u64,
    pub b: u64,
}

/// init_xorshift!(seed_ptr_expr, idx_expr) -> Xor
#[macro_export]
macro_rules! init_xorshift {
    ($seed:expr, $idx:expr) => {{
        let __seed: *const u8 = $seed;
        let __idx: u64 = $idx;
        let k0 = unsafe { *(__seed.add(0) as *const u64) };
        let k1 = unsafe { *(__seed.add(8) as *const u64) };
        let k2 = unsafe { *(__seed.add(16) as *const u64) };
        let k3 = unsafe { *(__seed.add(24) as *const u64) };

        let mut z0 = k0 ^ k2;
        z0 = z0.wrapping_add(__idx);
        z0 = (z0 ^ (z0 >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z0 = (z0 ^ (z0 >> 27)).wrapping_mul(0x94d049bb133111eb);
        let a = z0 ^ (z0 >> 31);

        let mut z1 = k1 ^ k3;
        z1 = z1.wrapping_add(__idx).wrapping_add(0x9e3779b97f4a7c15);
        z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z1 = (z1 ^ (z1 >> 27)).wrapping_mul(0x94d049bb133111eb);
        let b = z1 ^ (z1 >> 31);

        $crate::kernels::xorshift::Xor { a, b }
    }};
}

/// xorshift128p!(x_mut_ref) -> u64
#[macro_export]
macro_rules! xorshift128p {
    ($x:expr) => {{
        let __x: &mut $crate::kernels::xorshift::Xor = $x;
        let mut s1 = __x.a;
        let s0 = __x.b;
        __x.a = s0;
        s1 ^= s1 << 23;
        __x.b = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        __x.b.wrapping_add(s0)
    }};
}
