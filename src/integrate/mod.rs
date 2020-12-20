/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use alga::general::ComplexField;
use num_traits::FromPrimitive;

/// Numerically integrate a function over an interval.
///
/// Given a function and end points, numerically integrate
/// using Romberg integration. Uses `n` steps.
pub fn integrate_fixed<N: ComplexField>(
    left: N::RealField,
    right: N::RealField,
    f: fn(N::RealField) -> N,
    n: usize,
) -> Result<N, String> {
    if left >= right {
        return Err("integrate: left must be less than right".to_owned());
    }

    let mut h = right - left;

    let mut prev_rows = vec![N::zero(); n];
    prev_rows[0] = N::from_real(h) * N::from_f64(0.5).unwrap() * (f(left) + f(right));
    let mut next = vec![N::zero(); n];

    for i in 2..=n {
        let mut acc = N::zero();
        for k in 1..=(1 << (i - 2)) {
            acc += f(left + N::RealField::from_f64(k as f64 - 0.5).unwrap() * h);
        }
        acc *= N::from_real(h);
        acc += prev_rows[0];
        acc *= N::from_f64(0.5).unwrap();
        next[0] = acc;
        for j in 2..=i {
            next[j - 1] = next[j - 2]
                + (next[j - 2] - prev_rows[j - 2])
                    / N::from_i32((4 as i32).pow(j as u32 - 1) - 1).unwrap();
        }

        h *= N::RealField::from_f64(0.5).unwrap();
        prev_rows[..i].clone_from_slice(&next[..i]);
    }

    Ok(prev_rows[n - 1])
}
