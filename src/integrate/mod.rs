/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use alga::general::{ComplexField, RealField};
use num_traits::FromPrimitive;

/// Numerically integrate a function over an interval within a tolerance.
///
/// Given a function and end points, numerically integrate using adaptive
/// simpson's rule until the error is within tolerance or the maximum
/// iterations are exceeded.
pub fn integrate<N: ComplexField>(
    left: N::RealField,
    right: N::RealField,
    f: fn(N::RealField) -> N,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String> {
    if left >= right {
        return Err("integrate: left must be less than right".to_owned());
    }
    if !tol.is_sign_positive() {
        return Err("integrate: tolerance must be positive".to_owned());
    }
    let mut area = N::zero();
    let mut i = 1;
    let mut tol_i = vec![N::RealField::from_i32(10).unwrap() * tol];
    let mut left_i = vec![left];
    let mut step_i = vec![(right - left) * N::RealField::from_f64(0.5).unwrap()];
    let mut f_ai = vec![f(left)];
    let mut f_ci = vec![f(left + step_i[0])];
    let mut f_bi = vec![f(right)];
    let mut sum_i = vec![
        N::from_real(step_i[0])
            * (f_ai[0] + N::from_i32(4).unwrap() * f_ci[0] + f_bi[0])
            * N::from_f64(1.0 / 3.0).unwrap(),
    ];
    let mut l_i = vec![1];

    while i > 0 {
        let f_d = f(left_i[i - 1] + N::RealField::from_f64(0.5).unwrap() * step_i[i - 1]);
        let f_e = f(left_i[i - 1] + N::RealField::from_f64(1.5).unwrap() * step_i[i - 1]);
        let s1 = N::from_real(step_i[i - 1])
            * (f_ai[i - 1] + N::from_i32(4).unwrap() * f_d + f_ci[i - 1])
            * N::from_f64(1.0 / 6.0).unwrap();
        let s2 = N::from_real(step_i[i - 1])
            * (f_ci[i - 1] + N::from_i32(4).unwrap() * f_e + f_bi[i - 1])
            * N::from_f64(1.0 / 6.0).unwrap();
        let v_1 = left_i[i - 1];
        let v_2 = f_ai[i - 1];
        let v_3 = f_ci[i - 1];
        let v_4 = f_bi[i - 1];
        let v_5 = step_i[i - 1];
        let v_6 = tol_i[i - 1];
        let v_7 = sum_i[i - 1];
        let v_8 = l_i[i - 1];

        i -= 1;

        if (s1 + s2 - v_7).abs() < v_6 {
            area += s1 + s2;
        } else {
            println!("{} {}", (s1 + s2 - v_7).abs(), v_6);
            if v_8 >= n_max {
                println!("{}", v_8);
                return Err("integrate: maximum iterations exceeded".to_owned());
            }

            i += 1;
            if i > left_i.len() {
                left_i.push(v_1 + v_5);
                f_ai.push(v_3);
                f_ci.push(f_e);
                f_bi.push(v_4);
                step_i.push(N::RealField::from_f64(0.5).unwrap() * v_5);
                tol_i.push(N::RealField::from_f64(0.5).unwrap() * v_6);
                sum_i.push(s2);
                l_i.push(v_8 + 1);
            } else {
                left_i[i - 1] = v_1 + v_5;
                f_ai[i - 1] = v_3;
                f_ci[i - 1] = f_e;
                f_bi[i - 1] = v_4;
                step_i[i - 1] = N::RealField::from_f64(0.5).unwrap() * v_5;
                tol_i[i - 1] = N::RealField::from_f64(0.5).unwrap() * v_6;
                sum_i[i - 1] = s2;
                l_i[i - 1] = v_8 + 1;
            }

            i += 1;
            if i > left_i.len() {
                left_i.push(v_1);
                f_ai.push(v_2);
                f_ci.push(f_d);
                f_bi.push(v_3);
                step_i.push(step_i[i - 2]);
                tol_i.push(tol_i[i - 2]);
                sum_i.push(s1);
                l_i.push(l_i[i - 2]);
            } else {
                left_i[i - 1] = v_1;
                f_ai[i - 1] = v_2;
                f_ci[i - 1] = f_d;
                f_bi[i - 1] = v_3;
                step_i[i - 1] = step_i[i - 2];
                tol_i[i - 1] = tol_i[i - 2];
                sum_i[i - 1] = sum_i[i - 2];
                l_i[i - 1] = l_i[i - 2];
            }
        }
    }

    Ok(area)
}

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
        return Err("integrate_fixed: left must be less than right".to_owned());
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
