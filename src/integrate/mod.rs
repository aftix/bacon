/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::{ComplexField, RealField};
use num_traits::{FromPrimitive, One, Zero};
use std::f64;

mod gaussian;
pub use gaussian::*;
mod tables;

use tables::WEIGHTS_DE;

// Taken and modified from https://github.com/Eh2406/quadrature/blob/master/src/double_exponential/mod.rs
// published under the BSD license
fn integrate_core<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    let mut error_estimate = N::RealField::one() + tol;
    let mut num_function_evaluations = 1;
    let mut current_delta = N::RealField::zero();

    let half = N::from_f64(0.5).unwrap();
    let one_point_nine = N::from_f64(1.9).unwrap().real();
    let two_point_one = N::from_f64(2.1).unwrap().real();
    let pi = N::from_f64(f64::consts::PI).unwrap();

    let mut integral = pi * f(N::RealField::zero());

    for &weight in &WEIGHTS_DE {
        let new_contribution = weight
            .iter()
            .map(|&(w, x)| {
                let x = N::from_f64(x).unwrap().real();
                N::from_f64(w).unwrap() * (f(x) + f(-x))
            })
            .fold(N::zero(), |sum, x| sum + x);
        num_function_evaluations += 2 * weight.len();

        // difference in consecutive integral estimates
        let previous_delta_ln = current_delta.ln();
        current_delta = (half * integral - new_contribution).abs();
        integral = half * integral + new_contribution;

        // Once convergence kicks in, error is approximately squared at each step.
        // Determine whether we're in the convergent region by looking at the trend in the error.
        if num_function_evaluations <= 13 {
            // level <= 1
            continue; // previousDelta meaningless, so cannot check convergence.
        }

        // Exact comparison with zero is harmless here.  Could possibly be replaced with
        // a small positive upper limit on the size of currentDelta, but determining
        // that upper limit would be difficult.  At worse, the loop is executed more
        // times than necessary.  But no infinite loop can result since there is
        // an upper bound on the loop variable.
        if current_delta == N::RealField::zero() {
            error_estimate = N::RealField::zero();
            break;
        }
        // previousDelta != 0 or would have been kicked out previously
        let r = current_delta.ln() / previous_delta_ln;

        if r > one_point_nine && r < two_point_one {
            // If convergence theory applied perfectly, r would be 2 in the convergence region.
            // r close to 2 is good enough. We expect the difference between this integral estimate
            // and the next one to be roughly delta^2.
            error_estimate = current_delta * current_delta;
        } else {
            // Not in the convergence region.  Assume only that error is decreasing.
            error_estimate = current_delta;
        }

        if error_estimate < tol {
            break;
        }
    }

    if error_estimate < tol {
        Ok(integral)
    } else {
        Err("integrate: maximum iterations exceeded".to_owned())
    }
}

pub fn integrate<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    left: N::RealField,
    right: N::RealField,
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if left >= right {
        return Err("integrate: left must be less than right".to_owned());
    }
    if !tol.is_sign_positive() {
        return Err("integrate: tolerance must be positive".to_owned());
    }

    let half = N::from_f64(0.5).unwrap().real();

    let scale = (right - left) * half;
    let shift = (right + left) * half;
    let scale_cmplx = N::from_real(scale);

    let fun = |x: N::RealField| -> N {
        let out = f(scale * x + shift);
        if out.is_finite() {
            out
        } else {
            N::zero()
        }
    };

    Ok(integrate_core(fun, tol)? * scale_cmplx)
}

/// Numerically integrate a function over an interval within a tolerance.
///
/// Given a function and end points, numerically integrate using adaptive
/// simpson's rule until the error is within tolerance or the maximum
/// iterations are exceeded.
pub fn integrate_simpson<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    left: N::RealField,
    right: N::RealField,
    mut f: F,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if left >= right {
        return Err("integrate: left must be less than right".to_owned());
    }
    if !tol.is_sign_positive() {
        return Err("integrate: tolerance must be positive".to_owned());
    }

    let sixth = N::from_f64(1.0 / 6.0).unwrap();
    let third = N::from_f64(1.0 / 3.0).unwrap();
    let half_real = N::from_f64(0.5).unwrap().real();
    let one_and_a_half_real = N::from_f64(1.5).unwrap().real();
    let four = N::from_i32(4).unwrap();

    let mut area = N::zero();
    let mut i = 1;
    let mut tol_i = vec![N::from_i32(10).unwrap().real() * tol];
    let mut left_i = vec![left];
    let mut step_i = vec![(right - left) * half_real];
    let mut f_ai = vec![f(left)];
    let mut f_ci = vec![f(left + step_i[0])];
    let mut f_bi = vec![f(right)];
    let mut sum_i = vec![N::from_real(step_i[0]) * (f_ai[0] + four * f_ci[0] + f_bi[0]) * third];
    let mut l_i = vec![1];

    while i > 0 {
        let f_d = f(left_i[i - 1] + half_real * step_i[i - 1]);
        let f_e = f(left_i[i - 1] + one_and_a_half_real * step_i[i - 1]);
        let s1 = N::from_real(step_i[i - 1]) * (f_ai[i - 1] + four * f_d + f_ci[i - 1]) * sixth;
        let s2 = N::from_real(step_i[i - 1]) * (f_ci[i - 1] + four * f_e + f_bi[i - 1]) * sixth;
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
            if v_8 >= n_max {
                return Err("integrate: maximum iterations exceeded".to_owned());
            }

            i += 1;
            if i > left_i.len() {
                left_i.push(v_1 + v_5);
                f_ai.push(v_3);
                f_ci.push(f_e);
                f_bi.push(v_4);
                step_i.push(half_real * v_5);
                tol_i.push(half_real * v_6);
                sum_i.push(s2);
                l_i.push(v_8 + 1);
            } else {
                left_i[i - 1] = v_1 + v_5;
                f_ai[i - 1] = v_3;
                f_ci[i - 1] = f_e;
                f_bi[i - 1] = v_4;
                step_i[i - 1] = half_real * v_5;
                tol_i[i - 1] = half_real * v_6;
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
pub fn integrate_fixed<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    left: N::RealField,
    right: N::RealField,
    mut f: F,
    n: usize,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if left >= right {
        return Err("integrate_fixed: left must be less than right".to_owned());
    }

    let half = N::from_f64(0.5).unwrap();
    let half_real = N::from_f64(0.5).unwrap().real();
    let four = N::from_i32(4).unwrap();

    let mut h = right - left;

    let mut prev_rows = vec![N::zero(); n];
    prev_rows[0] = N::from_real(h) * half * (f(left) + f(right));
    let mut next = vec![N::zero(); n];

    for i in 2..=n {
        let mut acc = N::zero();
        for k in 1..=(1 << (i - 2)) {
            acc += f(left + N::from_f64(k as f64 - 0.5).unwrap().real() * h);
        }
        acc *= N::from_real(h);
        acc += prev_rows[0];
        acc *= half;
        next[0] = acc;
        for j in 2..=i {
            next[j - 1] = next[j - 2]
                + (next[j - 2] - prev_rows[j - 2]) / (four.powi(j as i32 - 1) - N::one());
        }

        h *= half_real;
        prev_rows[..i].clone_from_slice(&next[..i]);
    }

    Ok(prev_rows[n - 1])
}
