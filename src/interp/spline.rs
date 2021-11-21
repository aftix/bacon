/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::polynomial::Polynomial;
use nalgebra::{ComplexField, RealField};
use num_traits::FromPrimitive;

#[derive(Debug, Clone)]
pub struct CubicSpline<N: ComplexField + FromPrimitive + Copy>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    cubics: Vec<Polynomial<N>>,
    ranges: Vec<(N::RealField, N::RealField)>,
}

impl<N: ComplexField + FromPrimitive + Copy> CubicSpline<N>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    pub fn evaluate(&self, x: N::RealField) -> Result<N, String> {
        if self.cubics.is_empty() {
            return Err("CubicSpline evaluate: Empty spline".to_owned());
        }

        for (ind, range) in self.ranges.iter().enumerate() {
            if x >= range.0 && x <= range.1 {
                return Ok(self.cubics[ind].evaluate(N::from_real(x)));
            }
        }

        Err(format!("CubicSpline evaluate: {} outside of range", x))
    }

    pub fn evaluate_derivative(&self, x: N::RealField) -> Result<(N, N), String> {
        if self.cubics.is_empty() {
            return Err("CubicSpline evaluate: Empty spline".to_owned());
        }

        for (ind, range) in self.ranges.iter().enumerate() {
            if x >= range.0 && x <= range.1 {
                return Ok(self.cubics[ind].evaluate_derivative(N::from_real(x)));
            }
        }

        Err(format!("CubicSpline evaluate: {} outside of range", x))
    }
}

/// Create a free cubic spline interpolating the given points.
///
/// Given a set of ordered points, produce a piecewise function of
/// cubic polynomials that interpolate the points given the second derivative
/// of the piecewise function at the end points is zero and the piecewise function
/// is smooth.
///
/// # Params
/// `xs` x points. Must be real because cubic splines keep track of ranges within
/// which it interpolates. Must be sorted.
///
/// `ys` y points. Can be complex. ys[i] must match with xs[i].
///
/// `tol` the tolerance of the polynomials
///
/// # Examples
/// ```
/// use bacon_sci::interp::spline_free;
/// fn example() {
///     let xs: Vec<_> = (0..=10).map(|x| x as f64).collect();
///     let ys: Vec<_> = xs.iter().map(|x| x.exp()).collect();
///
///     let spline = spline_free(&xs, &ys, 1e-8).unwrap();
///     for i in 0..1000 {
///         let i = i as f64 * 0.001;
///         assert!((spline.evaluate(i).unwrap() - i.exp()).abs() / i.exp() < 0.25);
///     }
/// }
/// ```
pub fn spline_free<N: ComplexField + FromPrimitive + Copy>(
    xs: &[N::RealField],
    ys: &[N],
    tol: N::RealField,
) -> Result<CubicSpline<N>, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if xs.len() != ys.len() {
        return Err("spline_free: xs and ys must be same length".to_owned());
    }
    if xs.len() < 2 {
        return Err("spline_free: need at least two points".to_owned());
    }

    let mut hs = Vec::with_capacity(xs.len() - 1);
    for i in 0..xs.len() - 1 {
        hs.push(xs[i + 1] - xs[i]);
    }

    if hs.iter().any(|h| !h.is_sign_positive()) {
        return Err("spline_free: xs must be sorted".to_owned());
    }

    let third = N::from_f64(1.0 / 3.0).unwrap();
    let two = N::from_i32(2).unwrap();
    let three = N::from_i32(3).unwrap();

    let mut alphas = Vec::with_capacity(xs.len() - 1);
    alphas.push(N::zero());
    for i in 1..xs.len() - 1 {
        alphas.push(
            (three / N::from_real(hs[i])) * (ys[i + 1] - ys[i])
                - (three / N::from_real(hs[i - 1])) * (ys[i] - ys[i - 1]),
        );
    }

    let mut l = Vec::with_capacity(xs.len() - 1);
    let mut mu = Vec::with_capacity(xs.len() - 1);
    let mut z = Vec::with_capacity(xs.len() - 1);

    l.push(N::one());
    mu.push(N::zero());
    z.push(N::zero());

    for i in 1..xs.len() - 1 {
        l.push(two * N::from_real(xs[i + 1] - xs[i - 1]) - N::from_real(hs[i - 1]) * mu[i - 1]);
        mu.push(N::from_real(hs[i]) / l[i]);
        z.push((alphas[i] - N::from_real(hs[i - 1]) * z[i - 1]) / l[i]);
    }

    l.push(N::one());
    z.push(N::zero());

    let mut c_coefficient = vec![N::zero(); xs.len()];
    let mut b_coefficient = vec![N::zero(); xs.len()];
    let mut d_coefficient = vec![N::zero(); xs.len()];
    for i in (0..xs.len() - 1).rev() {
        c_coefficient[i] = z[i] - mu[i] * c_coefficient[i + 1];
        b_coefficient[i] = (ys[i + 1] - ys[i]) / N::from_real(hs[i])
            - N::from_real(hs[i]) * (c_coefficient[i + 1] + two * c_coefficient[i]) * third;
        d_coefficient[i] =
            (c_coefficient[i + 1] - c_coefficient[i]) / (three * N::from_real(hs[i]));
    }

    let mut polynomials = Vec::with_capacity(xs.len() - 1);
    let mut ranges = Vec::with_capacity(xs.len() - 1);

    for i in 0..xs.len() - 1 {
        // Horner's method to build polynomial
        let term = polynomial![N::one(), N::from_real(-xs[i])];
        let mut poly = &term * d_coefficient[i];
        poly.set_tolerance(tol)?;
        poly += c_coefficient[i];
        poly *= &term;
        poly += b_coefficient[i];
        poly *= term;
        poly += ys[i];
        polynomials.push(poly);
        ranges.push((xs[i], xs[i + 1]));
    }

    Ok(CubicSpline {
        cubics: polynomials,
        ranges,
    })
}

/// Create a clamped cubic spline interpolating the given points.
///
/// Given a set of ordered points, produce a piecewise function of
/// cubic polynomials that interpolate the points given the first derivative
/// of the piecewise function at the end points is the same as the given values
/// and the piecewise function is smooth.
///
/// # Params
/// `xs` x points. Must be real because cubic splines keep track of ranges within
/// which it interpolates. Must be sorted.
///
/// `ys` y points. Can be complex. ys[i] must match with xs[i].
///
/// `(f_0, f_n)` The derivative values at the end points.
///
/// `tol` the tolerance of the polynomials
///
/// # Examples
/// ```
/// use bacon_sci::interp::spline_clamped;
/// fn example() {
///     let xs: Vec<_> = (0..=10).map(|x| x as f64).collect();
///     let ys: Vec<_> = xs.iter().map(|x| x.exp()).collect();
///
///     let spline = spline_clamped(&xs, &ys, (1.0, (10.0f64).exp()), 1e-8).unwrap();
///     for i in 0..1000 {
///         let i = i as f64 * 0.001;
///         assert!((spline.evaluate(i).unwrap() - i.exp()).abs() / i.exp() < 0.25);
///     }
/// }
/// ```
pub fn spline_clamped<N: ComplexField + FromPrimitive + Copy>(
    xs: &[N::RealField],
    ys: &[N],
    (f_0, f_n): (N, N),
    tol: N::RealField,
) -> Result<CubicSpline<N>, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if xs.len() != ys.len() {
        return Err("spline_clamped: xs and ys must be same length".to_owned());
    }
    if xs.len() < 2 {
        return Err("spline_clamped: need at least two points".to_owned());
    }

    let mut hs = Vec::with_capacity(xs.len() - 1);
    for i in 0..xs.len() - 1 {
        hs.push(xs[i + 1] - xs[i]);
    }

    if hs.iter().any(|h| !h.is_sign_positive()) {
        return Err("spline_clamped: xs must be sorted".to_owned());
    }

    let third = N::from_f64(1.0 / 3.0).unwrap();
    let half = N::from_f64(0.5).unwrap();
    let two = N::from_i32(2).unwrap();
    let three = N::from_i32(3).unwrap();

    let mut alphas = vec![N::zero(); xs.len()];
    alphas[0] = three * ((ys[1] - ys[0]) / N::from_real(hs[0]) - f_0);
    alphas[xs.len() - 1] =
        three * (f_n - (ys[xs.len() - 1] - ys[xs.len() - 2]) / N::from_real(hs[xs.len() - 2]));

    for i in 1..xs.len() - 1 {
        alphas[i] = three
            * ((ys[i + 1] - ys[i]) / N::from_real(hs[i])
                - (ys[i] - ys[i - 1]) / N::from_real(hs[i - 1]));
    }

    let mut l = Vec::with_capacity(xs.len() - 1);
    let mut mu = Vec::with_capacity(xs.len() - 1);
    let mut z = Vec::with_capacity(xs.len() - 1);

    l.push(two * N::from_real(hs[0]));
    mu.push(half);
    z.push(alphas[0] / l[0]);

    for i in 1..xs.len() - 1 {
        l.push(two * N::from_real(xs[i + 1] - xs[i - 1]) - N::from_real(hs[i - 1]) * mu[i - 1]);
        mu.push(N::from_real(hs[i]) / l[i]);
        z.push((alphas[i] - N::from_real(hs[i - 1]) * z[i - 1]) / l[i]);
    }

    l.push(N::from_real(hs[xs.len() - 2]) * (two - mu[xs.len() - 2]));
    z.push(
        (alphas[xs.len() - 1] - N::from_real(hs[xs.len() - 2]) * z[xs.len() - 2]) / l[xs.len() - 1],
    );

    let mut b_coefficient = vec![N::zero(); xs.len()];
    let mut c_coefficient = vec![N::zero(); xs.len()];
    let mut d_coefficient = vec![N::zero(); xs.len()];

    c_coefficient[xs.len() - 1] = z[xs.len() - 1];

    for i in (0..xs.len() - 1).rev() {
        c_coefficient[i] = z[i] - mu[i] * c_coefficient[i + 1];
        b_coefficient[i] = (ys[i + 1] - ys[i]) / N::from_real(hs[i])
            - N::from_real(hs[i]) * third * (c_coefficient[i + 1] + two * c_coefficient[i]);
        d_coefficient[i] =
            (c_coefficient[i + 1] - c_coefficient[i]) / (three * N::from_real(hs[i]));
    }

    let mut polynomials = Vec::with_capacity(xs.len() - 1);
    let mut ranges = Vec::with_capacity(xs.len() - 1);

    for i in 0..xs.len() - 1 {
        // Horner's method to build polynomial
        let term = polynomial![N::one(), N::from_real(-xs[i])];
        let mut poly = &term * d_coefficient[i];
        poly.set_tolerance(tol)?;
        poly += c_coefficient[i];
        poly *= &term;
        poly += b_coefficient[i];
        poly *= term;
        poly += ys[i];
        polynomials.push(poly);
        ranges.push((xs[i], xs[i + 1]));
    }

    Ok(CubicSpline {
        cubics: polynomials,
        ranges,
    })
}
