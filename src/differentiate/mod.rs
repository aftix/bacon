/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::ComplexField;
use num_traits::FromPrimitive;

/// Numerically find the derivative of a function at a point.
///
/// Given a function of a real field to a complex field, an input point x, and a
/// step size h, calculate the derivative of f at x using the five-point method.
///
/// Making h too small will lead to round-off error.
pub fn derivative<N: ComplexField + FromPrimitive + Copy>(
    f: impl Fn(N::RealField) -> N,
    x: N::RealField,
    h: N::RealField,
) -> N
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    (f(x - h - h) + N::from_f64(8.0).unwrap() * (f(x + h) - f(x - h)) - f(x + h + h))
        / (N::from_f64(12.0).unwrap() * N::from_real(h))
}

/// Numerically find the second derivative of a function at a point.
///
/// Given a function of a real field to a complex field, an input point x, and a
/// step size h, calculate the second derivative of f at x using the five-point
/// method.
///
/// Making h too small will lead to round-off error.
pub fn second_derivative<N: ComplexField + FromPrimitive + Copy>(
    f: impl Fn(N::RealField) -> N,
    x: N::RealField,
    h: N::RealField,
) -> N
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    (f(x - h) - N::from_f64(2.0).unwrap() * f(x) + f(x + h)) / N::from_real(h.powi(2))
}
