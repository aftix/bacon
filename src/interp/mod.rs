/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::polynomial::Polynomial;
use nalgebra::ComplexField;
use num_traits::FromPrimitive;

mod spline;
pub use spline::*;

/// Create a Lagrange interpolating polynomial.
///
/// Create an nth degree polynomial matching the n points (xs[i], ys[i])
/// using Neville's iterated method for Lagrange polynomials. The result will
/// match no derivatives.
///
/// # Examples
/// ```
/// use bacon_sci::interp::lagrange;
/// use bacon_sci::polynomial::Polynomial;
/// fn example() {
///     let xs: Vec<_> = (0..10).map(|i| i as f64).collect();
///     let ys: Vec<_> = xs.iter().map(|x| x.cos()).collect();
///     let poly = lagrange(&xs, &ys, 1e-6).unwrap();
///     for x in xs {
///         assert!((x.cos() - poly.evaluate(x)).abs() < 0.00001);
///     }
/// }
/// ```
pub fn lagrange<N: ComplexField + FromPrimitive + Copy>(
    xs: &[N],
    ys: &[N],
    tol: N::RealField,
) -> Result<Polynomial<N>, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if xs.len() != ys.len() {
        return Err("lagrange: slices have mismatched dimension".to_owned());
    }

    let mut qs = vec![Polynomial::with_tolerance(tol)?; xs.len() * xs.len()];
    for (ind, y) in ys.iter().enumerate() {
        qs[ind] = polynomial![*y];
    }

    for i in 1..xs.len() {
        let mut poly_2 = polynomial![N::one(), -xs[i]];
        poly_2.set_tolerance(tol)?;
        for j in 1..=i {
            let mut poly_1 = polynomial![N::one(), -xs[i - j]];
            poly_1.set_tolerance(tol)?;
            let idenom = N::one() / (xs[i] - xs[i - j]);
            let numer =
                &poly_1 * &qs[i + xs.len() * (j - 1)] - &poly_2 * &qs[(i - 1) + xs.len() * (j - 1)];
            qs[i + xs.len() * j] = numer * idenom;
        }
    }

    for i in 0..=qs[xs.len() * xs.len() - 1].order() {
        if qs[xs.len() * xs.len() - 1].get_coefficient(i).abs() < tol {
            qs[xs.len() * xs.len() - 1].purge_coefficient(i);
        }
    }

    qs[xs.len() * xs.len() - 1].purge_leading();
    Ok(qs[xs.len() * xs.len() - 1].clone())
}

pub fn hermite<N: ComplexField + FromPrimitive + Copy>(
    xs: &[N],
    ys: &[N],
    derivs: &[N],
    tol: N::RealField,
) -> Result<Polynomial<N>, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if xs.len() != ys.len() {
        return Err("hermite: slices have mismatched dimension".to_owned());
    }
    if xs.len() != derivs.len() {
        return Err("hermite: derivatives have mismatched dimension".to_owned());
    }

    let mut zs = vec![N::zero(); 2 * xs.len()];
    let mut qs = vec![N::zero(); 4 * xs.len() * xs.len()];

    for i in 0..xs.len() {
        zs[2 * i] = xs[i];
        zs[2 * i + 1] = xs[i];
        qs[2 * i] = ys[i];
        qs[2 * i + 1] = ys[i];
        qs[2 * i + 1 + (2 * xs.len())] = derivs[i];

        if i != 0 {
            qs[2 * i + (2 * xs.len())] = (qs[2 * i] - qs[2 * i - 1]) / (zs[2 * i] - zs[2 * i - 1]);
        }
    }

    for i in 2..2 * xs.len() {
        for j in 2..=i {
            qs[i + j * (2 * xs.len())] = (qs[i + (j - 1) * (2 * xs.len())]
                - qs[i - 1 + (j - 1) * (2 * xs.len())])
                / (zs[i] - zs[i - j]);
        }
    }

    let mut hermite = polynomial![N::zero()];
    for i in (1..2 * xs.len()).rev() {
        hermite += qs[i + i * (2 * xs.len())];
        hermite *= polynomial![N::one(), -xs[(i - 1) / 2]];
    }
    hermite += qs[0];

    for i in 0..=hermite.order() {
        if hermite.get_coefficient(i).abs() < tol {
            hermite.purge_coefficient(i);
        }
    }

    hermite.purge_leading();
    Ok(hermite)
}
