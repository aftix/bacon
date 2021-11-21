use nalgebra::{ComplexField, RealField};
use num_traits::{FromPrimitive, One, Zero};

use super::tables::{
    WEIGHTS_CHEBYSHEV, WEIGHTS_CHEBYSHEV_SECOND, WEIGHTS_HERMITE, WEIGHTS_LAGUERRE,
    WEIGHTS_LEGENDRE,
};

fn integrate_gaussian_core<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    let mut prev_err = N::RealField::one() + tol;
    let mut prev_area = N::zero();

    for weights in WEIGHTS_LEGENDRE {
        let area = weights
            .iter()
            .map(|(x, w)| {
                if *x == 0.0 {
                    N::from_f64(*w).unwrap() * f(N::from_f64(*x).unwrap().real())
                } else {
                    let x = N::from_f64(*x).unwrap().real();
                    N::from_f64(*w).unwrap() * (f(x) + f(-x))
                }
            })
            .fold(N::zero(), |sum, x| x + sum);

        let err = (area - prev_area).abs();
        if err < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = err;
    }

    Err("integrate_gaussian: Maximum iterations exceeded".to_owned())
}

/// Numerically integrate a function over an interval within a tolerance.
///
/// Given a function and end points, numerically intgerate using Gaussian-Legedre
/// Quadrature until two consecutive iterations are within tolerance or the maximum
/// number of iterations is exceeded.
pub fn integrate_gaussian<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    left: N::RealField,
    right: N::RealField,
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if !tol.is_sign_positive() {
        return Err("integrate_gaussian: tol must be positive".to_owned());
    }

    let half_real = N::from_f64(0.5).unwrap().real();
    let scale = half_real * (right - left);
    let shift = half_real * (right + left);
    let scale_cmplx = N::from_real(scale);

    let fun = |x: N::RealField| f(scale * x + shift);
    Ok(
        integrate_gaussian_core(fun, N::from_f64(0.25).unwrap().real() * tol / scale)?
            * scale_cmplx,
    )
}

/// Numerically integrate an integral of the form int_0^inf f(x) exp(-x) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Gaussian-Laguerre
/// Quadrature until two consecutive iterations are within tolerance or
/// the maximum number of iterations is exceeded.
pub fn integrate_laguerre<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if !tol.is_sign_positive() {
        return Err("integrate_laguerre: tol must be positive".to_owned());
    }

    let mut prev_area = N::zero();
    let mut prev_err = N::RealField::one() + tol;

    for weight in WEIGHTS_LAGUERRE {
        let area = weight
            .iter()
            .map(|(z, w)| N::from_f64(*w).unwrap() * f(N::from_f64(*z).unwrap().real()))
            .fold(N::zero(), |sum, x| sum + x);

        let err = (area - prev_area).abs();
        if err < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = err;
    }

    Err("integrate_laguerre: maximum number of iterations exceeded".to_owned())
}

/// Numerically integrate an integral of the form int_-inf^inf f(x) exp(-x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Gaussian-Hermite
/// Quadrature until two consecutive iterations are within tolerance or
/// the maximum number of iterations is exceeded.
pub fn integrate_hermite<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if !tol.is_sign_positive() {
        return Err("integrate_hermite: tol must be positive".to_owned());
    }

    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    for weight in WEIGHTS_HERMITE {
        let area = weight
            .iter()
            .map(|(z, w)| {
                if *z == 0.0 {
                    N::from_f64(*w).unwrap() * f(N::RealField::zero())
                } else {
                    let x = N::from_f64(*z).unwrap().real();
                    N::from_f64(*w).unwrap() * (f(x) + f(-x))
                }
            })
            .fold(N::zero(), |sum, x| sum + x);

        let err = (area - prev_area).abs();
        if err < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = err;
    }

    Err("integrate_hermite: maximum iterations exceeded".to_owned())
}

/// Numerically integrate an integral of the form int_-1^1 f(x) / sqrt(1 - x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Chebyshev-Gaussian Quadrature
/// of the first kind until two consecutive iterations are within tolerance
/// or the maximum number of iterations is exceeded.
pub fn integrate_chebyshev<N: ComplexField + FromPrimitive + Copy, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if !tol.is_sign_positive() {
        return Err("integrate_chebyshev: tol must be positive".to_owned());
    }

    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    for weight in WEIGHTS_CHEBYSHEV {
        let area = weight
            .iter()
            .map(|(z, w)| {
                if *z == 0.0 {
                    N::from_f64(*w).unwrap() * f(N::RealField::zero())
                } else {
                    let x = N::from_f64(*z).unwrap().real();
                    N::from_f64(*w).unwrap() * (f(x) + f(-x))
                }
            })
            .fold(N::zero(), |sum, x| sum + x);

        let err = (area - prev_area).abs();
        if err < tol && prev_err < tol {
            return Ok(area);
        }

        prev_err = err;
        prev_area = area;
    }

    Err("integrate_chebyshev: maximum iterations exceeded".to_owned())
}

/// Numerically integrate an integral of the form int_-1^1 f(x) sqrt(1 - x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Chebyshev-Gaussian Quadrature
/// of the second kind until two consecutive iterations are within tolerance
/// or the maximum number of iterations is exceeded.
pub fn integrate_chebyshev_second<
    N: ComplexField + FromPrimitive + Copy,
    F: FnMut(N::RealField) -> N,
>(
    mut f: F,
    tol: N::RealField,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    if !tol.is_sign_positive() {
        return Err("integrate_chebyshev_second: tol must be positive".to_owned());
    }
    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    for weight in WEIGHTS_CHEBYSHEV_SECOND {
        let area = weight
            .iter()
            .map(|(z, w)| {
                if *z == 0.0 {
                    N::from_f64(*w).unwrap() * f(N::RealField::zero())
                } else {
                    let x = N::from_f64(*z).unwrap().real();
                    N::from_f64(*w).unwrap() * (f(x) + f(-x))
                }
            })
            .fold(N::zero(), |sum, x| sum + x);

        let err = (area - prev_area).abs();
        if err < tol && prev_err < tol {
            return Ok(area);
        }

        prev_err = err;
        prev_area = area;
    }

    Err("integrate_chebyshev: maximum iterations exceeded".to_owned())
}
