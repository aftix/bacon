use alga::general::ComplexField;
use num_traits::{FromPrimitive, One};
use std::f64;
use std::iter::FromIterator;

/// Numerically integrate a function over an interval within a tolerance.
///
/// Given a function and end points, numerically intgerate using Gaussian-Legedre
/// Quadrature until two consecutive iterations are within tolerance or the maximum
/// number of iterations is exceeded.
pub fn integrate_gaussian<N: ComplexField, F: FnMut(N::RealField) -> N>(
    left: N::RealField,
    right: N::RealField,
    mut f: F,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String> {
    let mut p_0 = polynomial![N::one()];
    let mut p_1 = polynomial![N::one(), N::zero()];

    let half_real = N::RealField::from_f64(0.5).unwrap();
    let two = N::from_i32(2).unwrap();

    let scale = half_real * (right - left);
    let shift = half_real * (right + left);

    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    let mut i: u32 = 1;
    while i < n_max as u32 {
        let mut p_next = polynomial![N::from_u32(2 * i + 1).unwrap(), N::zero()] * &p_1;
        p_next -= &p_0 * N::from_u32(i).unwrap();
        p_next /= N::from_u32(i + 1).unwrap();

        p_0 = p_1;
        p_1 = p_next;
        let zeros = Vec::from_iter(p_1.roots(tol, n_max)?.iter().map(|c| N::from_real(c.re)));

        let mut area = N::zero();
        for zero in &zeros {
            let (_, deriv) = p_1.evaluate_derivative(*zero);
            let weight = two / ((N::one() - zero.powi(2)) * deriv.powi(2));
            area += weight * f(scale * zero.real() + shift);
        }
        area *= N::from_real(shift);

        let error = (area - prev_area).abs();
        if error < tol && prev_err < tol {
            return Ok(area);
        }

        prev_err = error;
        prev_area = area;
        i += 1;
    }

    Err("integrate_gaussian: maximum iterations exceeded".to_owned())
}

fn factorial(n: u32) -> u32 {
    let mut acc = 1;
    for i in 2..=n {
        acc *= i;
    }
    acc
}

/// Numerically integrate an integral of the form int_-inf^inf f(x) exp(-x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Gaussian-Hermite
/// Quadrature until two consecutive iterations are within tolerance or
/// the maximum number of iterations is exceeded.
pub fn integrate_hermite<N: ComplexField, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String> {
    let mut h_0 = polynomial![N::one()];
    let mut h_1 = polynomial![N::from_i32(2).unwrap(), N::zero()];
    let x_2 = h_1.clone();

    let sqrt_pi = N::from_f64(f64::consts::PI.sqrt()).unwrap();

    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    let mut i: u32 = 1;
    while i < n_max as u32 {
        let p_next = &x_2 * &h_1 - (&h_0 * N::from_u32(2 * i).unwrap());
        h_0 = h_1;
        h_1 = p_next;
        i += 1;

        let roots = Vec::from_iter(h_1.roots(tol, n_max)?.iter().map(|c| {
            if c.re.abs() < tol {
                N::zero()
            } else {
                N::from_real(c.re)
            }
        }));

        let mut area = N::zero();
        let two_power = N::from_u32(1 << (i - 1)).unwrap();
        for root in &roots {
            let weight = N::from_u32(factorial(i)).unwrap() * sqrt_pi * two_power
                / (N::from_u32(i.pow(2)).unwrap() * h_0.evaluate(*root).powi(2));
            area += weight * f(root.real());
        }

        let error = (area - prev_area).abs();
        if error < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = error;
    }

    Err("integrate_hermite: maximum iterations exceeded".to_owned())
}

/// Numerically integrate an integral of the form int_-1^1 f(x) / sqrt(1 - x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Chebyshev-Gaussian Quadrature
/// of the first kind until two consecutive iterations are within tolerance
/// or the maximum number of iterations is exceeded.
pub fn integrate_chebyshev<N: ComplexField, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String> {
    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    let mut i: u32 = 2;
    while i < n_max as u32 {
        let mut area = N::zero();
        let denom = 1.0 / (2 * i) as f64;
        let weight = N::from_f64(f64::consts::PI / i as f64).unwrap();
        for j in 1..=i {
            let x_i = (2 * j - 1) as f64 * f64::consts::PI * denom;
            let x_i = x_i.cos();
            area += f(N::RealField::from_f64(x_i).unwrap()) * weight;
        }

        let error = (area - prev_area).abs();
        if error < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = error;
        i += 1;
    }

    Err("integrate_chebyshev: maximum iterations exceeded".to_owned())
}

/// Numerically integrate an integral of the form int_-1^1 f(x) sqrt(1 - x^2) dx
/// within a tolerance.
///
/// Given a function, numerically integrate using Chebyshev-Gaussian Quadrature
/// of the second kind until two consecutive iterations are within tolerance
/// or the maximum number of iterations is exceeded.
pub fn integrate_chebyshev_second<N: ComplexField, F: FnMut(N::RealField) -> N>(
    mut f: F,
    tol: N::RealField,
    n_max: usize,
) -> Result<N, String> {
    let mut prev_err = tol + N::RealField::one();
    let mut prev_area = N::zero();

    let pi = N::from_f64(f64::consts::PI).unwrap();

    let mut i: u32 = 2;
    while i < n_max as u32 {
        let mut area = N::zero();
        let denom = N::from_f64(1.0 / (i + 1) as f64).unwrap();
        let weight = pi * denom;
        for j in 1..=i {
            let j = N::from_u32(j).unwrap();
            let x_i = (j * weight).cos();
            let weight = weight * (j * weight).sin().powi(2);
            area += f(x_i.real()) * weight;
        }

        let error = (area - prev_area).abs();
        if error < tol && prev_err < tol {
            return Ok(area);
        }

        prev_area = area;
        prev_err = error;
        i += 1;
    }

    Err("integrate_chebyshev: maximum iterations exceeded".to_owned())
}
