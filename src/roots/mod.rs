/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::{ComplexField, Const, DimMin, RealField, SMatrix, SVector};
use num_traits::{FromPrimitive, Zero};

mod polynomial;
pub use polynomial::*;

/// Use the bisection method to solve for a zero of an equation.
///
/// This function takes an interval and uses a binary search to find
/// where in that interval a function has a root. The signs of the function
/// at each end of the interval must be different
///
/// # Returns
/// Ok(root) when a root has been found, Err on failure
///
/// # Params
/// `(left, right)` The starting interval. f(left) * f(right) > 0.0
///
/// `f` The function to find the root for
///
/// `tol` The tolerance of the relative error between iterations.
///
/// `n_max` The maximum number of iterations to use.
///
/// # Examples
/// ```
/// use bacon_sci::roots::bisection;
///
/// fn cubic(x: f64) -> f64 {
///   x*x*x
/// }
/// //...
/// fn example() {
///   let solution = bisection((-1.0, 1.0), cubic, 0.001, 1000).unwrap();
/// }
/// ```
pub fn bisection<N, F>(
    (mut left, mut right): (N, N),
    mut f: F,
    tol: N,
    n_max: usize,
) -> Result<N, String>
where
    N: RealField + FromPrimitive + Copy,
    F: FnMut(N) -> N,
{
    if left >= right {
        return Err("Bisection: requirement: right > left".to_owned());
    }

    let mut n = 1;

    let mut f_a = f(left);
    if (f_a * f(right)).is_sign_positive() {
        return Err("Bisection: requirement: Signs must be different".to_owned());
    }

    let half = N::from_f64(0.5).unwrap();

    let mut half_interval = (left - right) * half;
    let mut middle = left + half_interval;

    if middle.abs() <= tol {
        return Ok(middle);
    }

    while n <= n_max {
        let f_p = f(middle);
        if (f_p * f_a).is_sign_positive() {
            left = middle;
            f_a = f_p;
        } else {
            right = middle;
        }

        half_interval = (right - left) * half;

        let middle_new = left + half_interval;

        if (middle - middle_new).abs() / middle.abs() < tol || middle_new.abs() < tol {
            return Ok(middle_new);
        }

        middle = middle_new;
        n += 1;
    }

    Err("Bisection: Maximum iterations exceeded".to_owned())
}

/// Use steffenson's method to find a fixed point
///
/// Use steffenson's method to find a value x so that f(x) = x, given
/// a starting point.
///
/// # Returns
/// `Ok(x)` so that `f(x) - x < tol` on success, `Err` on failure
///
/// # Params
/// `initial` inital guess for the fixed point
///
/// `f` Function to find the fixed point of
///
/// `tol` Tolerance from 0 to try and achieve
///
/// `n_max` maximum number of iterations
///
/// # Examples
/// ```
/// use bacon_sci::roots::steffensen;
/// fn cosine(x: f64) -> f64 {
///   x.cos()
/// }
/// //...
/// fn example() -> Result<(), String> {
///   let solution = steffensen(0.5f64, cosine, 0.0001, 1000)?;
///   Ok(())
/// }
pub fn steffensen<N>(mut initial: N, f: fn(N) -> N, tol: N, n_max: usize) -> Result<N, String>
where
    N: RealField + FromPrimitive + Copy,
{
    let mut n = 0;

    while n < n_max {
        let guess = f(initial);
        let new_guess = f(guess);
        let diff = initial
            - (guess - initial).powi(2) / (new_guess - N::from_f64(2.0).unwrap() * guess + initial);
        if (diff - initial).abs() <= tol {
            return Ok(diff);
        }
        initial = diff;
        n += 1;
    }

    Err("Steffensen: Maximum number of iterations exceeded".to_owned())
}

/// Use Newton's method to find a root of a vector function.
///
/// Using a vector function and its derivative, find a root based on an initial guess
/// using Newton's method.
///
/// # Returns
/// `Ok(vec)` on success, where `vec` is a vector input for which the function is
/// zero. `Err` on failure.
///
/// # Params
/// `initial` Initial guess of the root. Should be near actual root. Slice since this
/// function finds roots of vector functions.
///
/// `f` Vector function for which to find the root
///
/// `f_deriv` Derivative of `f`
///
/// `tol` tolerance for error between iterations of Newton's method
///
/// `n_max` Maximum number of iterations
///
/// # Examples
/// ```
/// use nalgebra::{SVector, SMatrix};
/// use bacon_sci::roots::newton;
/// fn cubic(x: &[f64]) -> SVector<f64, 1> {
///   SVector::<f64, 1>::from_iterator(x.iter().map(|x| x.powi(3)))
/// }
///
/// fn cubic_deriv(x: &[f64]) -> SMatrix<f64, 1, 1> {
///  SMatrix::<f64, 1, 1>::from_iterator(x.iter().map(|x| 3.0*x.powi(2)))
/// }
/// //...
/// fn example() {
///   let solution = newton(&[0.1], cubic, cubic_deriv, 0.001, 1000).unwrap();
/// }
/// ```
pub fn newton<N, F, G, const S: usize>(
    initial: &[N],
    mut f: F,
    mut jac: G,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<SVector<N, S>, String>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    F: FnMut(&[N]) -> SVector<N, S>,
    G: FnMut(&[N]) -> SMatrix<N, S, S>,
    Const<S>: DimMin<Const<S>, Output = Const<S>>,
{
    let mut guess = SVector::<N, S>::from_column_slice(initial);
    let mut norm = guess.dot(&guess).sqrt().abs();
    let mut n = 0;

    if norm <= tol {
        return Ok(guess);
    }

    while n < n_max {
        let f_val = -f(guess.as_slice());
        let f_deriv_val = jac(guess.as_slice());
        let lu = f_deriv_val.lu();
        match lu.solve(&f_val) {
            None => return Err("newton: failed to solve linear equation".to_owned()),
            Some(adjustment) => {
                let new_guess = guess + adjustment;
                let new_norm = new_guess.dot(&new_guess).sqrt().abs();
                if ((norm - new_norm) / norm).abs() <= tol || new_norm <= tol {
                    return Ok(new_guess);
                }

                norm = new_norm;
                guess = new_guess;
                n += 1;
            }
        }
    }

    Err("Newton: Maximum iterations exceeded".to_owned())
}

fn jac_finite_diff<N, F, const S: usize>(
    mut f: F,
    x: &mut SVector<N, S>,
    h: <N as ComplexField>::RealField,
) -> SMatrix<N, S, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    F: FnMut(&[N]) -> SVector<N, S>,
{
    let mut mat = SMatrix::<N, S, S>::zero();
    let h = N::from_real(h);
    let denom = N::one() / (N::from_i32(2).unwrap() * h);

    for col in 0..mat.row(0).len() {
        x[col] += h;
        let above = f(x.as_slice());
        x[col] -= h;
        x[col] -= h;
        let below = f(x.as_slice());
        x[col] += h;
        let jac_col = (above + below) * denom;
        for row in 0..mat.column(0).len() {
            mat[(row, col)] = jac_col[row];
        }
    }

    mat
}

/// Use secant method to find a root of a vector function.
///
/// Using a vector function and its derivative, find a root based on an initial guess
/// and finite element differences using Broyden's method.
///
/// # Returns
/// `Ok(vec)` on success, where `vec` is a vector input for which the function is
/// zero. `Err` on failure.
///
/// # Params
/// `initial` Initial guesses of the root. Should be near actual root. Slice since this
/// function finds roots of vector functions.
///
/// `f` Vector function for which to find the root
///
/// `tol` tolerance for error between iterations of Newton's method
///
/// `n_max` Maximum number of iterations
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use bacon_sci::roots::secant;
/// fn cubic(x: &[f64]) -> SVector<f64, 1> {
///   SVector::<f64, 1>::from_iterator(x.iter().map(|x| x.powi(3)))
/// }
/// //...
/// fn example() {
///   let solution = secant(&[0.1], cubic, 0.1, 0.001, 1000).unwrap();
/// }
/// ```
pub fn secant<N, F, const S: usize>(
    initial: &[N],
    mut func: F,
    h: <N as ComplexField>::RealField,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<SVector<N, S>, String>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    F: FnMut(&[N]) -> SVector<N, S>,
    Const<S>: DimMin<Const<S>, Output = Const<S>>,
{
    let mut n = 2;

    let mut guess = SVector::<N, S>::from_column_slice(initial);
    let mut func_eval = func(guess.as_slice());

    let jac = jac_finite_diff(&mut func, &mut guess, h);
    let lu = jac.lu();
    let try_inv = lu.try_inverse();
    let mut jac_inv = if let Some(inv) = try_inv {
        inv
    } else {
        return Err("Secant: Can not inverse finite element difference jacobian".to_owned());
    };

    let mut shift = -jac_inv * func_eval;
    guess += &shift;

    while n < n_max {
        let func_eval_last = func_eval;
        func_eval = func(guess.as_slice());
        let diff = func_eval - func_eval_last;
        let adjustment = -jac_inv * diff;
        let s_transpose = shift.transpose();
        let p = (-s_transpose * adjustment)[(0, 0)];
        let u = s_transpose * jac_inv;
        jac_inv += (shift + adjustment) * u / p;
        shift = -&jac_inv * func_eval;
        guess += &shift;
        if shift.norm().abs() <= tol {
            return Ok(guess);
        }
        n += 1;
    }

    Err("Secant: Maximum iterations exceeded".to_owned())
}

/// Use Brent's method to find the root of a function
///
/// The initial guesses must bracket the root. That is, the function evaluations of
/// the initial guesses must differ in sign.
///
/// # Examples
/// ```
/// use bacon_sci::roots::brent;
/// fn cubic(x: f64) -> f64 {
///     x.powi(3)
/// }
/// //...
/// fn example() {
///   let solution = brent((0.1, -0.1), cubic, 1e-5).unwrap();
/// }
/// ```
pub fn brent<N, F>(initial: (N, N), mut f: F, tol: N) -> Result<N, String>
where
    N: RealField + FromPrimitive + Copy,
    F: FnMut(N) -> N,
{
    if !tol.is_sign_positive() {
        return Err("brent: tolerance must be positive".to_owned());
    }

    let mut left = initial.0;
    let mut right = initial.1;
    let mut f_left = f(left);
    let mut f_right = f(right);

    // Make a the maximum
    if f_left.abs() < f_right.abs() {
        std::mem::swap(&mut left, &mut right);
        std::mem::swap(&mut f_left, &mut f_right);
    }

    if !(f_left * f_right).is_sign_negative() {
        return Err("brent: initial guesses do not bracket root".to_owned());
    }

    let two = N::from_i32(2).unwrap();
    let three = N::from_i32(3).unwrap();
    let four = N::from_i32(4).unwrap();

    let mut c = left;
    let mut f_c = f_left;
    let mut s = right - f_right * (right - left) / (f_right - f_left);
    let mut f_s = f(s);
    let mut mflag = true;
    let mut d = c;

    while !(f_right.abs() < tol || f_s.abs() < tol || (left - right).abs() < tol) {
        if (f_left - f_c).abs() < tol && (f_right - f_c).abs() < tol {
            s = (left * f_right * f_c) / ((f_left - f_right) * (f_left - f_c))
                + (right * f_left * f_c) / ((f_right - f_left) * (f_right - f_c))
                + (c * f_left * f_right) / ((f_c - f_left) * (f_c - f_right));
        } else {
            s = right - f_right * (right - left) / (f_right - f_left);
        }

        if !(s >= (three * left + right) / four && s <= right)
            || (mflag && (s - right).abs() >= (right - c) / two)
            || (!mflag && (s - right).abs() >= (c - d).abs() / two)
            || (mflag && (right - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol)
        {
            s = (left + right) / two;
            mflag = true;
        } else {
            mflag = false;
        }

        f_s = f(s);
        d = c;
        c = right;
        f_c = f_right;
        if (f_left * f_s).is_sign_negative() {
            right = s;
            f_right = f_s;
        } else {
            left = s;
            f_left = f_s;
        }

        if f_left.abs() < f_right.abs() {
            std::mem::swap(&mut left, &mut right);
            std::mem::swap(&mut f_left, &mut f_right);
        }
    }

    if f_s.abs() < tol {
        Ok(s)
    } else {
        Ok(right)
    }
}

/// Find the root of an equation using the ITP method.
///
/// The initial guess must bracket the root, that is the
/// function evaluations must differ in sign between the
/// two initial guesses. k_1 is a parameter in (0, infty).
/// k_2 is a paramater in (1, 1 + golden_ratio). n_0 is a parameter
/// in [0, infty). This method gives the worst case performance of the
/// bisection method (which has the best worst case performance) with
/// better average convergance.
///
/// # Examples
/// ```
/// use bacon_sci::roots::itp;
/// fn cubic(x: f64) -> f64 {
///     x.powi(3)
/// }
/// //...
/// fn example() {
///   let solution = itp((0.1, -0.1), cubic, 0.1, 2.0, 0.99, 1e-5).unwrap();
/// }
/// ```
pub fn itp<N, F>(initial: (N, N), mut f: F, k_1: N, k_2: N, n_0: N, tol: N) -> Result<N, String>
where
    N: RealField + FromPrimitive + Copy,
    F: FnMut(N) -> N,
{
    if !tol.is_sign_positive() {
        return Err("itp: tolerance must be positive".to_owned());
    }

    if !k_1.is_sign_positive() {
        return Err("itp: k_1 must be positive".to_owned());
    }

    if k_2 <= N::one() || k_2 >= (N::one() + N::from_f64(0.5 * (1.0 + 5.0_f64.sqrt())).unwrap()) {
        return Err("itp: k_2 must be in (1, 1 + golden_ratio)".to_owned());
    }

    let mut left = initial.0;
    let mut right = initial.1;
    let mut f_left = f(left);
    let mut f_right = f(right);

    if !(f_left * f_right).is_sign_negative() {
        return Err("itp: initial guesses must bracket root".to_owned());
    }

    if f_left.is_sign_positive() {
        std::mem::swap(&mut left, &mut right);
        std::mem::swap(&mut f_left, &mut f_right);
    }

    let two = N::from_i32(2).unwrap();

    let n_half = ((right - left).abs() / (two * tol)).log2().ceil();
    let n_max = n_half + n_0;
    let mut j = 0;

    while (right - left).abs() > two * tol {
        let x_half = (left + right) / two;
        let r = tol * two.powf(n_max + n_0 - N::from_i32(j).unwrap()) - (right - left) / two;
        let x_f = (f_right * left - f_left * right) / (f_right - f_left);
        let sigma = (x_half - x_f) / (x_half - x_f).abs();
        let delta = k_1 * (right - left).powf(k_2);
        let x_t = if delta <= (x_half - x_f).abs() {
            x_f + sigma * delta
        } else {
            x_half
        };
        let x_itp = if (x_t - x_half).abs() <= r {
            x_t
        } else {
            x_half - sigma * r
        };
        let f_itp = f(x_itp);
        if f_itp.is_sign_positive() {
            right = x_itp;
            f_right = f_itp;
        } else if f_itp.is_sign_negative() {
            left = x_itp;
            f_left = f_itp;
        } else {
            left = x_itp;
            right = x_itp;
        }
        j += 1;
    }

    Ok((left + right) / two)
}
