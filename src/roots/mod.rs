/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use alga::general::{ComplexField, RealField};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, VectorN};

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
pub fn bisection<N: RealField, F: FnMut(N) -> N>(
    (mut left, mut right): (N, N),
    mut f: F,
    tol: N,
    n_max: usize,
) -> Result<N, String> {
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
pub fn steffensen<N: RealField + From<f64> + Copy>(
    mut initial: N,
    f: fn(N) -> N,
    tol: N,
    n_max: usize,
) -> Result<N, String> {
    let mut n = 0;

    while n < n_max {
        let guess = f(initial);
        let new_guess = f(guess);
        let diff =
            initial - (guess - initial).powi(2) / (new_guess - N::from(2.0) * guess + initial);
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
/// use nalgebra::{VectorN, U1};
/// use bacon_sci::roots::newton;
/// fn cubic(x: &[f64]) -> VectorN<f64, U1> {
///   VectorN::<f64, U1>::from_iterator(x.iter().map(|x| x.powi(3)))
/// }
///
/// fn cubic_deriv(x: &[f64]) -> VectorN<f64, U1> {
///  VectorN::<f64, U1>::from_iterator(x.iter().map(|x| 3.0*x.powi(2)))
/// }
/// //...
/// fn example() {
///   let solution = newton(&[0.1], cubic, cubic_deriv, 0.001, 1000).unwrap();
/// }
/// ```
pub fn newton<
    N: ComplexField,
    S: DimName,
    F: FnMut(&[N]) -> VectorN<N, S>,
    G: FnMut(&[N]) -> VectorN<N, S>,
>(
    initial: &[N],
    mut f: F,
    mut f_deriv: G,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<VectorN<N, S>, String>
where
    DefaultAllocator: Allocator<N, S>,
{
    let mut guess = VectorN::from_column_slice(initial);
    let mut norm = guess.dot(&guess).sqrt().abs();
    let mut n = 0;

    if norm <= tol {
        return Ok(guess);
    }

    while n < n_max {
        let f_val = f(guess.as_slice());
        let f_deriv_val = f_deriv(guess.as_slice());
        let adjustment = VectorN::from_iterator(
            f_val
                .column(0)
                .iter()
                .zip(f_deriv_val.column(0).iter())
                .map(|(f, f_d)| *f / *f_d),
        );
        let new_guess = &guess - adjustment;
        let new_norm = new_guess.dot(&new_guess).sqrt().abs();
        if ((norm - new_norm) / norm).abs() <= tol || new_norm <= tol {
            return Ok(new_guess);
        }

        norm = new_norm;
        guess = new_guess;
        n += 1;
    }

    Err("Newton: Maximum iterations exceeded".to_owned())
}

/// Use secant method to find a root of a vector function.
///
/// Using a vector function and its derivative, find a root based on two initial guesses
/// using secant method.
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
/// use nalgebra::{VectorN, U1};
/// use bacon_sci::roots::secant;
/// fn cubic(x: &[f64]) -> VectorN<f64, U1> {
///   VectorN::<f64, U1>::from_iterator(x.iter().map(|x| x.powi(3)))
/// }
/// //...
/// fn example() {
///   let solution = secant((&[0.1], &[-0.1]), cubic, 0.001, 1000).unwrap();
/// }
/// ```
pub fn secant<N: ComplexField, S: DimName, F: FnMut(&[N]) -> VectorN<N, S>>(
    initial: (&[N], &[N]),
    mut f: F,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<VectorN<N, S>, String>
where
    DefaultAllocator: Allocator<N, S>,
{
    let mut n = 0;

    let mut left = VectorN::from_column_slice(initial.0);
    let mut right = VectorN::from_column_slice(initial.1);

    let mut left_val = f(initial.0);
    let mut right_val = f(initial.1);

    let mut norm = right.dot(&right).sqrt().abs();
    if norm <= tol {
        return Ok(right);
    }

    while n <= n_max {
        let adjustment = VectorN::from_iterator(right_val.iter().enumerate().map(|(i, q)| {
            *q * (*right.get(i).unwrap() - *left.get(i).unwrap()) / (*q - *left_val.get(i).unwrap())
        }));
        let new_guess = &right - adjustment;
        let new_norm = new_guess.dot(&new_guess).sqrt().abs();
        if ((norm - new_norm) / norm).abs() <= tol || new_norm <= tol {
            return Ok(new_guess);
        }

        norm = new_norm;
        left_val = right_val;
        left = right;
        right = new_guess;
        right_val = f(right.as_slice());
        n += 1;
    }

    Err("Secant: Maximum iterations exceeded".to_owned())
}

/// Use Brent's method to find the root of a function
///
/// The initial guesses must bracket the root. That is, the initial guesses
/// must differ in sign.
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
pub fn brent<N: RealField, F: FnMut(N) -> N>(
    initial: (N, N),
    mut f: F,
    tol: N,
) -> Result<N, String> {
    if !tol.is_sign_positive() {
        return Err("brent: tolerance must be positive".to_owned());
    }

    let mut a = initial.0;
    let mut b = initial.1;
    let mut f_a = f(a);
    let mut f_b = f(b);

    // Make a the maximum
    if f_a.abs() < f_b.abs() {
        let tmp = a;
        a = b;
        b = tmp;
        let tmp = f_a;
        f_a = f_b;
        f_b = tmp;
    }

    if !(f_a * f_b).is_sign_negative() {
        println!("{} {}", f_a, f_b);
        return Err("brent: initial guesses do not bracket root".to_owned());
    }

    let two = N::from_i32(2).unwrap();
    let three = N::from_i32(3).unwrap();
    let four = N::from_i32(4).unwrap();

    let mut c = a;
    let mut f_c = f_a;
    let mut s = b - f_b * (b - a) / (f_b - f_a);
    let mut f_s = f(s);
    let mut mflag = true;
    let mut d = c;

    while !(f_b.abs() < tol || f_s.abs() < tol || (a - b).abs() < tol) {
        if (f_a - f_c).abs() < tol && (f_b - f_c).abs() < tol {
            s = (a * f_b * f_c) / ((f_a - f_b) * (f_a - f_c))
                + (b * f_a * f_c) / ((f_b - f_a) * (f_b - f_c))
                + (c * f_a * f_b) / ((f_c - f_a) * (f_c - f_b));
        } else {
            s = b - f_b * (b - a) / (f_b - f_a);
        }

        if !(s >= (three * a + b) / four && s <= b)
            || (mflag && (s - b).abs() >= (b - c) / two)
            || (!mflag && (s - b).abs() >= (c - d).abs() / two)
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol)
        {
            s = (a + b) / two;
            mflag = true;
        } else {
            mflag = false;
        }

        f_s = f(s);
        d = c;
        c = b;
        f_c = f_b;
        if (f_a * f_s).is_sign_negative() {
            b = s;
            f_b = f_s;
        } else {
            a = s;
            f_a = f_s;
        }

        if f_a.abs() < f_b.abs() {
            let tmp = a;
            a = b;
            b = tmp;
            let tmp = f_a;
            f_a = f_b;
            f_b = tmp;
        }
    }

    if f_s.abs() < tol {
        Ok(s)
    } else {
        Ok(b)
    }
}
