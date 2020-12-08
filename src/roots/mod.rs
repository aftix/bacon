// MIT License
//
// Copyright (c) 2020 Wyatt Campbell
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use nalgebra::DVector;

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
/// fn cubic(x: f64) -> f64 {
///   x*x*x
/// }
/// ...
/// let solution = roots::bisection((-1.0, 1.0), cubic, 0.001, 1000).unwrap();
/// ```
pub fn bisection((mut left, mut right): (f64, f64), f: fn(f64) -> f64, tol: f64, n_max: usize) -> Result<f64, String> {
  if left >= right {
    return Err("requirement: right > left".to_owned());
  }

  let mut n = 1;

  let mut f_a = f(left);
  if f_a * f(right) > 0.0 {
    return Err("requirement: Signs must be different".to_owned());
  }

  if f_a * f(right) > 0.0 {
    return Err("sgn(f(a)) != sgn(f(b))".to_owned());
  }

  let mut half_interval = (left - right) * 0.5;
  let mut middle = left + half_interval;

  if approx_eq!(f64, 0.0, middle, epsilon = tol) {
    return Ok(middle);
  }

  while n <= n_max {
    let f_p = f(middle);
    if f_p * f_a > 0.0 {
      left = middle;
      f_a = f_p;
    } else {
      right = middle;
    }

    half_interval = (right - left) * 0.5;

    let middle_new = left + half_interval;

    if (middle - middle_new).abs() / middle.abs() < tol {
      return Ok(middle_new);
    }

    middle = middle_new;
    n += 1;
  }

  Err("Maximum iterations exceeded".to_owned())
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
/// fn cubic(x: &[f64]) -> DVector<f64> {
///   DVector::from_iterator(x.len(), x.iter().map(|x| x.powi(3)))
/// }
///
/// fn cubic_deriv(x: &[f64]) -> DVector<f64> {
///   DVector::from_iterator(x.len(), x.iter.map(|x| 3.0*x.powi(2)))
/// }
/// ...
/// let solution = roots::newton(&[0.1], cubic, cubic_deriv, 0.001, 1000).unwrap();
/// ```
pub fn newton(
  initial: &[f64],
  f: fn(&[f64]) -> DVector<f64>,
  f_deriv: fn(&[f64]) -> DVector<f64>,
  tol: f64,
  n_max: usize
) -> Result<DVector<f64>, String> {
  let mut guess = DVector::from_column_slice(initial);
  let mut norm = guess.norm();
  let mut n = 0;

  if approx_eq!(f64, 0.0, norm, epsilon = tol) {
    return Ok(guess);
  }

  while n < n_max {
    let f_val = f(guess.column(0).as_slice());
    let f_deriv_val = f_deriv(guess.column(0).as_slice());
    let adjustment = DVector::from_iterator(
      guess.column(0).len(),
      f_val.column(0).iter().zip(f_deriv_val.column(0).iter()).map(|(f, f_d)| f / f_d)
    );
    let new_guess = &guess - adjustment;
    let new_norm = new_guess.norm();
    if ((norm - new_norm) / norm).abs() < tol || approx_eq!(f64, 0.0, new_norm, epsilon = tol){
      return Ok(new_guess);
    }

    norm = new_norm;
    guess = new_guess;
    n += 1;
  }

  Err("Maximum iterations exceeded".to_owned())
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
/// fn cubic(x: &[f64]) -> DVector<f64> {
///   DVector::from_iterator(x.len(), x.iter().map(|x| x.powi(3)))
/// }
/// ...
/// let solution = roots::secant((&[0.1], &[-0.1]), cubic, 0.001, 1000).unwrap();
/// ```
pub fn secant(
  initial: (&[f64], &[f64]),
  f: fn(&[f64]) -> DVector<f64>,
  tol: f64,
  n_max: usize
) -> Result<DVector<f64>, String> {
  let mut n = 0;

  let mut left = DVector::from_column_slice(initial.0);
  let mut right = DVector::from_column_slice(initial.1);

  let mut left_val = f(initial.0);
  let mut right_val = f(initial.1);

  let mut norm = right.norm();
  if approx_eq!(f64, 0.0, norm, epsilon = tol) {
    return Ok(right);
  }

  while n <= n_max {
    let adjustment = DVector::from_iterator(
      right_val.iter().len(),
      right_val.iter().enumerate().map(|(i, q)| {
        q * (*right.get(i).unwrap() - *left.get(i).unwrap()) / (q - *left_val.get(i).unwrap())
      })
    );
    let new_guess = &right - adjustment;
    let new_norm = new_guess.norm();
    if ((norm - new_norm) / norm).abs() < tol || approx_eq!(f64, 0.0, new_norm, epsilon = tol) {
      return Ok(new_guess);
    }

    norm = new_norm;
    left_val = right_val;
    left = right;
    right = new_guess;
    right_val = f(right.column(0).as_slice());
    n += 1;
  }

  Err("Maximum iterations exceeded".to_owned())
}
