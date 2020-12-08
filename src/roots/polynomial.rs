use crate::polynomial::Polynomial;
use alga::general::ComplexField;

/// Use Newton's method on a polynomial.
///
/// Given an initial guess of a polynomial root, use Newton's method to
/// approximate within tol.
///
/// # Returns
/// `Ok(root)` when a root has been found, `Err` on failure
///
/// # Params
/// `initial` initial estimate of the root
///
/// `poly` the polynomial to solve the root for
///
/// `tol` The tolerance of relative error between iterations
///
/// `n_max` the maximum number of iterations
pub fn newton_polynomial<N: ComplexField+From<f64>+Copy>(
  initial: N,
  poly: &Polynomial<N>,
  tol: <N as ComplexField>::RealField,
  n_max: usize
) -> Result<N, String> {
  let mut n = 0;

  let mut guess = initial;


  let mut norm = guess.abs();
  if norm <= tol {
    return Ok(guess);
  }

  while n < n_max {
    let (f_val, f_deriv_val) = poly.evaluate_derivative(guess);
    let new_guess = guess - (f_val / f_deriv_val);
    let new_norm = new_guess.abs();
    if ((norm - new_norm) / norm).abs() <= tol || new_norm <= tol {
      return Ok(new_guess);
    }

    norm = new_norm;
    guess = new_guess;
    n += 1;
  }

  Err("Newton_polynomial: maximum iterations exceeded".to_owned())
}
