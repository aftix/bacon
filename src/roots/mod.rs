use nalgebra::DVector;

pub fn bisection((mut left, mut right): (f64, f64), f: fn(f64) -> f64, tol: f64, n_max: usize) -> Result<f64, String> {
  if left >= right {
    return Err("requirement: right > left".to_owned());
  }

  let mut n = 1;

  let mut f_a = f(left);

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
