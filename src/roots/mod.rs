
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

  'out: while n <= n_max {
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
      break 'out;
    }

    middle = middle_new;
    n += 1;
  }

  if n > n_max {
    return Err("Maximum iterations exceeded".to_owned());
  }

  Ok(middle)
}
