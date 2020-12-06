use std::f64;
use crate::roots;

fn cubic(x: f64) -> f64 {
  x*x*x
}

fn sqrt_two(x: f64) -> f64 {
  x*x - 2.0
}

fn exp_xsq(x: f64) -> f64 {
  x.exp() - x*x
}

#[test]
fn bisection_cubic() {
  let a = -1.0;
  let b = 2.0;
  let tol = 0.0001;

  let solution = roots::bisection((a, b), cubic, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, 0.0, epsilon=tol));
}

#[test]
fn bisection_sqrt2() {
  let a = 0.0;
  let b = 2.0;
  let tol = 0.0001;

  let solution = roots::bisection((a, b), sqrt_two, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, f64::consts::SQRT_2, epsilon=tol));
}

#[test]
fn bisection_expxx() {
  let a = -2.0;
  let b = 2.0;
  let tol = 0.000001;

  let solution = roots::bisection((a, b), exp_xsq, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, -0.703467, epsilon=0.0000005));
}
