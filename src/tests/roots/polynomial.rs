use crate::polynomial::Polynomial;
use crate::roots::{muller_polynomial, newton_polynomial};

#[test]
fn newton() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(2, 1.0);
  poly.set_coefficient(0, -1.0);

  let solution = newton_polynomial(1.5, &poly, 0.0001, 1000).unwrap();
  assert!(approx_eq!(f64, solution, 1.0, epsilon=0.0001));
  let solution = newton_polynomial(-1.5, &poly, 0.0001, 1000).unwrap();
  assert!(approx_eq!(f64, solution, -1.0, epsilon=0.0001));

  poly.set_coefficient(2, 2.0);
  poly.set_coefficient(3, 5.0);
  poly.set_coefficient(1, -2.0);
  let solution = newton_polynomial(2.0, &poly, 0.0001, 1000).unwrap();
  assert!(approx_eq!(f64, solution, 0.66157, epsilon=0.00001));
}

#[test]
fn muller() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(2, 1.0);
  poly.set_coefficient(0, -1.0);

  let solution = muller_polynomial(
    (0.0, 0.5, 1.5),
    &poly,
    0.0001,
    1000
  ).unwrap();
  assert!(approx_eq!(f64, solution.re, 1.0, epsilon=0.0001));
  assert!(approx_eq!(f64, solution.im, 0.0, epsilon=0.0001));
  let solution = muller_polynomial(
    (0.0, -0.5, -1.5),
    &poly,
    0.0001,
    1000
  ).unwrap();
  assert!(approx_eq!(f64, solution.re, -1.0, epsilon=0.0001));
  assert!(approx_eq!(f64, solution.im, 0.0, epsilon=0.0001));

  poly.set_coefficient(2, 2.0);
  poly.set_coefficient(3, 5.0);
  poly.set_coefficient(1, -2.0);
  let solution = muller_polynomial(
    (0.0, 1.0, 2.0),
    &poly,
    0.0001,
    1000
  ).unwrap();
  assert!(approx_eq!(f64, solution.re, 0.66157, epsilon=0.00001));
  assert!(approx_eq!(f64, solution.im, 0.0, epsilon=0.00001));
}
