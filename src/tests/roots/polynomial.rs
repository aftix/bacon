use crate::polynomial::Polynomial;
use crate::roots::newton_polynomial;

#[test]
fn newton() {
  let mut poly: Polynomial<f64> = Polynomial::<f64>::new();
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
