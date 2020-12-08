use alga::general::{ComplexField};

/// Polynomial on a ComplexField.
#[derive(Debug,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct Polynomial<N: ComplexField+From<f64>+Copy> {
  // Index 0 is constant, 1 is linear, etc.
  coefficients: Vec<N>,
}

impl<N: ComplexField+From<f64>+Copy> Polynomial<N> {
  /// Returns the zero polynomial on a given field
  pub fn new<U: ComplexField+From<f64>+Copy>() -> Polynomial<U> {
    Polynomial{
      coefficients: vec![U::from(0.0)]
    }
  }

  /// Evaluate a polynomial at a value
  pub fn evaluate(&self, x: N) -> N {
    let mut acc = *self.coefficients.last().unwrap();
    for val in self.coefficients.iter().rev().skip(1) {
      acc *= x;
      acc += *val;
    }

    acc
  }

  /// Evaluate a polynomial and its derivative at a value
  pub fn evaluate_derivative(&self, x: N) -> (N, N) {
    // Start with biggest coefficients
    let mut acc_eval = *self.coefficients.last().unwrap();
    let mut acc_deriv = *self.coefficients.last().unwrap();
    // For every coefficient except the constant and largest
    for val in self.coefficients.iter().skip(1).rev().skip(1) {
      acc_eval = acc_eval * x + *val;
      acc_deriv = acc_deriv * x + acc_eval;
    }
    // Do the constant for the polynomial evaluation
    acc_eval = x * acc_eval + self.coefficients[0];

    (acc_eval, acc_deriv)
  }

  /// Set a coefficient of a power in the polynomial
  pub fn set_coefficient(&mut self, power: u32, coefficient: N) {
    while (power + 1) > self.coefficients.len() as u32 {
      self.coefficients.push(N::from(0.0));
    }
    self.coefficients[power as usize] = coefficient;
  }


  /// Remove the coefficient of a power in the polynomial
  pub fn purge_coefficient(&mut self, power: u32) {
    if power == self.coefficients.len() as u32 {
      self.coefficients.pop();
    } else if power < self.coefficients.len() as u32 {
      self.coefficients[power as usize] = N::from(0.0);
    }
  }

  /// Get the derivative of the polynomial
  pub fn derivative(&self) -> Polynomial<N> {
    if self.coefficients.len() == 1 {
      return Polynomial {
        coefficients: vec![N::from(0.0),],
      };
    }

    let mut deriv_coeff = Vec::with_capacity(self.coefficients.len() - 1);

    for (i, val) in self.coefficients.iter().enumerate().skip(1) {
      deriv_coeff.push(N::from(i as f64) * *val);
    }

    Polynomial {
      coefficients: deriv_coeff
    }
  }
}
