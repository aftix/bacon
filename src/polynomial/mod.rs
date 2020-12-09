use alga::general::{ComplexField};
use std::ops;

/// Polynomial on a ComplexField.
#[derive(Debug,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct Polynomial<N: ComplexField+From<f64>+Copy> {
  // Index 0 is constant, 1 is linear, etc.
  coefficients: Vec<N>,
}

impl<N: ComplexField+From<f64>+Copy> Polynomial<N> {
  /// Returns the zero polynomial on a given field
  pub fn new() -> Self {
    Polynomial{
      coefficients: vec![N::from(0.0)]
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
    match self.coefficients.len() as u32 {
      len if len == power => { self.coefficients.pop(); },
      _ => { self.coefficients[power as usize] = N::from(0.0); },
    };
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

impl<N: ComplexField+From<f64>+Copy> Default for Polynomial<N> {
  fn default() -> Self {
    Self::new()
  }
}

// Operator overloading

impl<N: ComplexField+From<f64>+Copy> ops::Add<N> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(mut self, rhs: N) -> Polynomial<N> {
    self.coefficients[0] += rhs;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Add<N> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(self, rhs: N) -> Polynomial<N> {
    let mut coefficients = Vec::from(self.coefficients.as_slice());
    coefficients[0] += rhs;
    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Add<Polynomial<N>> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(mut self, rhs: Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val += rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(*val);
    }

    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Add<&Polynomial<N>> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(mut self, rhs: &Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val += rhs.coefficients[ind];
    }

    // Will only run if rhs has higher order
    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(*val);
    }

    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Add<Polynomial<N>> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(self, rhs: Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    let mut coefficients = Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
    for (ind, val) in self.coefficients.iter().take(min_order).enumerate() {
      coefficients.push(*val + rhs.coefficients[ind]);
    }

    // Only one loop will run
    for val in self.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Add<&Polynomial<N>> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn add(self, rhs: &Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    let mut coefficients = Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
    for (ind, val) in self.coefficients.iter().take(min_order).enumerate() {
      coefficients.push(*val + rhs.coefficients[ind]);
    }

    // Only one loop will run
    for val in self.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::AddAssign<N> for Polynomial<N> {
  fn add_assign(&mut self, rhs: N) {
    self.coefficients[0] += rhs;
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::AddAssign<Polynomial<N>> for Polynomial<N> {
  fn add_assign(&mut self, rhs: Polynomial<N>) {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val += rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(*val);
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::AddAssign<&Polynomial<N>> for Polynomial<N> {
  fn add_assign(&mut self, rhs: &Polynomial<N>) {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val += rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(*val);
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<N> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(mut self, rhs: N) -> Polynomial<N> {
    self.coefficients[0] -= rhs;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<N> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(self, rhs: N) -> Polynomial<N> {
    let mut coefficients = Vec::from(self.coefficients.as_slice());
    coefficients[0] -= rhs;
    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<Polynomial<N>> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(mut self, rhs: Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val -= rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(-*val);
    }

    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<Polynomial<N>> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(self, rhs: Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    let mut coefficients = Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
    for (ind, val) in self.coefficients.iter().take(min_order).enumerate() {
      coefficients.push(*val - rhs.coefficients[ind]);
    }

    // Only one for loop runs
    for val in self.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      coefficients.push(-*val);
    }

    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<&Polynomial<N>> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(mut self, rhs: &Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val -= rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(-*val);
    }

    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Sub<&Polynomial<N>> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn sub(self, rhs: &Polynomial<N>) -> Polynomial<N> {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    let mut coefficients = Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
    for (ind, val) in self.coefficients.iter().take(min_order).enumerate() {
      coefficients.push(*val - rhs.coefficients[ind]);
    }

    // Only one for loop runs
    for val in self.coefficients.iter().skip(min_order) {
      coefficients.push(*val);
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      coefficients.push(-*val);
    }

    Polynomial {
      coefficients
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::SubAssign<N> for Polynomial<N> {
  fn sub_assign(&mut self, rhs: N) {
    self.coefficients[0] -= rhs;
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::SubAssign<Polynomial<N>> for Polynomial<N> {
  fn sub_assign(&mut self, rhs: Polynomial<N>) {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val -= rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(-*val);
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::SubAssign<&Polynomial<N>> for Polynomial<N> {
  fn sub_assign(&mut self, rhs: &Polynomial<N>) {
    let min_order = self.coefficients.len().min(rhs.coefficients.len());
    for (ind, val) in self.coefficients.iter_mut().take(min_order).enumerate() {
      *val -= rhs.coefficients[ind];
    }

    for val in rhs.coefficients.iter().skip(min_order) {
      self.coefficients.push(-*val);
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Mul<N> for Polynomial<N> {
  type Output = Polynomial<N>;

  fn mul(mut self, rhs: N) -> Polynomial<N> {
    for val in &mut self.coefficients {
      *val *= rhs;
    }
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> ops::Mul<N> for &Polynomial<N> {
  type Output = Polynomial<N>;

  fn mul(self, rhs: N) -> Polynomial<N> {
    let mut coefficients = Vec::with_capacity(self.coefficients.len());
    for val in &self.coefficients {
      coefficients.push(*val * rhs);
    }
    Polynomial {
      coefficients
    }
  }
}
