use alga::general::*;
use num_complex::Complex;
use num_traits::Zero;
use std::iter::FromIterator;
use std::ops;

/// Polynomial on a ComplexField.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Polynomial<N: ComplexField> {
    // Index 0 is constant, 1 is linear, etc.
    coefficients: Vec<N>,
}

#[macro_export]
macro_rules! polynomial {
  ( $( $x:expr ),* ) => {
    $crate::polynomial::Polynomial::from_slice(&[$($x),*])
  }
}

impl<N: ComplexField> Polynomial<N> {
    /// Returns the zero polynomial on a given field
    pub fn new() -> Self {
        Polynomial {
            coefficients: vec![N::from_f64(0.0).unwrap()],
        }
    }

    /// Returns the zero polynomial on a given field with preallocated memory
    pub fn with_capacity(capacity: usize) -> Polynomial<N> {
        let mut coefficients = Vec::with_capacity(capacity);
        coefficients.push(N::zero());
        Polynomial { coefficients }
    }

    /// Create a polynomial from a slice, with the first element of the slice being the highest power
    pub fn from_slice(data: &[N]) -> Polynomial<N> {
        if data.is_empty() {
            return Polynomial {
                coefficients: vec![N::zero()],
            };
        }
        Polynomial {
            coefficients: Vec::from_iter(data.iter().rev().copied()),
        }
    }

    /// Get the order of the polynomial
    pub fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Get the coefficient of a power
    pub fn get_coefficient(&self, ind: usize) -> N {
        if ind >= self.coefficients.len() {
            N::zero()
        } else {
            self.coefficients[ind]
        }
    }

    /// Make a polynomial complex
    pub fn make_complex(&self) -> Polynomial<Complex<<N as ComplexField>::RealField>> {
        let mut coefficients: Vec<Complex<N::RealField>> =
            Vec::with_capacity(self.coefficients.len());
        for val in &self.coefficients {
            coefficients.push(Complex::<N::RealField>::new(val.real(), val.imaginary()));
        }
        Polynomial { coefficients }
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
        if self.coefficients.len() == 1 {
            return (self.coefficients[0], N::zero());
        }
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
            self.coefficients.push(N::from_f64(0.0).unwrap());
        }
        self.coefficients[power as usize] = coefficient;
    }

    /// Remove the coefficient of a power in the polynomial
    pub fn purge_coefficient(&mut self, power: usize) {
        match self.coefficients.len() {
            len if len == power => {
                self.coefficients.pop();
            }
            _ => {
                self.coefficients[power] = N::from_f64(0.0).unwrap();
            }
        };
    }

    /// Get the derivative of the polynomial
    pub fn derivative(&self) -> Polynomial<N> {
        if self.coefficients.len() == 1 {
            return Polynomial {
                coefficients: vec![N::from_f64(0.0).unwrap()],
            };
        }

        let mut deriv_coeff = Vec::with_capacity(self.coefficients.len() - 1);

        for (i, val) in self.coefficients.iter().enumerate().skip(1) {
            deriv_coeff.push(N::from_f64(i as f64).unwrap() * *val);
        }

        Polynomial {
            coefficients: deriv_coeff,
        }
    }

    /// Get the antiderivative of the polynomial with specified constant
    pub fn antiderivative(&self, constant: N) -> Polynomial<N> {
        let mut coefficients = Vec::with_capacity(self.coefficients.len() + 1);
        coefficients.push(constant);
        for (ind, val) in self.coefficients.iter().enumerate() {
            coefficients.push(*val * N::from_f64(1.0 / (ind + 1) as f64).unwrap());
        }
        Polynomial { coefficients }
    }

    /// Integrate this polynomial between to starting points
    pub fn integrate(&self, lower: N, upper: N) -> N {
        let poly_anti = self.antiderivative(N::zero());
        println!("{:?}", poly_anti);
        poly_anti.evaluate(upper) - poly_anti.evaluate(lower)
    }

    /// Divide this polynomial by another, getting a quotient and remainder, using tol to check for 0
    pub fn divide(
        &self,
        divisor: &Polynomial<N>,
        tol: <N as ComplexField>::RealField,
    ) -> Result<(Polynomial<N>, Polynomial<N>), String> {
        if divisor.coefficients.len() == 1
            && divisor.coefficients[0].real().abs() < tol
            && divisor.coefficients[0].imaginary().abs() < tol
        {
            return Err("Polynomial division: Can not divide by 0".to_owned());
        }

        let mut quotient = Polynomial::new();
        let mut remainder = Polynomial::from_iter(self.coefficients.iter().copied());
        let mut temp = Polynomial::new();

        while remainder.coefficients.len() >= divisor.coefficients.len()
            && !(remainder.coefficients.len() == 1
                && remainder.coefficients[0].real().abs() < tol
                && remainder.coefficients[0].imaginary().abs() < tol)
        {
            // Get the power left over from dividing lead terms
            let order = remainder.coefficients.len() - divisor.coefficients.len();
            // Make a vector that is just the lead coefficients divided at the right power
            temp.coefficients = vec![N::zero(); order + 1];
            temp.coefficients[order] =
                *remainder.coefficients.last().unwrap() / *divisor.coefficients.last().unwrap();
            // Add the division to the quotient
            quotient += &temp;
            // Get the amount to shift divisor by
            let padding = temp.coefficients.len() - 1;
            // Multiply every coefficient in divisor by temp's coefficient
            temp = Polynomial::from_iter(
                divisor
                    .coefficients
                    .iter()
                    .map(|c| *c * *temp.coefficients.last().unwrap()),
            );
            // Shift the coefficients to multiply by the right power of x
            for _ in 0..padding {
                temp.coefficients.insert(0, N::zero());
            }
            // remainder -= temp x d;
            remainder -= &temp;
            while remainder.coefficients.len() > 1
                && remainder.coefficients.last().unwrap().real().abs() < tol
                && remainder.coefficients.last().unwrap().imaginary().abs() < tol
            {
                remainder.coefficients.pop();
            }
        }

        Ok((quotient, remainder))
    }
}

impl<N: ComplexField> FromIterator<N> for Polynomial<N> {
    fn from_iter<I: IntoIterator<Item = N>>(iter: I) -> Polynomial<N> {
        Polynomial {
            coefficients: Vec::from_iter(iter),
        }
    }
}

impl<N: ComplexField> Default for Polynomial<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> AbstractMagma<Additive> for Polynomial<N> {
    fn operate(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

impl<N: ComplexField> Zero for Polynomial<N> {
    fn zero() -> Polynomial<N> {
        Polynomial::new()
    }

    fn is_zero(&self) -> bool {
        for val in &self.coefficients {
            if !val.is_zero() {
                return false;
            }
        }
        true
    }
}

// TODO: Add other alga traits

// Operator overloading

impl<N: ComplexField> ops::Add<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn add(mut self, rhs: N) -> Polynomial<N> {
        self.coefficients[0] += rhs;
        self
    }
}

impl<N: ComplexField> ops::Add<N> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn add(self, rhs: N) -> Polynomial<N> {
        let mut coefficients = Vec::from(self.coefficients.as_slice());
        coefficients[0] += rhs;
        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::Add<Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Add<&Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Add<Polynomial<N>> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn add(self, rhs: Polynomial<N>) -> Polynomial<N> {
        let min_order = self.coefficients.len().min(rhs.coefficients.len());
        let mut coefficients =
            Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
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

        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::Add<&Polynomial<N>> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn add(self, rhs: &Polynomial<N>) -> Polynomial<N> {
        let min_order = self.coefficients.len().min(rhs.coefficients.len());
        let mut coefficients =
            Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
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

        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::AddAssign<N> for Polynomial<N> {
    fn add_assign(&mut self, rhs: N) {
        self.coefficients[0] += rhs;
    }
}

impl<N: ComplexField> ops::AddAssign<Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::AddAssign<&Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Sub<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn sub(mut self, rhs: N) -> Polynomial<N> {
        self.coefficients[0] -= rhs;
        self
    }
}

impl<N: ComplexField> ops::Sub<N> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn sub(self, rhs: N) -> Polynomial<N> {
        let mut coefficients = Vec::from(self.coefficients.as_slice());
        coefficients[0] -= rhs;
        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::Sub<Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Sub<Polynomial<N>> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn sub(self, rhs: Polynomial<N>) -> Polynomial<N> {
        let min_order = self.coefficients.len().min(rhs.coefficients.len());
        let mut coefficients =
            Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
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

        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::Sub<&Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Sub<&Polynomial<N>> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn sub(self, rhs: &Polynomial<N>) -> Polynomial<N> {
        let min_order = self.coefficients.len().min(rhs.coefficients.len());
        let mut coefficients =
            Vec::with_capacity(self.coefficients.len().max(rhs.coefficients.len()));
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

        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::SubAssign<N> for Polynomial<N> {
    fn sub_assign(&mut self, rhs: N) {
        self.coefficients[0] -= rhs;
    }
}

impl<N: ComplexField> ops::SubAssign<Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::SubAssign<&Polynomial<N>> for Polynomial<N> {
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

impl<N: ComplexField> ops::Mul<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn mul(mut self, rhs: N) -> Polynomial<N> {
        for val in &mut self.coefficients {
            *val *= rhs;
        }
        self
    }
}

impl<N: ComplexField> ops::Mul<N> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn mul(self, rhs: N) -> Polynomial<N> {
        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        for val in &self.coefficients {
            coefficients.push(*val * rhs);
        }
        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::MulAssign<N> for Polynomial<N> {
    fn mul_assign(&mut self, rhs: N) {
        for val in self.coefficients.iter_mut() {
            *val *= rhs;
        }
    }
}

impl<N: ComplexField> ops::Div<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn div(mut self, rhs: N) -> Polynomial<N> {
        for val in &mut self.coefficients {
            *val /= rhs;
        }
        self
    }
}

impl<N: ComplexField> ops::Div<N> for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn div(self, rhs: N) -> Polynomial<N> {
        let mut coefficients = Vec::from(self.coefficients.as_slice());
        for val in &mut coefficients {
            *val /= rhs;
        }
        Polynomial { coefficients }
    }
}

impl<N: ComplexField> ops::DivAssign<N> for Polynomial<N> {
    fn div_assign(&mut self, rhs: N) {
        for val in &mut self.coefficients {
            *val /= rhs;
        }
    }
}

impl<N: ComplexField> ops::Neg for Polynomial<N> {
    type Output = Polynomial<N>;

    fn neg(mut self) -> Polynomial<N> {
        for val in &mut self.coefficients {
            *val = -*val;
        }
        self
    }
}

impl<N: ComplexField> ops::Neg for &Polynomial<N> {
    type Output = Polynomial<N>;

    fn neg(self) -> Polynomial<N> {
        Polynomial {
            coefficients: Vec::from_iter(self.coefficients.iter().map(|c| -*c)),
        }
    }
}
