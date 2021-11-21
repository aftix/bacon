use crate::polynomial::Polynomial;
use nalgebra::ComplexField;
use num_complex::Complex;
use num_traits::{FromPrimitive, Zero};

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
///
/// # Examples
/// ```
/// use bacon_sci::polynomial::Polynomial;
/// use bacon_sci::roots::newton_polynomial;
/// fn example() {
///   let mut polynomial = Polynomial::new();
///   polynomial.set_coefficient(2, 1.0);
///   polynomial.set_coefficient(0, -1.0);
///   let solution = newton_polynomial(0.5, &polynomial, 0.0001, 1000).unwrap();
/// }
/// ```
pub fn newton_polynomial<N: ComplexField + FromPrimitive + Copy>(
    initial: N,
    poly: &Polynomial<N>,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<N, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
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

/// Use Mueller's method on a polynomial. Note this usually requires complex numbers.
///
/// Givin three initial guesses of a polynomial root, use Muller's method to
/// approximate within tol.
///
/// # Returns
/// `Ok(root)` when a root has been found, `Err` on failure
///
/// # Params
/// `initial` Triplet of initial guesses
///
/// `poly` the polynomial to solve the root for
///
/// `tol` the tolerance of relative error between iterations
///
/// `n_max` Maximum number of iterations
/// # Examples
/// ```
/// use bacon_sci::polynomial::Polynomial;
/// use bacon_sci::roots::muller_polynomial;
/// fn example() {
///   let mut polynomial = Polynomial::new();
///   polynomial.set_coefficient(2, 1.0);
///   polynomial.set_coefficient(0, -1.0);
///   let solution = muller_polynomial((0.0, 1.5, 2.0), &polynomial, 0.0001, 1000).unwrap();
/// }
/// ```
pub fn muller_polynomial<N: ComplexField + FromPrimitive + Copy>(
    initial: (N, N, N),
    poly: &Polynomial<N>,
    tol: <N as ComplexField>::RealField,
    n_max: usize,
) -> Result<Complex<<N as ComplexField>::RealField>, String>
where
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    let poly = poly.make_complex();
    let mut n = 0;
    let mut poly_0 = Complex::<N::RealField>::new(initial.0.real(), initial.0.imaginary());
    let mut poly_1 = Complex::<N::RealField>::new(initial.1.real(), initial.1.imaginary());
    let mut poly_2 = Complex::<N::RealField>::new(initial.2.real(), initial.1.imaginary());
    let mut h_1 = poly_1 - poly_0;
    let mut h_2 = poly_2 - poly_1;
    let poly_1_evaluated = poly.evaluate(poly_1);
    let mut poly_2_evaluated = poly.evaluate(poly_2);
    let mut delta_1 = (poly_1_evaluated - poly.evaluate(poly_0)) / h_1;
    let mut delta_2 = (poly_2_evaluated - poly_1_evaluated) / h_2;
    let mut delta = (delta_2 - delta_1) / (h_2 + h_1);

    let negtwo = N::RealField::from_i32(-2).unwrap();
    let four = N::RealField::from_i32(4).unwrap();

    while n < n_max {
        let b_coefficient = delta_2 + h_2 * delta;
        let determinate = (b_coefficient.powi(2)
            - Complex::<N::RealField>::new(four, N::RealField::zero()) * poly_2_evaluated * delta)
            .sqrt();
        let error = if (b_coefficient - determinate).abs() < (b_coefficient + determinate).abs() {
            b_coefficient + determinate
        } else {
            b_coefficient - determinate
        };
        let step =
            Complex::<N::RealField>::new(negtwo, N::RealField::zero()) * poly_2_evaluated / error;
        let p = poly_2 + step;

        if step.abs() <= tol {
            return Ok(p);
        }

        poly_0 = poly_1;
        poly_1 = poly_2;
        poly_2 = p;
        poly_2_evaluated = poly.evaluate(p);
        h_1 = poly_1 - poly_0;
        h_2 = poly_2 - poly_1;
        let poly_1_evaluated = poly.evaluate(poly_1);
        delta_1 = (poly_1_evaluated - poly.evaluate(poly_0)) / h_1;
        delta_2 = (poly_2_evaluated - poly_1_evaluated) / h_2;
        delta = (delta_2 - delta_1) / (h_1 + h_2);

        n += 1;
    }

    Err("Muller: maximum iterations exceeded".to_owned())
}
