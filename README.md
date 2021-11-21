Scientific Computing in Rust

# Features
* Initial value problem solving
* Root finding algorithms
* Polynomials
* Polynomial Interpolation
* Scientific Constants
* Special functions/polynomials
* Numeric quadrature
* Numeric differentiation

Explanations of the features can be found [here](https://aftix.xyz/home/bacon).

# Initial Value Problems

There are two adaptive Runge-Kutta methods, two
Adams predictor-correctors, and two adaptive Backwards Differentiation
Formulas implemented. The interface to all of the solvers is the same.
As an example, this code solves `y' = y` using the Runge-Kutta-Fehlberg
method.

```rust
use bacon_sci::ivp::{RK45, RungeKuttaSolver};
use nalgebra::SVector;

fn deriv(_t: f64, y: &[f64], _params: &mut ()) -> Result<SVector<f64, 1>, String> {
    Ok(SVector::<f64, 1>::from_column_slice(y))
}

fn solve() -> Result<(), String> {
    let solver = RK45::new()
                    .with_dt_min(0.01)?
                    .with_dt_max(0.1)?
                    .with_tolerance(1e-4)?
                    .with_initial_conditions(&[1.0])?
                    .with_start(0.0)?
                    .with_end(10.0)?
                    .build();
    let _solution = solver.solve_ivp(deriv, &mut ())?;
    Ok(())
}
```

There is also a `solve_ivp` function in `bacon_sci::ivp` that tries a fifth-order
predictor-corrector followed by the Runge-Kutta-Fehlberg method followed by
BDF6.

# Root Finding Algorithms

`bacon_sci::roots` implements the bisection method, Newton's method,
the secant method, Newton's method for polynomials, and Müller's method
for polynomials.

As an example, the following code snippet finds the root of `x^3` using
initial guesses of `0.1` and `-0.1`.

```rust
use bacon_sci::roots::secant;
use nalgebra::SVector;

fn cubic(x: &[f64]) -> SVector<f64, 1> {
    SVector::<f64, 1>::from_iterator(x.iter.map(|x| x.powi(3)))
}

fn solve() -> f64 {
    secant((&[-0.1], &[0.1]), cubic, 0.001, 1000).unwrap()
}
```

# Polynomials and Polynomial Interpolation

`bacon_sci::polynomial` implements a polynomial struct. `bacon_sci::interp` implements
Lagrange interpolation, Hermite interpolation, and cubic spline interpolation.

# Scientific Constants

Several scientific constants are defined in `bacon_sci::constants`. The data
comes from NIST. The 2018 CODATA complete listing is available as a hashmap.

# Special Functions and Polynomials

Currently, `bacon_sci::special` allows you to get Legendre polynomials, Hermite polynomials,
Laguerre polynomials, and Chebyshev polynomials.

# Numeric Differentiation and Quadrature

`bacon_sci::differentiate` allows first- and second-derivative evaluation numerically.
`bacon_sci::integrate` implements Tanh-Sinh quadrature, adaptive Simpson's rule,
Romberg integration, and several adaptive Gaussian integration schemes.
