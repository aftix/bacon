/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::DVector;
use alga::general::{ComplexField, RealField};

/// This trait allows a struct to be used in the Runge-Kutta solver.
///
/// # Examples
/// See `struct RungeKutta` and `struct RungeKuttaFehlberg` for examples of implementing
/// this trait.
pub trait RungeKuttaSolver {
  type Complex: ComplexField+From<f64>;

  /// Returns a slice of coeffecients to multiply the time step by when getting
  /// intermediate results. Upper-left portion of Butch Tableaux
  fn t_coefficients() -> Vec<<Self::Complex as ComplexField>::RealField>;

  /// Returns the coefficients to use on the k_i's when finding another
  /// k_i. Upper-right portion of the Butch Tableax. Should be
  /// an NxN-1 matrix, where N is the order of the Runge-Kutta Method (Or order+1 for
  /// adaptive methods)
  fn k_coefficients() -> Vec<Vec<<Self::Complex as ComplexField>::RealField>>;

  /// Coefficients to use when calculating the final step to take.
  /// These are the weights of the weighted average of k_i's. Bottom
  /// portion of the Butch Tableaux. For adaptive methods, this is the first
  /// row of the bottom portion.
  fn avg_coefficients() -> Vec<<Self::Complex as ComplexField>::RealField>;

  /// Used for adaptive methods only. Coefficients to use on
  /// the k_i's to find the error between the two orders
  /// of Runge-Kutta methods. In the Butch Tableaux, this is
  /// the first row of the bottom portion minus the second row.
  fn error_coefficients() -> Vec<<Self::Complex as ComplexField>::RealField> {
    vec![Self::Complex::from(0.0).real()]
  }

  /// Return this method's current time step
  fn dt(&self) -> <Self::Complex as ComplexField>::RealField;

  /// Returns whether or not this method is adaptive
  fn adaptive() -> bool {
    false
  }

  /// Used for adaptive solvers to update the timestep based on
  /// the current error.
  ///
  /// # Returns
  /// Returns Ok(true) if this step should be accepted, Ok(false) if this step
  /// should be rejected, and Err if the timestep became less than the minimum value.
  fn update_dt(&mut self, _error: <Self::Complex as ComplexField>::RealField) -> Result<bool, String> {
    Ok(true)
  }
}

type DerivativeFunc<Complex, Real, T> = fn(Real, &[Complex], &mut T) -> DVector<Complex>;
type ReturnType<Complex, Real> = Result<Vec<(Real, DVector<Complex>)>, String>;
/// Use a Runge-Kutta method to solve an initial value problem.
///
/// This function takes a Runge-Kutta solver, adaptive or not,
/// and solves an initial value problem defined by `y_0` as the initial
/// value and `derivs` as the derivative function.
///
/// # Return
/// On success, an `Ok(vec)` where `vec` is a vector of steps
/// of the form `(t_n, y_n)` with y_n being a vector equal in
/// dimension to `y_0`.
///
/// # Params
/// `solver` A solver implementing `RungeKuttaSolver`
///
/// `(t_initial, t_final)` Interval to solve the initial value problem on
///
/// `y_0` initial values for the ivp
///
/// `derivs` Derivative function. Should take the arguments `(time, slice of all y_n's, params)` where
/// y_n is the value of the initial value problem at time `time`.
///
/// `params` Mutable reference to a type that implements `Clone`. `params` is cloned
/// for all intermediate steps done by the solver so that `derivs` at `t_n+1` gets
/// the params passed from `derivs` at `t_n`, not some intermediate `k` step.
///
/// # Examples
///
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{RungeKutta, runge_kutta};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
///
/// //...
/// fn example() {
///   let rk = RungeKutta::default().with_dt(0.01).build();
///   let path = runge_kutta(rk, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
pub fn runge_kutta<S: RungeKuttaSolver, T: Clone>(
  mut solver: S,
  (t_initial, t_final): (<S::Complex as ComplexField>::RealField,<S::Complex as ComplexField>::RealField),
  y_0: &[S::Complex],
  derivs: DerivativeFunc<S::Complex, <S::Complex as ComplexField>::RealField, T>,
  params: &mut T
) -> ReturnType<S::Complex, <S::Complex as ComplexField>::RealField> {
  let mut state = DVector::from_column_slice(y_0);
  let mut path = vec![(t_initial, state.clone())];

  let num_k = S::avg_coefficients().len();

  let mut time = t_initial;

  while time < t_final {
    let old_params = params.clone();
    let mut k: Vec<DVector<S::Complex>> = vec![];

    let mut new_params = old_params.clone();
    for ind in 0..num_k {
      state = path.last().unwrap().1.clone();
      for (j, k) in k.iter().enumerate() {
        state += k * S::Complex::from_real(S::k_coefficients()[ind][j]);
      }
      k.push(
        derivs(
          time + S::t_coefficients()[ind]*solver.dt(),
          state.column(0).as_slice(),
          params
        ) * S::Complex::from_real(solver.dt())
      );
      if ind == 0 {
        new_params = params.clone();
      }
      *params = old_params.clone();
    }
    *params = new_params.clone();

    state = path.last().unwrap().1.clone();
    for (ind, k) in k.iter().enumerate() {
      state += k * S::Complex::from_real(S::avg_coefficients()[ind]);
    }

    let mut error: <S::Complex as ComplexField>::RealField = S::Complex::from(0.0).real();
    if S::adaptive() {
      let mut error_vec = k[0].clone() * S::Complex::from_real(S::error_coefficients()[0]);
      for (ind, k) in k.iter().enumerate().skip(1) {
        error_vec += k * S::Complex::from_real(S::error_coefficients()[ind]);
      }
      error = error_vec.dot(&error_vec).real() / solver.dt();
    }

    let old_dt = solver.dt();
    if solver.update_dt(error)? {
      time += old_dt;
      path.push((time, state.clone()));
    }
  }

  Ok(path)
}

/// Solver for the fourth order Runge-Kutta method
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{RungeKutta, runge_kutta};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// //...
/// fn example() {
///   let rk = RungeKutta::default().with_dt(0.01).build();
///   let path = runge_kutta(rk, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKutta<N: ComplexField+From<f64>+Copy> {
  dt: <N as ComplexField>::RealField,
}

/// Builds a RungeKutta solver
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaBuilder<N: ComplexField+From<f64>+Copy> {
  solver: RungeKutta<N>,
}

impl<N: ComplexField+From<f64>+Copy> RungeKutta<N> {
  /// Get a builder to make a RungeKutta solver
  pub fn default() -> RungeKuttaBuilder<N> {
    RungeKuttaBuilder {
      solver: RungeKutta{
        dt: N::from(0.01).real(),
      }
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> RungeKuttaBuilder<N> {
  /// Make a RungeKutta solver out of this builder
  pub fn build(self) -> RungeKutta<N> {
    self.solver
  }

  /// Set the timestep for the RungeKutta solver
  pub fn with_dt(&mut self, dt: <N as ComplexField>::RealField) -> &mut RungeKuttaBuilder<N> {
    if dt <= N::from(0.0).real() {
      panic!("dt must be positive");
    }
    self.solver.dt = dt;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> RungeKuttaSolver for RungeKutta<N> {
  type Complex = N;

  fn t_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(0.0).real(),
      N::from(0.5).real(),
      N::from(0.5).real(),
      N::from(1.0).real(),
    ]
  }

  fn k_coefficients() -> Vec<Vec<<N as ComplexField>::RealField>> {
    vec![
      vec![N::from(0.0).real(), N::from(0.0).real(), N::from(0.0).real()],
      vec![N::from(0.5).real(), N::from(0.0).real(), N::from(0.0).real()],
      vec![N::from(0.0).real(), N::from(0.5).real(), N::from(0.0).real()],
      vec![N::from(0.0).real(), N::from(0.0).real(), N::from(1.0).real()],
    ]
  }

  fn avg_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(1.0/6.0).real(),
      N::from(1.0/3.0).real(),
      N::from(1.0/3.0).real(),
      N::from(1.0/6.0).real(),
    ]
  }

  fn dt(&self) -> <N as ComplexField>::RealField {
    self.dt
  }
}

/// Solver for the Runge-Kutta-Fehlberg Solver
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{runge_kutta, RungeKuttaFehlberg};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// //...
/// fn example() {
///   let rkf = RungeKuttaFehlberg::default().with_dt_min(0.001).with_dt_max(0.01).with_tolerance(0.01).build();
///   let path = runge_kutta(rkf, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaFehlberg<N: ComplexField+From<f64>+Copy> {
  dt: <N as ComplexField>::RealField,
  dt_min: <N as ComplexField>::RealField,
  dt_max: <N as ComplexField>::RealField,
  tolerance: <N as ComplexField>::RealField,
}

/// Builder for a RungeKuttaFehlberg solver
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaFehlbergBuilder<N: ComplexField+From<f64>+Copy> {
  solver: RungeKuttaFehlberg<N>,
}

impl<N: ComplexField+From<f64>+Copy> RungeKuttaFehlberg<N> {
  /// Get a builder for a new RungeKuttaFehlberg solver
  pub fn default() -> RungeKuttaFehlbergBuilder<N> {
    RungeKuttaFehlbergBuilder {
      solver: RungeKuttaFehlberg {
        dt: N::from(0.01).real(),
        dt_max: N::from(0.1).real(),
        dt_min: N::from(0.001).real(),
        tolerance: N::from(0.001).real(),
      }
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> RungeKuttaFehlbergBuilder<N> {
  /// Build this RungeKuttaFehlberg solver
  pub fn build(mut self) -> RungeKuttaFehlberg<N> {
    if self.solver.dt_min >= self.solver.dt_max {
      panic!("dt_min must be <= dt_max");
    }
    self.solver.dt = self.solver.dt_max;
    self.solver
  }

  /// Set the minimum timestep for this solver
  pub fn with_dt_min(&mut self, dt_min: <N as ComplexField>::RealField) -> &mut RungeKuttaFehlbergBuilder<N> {
    if !dt_min.is_sign_positive() {
      panic!("dt_min must be positive");
    }
    self.solver.dt_min = dt_min;
    self
  }

  /// Set the maximum timestep for this solver.
  pub fn with_dt_max(&mut self, dt_max: <N as ComplexField>::RealField) -> &mut RungeKuttaFehlbergBuilder<N> {
    if !dt_max.is_sign_positive() {
      panic!("dt_max must be positive");
    }
    self.solver.dt_max = dt_max;
    self
  }

  /// Set the error tolerance for this solver
  pub fn with_tolerance(&mut self, tol: <N as ComplexField>::RealField) -> &mut RungeKuttaFehlbergBuilder<N> {
    if !tol.is_sign_positive() {
      panic!("tolerance must be positive");
    }
    self.solver.tolerance = tol;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> RungeKuttaSolver for RungeKuttaFehlberg<N> {
  type Complex = N;

  fn t_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(0.0).real(),
      N::from(0.25).real(),
      N::from(3.0/8.0).real(),
      N::from(12.0/13.0).real(),
      N::from(1.0).real(),
      N::from(0.5).real(),
    ]
  }

  fn k_coefficients() -> Vec<Vec<<N as ComplexField>::RealField>> {
    vec![
      vec![
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
      ],
      vec![
        N::from(1.0/4.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
      ],
      vec![
        N::from(3.0/32.0).real(),
        N::from(9.0/32.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
      ],
      vec![
        N::from(1932.0/2197.0).real(),
        N::from(-7200.0/2197.0).real(),
        N::from(7296.0/2197.0).real(),
        N::from(0.0).real(),
        N::from(0.0).real(),
      ],
      vec![
        N::from(439.0/216.0).real(),
        N::from(-8.0).real(),
        N::from(3680.0/513.0).real(),
        N::from(-845.0/4104.0).real(),
        N::from(0.0).real(),
      ],
      vec![
        N::from(-8.0/27.0).real(),
        N::from(2.0).real(),
        N::from(-3544.0/2565.0).real(),
        N::from(1859.0/4104.0).real(),
        N::from(-11.0/40.0).real(),
      ]
    ]
  }

  fn avg_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(25.0/216.0).real(),
      N::from(0.0).real(),
      N::from(1408.0/2565.0).real(),
      N::from(2197.0/4104.0).real(),
      N::from(-1.0/5.0).real(),
      N::from(0.0).real(),
    ]
  }

  fn error_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(1.0/360.0).real(),
      N::from(0.0).real(),
      N::from(-128.0/4275.0).real(),
      N::from(-2197.0/75240.0).real(),
      N::from(1.0/50.0).real(),
      N::from(2.0/55.0).real(),
    ]
  }

  fn dt(&self) -> <N as ComplexField>::RealField {
    self.dt
  }

  fn update_dt(&mut self, error: <N as ComplexField>::RealField) -> Result<bool, String>{
    let delta = N::from(0.84).real() * (self.tolerance/error).powf(N::from(0.25).real());
    if delta <= N::from(0.1).real() {
      self.dt *= N::from(0.1).real();
    } else if delta >= N::from(4.0).real() {
      self.dt *= N::from(4.0).real();
    } else {
      self.dt *= delta;
    }

    if self.dt > self.dt_max {
      self.dt = self.dt_max;
    }

    if self.dt < self.dt_min {
      Err("Mininum dt exceeded".to_owned())
    } else {
      Ok(error <= self.tolerance)
    }
  }

  fn adaptive() -> bool {
    true
  }
}
