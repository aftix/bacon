/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::DVector;
use std::collections::VecDeque;
use alga::general::{ComplexField, RealField};

/// This trait allows a struct to be used in the Adams-Bashforth/Predictor-corrector solver.
///
/// # Examples
/// See `struct AdamsBashforth` and `struct PredictorCorrector` for examples of
/// implementing this trait.
pub trait AdamsSolver {
  type Complex: ComplexField+From<f64>;

  /// Returns a slice of coefficients to weight the previous
  /// steps of the explicit solver with. The length of the slice
  /// is the order of the predictor.
  /// The first element is the weight of the n-1 step,
  /// the next is the n-2 step, etc.
  fn predictor_coefficients() -> Vec<<Self::Complex as ComplexField>::RealField>;

  /// Returns a slice of coefficients to weight the
  /// previous steps of the implicit solver with. The length
  /// of the slice is the order of the corrector.
  /// The length must be at most 1 more than the predictor
  /// coefficients (so if the predictor uses previous function
  /// evaluations up to n-k then the corrector can only use up to n-k).
  fn corrector_coefficients() -> Vec<<Self::Complex as ComplexField>::RealField> {
    vec![]
  }

  /// Returns true if this is an adaptive predictor-corrector method.
  /// If this is false, corrector_coefficients is never called.
  fn predictor_corrector() -> bool {
    false
  }

  /// Returns the current timestep
  fn dt(&self) -> <Self::Complex as ComplexField>::RealField;

  /// For predictor-correctors, update the current timestep based on the error.
  ///
  /// # Return
  /// Returns an Ok(0) if the last step failed, Ok(1) if the last step
  /// passed with no update to dt, and Ok(2) if the last step passed
  /// with an update to dt. Returns Err if the timestep goes under the
  /// minimum timestep specified.
  fn update_dt(&mut self, _error: <Self::Complex as ComplexField>::RealField) -> Result<u8, String> {
    Ok(1)
  }

  /// For predictor-corrector methods, the value to weight the error by.
  fn error_coefficient() -> <Self::Complex as ComplexField>::RealField {
    Self::Complex::from(0.0).real()
  }
}

type DerivativeFunc<Complex, Real, T> = fn(Real, &[Complex], &mut T) -> DVector<Complex>;
type ReturnType<Complex, Real> = Result<Vec<(Real, DVector<Complex>)>, String>;
/// Use an Adams-Bashforth method or an
/// Adams-Bashforth/Adams-Moulton predictor-corrector to solve an IVP
///
/// This function takes an Adams-Bashforth or Adams-Bashforth/Adams-Moulton predictor-corrector
/// solver and solves an initial value problem defined by `y_0' as the initial value
/// and `derivs` as the derivative function. The initial few steps to start the
/// solver and the initial steps taken after a change in dt are done using
/// the classic Runge-Kutta fourth order method.
///
/// # Return
/// On success, an `Ok(vec)` where `vec` is a vector of steps of the
/// form `(t_n, y_n)` with y_n being a vector equal in dimension to
/// `y_0`.
///
/// # Params
/// `solver` a solver implementing `AdamsSolver`.
///
/// `(t_initial, t_final)` Interval to solve the intial value problem on
///
/// `y_0` initial values for the ivp
///
/// `derivs` Derivative function. Should take the argument `(time, slice of all y_n's, params)`
/// where y_n is the value of the initial value problem at time `time`.
///
/// `params` Mutable reference to a type that implements `Clone`. `params` is cloned
/// for all intermediate steps done by the solver so that `derivs` at `t_n+1` gets
/// the params passed from `derivs` at `t_n`, not some intermediate `k` step.
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{AdamsBashforth,adams};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// //...
/// fn example() {
///   let adam = AdamsBashforth::default().with_dt(0.01).build();
///   let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
pub fn adams<S: AdamsSolver, T: Clone>(
  mut solver: S,
  (t_initial, t_final): (<S::Complex as ComplexField>::RealField,<S::Complex as ComplexField>::RealField),
  y_0: &[S::Complex],
  derivs: DerivativeFunc<S::Complex, <S::Complex as ComplexField>::RealField, T>,
  params: &mut T
) -> ReturnType<S::Complex, <S::Complex as ComplexField>::RealField> {
  let state = DVector::from_column_slice(y_0);
  let mut path = vec![(t_initial, state)];

  let mut params_considering = params.clone();

  let mut considering = VecDeque::with_capacity(S::predictor_coefficients().len());

  let rk = super::RungeKutta::<S::Complex>::default().with_dt(solver.dt()).build();
  let initial = super::runge_kutta(
      rk,
      (
        t_initial,
        t_initial + S::Complex::from(S::predictor_coefficients().len() as f64).real() * solver.dt()
      ),
      y_0,
      derivs,
      params
    )?;
  for i in initial {
    considering.push_back(i);
  }
  while considering.len() > S::predictor_coefficients().len() {
    considering.pop_front();
  }

  let mut memory = VecDeque::with_capacity(considering.len());
  for (i, state) in considering.iter().enumerate() {
    if i == 0 {
      memory.push_back(derivs(state.0, y_0, &mut params_considering));
    } else {
      memory.push_back(derivs(state.0, considering[i - 1].1.column(0).as_slice(), &mut params_considering));
    }
  }

  let mut last = false;
  let mut nflag = true;
  let mut time = considering.back().unwrap().0;

  if time > t_final {
    for c in &considering {
      path.push(c.clone());
    }
    return Ok(path);
  }

  'out: loop {
    let mut predictor = considering.back().unwrap().1.clone();
    for i in 0..considering.len() {
      predictor += &memory[considering.len() - i - 1] * S::Complex::from_real(S::predictor_coefficients()[i] *solver.dt());
    }

    let implicit = derivs(time + solver.dt(), predictor.column(0).as_slice(), &mut params_considering);

    if S::predictor_corrector() {
      let mut corrector = considering.back().unwrap().1.clone();
      for i in 0..considering.len() {
        corrector += &memory[considering.len() - i - 1] * S::Complex::from_real(S::corrector_coefficients()[i+1] * solver.dt());
      }
      corrector += &implicit * S::Complex::from_real(S::corrector_coefficients()[0] * solver.dt());

      let error = &corrector - &predictor;
      let error = error.dot(&error).real().sqrt() * S::error_coefficient() / solver.dt();

      let dt_old = solver.dt();
      let result = solver.update_dt(error)?;
      if result > 0 {
        if nflag {
          for c in &considering {
            path.push(c.clone());
          }
          nflag = false;
        }
        time += dt_old;
        path.push((time, corrector.clone()));
        memory.pop_front();
        memory.push_back(implicit);
        considering.pop_front();
        considering.push_back((time, corrector.clone()));
        if last {
          break 'out;
        }
        if result == 2 || time + dt_old > t_final {
          let mut dt_old = solver.dt();
          if time + S::Complex::from(4.0).real()*dt_old > t_final {
            last = true;
            dt_old = (t_final - time) / (S::Complex::from(4.0).real() * dt_old);
          }

          considering.clear();
          memory.clear();
          *params = params_considering.clone();
          let rk = super::RungeKutta::default().with_dt(dt_old).build();
          let initial = super::runge_kutta(
              rk,
              (time, time + S::Complex::from(S::predictor_coefficients().len() as f64).real() * dt_old),
              y_0,
              derivs,
              params
            )?;
          for i in initial {
            considering.push_back(i);
          }
          while considering.len() > S::predictor_coefficients().len() {
            considering.pop_front();
          }

          for (i, state) in considering.iter().enumerate() {
            if i == 0 {
              memory.push_back(derivs(state.0, y_0, &mut params_considering));
            } else {
              memory.push_back(derivs(state.0, considering[i - 1].1.column(0).as_slice(), &mut params_considering));
            }
          }

          nflag = true;
        }
      } else if nflag {
        considering.clear();
        memory.clear();
        *params = params_considering.clone();
        let rk = super::RungeKutta::default().with_dt(solver.dt()).build();
        let initial = super::runge_kutta(
            rk,
            (time, time + S::Complex::from(S::predictor_coefficients().len() as f64).real() * solver.dt()),
            y_0,
            derivs,
            params
          )?;
        for i in initial {
          considering.push_back(i);
        }
        while considering.len() > S::predictor_coefficients().len() {
          considering.pop_front();
        }

        for (i, state) in considering.iter().enumerate() {
          if i == 0 {
            memory.push_back(derivs(state.0, y_0, &mut params_considering));
          } else {
            memory.push_back(derivs(state.0, considering[i - 1].1.column(0).as_slice(), &mut params_considering));
          }
        }
      }
    } else {
      if last {
        break 'out;
      }
      if nflag {
        for c in &considering {
          path.push(c.clone());
        }
        nflag = false;
      }

      time += solver.dt();
      path.push((time, predictor.clone()));
      if time > t_final {
        last = true;
      }

      memory.pop_front();
      memory.push_back(implicit);
      considering.pop_front();
      considering.push_back((time, predictor));

    }

  }

  Ok(path)
}

/// Solver for the fourth order Adams-Basforth method
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{adams,AdamsBashforth};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// ///...
/// fn example() {
///   let adam = AdamsBashforth::default().with_dt(0.01).build();
///   let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforth<N: ComplexField+From<f64>+Copy> {
  dt: <N as ComplexField>::RealField,
}

/// Builds an AdamsBashforth
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforthBuilder<N: ComplexField+From<f64>+Copy> {
  solver: AdamsBashforth<N>,
}

impl<N: ComplexField+From<f64>+Copy> AdamsBashforth<N> {
  /// Get a builder to make an AdamsBashforth solver
  pub fn default() -> AdamsBashforthBuilder<N> {
    AdamsBashforthBuilder {
      solver: AdamsBashforth{
        dt: N::from(0.01).real(),
      },
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> AdamsBashforthBuilder<N> {
  /// Make an AdamsBashforth solver
  pub fn build(self) -> AdamsBashforth<N> {
    self.solver
  }

  /// Set the timestep for this solver
  pub fn with_dt(&mut self, dt: <N as ComplexField>::RealField) -> &mut AdamsBashforthBuilder<N> {
    self.solver.dt = dt;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> AdamsSolver for AdamsBashforth<N> {
  type Complex = N;

  fn predictor_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(55.0/24.0).real(),
      N::from(-59.0/24.0).real(),
      N::from(37.0/24.0).real(),
      N::from(-9.0/24.0).real(),
    ]
  }

  fn dt(&self) -> <N as ComplexField>::RealField {
    self.dt
  }
}

/// Fourth order predictor-corrector solver
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon::ivp::{adams,PredictorCorrector};
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// //...
/// fn example() {
///   let adam = PredictorCorrector::default().with_dt_max(0.01).with_dt_min(0.0001).with_tolerance(0.001).build();
///   let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// }
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrector<N: ComplexField+From<f64>+Copy> {
  dt: <N as ComplexField>::RealField,
  dt_max: <N as ComplexField>::RealField,
  dt_min: <N as ComplexField>::RealField,
  tolerance: <N as ComplexField>::RealField,
}

/// Builder for a PredictorCorrector solver
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrectorBuilder<N: ComplexField+From<f64>+Copy> {
  solver: PredictorCorrector<N>
}

impl<N: ComplexField+From<f64>+Copy> PredictorCorrector<N> {
  /// Make a builder to get a PredictorCorrector solver
  pub fn default() -> PredictorCorrectorBuilder<N> {
    PredictorCorrectorBuilder {
      solver: PredictorCorrector {
        dt: N::from(0.01).real(),
        dt_min: N::from(0.01).real(),
        dt_max: N::from(0.1).real(),
        tolerance: N::from(0.005).real(),
      }
    }
  }
}

impl<N: ComplexField+From<f64>+Copy> PredictorCorrectorBuilder<N> {
  /// Get a PredictorCorrector solver
  pub fn build(mut self) -> PredictorCorrector<N> {
    if self.solver.dt_min >= self.solver.dt_max {
      panic!("dt_min must be <= dt_max");
    }
    self.solver.dt = self.solver.dt_max;
    self.solver
  }

  /// Set the minimum timestep for this solver
  pub fn with_dt_min(&mut self, dt_min: <N as ComplexField>::RealField) -> &mut PredictorCorrectorBuilder<N> {
    if !dt_min.is_sign_positive() {
      panic!("dt_min must be positive");
    }
    self.solver.dt_min = dt_min;
    self
  }

  /// Set the maximum timestep for this solver
  pub fn with_dt_max(&mut self, dt_max: <N as ComplexField>::RealField) -> &mut PredictorCorrectorBuilder<N> {
    if !dt_max.is_sign_positive() {
      panic!("dt_max must be positive");
    }
    self.solver.dt_max = dt_max;
    self
  }

  /// Set the error tolerance for this solver
  pub fn with_tolerance(&mut self, tol: <N as ComplexField>::RealField) -> &mut PredictorCorrectorBuilder<N> {
    if !tol.is_sign_positive() {
      panic!("tolerance must be positive");
    }
    self.solver.tolerance = tol;
    self
  }
}

impl<N: ComplexField+From<f64>+Copy> AdamsSolver for PredictorCorrector<N> {
  type Complex = N;

  fn predictor_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(55.0/24.0).real(),
      N::from(-59.0/24.0).real(),
      N::from(37.0/24.0).real(),
      N::from(-9.0/24.0).real(),
    ]
  }

  fn dt(&self) -> <N as ComplexField>::RealField {
    self.dt
  }

  fn corrector_coefficients() -> Vec<<N as ComplexField>::RealField> {
    vec![
      N::from(9.0/24.0).real(),
      N::from(19.0/24.0).real(),
      N::from(-5.0/24.0).real(),
      N::from(1.0/24.0).real(),
      N::from(0.0).real(),
    ]
  }

  fn error_coefficient() -> <N as ComplexField>::RealField {
    N::from(19.0/270.0).real()
  }

  fn update_dt(&mut self, error: <N as ComplexField>::RealField) -> Result<u8, String> {
    if error > self.tolerance {
      let q = (self.tolerance/(N::from(2.0).real()*error)).powf(N::from(0.25).real());
      if q <= N::from(0.1).real() {
        self.dt *= N::from(0.1).real();
      } else {
        self.dt *= q;
      }

      if self.dt < self.dt_min {
        return Err("minimum dt exceeded".to_owned());
      }

      return Ok(0);
    }

    if error < N::from(0.1).real() * self.tolerance {
      let q = (self.tolerance/(N::from(2.0).real()*error)).powf(N::from(0.25).real());
      if q >= N::from(4.0).real() {
        self.dt *= N::from(4.0).real();
      } else {
        self.dt *= q;
      }
      if self.dt > self.dt_max {
        self.dt = self.dt_max;
      }

      return Ok(2);
    }

    Ok(1)
  }
}
