// MIT License
//
// Copyright (c) 2020 Wyatt Campbell
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use nalgebra::DVector;
use std::collections::VecDeque;

/// This trait allows a struct to be used in the Adams-Bashforth/Predictor-corrector solver.
///
/// # Examples
/// See `struct AdamsBashforth` and `struct PredictorCorrector` for examples of
/// implementing this trait.
pub trait AdamsSolver {
  /// Returns a slice of coefficients to weight the previous
  /// steps of the explicit solver with. The length of the slice
  /// is the order of the predictor.
  /// The first element is the weight of the n-1 step,
  /// the next is the n-2 step, etc.
  fn predictor_coefficients() -> &'static [f64];

  /// Returns a slice of coefficients to weight the
  /// previous steps of the implicit solver with. The length
  /// of the slice is the order of the corrector.
  /// The length must be at most 1 more than the predictor
  /// coefficients (so if the predictor uses previous function
  /// evaluations up to n-k then the corrector can only use up to n-k).
  fn corrector_coefficients() -> &'static [f64] {
    &[]
  }

  /// Returns true if this is an adaptive predictor-corrector method.
  /// If this is false, corrector_coefficients is never called.
  fn predictor_corrector() -> bool {
    false
  }

  /// Returns the current timestep
  fn dt(&self) -> f64;

  // 0 = fail, 1 = pass with no update, 2 = pass with update
  /// For predictor-correctors, update the current timestep based on the error.
  ///
  /// # Return
  /// Returns an Ok(0) if the last step failed, Ok(1) if the last step
  /// passed with no update to dt, and Ok(2) if the last step passed
  /// with an update to dt. Returns Err if the timestep goes under the
  /// minimum timestep specified.
  fn update_dt(&mut self, _error: f64) -> Result<u8, String> {
    Ok(1)
  }

  /// For predictor-corrector methods, the value to weight the error by.
  fn error_coefficient() -> f64 {
    0.0
  }
}

// Use an adams method to solve an IVP
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
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_iterator(y.len(), y.iter())
/// }
/// ...
/// let adam = AdamsBashforth::default().with_dt(0.01).build();
/// let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// ```
pub fn adams<T: Clone, S: AdamsSolver>(
  mut solver: S,
  (t_initial, t_final): (f64,f64),
  y_0: &[f64],
  derivs: fn(f64, &[f64], &mut T) -> DVector<f64>,
  params: &mut T
) -> Result<Vec<(f64, DVector<f64>)>, String> {
  let state = DVector::from_column_slice(y_0);
  let mut path = vec![(t_initial, state)];

  let mut params_considering = params.clone();

  let mut considering: VecDeque<(f64, DVector<f64>)> = VecDeque::with_capacity(S::predictor_coefficients().len());

  let rk = super::RungeKutta::default().with_dt(solver.dt()).build();
  let initial = super::runge_kutta(
      rk,
      (t_initial, t_initial + S::predictor_coefficients().len() as f64 * solver.dt()),
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

  let mut memory: VecDeque<DVector<f64>> = VecDeque::with_capacity(considering.len());
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
      predictor += solver.dt() * &memory[considering.len() - i - 1] * S::predictor_coefficients()[i];
    }

    let implicit = derivs(time + solver.dt(), predictor.column(0).as_slice(), &mut params_considering);

    if S::predictor_corrector() {
      let mut corrector = considering.back().unwrap().1.clone();
      for i in 0..considering.len() {
        corrector += solver.dt() * &memory[considering.len() - i - 1] * S::corrector_coefficients()[i+1];
      }
      corrector += solver.dt() * &implicit * S::corrector_coefficients()[0];

      let error = (&corrector - &predictor).norm() * S::error_coefficient() / solver.dt();

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
          if time + 4.0*dt_old > t_final {
            last = true;
            dt_old = (t_final - time) / (4.0 * dt_old);
          }

          considering.clear();
          memory.clear();
          *params = params_considering.clone();
          let rk = super::RungeKutta::default().with_dt(dt_old).build();
          let initial = super::runge_kutta(
              rk,
              (time, time + S::predictor_coefficients().len() as f64 * dt_old),
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
            (time, time + S::predictor_coefficients().len() as f64 * solver.dt()),
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
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_iterator(y.len(), y.iter())
/// }
/// ...
/// let adam = AdamsBashforth::default().with_dt(0.01).build();
/// let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforth {
  dt: f64,
}

/// Builds an AdamsBashforth
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforthBuilder {
  solver: AdamsBashforth,
}

impl AdamsBashforth {
  /// Get a builder to make an AdamsBashforth solver
  pub fn default() -> AdamsBashforthBuilder {
    AdamsBashforthBuilder {
      solver: AdamsBashforth{
        dt: 0.01,
      },
    }
  }
}

impl AdamsBashforthBuilder {
  /// Make an AdamsBashforth solver
  pub fn build(self) -> AdamsBashforth {
    self.solver
  }

  /// Set the timestep for this solver
  pub fn with_dt(&mut self, dt: f64) -> &mut AdamsBashforthBuilder {
    self.solver.dt = dt;
    self
  }
}

impl AdamsSolver for AdamsBashforth {
  fn predictor_coefficients() -> &'static [f64] {
    &[55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0]
  }

  fn dt(&self) -> f64 {
    self.dt
  }
}

// Fourth order adams predictor corrector
/// Fourth order predictor-corrector solver
///
/// # Examples
/// ```
/// fn derivatives(_time: f64, y: &[f64], _params: &mut ()) -> DVector<f64> {
///   DVector::from_iterator(y.len(), y.iter())
/// }
/// ...
/// let adam = PredictorCorrector::default().with_dt_max(0.01).with_dt_min(0.0001).tolerance(0.001).build();
/// let path = adams(adam, (0.0, 1.0), &[1.0], derivatives, &mut ());
/// ```
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrector {
  dt: f64,
  dt_max: f64,
  dt_min: f64,
  tolerance: f64
}

/// Builder for a PredictorCorrector solver
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrectorBuilder {
  solver: PredictorCorrector
}

impl PredictorCorrector {
  /// Make a builder to get a PredictorCorrector solver
  pub fn default() -> PredictorCorrectorBuilder {
    PredictorCorrectorBuilder {
      solver: PredictorCorrector {
        dt: 0.01,
        dt_min: 0.01,
        dt_max: 0.1,
        tolerance: 0.005,
      }
    }
  }
}

impl PredictorCorrectorBuilder {
  /// Get a PredictorCorrector solver
  pub fn build(mut self) -> PredictorCorrector {
    if self.solver.dt_min >= self.solver.dt_max {
      panic!("dt_min must be <= dt_max");
    }
    self.solver.dt = self.solver.dt_max;
    self.solver
  }

  /// Set the minimum timestep for this solver
  pub fn with_dt_min(&mut self, dt_min: f64) -> &mut PredictorCorrectorBuilder {
    if dt_min <= 0.0 {
      panic!("dt_min must be positive");
    }
    self.solver.dt_min = dt_min;
    self
  }

  /// Set the maximum timestep for this solver
  pub fn with_dt_max(&mut self, dt_max: f64) -> &mut PredictorCorrectorBuilder {
    if dt_max <= 0.0 {
      panic!("dt_max must be positive");
    }
    self.solver.dt_max = dt_max;
    self
  }

  /// Set the error tolerance for this solver
  pub fn with_tolerance(&mut self, tol: f64) -> &mut PredictorCorrectorBuilder {
    if tol <= 0.0 {
      panic!("tolerance must be positive");
    }
    self.solver.tolerance = tol;
    self
  }
}

impl AdamsSolver for PredictorCorrector {
  fn predictor_coefficients() -> &'static [f64] {
    &[55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0]
  }

  fn dt(&self) -> f64 {
    self.dt
  }

  fn corrector_coefficients() -> &'static [f64] {
    &[9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0, 0.0]
  }

  fn error_coefficient() -> f64 {
    19.0/270.0
  }

  fn update_dt(&mut self, error: f64) -> Result<u8, String> {
    if error > self.tolerance {
      let q = (self.tolerance/(2.0*error)).powf(0.25);
      if q <= 0.1 {
        self.dt *= 0.1;
      } else {
        self.dt *= q;
      }

      if self.dt < self.dt_min {
        return Err("minimum dt exceeded".to_owned());
      }

      return Ok(0);
    }

    if error < 0.1 * self.tolerance {
      let q = (self.tolerance/(2.0*error)).powf(0.25);
      if q >= 4.0 {
        self.dt *= 4.0;
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
