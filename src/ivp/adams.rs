use nalgebra::DVector;
use std::collections::VecDeque;

pub trait AdamsSolver {
  fn predictor_coefficients() -> &'static [f64];

  fn corrector_coefficients() -> &'static [f64] {
    &[]
  }

  fn predictor_corrector() -> bool {
    false
  }

  fn dt(&self) -> f64;

  // 0 = fail, 1 = pass with no update, 2 = pass with update
  fn update_dt(&mut self, _error: f64) -> Result<u8, String> {
    Ok(1)
  }

  fn error_coefficient() -> f64 {
    0.0
  }
}

// Use an adams method to solve an IVP
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
        path.push((time + dt_old, corrector.clone()));
        memory.pop_front();
        memory.push_back(implicit);
        considering.pop_front();
        considering.push_back((time + dt_old, corrector.clone()));
        if last {
          break 'out;
        }
        if result == 2 || time + dt_old > t_final {
          let mut dt_old = solver.dt();
          if time + 4.0*solver.dt() > t_final {
            last = true;
            dt_old = (t_final - time) / (4.0 * solver.dt());
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

          time = considering.back().unwrap().0;
        } else {
          time += dt_old;
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

// Fourth order adams bashforth solver
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforth {
  dt: f64,
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct AdamsBashforthBuilder {
  solver: AdamsBashforth,
}

impl AdamsBashforth {
  pub fn default() -> AdamsBashforthBuilder {
    AdamsBashforthBuilder {
      solver: AdamsBashforth{
        dt: 0.01,
      },
    }
  }
}

impl AdamsBashforthBuilder {
  pub fn build(self) -> AdamsBashforth {
    self.solver
  }

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
#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrector {
  dt: f64,
  dt_max: f64,
  dt_min: f64,
  tolerance: f64
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct PredictorCorrectorBuilder {
  solver: PredictorCorrector
}

impl PredictorCorrector {
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
  pub fn build(mut self) -> PredictorCorrector {
    if self.solver.dt_min >= self.solver.dt_max {
      panic!("dt_min must be <= dt_max");
    }
    self.solver.dt = self.solver.dt_max;
    self.solver
  }

  pub fn with_dt(&mut self, dt: f64) -> &mut PredictorCorrectorBuilder {
    if dt <= 0.0 {
      panic!("dt must be positive");
    }
    self.solver.dt = dt;
    self
  }

  pub fn with_dt_min(&mut self, dt_min: f64) -> &mut PredictorCorrectorBuilder {
    if dt_min <= 0.0 {
      panic!("dt_min must be positive");
    }
    self.solver.dt_min = dt_min;
    self
  }

  pub fn with_dt_max(&mut self, dt_max: f64) -> &mut PredictorCorrectorBuilder {
    if dt_max <= 0.0 {
      panic!("dt_max must be positive");
    }
    self.solver.dt_max = dt_max;
    self
  }

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
