use nalgebra::DVector;

pub trait RungeKuttaSolver {
  // Upper left portion of the tableaux
  fn t_coefficients() -> &'static [f64];

  // upper right portion of the tableaux
  fn k_coefficients() -> &'static [&'static [f64]];

  // Coeffecients to use when combining the partial steps with in the weighted averaging
  fn avg_coefficients() -> &'static [f64];

  // Coeffecients for finding the error in adaptive methods
  // Only used when adaptive is on, so non-adaptive methods don't
  //  need to match dimension with the other coefficients
  fn error_coefficients() -> &'static [f64] {
    &[0.0]
  }

  // Get the time step
  fn dt(&self) -> f64;

  // Is this solver adaptive time step?
  fn adaptive() -> bool {
    false
  }

  // For adaptive solvers, update dt
  // Returns Ok(true) if state should be accepted
  // Returns Ok(false) if state should be rejected
  // Returns Err on error
  fn update_dt(&mut self, _error: f64) -> Result<bool, String> {
    Ok(true)
  }
}

// Use a runge-kutta method to solve an IVP
pub fn runge_kutta<T: Clone, S: RungeKuttaSolver>(
  mut solver: S,
  (t_initial, t_final): (f64,f64),
  y_0: &[f64],
  derivs: fn(f64, &[f64], &mut T) -> DVector<f64>,
  params: &mut T
) -> Result<Vec<(f64, DVector<f64>)>, String> {
  let mut state = DVector::from_column_slice(y_0);
  let mut path = vec![(t_initial, state.clone())];

  let num_k = S::avg_coefficients().len();

  let mut time = t_initial;

  while time < t_final {
    let old_params = params.clone();
    let mut k: Vec<DVector<f64>> = vec![];

    let mut new_params = old_params.clone();
    for ind in 0..num_k {
      state = path.last().unwrap().1.clone();
      for (j, k) in k.iter().enumerate() {
        state += S::k_coefficients()[ind][j] * k;
      }
      k.push(
        derivs(
          time + S::t_coefficients()[ind]*solver.dt(),
          state.column(0).as_slice(),
          params
        ) * solver.dt()
      );
      if ind == 0 {
        new_params = params.clone();
      }
      *params = old_params.clone();
    }
    *params = new_params.clone();

    state = path.last().unwrap().1.clone();
    for (ind, k) in k.iter().enumerate() {
      state += k * S::avg_coefficients()[ind];
    }

    let mut error = 0.0;
    if S::adaptive() {
      let mut error_vec = k[0].clone() * S::error_coefficients()[0];
      for (ind, k) in k.iter().enumerate().skip(1) {
        error_vec += k * S::error_coefficients()[ind];
      }
      error = error_vec.norm() / solver.dt();
    }

    let old_dt = solver.dt();
    if solver.update_dt(error)? {
      time += old_dt;
      path.push((time, state.clone()));
    }
  }

  Ok(path)
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKutta {
  dt: f64
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaBuilder {
  solver: RungeKutta,
}

impl RungeKutta {
  pub fn default() -> RungeKuttaBuilder {
    RungeKuttaBuilder {
      solver: RungeKutta{
        dt: 0.01,
      }
    }
  }
}

impl RungeKuttaBuilder {
  pub fn build(self) -> RungeKutta {
    self.solver
  }

  pub fn with_dt(&mut self, dt: f64) -> &mut RungeKuttaBuilder {
    if dt <= 0.0 {
      panic!("dt must be positive");
    }
    self.solver.dt = dt;
    self
  }
}

impl RungeKuttaSolver for RungeKutta {
  fn t_coefficients() -> &'static [f64] {
    &[0.0, 0.5, 0.5, 1.0]
  }

  fn k_coefficients() -> &'static [&'static [f64]] {
    &[
      &[0.0, 0.0, 0.0],
      &[0.5, 0.0, 0.0],
      &[0.0, 0.5, 0.0],
      &[0.0, 0.0, 1.0]
    ]
  }

  fn avg_coefficients() -> &'static [f64] {
    &[1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
  }

  fn dt(&self) -> f64 {
    self.dt
  }
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaFehlberg {
  dt: f64,
  dt_min: f64,
  dt_max: f64,
  tolerance: f64
}

#[derive(Debug,Copy,Clone)]
#[cfg_attr(feature="serialize",derive(Serialize,Deserialize))]
pub struct RungeKuttaFehlbergBuilder {
  solver: RungeKuttaFehlberg,
}

impl RungeKuttaFehlberg {
  pub fn default() -> RungeKuttaFehlbergBuilder {
    RungeKuttaFehlbergBuilder {
      solver: RungeKuttaFehlberg {
        dt: 0.01,
        dt_max: 0.1,
        dt_min: 0.001,
        tolerance: 0.001,
      }
    }
  }
}

impl RungeKuttaFehlbergBuilder {
  pub fn build(mut self) -> RungeKuttaFehlberg {
    if self.solver.dt_min >= self.solver.dt_max {
      panic!("dt_min must be <= dt_max");
    }
    self.solver.dt = self.solver.dt_max;
    self.solver
  }

  pub fn with_dt(&mut self, dt: f64) -> &mut RungeKuttaFehlbergBuilder {
    if dt <= 0.0 {
      panic!("dt must be positive");
    }
    self.solver.dt = dt;
    self
  }

  pub fn with_dt_min(&mut self, dt_min: f64) -> &mut RungeKuttaFehlbergBuilder {
    if dt_min <= 0.0 {
      panic!("dt_min must be positive");
    }
    self.solver.dt_min = dt_min;
    self
  }

  pub fn with_dt_max(&mut self, dt_max: f64) -> &mut RungeKuttaFehlbergBuilder {
    if dt_max <= 0.0 {
      panic!("dt_max must be positive");
    }
    self.solver.dt_max = dt_max;
    self
  }

  pub fn with_tolerance(&mut self, tol: f64) -> &mut RungeKuttaFehlbergBuilder {
    if tol <= 0.0 {
      panic!("tolerance must be positive");
    }
    self.solver.tolerance = tol;
    self
  }
}

impl RungeKuttaSolver for RungeKuttaFehlberg {
  fn t_coefficients() -> &'static [f64] {
    &[0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5]
  }

  fn k_coefficients() -> &'static [&'static [f64]] {
    &[
      &[0.0, 0.0, 0.0, 0.0, 0.0],
      &[1.0/4.0, 0.0, 0.0, 0.0, 0.0],
      &[3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0],
      &[1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0],
      &[439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
      &[-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0]
    ]
  }

  fn avg_coefficients() -> &'static [f64] {
    &[25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]
  }

  fn error_coefficients() -> &'static [f64] {
    &[1.0/360.0, 0.0, -128.0/4275.0, -2197.0/75240.0, 1.0/50.0, 2.0/55.0]
  }

  fn dt(&self) -> f64 {
    self.dt
  }

  fn update_dt(&mut self, error: f64) -> Result<bool, String>{
    let delta = 0.84 * (self.tolerance/error).powf(0.25);
    if delta <= 0.1 {
      self.dt *= 0.1;
    } else if delta >= 4.0 {
      self.dt *= 4.0;
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
