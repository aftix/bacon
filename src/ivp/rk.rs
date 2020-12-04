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

  // For adaptive solvers, get the error tolerance
  fn tolerance(&self) -> f64 {
    0.0
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
      for j in 0..ind {
        state += S::k_coefficients()[ind][j] * &k[j];
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
      for ind in 1..k.len() {
        error_vec += &k[ind] * S::error_coefficients()[ind];
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
  pub fn new() -> RungeKuttaBuilder {
    RungeKuttaBuilder {
      solver: RungeKutta{
        dt: 0.01,
      }
    }
  }
}

impl RungeKuttaBuilder {
  pub fn build(&self) -> RungeKutta {
    RungeKutta {
      dt: self.solver.dt,
    }
  }

  pub fn with_dt(&mut self, dt: f64) -> &mut RungeKuttaBuilder {
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
