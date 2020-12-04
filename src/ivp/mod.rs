use nalgebra::{DVector};

pub mod rk;
pub use rk::*;

pub fn euler<T>(
  (t_initial, t_final): (f64, f64),
  y_0: &[f64],
  dt: f64,
  derivs: fn(f64, &[f64], &mut T) -> DVector<f64>,
  params: &mut T
) -> Vec<(f64, DVector<f64>)> {
  let mut state = DVector::from_column_slice(y_0);
  let mut path = vec![(t_initial, state.clone())];

  let mut time = t_initial;

  while time < t_final {
    let f = derivs(time, state.column(0).as_slice(), params);
    state += dt * f;
    time += dt;
    path.push((time, state.clone()));
  }

  path
}
