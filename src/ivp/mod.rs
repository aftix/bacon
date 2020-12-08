use nalgebra::{DVector};

pub mod rk;
pub mod adams;
pub use rk::*;
pub use adams::*;

/// Solves an initial value problem using euler's method. Don't use this.
///
/// Uses the basic recurrance relation y_(n+1) = y_n + dt*f(t, y_n) to solve
/// an initial value problem.
///
/// # Params
/// `(t_initial, t_final)` Interval to solve the ivp over
///
/// `y_0` Initial value, slice
///
/// `dt`  Timestep to use between iterations
///
/// `derivs` Derivative function for initial value problem. The arguments to
///   this functions should be (time, slice of current y_n's, mutable parameter object)
///
/// `params` Mutable reference to pass to `derivs`.
///
/// # Return
/// Returns a Vector of the steps taken to solve the initial value problem.
/// The format of the vector is (t_n, y_n)
///
/// # Example
///
/// ```
/// // Derivative of y_i = exp(y_i)
/// fn derivative(_t: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
///   DVector::from_iterator(y.len(), y.iter())
/// }
/// ...
/// let path = euler((0.0, 1.0), &[1.0], 0.01, derivative, &mut ());
/// ```
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
