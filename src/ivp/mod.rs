/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use alga::general::{ComplexField, RealField};
use nalgebra::DVector;

pub mod adams;
pub mod rk;
pub use adams::*;
pub use rk::*;

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
/// use nalgebra::DVector;
/// use bacon_sci::ivp::euler;
/// // Derivative of y_i = exp(y_i)
/// fn derivative(_t: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
///   DVector::from_column_slice(y)
/// }
/// //...
/// fn example() {
///   let path = euler((0.0, 1.0), &[1.0], 0.01, derivative, &mut ());
/// }
/// ```
pub fn euler<N: RealField, M: ComplexField + From<N>, T>(
    (t_initial, t_final): (N, N),
    y_0: &[M],
    dt: N,
    derivs: fn(N, &[M], &mut T) -> DVector<M>,
    params: &mut T,
) -> Vec<(N, DVector<M>)> {
    let mut state = DVector::from_column_slice(y_0);
    let mut path = vec![(t_initial, state.clone())];

    let mut time = t_initial;

    while time < t_final {
        let f = derivs(time, state.column(0).as_slice(), params);
        state += f * M::from(dt);
        time += dt;
        path.push((time, state.clone()));
    }

    path
}
