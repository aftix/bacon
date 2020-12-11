/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::ivp::{Euler, IVPSolver};
use nalgebra::DVector;

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(DVector::from_column_slice(y))
}

fn quadratic_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(DVector::from_iterator(y.len(), [-2.0 * t].repeat(y.len())))
}

// Test euler method on y = exp(x), y' = exp(x) = y, y(0) = 1
#[test]
fn euler_test_exp() {
    let t_initial = 0.0;
    let t_final = 1.0;
    let dt = 0.0005;

    let mut euler = Euler::new()
        .with_dt_max(dt)
        .unwrap()
        .with_start(t_initial)
        .unwrap()
        .with_end(t_final)
        .unwrap()
        .with_initial_conditions(&[1.0])
        .unwrap()
        .build();

    let path = euler.solve_ivp(exp_deriv, &mut ()).unwrap();

    for step in &path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.exp(),
            epsilon = 0.001
        ));
    }
}

// Test euler method on y = 1 - x^2, y' = -2x, y(0) = 1
#[test]
fn euler_test_quadratic() {
    let t_initial = 0.0;
    let t_final = 1.0;
    let dt = 0.0001;

    let mut euler = Euler::new()
        .with_dt_max(dt)
        .unwrap()
        .with_start(t_initial)
        .unwrap()
        .with_end(t_final)
        .unwrap()
        .with_initial_conditions(&[1.0])
        .unwrap()
        .build();

    let path = euler.solve_ivp(quadratic_deriv, &mut ()).unwrap();

    for step in &path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            1.0 - step.0.powi(2),
            epsilon = 0.001
        ));
    }
}
