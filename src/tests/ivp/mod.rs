/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

mod adams;
mod bdf;
mod euler;
mod rk;

use crate::ivp::solve_ivp;
use nalgebra::DVector;

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(DVector::from_column_slice(y))
}

fn quadratic_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(DVector::from_iterator(y.len(), [-2.0 * t].repeat(y.len())))
}

fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(DVector::from_iterator(y.len(), y.iter().map(|_| t.cos())))
}

fn unstable_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<DVector<f64>, String> {
    Ok(-DVector::from_column_slice(y))
}

#[test]
fn test_ivp_exp() {
    let t_initial = 0.0;
    let t_final = 7.0;

    let path = solve_ivp(
        (t_initial, t_final),
        (0.1, 0.001),
        &[1.0],
        &exp_deriv,
        0.00001,
        &mut (),
    )
    .unwrap();

    for step in path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.exp(),
            epsilon = 0.01
        ));
    }
}

#[test]
fn test_ivp_quadratic() {
    let t_initial = 0.0;
    let t_final = 10.0;

    let path = solve_ivp(
        (t_initial, t_final),
        (0.1, 0.001),
        &[1.0],
        &quadratic_deriv,
        0.00001,
        &mut (),
    )
    .unwrap();

    for step in path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            1.0 - step.0.powi(2),
            epsilon = 0.01
        ));
    }
}

#[test]
fn test_ivp_sin() {
    let t_initial = 0.0;
    let t_final = 7.0;

    let path = solve_ivp(
        (t_initial, t_final),
        (0.1, 0.001),
        &[0.0],
        &sine_deriv,
        0.00001,
        &mut (),
    )
    .unwrap();

    for step in path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.sin(),
            epsilon = 0.01
        ));
    }
}

#[test]
fn test_ivp_unstable() {
    let t_initial = 0.0;
    let t_final = 10.0;

    let path = solve_ivp(
        (t_initial, t_final),
        (0.1, 0.001),
        &[1.0],
        &unstable_deriv,
        0.00001,
        &mut (),
    )
    .unwrap();

    for step in path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            (-step.0).exp(),
            epsilon = 0.01
        ));
    }
}
