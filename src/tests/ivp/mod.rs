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
use nalgebra::{VectorN, U1, U2};

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_column_slice(y))
}

fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_column_slice(&[-2.0 * t]))
}

fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_iterator(y.iter().map(|_| t.cos())))
}

fn unstable_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(-VectorN::<f64, U1>::from_column_slice(y))
}

fn cos_deriv(_t: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U2>, String> {
    Ok(VectorN::<f64, U2>::from_column_slice(&[y[1], -y[0]]))
}

#[test]
fn test_ivp_cos() {
    let t_initial = 0.0;
    let t_final = 7.0;

    let path = solve_ivp(
        (t_initial, t_final),
        (0.1, 0.001),
        &[1.0, 0.0],
        cos_deriv,
        1e-4,
        &mut (),
    )
    .unwrap();

    for step in path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.cos(),
            epsilon = 0.01
        ));
    }
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
