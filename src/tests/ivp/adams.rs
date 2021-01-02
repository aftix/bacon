/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::ivp::*;
use nalgebra::{VectorN, U1};

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_column_slice(y))
}

fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_column_slice(&[-2.0 * t]))
}

fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<VectorN<f64, U1>, String> {
    Ok(VectorN::<f64, U1>::from_iterator(y.iter().map(|_| t.cos())))
}

// Test predictor-corrector for y=exp(t)
#[test]
fn pc_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 2.0;

    let solver = Adams::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.exp(),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn pc_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;

    let solver = Adams::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.001)?
        .with_tolerance(0.1)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&quadratic_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    1.0 - step.0.powi(2),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn pc_sine() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 6.28;

    let solver = Adams::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.001)?
        .with_tolerance(0.01)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[0.0])?
        .build();

    let path = solver.solve_ivp(&sine_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.sin(),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn adams2_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 2.0;

    let solver = Adams2::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.exp(),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn adams2_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;

    let solver = Adams2::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.001)?
        .with_tolerance(0.1)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&quadratic_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    1.0 - step.0.powi(2),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn adams2_sine() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 6.28;

    let solver = Adams2::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.001)?
        .with_tolerance(0.01)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[0.0])?
        .build();

    let path = solver.solve_ivp(&sine_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.sin(),
                    epsilon = 0.01
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}
