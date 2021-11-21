/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::ivp::*;
use nalgebra::SVector;

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<SVector<f64, 1>, String> {
    Ok(SVector::<f64, 1>::from_column_slice(y))
}

fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<SVector<f64, 1>, String> {
    Ok(SVector::<f64, 1>::from_column_slice(&[-2.0 * t]))
}

fn sine_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<SVector<f64, 1>, String> {
    Ok(SVector::<f64, 1>::from_column_slice(&[t.cos()]))
}

// Test runge-kutta-fehlberg method on y = exp(t)
#[test]
fn rkf_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = RK45::new()
        .with_dt_min(0.001)?
        .with_dt_max(0.01)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_tolerance(0.00001)?
        .with_initial_conditions(&[1.0])?;

    let path = solver.solve_ivp(&exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.exp(),
                    epsilon = 0.001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

// Test runge-kutta-fehlberg method on y = 1 - t^2
#[test]
fn rkf_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = RK45::new()
        .with_dt_min(0.0001)?
        .with_dt_max(0.1)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_tolerance(1e-5)?
        .with_initial_conditions(&[1.0])?;

    let path = solver.solve_ivp(&quadratic_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                println!("{} {}", step.1.column(0)[0], 1.0 - step.0.powi(2));
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    1.0 - step.0.powi(2),
                    epsilon = 0.0001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

// Test runge-kutta-fehlberg method on y = sin(t)
#[test]
fn rkf_test_sine() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = RK45::new()
        .with_dt_min(0.001)?
        .with_dt_max(0.01)?
        .with_tolerance(0.0001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[0.0])?;

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
fn bs_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;

    let solver = RK23::new()
        .with_dt_min(0.001)?
        .with_dt_max(0.01)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_tolerance(0.01)?
        .with_initial_conditions(&[1.0])?;

    let path = solver.solve_ivp(&exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.exp(),
                    epsilon = 0.001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn bs_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = RK23::new()
        .with_dt_min(0.0001)?
        .with_dt_max(0.1)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_tolerance(1e-5)?
        .with_initial_conditions(&[1.0])?;

    let path = solver.solve_ivp(&quadratic_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                println!("{} {}", step.1.column(0)[0], 1.0 - step.0.powi(2));
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    1.0 - step.0.powi(2),
                    epsilon = 0.0001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

#[test]
fn bs_test_sine() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = RK23::new()
        .with_dt_min(0.001)?
        .with_dt_max(0.01)?
        .with_tolerance(0.0001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[0.0])?;

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
