/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::ivp::*;
use nalgebra::DVector;

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
    DVector::from_column_slice(y)
}

fn quadratic_deriv(t: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
    DVector::from_iterator(y.len(), [-2.0 * t].repeat(y.len()))
}

fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
    DVector::from_iterator(y.len(), [t.cos()].repeat(y.len()))
}

// Test adams-bashforth on y=exp(t)
#[test]
fn adams_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;
    let dt = 0.001;

    let solver = AdamsBashforth::default().with_dt(dt)?.build()?;

    let path = adams(solver, (t_initial, t_final), &[1.0], exp_deriv, &mut ());

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

// Test adams-bashforth on y=1 - t^2
#[test]
fn adams_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;
    let dt = 0.001;

    let solver = AdamsBashforth::default().with_dt(dt)?.build()?;

    let path = adams(
        solver,
        (t_initial, t_final),
        &[1.0],
        quadratic_deriv,
        &mut (),
    );

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    1.0 - step.0.powi(2),
                    epsilon = 0.001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

// Test adams-bashforth on y=sine(t)
#[test]
fn adams_test_sine() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;
    let dt = 0.001;

    let solver = AdamsBashforth::default().with_dt(dt)?.build()?;

    let path = adams(solver, (t_initial, t_final), &[0.0], sine_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                assert!(approx_eq!(
                    f64,
                    step.1.column(0)[0],
                    step.0.sin(),
                    epsilon = 0.001
                ));
            }
        }
        Err(string) => panic!("Result not Ok: {}", string),
    }

    Ok(())
}

// Test predictor-corrector for y=exp(t)
#[test]
fn pc_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 5.0;

    let solver = PredictorCorrector::default()
        .with_dt_min(1e-5)?
        .with_dt_max(0.001)?
        .with_tolerance(1e-10)?
        .build()?;

    let path = adams(solver, (t_initial, t_final), &[1.0], exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                println!("{} {}", step.1.column(0)[0], step.0.exp());
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

// TODO MORE PC TESTS
