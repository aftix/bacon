/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::ivp::*;
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
fn bdf_test_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 7.0;

    let solver = BDF6::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.00001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&exp_deriv, &mut ()).unwrap();

    for step in &path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.exp(),
            epsilon = 0.01
        ));
    }

    Ok(())
}

#[test]
fn bdf_test_unstable() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 10.0;

    let solver = BDF6::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.00001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&unstable_deriv, &mut ()).unwrap();

    for step in &path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            (-step.0).exp(),
            epsilon = 0.01
        ));
    }

    Ok(())
}

#[test]
fn bdf_test_quadratic() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 2.0;

    let solver = BDF6::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.00001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[1.0])?
        .build();

    let path = solver.solve_ivp(&quadratic_deriv, &mut ()).unwrap();

    for step in &path {
        println!("{} {}", step.1.column(0)[0], 1.0 - step.0.powi(2));
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            1.0 - step.0.powi(2),
            epsilon = 0.01
        ));
    }

    Ok(())
}

#[test]
fn bdf_test_sin() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 6.0;

    let solver = BDF6::new()
        .with_dt_min(1e-5)?
        .with_dt_max(0.1)?
        .with_tolerance(0.00001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_initial_conditions(&[0.0])?
        .build();

    let path = solver.solve_ivp(&sine_deriv, &mut ()).unwrap();

    for step in &path {
        assert!(approx_eq!(
            f64,
            step.1.column(0)[0],
            step.0.sin(),
            epsilon = 0.01
        ));
    }

    Ok(())
}
