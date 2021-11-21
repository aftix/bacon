/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::roots;
use nalgebra::{SMatrix, SVector};
use std::f64;

mod polynomial;

fn cubic(x: f64) -> f64 {
    x * x * x
}

fn sqrt_two(x: f64) -> f64 {
    x * x - 2.0
}

fn exp_xsq(x: f64) -> f64 {
    x.exp() - x * x
}

fn exp_sqrt(x: f64) -> f64 {
    -x.exp().sqrt()
}

fn cosine(x: f64) -> f64 {
    x.cos()
}

fn cosine_fixed(x: f64) -> f64 {
    x.cos() - x
}

fn poly(x: f64) -> f64 {
    (10.0 / (x + 4.0)).sqrt()
}

fn newton_complex(x: &[f64]) -> SVector<f64, 3> {
    SVector::<f64, 3>::from_column_slice(&[
        3.0 * x[0] - (x[1] * x[2]).cos() - 0.5,
        x[0].powi(2) - 81.0 * (x[1] + 0.1).powi(2) + x[2].sin() + 1.06,
        (-x[0] * x[1]).exp() + 20.0 * x[2] + (f64::consts::PI * 10.0 - 3.0) / 3.0,
    ])
}

fn jac_complex(x: &[f64]) -> SMatrix<f64, 3, 3> {
    SMatrix::<f64, 3, 3>::new(
        3.0,
        x[2] * (x[1] * x[2]).sin(),
        x[1] * (x[1] * x[2]).sin(),
        2.0 * x[0],
        -162.0 * (x[1] + 0.1),
        x[2].cos(),
        -x[1] * (-x[0] * x[1]).exp(),
        -x[0] * (-x[0] * x[1]).exp(),
        20.0,
    )
}

// Solve x^n = exp(x) where n is 2+index (since x = exp(x) has no solution)
fn exp_newton(x: &[f64]) -> SVector<f64, 2> {
    SVector::<f64, 2>::from_iterator(
        x.iter()
            .enumerate()
            .map(|(i, x)| x.exp() - x.powi(i as i32 + 2)),
    )
}

fn exp_newton_deriv(x: &[f64]) -> SMatrix<f64, 2, 2> {
    SMatrix::<f64, 2, 2>::new(
        x[0].exp() - 2.0 * x[0],
        0.0,
        0.0,
        x[1].exp() - 3.0 * x[1].powi(2),
    )
}

fn cos_secant(x: &[f64]) -> SVector<f64, 1> {
    SVector::<f64, 1>::from_iterator(x.iter().map(|x| x.cos() - x))
}

#[test]
fn bisection_cubic() {
    let a = -1.0;
    let b = 2.0;
    let tol = 0.0001;

    let solution = roots::bisection((a, b), cubic, tol, 1000).unwrap();

    assert!(approx_eq!(f64, solution, 0.0, epsilon = tol));
}

#[test]
fn bisection_sqrt2() {
    let a = 0.0;
    let b = 2.0;
    let tol = 0.0001;

    let solution = roots::bisection((a, b), sqrt_two, tol, 1000).unwrap();

    assert!(approx_eq!(
        f64,
        solution,
        f64::consts::SQRT_2,
        epsilon = tol
    ));
}

#[test]
fn bisection_expxx() {
    let a = -2.0;
    let b = 2.0;
    let tol = 0.000001;

    let solution = roots::bisection((a, b), exp_xsq, tol, 1000).unwrap();

    assert!(approx_eq!(f64, solution, -0.703467, epsilon = 0.0000005));
}

#[test]
fn newton_exp() {
    let start = [0.7, 1.8];
    let tol = 0.0001;
    let solution = roots::newton(&start, exp_newton, exp_newton_deriv, tol, 1000).unwrap();
    assert!(approx_eq!(
        f64,
        *solution.get(0).unwrap(),
        -0.703467,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        *solution.get(1).unwrap(),
        1.85718,
        epsilon = 0.00001
    ));

    let start = [0.7, 4.5];
    let solution = roots::newton(&start, exp_newton, exp_newton_deriv, tol, 1000).unwrap();
    assert!(approx_eq!(
        f64,
        *solution.get(0).unwrap(),
        -0.703467,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        *solution.get(1).unwrap(),
        4.5364,
        epsilon = 0.00001
    ));
}

#[allow(clippy::approx_constant)]
#[test]
fn test_newton() {
    let start = [0.1, 0.1, -0.1];
    let solution = roots::newton(&start, newton_complex, jac_complex, 1e-5, 1000).unwrap();
    assert!(approx_eq!(f64, solution[0], 0.5, epsilon = 1e-5));
    assert!(approx_eq!(f64, solution[1], 0.0, epsilon = 1e-5));
    assert!(approx_eq!(f64, solution[2], -0.52359877, epsilon = 1e-5));
}

#[allow(clippy::approx_constant)]
#[test]
fn test_secant() {
    let start = [0.1, 0.1, -0.1];
    let solution = roots::secant(&start, newton_complex, 0.1, 1e-5, 1000).unwrap();
    assert!(approx_eq!(f64, solution[0], 0.5, epsilon = 1e-5));
    assert!(approx_eq!(f64, solution[1], 0.0, epsilon = 1e-5));
    assert!(approx_eq!(f64, solution[2], -0.52359877, epsilon = 1e-5));
}

#[test]
fn secant_exp() {
    let start = [0.7, 1.8];
    let tol = 1e-6;
    let solution = roots::secant(&start, exp_newton, 0.1, tol, 1000).unwrap();
    assert!(approx_eq!(
        f64,
        *solution.get(0).unwrap(),
        -0.703467,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        *solution.get(1).unwrap(),
        1.85718,
        epsilon = 0.00001
    ));

    let start = [0.7, 4.7];
    let solution = roots::secant(&start, exp_newton, 0.1, tol, 1000).unwrap();
    assert!(approx_eq!(
        f64,
        *solution.get(0).unwrap(),
        -0.703467,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        *solution.get(1).unwrap(),
        4.5364,
        epsilon = 0.00001
    ));
}

#[test]
fn secant_cos() {
    let start = [0.6];
    let tol = 0.0001;
    let solution = roots::secant(&start, cos_secant, 0.1, tol, 1000).unwrap();
    assert!(approx_eq!(
        f64,
        *solution.get(0).unwrap(),
        0.739085,
        epsilon = 0.000001
    ));
}

#[test]
fn steffensen_exp() {
    let solution = roots::steffensen(-0.7f64, exp_sqrt, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, -0.703467, epsilon = 0.000001));
}

#[test]
fn steffensen_cos() {
    let solution = roots::steffensen(0.8f64, cosine, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, 0.739085, epsilon = 0.000001));
}

#[test]
fn steffenson_poly() {
    let solution = roots::steffensen(1.2f64, poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, 1.3652, epsilon = 0.0001));
}

#[test]
fn brent_exp() {
    let solution = roots::brent((-2f64, 2f64), exp_xsq, 1e-7).unwrap();
    assert!(approx_eq!(f64, solution, -0.703467, epsilon = 0.0000005));
}

#[test]
fn brent_cos() {
    let solution = roots::brent((0.0, 1.0), cosine_fixed, 1e-7).unwrap();
    assert!(approx_eq!(f64, solution, 0.739085, epsilon = 0.000001));
}

#[test]
fn brent_sqrt() {
    let solution = roots::brent((1f64, 2f64), sqrt_two, 1e-7).unwrap();
    assert!(approx_eq!(
        f64,
        solution,
        f64::consts::SQRT_2,
        epsilon = 1e-7
    ));
}

#[test]
fn itp_exp() {
    let solution = roots::itp((2f64, -2f64), exp_xsq, 0.1, 2f64, 0.99, 1e-7).unwrap();
    assert!(approx_eq!(f64, solution, -0.703467, epsilon = 0.0000005));
}

#[test]
fn itp_cos() {
    let solution = roots::itp((0.1, 1.0), cosine_fixed, 0.1, 2.0, 0.99, 1e-7).unwrap();
    println!("{}", solution);
    assert!(approx_eq!(f64, solution, 0.739085, epsilon = 0.000001));
}

#[test]
fn itp_sqrt() {
    let solution = roots::itp((1f64, 2f64), sqrt_two, 0.1, 2.0, 0.99, 1e-7).unwrap();
    assert!(approx_eq!(
        f64,
        solution,
        f64::consts::SQRT_2,
        epsilon = 1e-7
    ));
}
