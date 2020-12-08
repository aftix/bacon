// MIT License
//
// Copyright (c) 2020 Wyatt Campbell
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::roots;
use std::f64;
use nalgebra::DVector;

fn cubic(x: f64) -> f64 {
  x*x*x
}

fn sqrt_two(x: f64) -> f64 {
  x*x - 2.0
}

fn exp_xsq(x: f64) -> f64 {
  x.exp() - x*x
}

// Solve x^n = exp(x) where n is 2+index (since x = exp(x) has no solution)
fn exp_newton(x: &[f64]) -> DVector<f64> {
  DVector::from_iterator(x.len(), x.iter().enumerate().map(|(i, x)| x.exp() - x.powi(i as i32 + 2)))
}

fn exp_newton_deriv(x: &[f64]) -> DVector<f64> {
  DVector::from_iterator(x.len(), x.iter().enumerate().map(|(i, x)| x.exp() - ((i + 2) as f64) * x.powi(i as i32 + 1)))
}

fn cos_secant(x: &[f64]) -> DVector<f64> {
  DVector::from_iterator(x.len(), x.iter().map(|x| x.cos() - x))
}

#[test]
fn bisection_cubic() {
  let a = -1.0;
  let b = 2.0;
  let tol = 0.0001;

  let solution = roots::bisection((a, b), cubic, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, 0.0, epsilon=tol));
}

#[test]
fn bisection_sqrt2() {
  let a = 0.0;
  let b = 2.0;
  let tol = 0.0001;

  let solution = roots::bisection((a, b), sqrt_two, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, f64::consts::SQRT_2, epsilon=tol));
}

#[test]
fn bisection_expxx() {
  let a = -2.0;
  let b = 2.0;
  let tol = 0.000001;

  let solution = roots::bisection((a, b), exp_xsq, tol, 1000).unwrap();

  assert!(approx_eq!(f64, solution, -0.703467, epsilon=0.0000005));
}

#[test]
fn newton_exp() {
  let start = [0.7, 1.8];
  let tol = 0.0001;
  let solution = roots::newton(&start, exp_newton, exp_newton_deriv, tol, 1000).unwrap();
  assert!(approx_eq!(f64, *solution.get(0).unwrap(), -0.703467, epsilon=0.000001));
  assert!(approx_eq!(f64, *solution.get(1).unwrap(), 1.85718, epsilon=0.00001));

  let start = [0.7, 4.5];
  let solution = roots::newton(&start, exp_newton, exp_newton_deriv, tol, 1000).unwrap();
  assert!(approx_eq!(f64, *solution.get(0).unwrap(), -0.703467, epsilon=0.000001));
  assert!(approx_eq!(f64, *solution.get(1).unwrap(), 4.5364, epsilon=0.00001));
}

#[test]
fn secant_exp() {
  let start_1 = [0.7, 1.8];
  let start_2 = [0.8, 1.9];
  let tol = 0.0001;
  let solution = roots::secant((&start_1, &start_2), exp_newton, tol, 1000).unwrap();
  assert!(approx_eq!(f64, *solution.get(0).unwrap(), -0.703467, epsilon=0.000001));
  assert!(approx_eq!(f64, *solution.get(1).unwrap(), 1.85718, epsilon=0.00001));

  let start_1 = [0.7, 4.4];
  let start_2 = [0.8, 4.6];
  let solution = roots::secant((&start_1, &start_2), exp_newton, tol, 1000).unwrap();
  assert!(approx_eq!(f64, *solution.get(0).unwrap(), -0.703467, epsilon=0.000001));
  assert!(approx_eq!(f64, *solution.get(1).unwrap(), 4.5364, epsilon=0.00001));
}

#[test]
fn secant_cos() {
  let start_1 = [0.6];
  let start_2 = [1.0];
  let tol = 0.0001;
  let solution = roots::secant((&start_1, &start_2), cos_secant, tol, 1000).unwrap();
  assert!(approx_eq!(f64, *solution.get(0).unwrap(), 0.739085, epsilon=0.000001));
}
