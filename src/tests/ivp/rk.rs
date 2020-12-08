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

// Test runge-kutta method on y = exp(t)
#[test]
fn rk_test_exp() {
  let t_initial = 0.0;
  let t_final = 10.0;
  let dt = 0.01;

  let solver = RungeKutta::default().with_dt(dt).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[1.0], exp_deriv, &mut ());

  match path {
    Ok(path) => {
      for step in &path {
        assert!(approx_eq!(f64, step.1.column(0)[0], step.0.exp(), epsilon = 0.001));
      }
    },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// Test runge-kutta method on y = 1 - t^2
#[test]
fn rk_test_quadratic() {
  let t_initial = 0.0;
  let t_final = 10.0;
  let dt = 0.01;

  let solver = RungeKutta::default().with_dt(dt).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[1.0], quadratic_deriv, &mut ());

  match path {
    Ok(path) => {
        for step in &path {
          assert!(approx_eq!(f64, step.1.column(0)[0], 1.0 - step.0.powi(2), epsilon = 0.001));
        }
      },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// Test runge-kutta method on y = sin(t)
#[test]
fn rk_test_sine() {
  let t_initial = 0.0;
  let t_final = 10.0;
  let dt = 0.01;

  let solver = RungeKutta::default().with_dt(dt).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[0.0], sine_deriv, &mut ());

  match path {
    Ok(path) => {
        for step in &path {
          assert!(approx_eq!(f64, step.1.column(0)[0], step.0.sin(), epsilon = 0.001));
        }
      },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// Test runge-kutta-fehlberg method on y = exp(t)
#[test]
fn rkf_test_exp() {
  let t_initial = 0.0;
  let t_final = 10.0;

  let solver = RungeKuttaFehlberg::default().with_dt_min(0.001).with_dt_max(0.01).with_tolerance(0.0001).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[1.0], exp_deriv, &mut ());

  match path {
    Ok(path) => {
        for step in &path {
          assert!(approx_eq!(f64, step.1.column(0)[0], step.0.exp(), epsilon = 0.001));
        }
      },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// Test runge-kutta-fehlberg method on y = 1 - t^2
#[test]
fn rkf_test_quadratic() {
  let t_initial = 0.0;
  let t_final = 20.0;

  let solver = RungeKuttaFehlberg::default().with_dt_min(0.001).with_dt_max(0.01).with_tolerance(0.0001).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[1.0], quadratic_deriv, &mut ());

  match path {
    Ok(path) => {
        for step in &path {
          assert!(approx_eq!(f64, step.1.column(0)[0], 1.0 - step.0.powi(2), epsilon = 0.001));
        }
      },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// Test runge-kutta-fehlberg method on y = sin(t)
#[test]
fn rkf_test_sine() {
  let t_initial = 0.0;
  let t_final = 20.0;

  let solver = RungeKuttaFehlberg::default().with_dt_min(0.001).with_dt_max(0.01).with_tolerance(0.0001).build();

  let path = runge_kutta(solver, (t_initial, t_final), &[0.0], sine_deriv, &mut ());

  match path {
    Ok(path) => {
        for step in &path {
          assert!(approx_eq!(f64, step.1.column(0)[0], step.0.sin(), epsilon = 0.001));
        }
      },
    Err(string) => panic!("Result not Ok: {}", string)
  }
}

// TODO: Add more runge-kutta tests
