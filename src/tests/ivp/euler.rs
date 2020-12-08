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

use crate::ivp;
use nalgebra::DVector;

fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
  DVector::from_column_slice(y)
}

fn quadratic_deriv(t: f64, y: &[f64], _: &mut ()) -> DVector<f64> {
  DVector::from_iterator(y.len(), [-2.0 * t].repeat(y.len()))
}

// Test euler method on y = exp(x), y' = exp(x) = y, y(0) = 1
#[test]
fn euler_test_exp() {
  let t_initial = 0.0;
  let t_final = 1.0;
  let dt = 0.0005;

  let path = ivp::euler((t_initial, t_final), &[1.0], dt, exp_deriv, &mut ());

  for step in &path {
    assert!(approx_eq!(f64, step.1.column(0)[0], step.0.exp(), epsilon = 0.001));
  }
}

// Test euler method on y = 1 - x^2, y' = -2x, y(0) = 1
#[test]
fn euler_test_quadratic() {
  let t_initial = 0.0;
  let t_final = 1.0;
  let dt = 0.0001;

  let path = ivp::euler((t_initial, t_final), &[1.0], dt, quadratic_deriv, &mut ());

  for step in &path {
    assert!(approx_eq!(f64, step.1.column(0)[0], 1.0 - step.0.powi(2), epsilon = 0.001));
  }
}
