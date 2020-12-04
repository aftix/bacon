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

  let solver = RungeKutta::new().with_dt(dt).build();

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

  let solver = RungeKutta::new().with_dt(dt).build();

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

  let solver = RungeKutta::new().with_dt(dt).build();

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
