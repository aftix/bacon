use crate::integrate::{
    integrate, integrate_chebyshev, integrate_chebyshev_second, integrate_fixed,
    integrate_gaussian, integrate_hermite,
};
use std::f64;

fn exp(x: f64) -> f64 {
    x.exp()
}

fn sin(x: f64) -> f64 {
    x.sin()
}

fn sinsin(x: f64) -> f64 {
    x.sin().sin()
}

fn one(_: f64) -> f64 {
    1.0
}

fn x(x: f64) -> f64 {
    x
}

fn xsquared(x: f64) -> f64 {
    x.powi(2)
}

#[test]
fn test_integrate() {
    let area = integrate(0.0, f64::consts::PI, sin, 0.00001, 50).unwrap();
    assert!(approx_eq!(f64, area, 2.0, epsilon = 0.0001));
    let area = integrate(0.0, 10.0, exp, 0.00001, 50).unwrap();
    assert!(approx_eq!(
        f64,
        area,
        (10.0 as f64).exp() - 1.0,
        epsilon = 0.0001
    ));
    let area = integrate(0.0, f64::consts::PI, sinsin, 0.00001, 50).unwrap();
    assert!(approx_eq!(f64, area, 1.78649, epsilon = 0.0001));
}

#[test]
fn test_integrate_fixed() {
    let area = integrate_fixed(0.0, f64::consts::PI, sin, 6).unwrap();
    assert!(approx_eq!(f64, area, 2.0, epsilon = 0.01));
    let area = integrate_fixed(0.0, 10.0, exp, 10).unwrap();
    assert!(approx_eq!(
        f64,
        area,
        (10.0 as f64).exp() - 1.0,
        epsilon = 0.0001
    ));
    let area = integrate_fixed(0.0, f64::consts::PI, sinsin, 10).unwrap();
    assert!(approx_eq!(f64, area, 1.78649, epsilon = 0.00001));
}

#[test]
fn test_integrate_gaussian() {
    let area = integrate_gaussian(0.0, f64::consts::PI, sin, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 2.0, epsilon = 0.0001));
    let area = integrate_gaussian(0.0, 10.0, exp, 0.00001, 50).unwrap();
    assert!(approx_eq!(
        f64,
        area,
        (10.0 as f64).exp() - 1.0,
        epsilon = 0.0001
    ));
    let area = integrate_gaussian(0.0, f64::consts::PI, sinsin, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 1.78649, epsilon = 0.0001));
}

#[test]
fn test_integrate_hermite() {
    let area = integrate_hermite(one, 0.00001, 10).unwrap();
    assert!(approx_eq!(
        f64,
        area,
        f64::consts::PI.sqrt(),
        epsilon = 0.0001
    ));
    let area = integrate_hermite(x, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
    let area = integrate_hermite(xsquared, 0.00001, 10).unwrap();
    assert!(approx_eq!(
        f64,
        area,
        f64::consts::PI.sqrt() / 2.0,
        epsilon = 0.0001
    ));
    let area = integrate_hermite(sin, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
}

#[test]
fn test_integrate_chebyshev() {
    let area = integrate_chebyshev(sin, 0.00001, 100).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
    let area = integrate_chebyshev(exp, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 3.97746, epsilon = 0.0001));
    let area = integrate_chebyshev(sinsin, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
    let area = integrate_chebyshev(xsquared, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 1.5708, epsilon = 0.001));
}

#[test]
fn test_integrate_chebyshev_second() {
    let area = integrate_chebyshev_second(sin, 0.00001, 100).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
    let area = integrate_chebyshev_second(exp, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 1.7755, epsilon = 0.0001));
    let area = integrate_chebyshev_second(sinsin, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.0001));
    let area = integrate_chebyshev_second(xsquared, 0.00001, 10).unwrap();
    assert!(approx_eq!(f64, area, 0.3927, epsilon = 0.001));
}
