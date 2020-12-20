use crate::integrate::integrate_fixed;
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
