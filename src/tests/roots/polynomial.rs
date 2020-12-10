use crate::polynomial::Polynomial;
use crate::roots::{muller_polynomial, newton_polynomial};
use num_complex::Complex64;

#[test]
fn newton() {
    let mut poly = Polynomial::new();
    poly.set_coefficient(2, 1.0);
    poly.set_coefficient(0, -1.0);

    let solution = newton_polynomial(1.5, &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, 1.0, epsilon = 0.0001));
    let solution = newton_polynomial(-1.5, &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, -1.0, epsilon = 0.0001));

    poly.set_coefficient(2, 2.0);
    poly.set_coefficient(3, 5.0);
    poly.set_coefficient(1, -2.0);
    let solution = newton_polynomial(2.0, &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution, 0.66157, epsilon = 0.00001));
}

#[test]
fn muller() {
    let mut poly = polynomial![1.0, 0.0, -1.0];

    let solution = muller_polynomial((0.0, 0.5, 1.5), &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution.re, 1.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, solution.im, 0.0, epsilon = 0.0001));
    let solution = muller_polynomial((0.0, -0.5, -1.5), &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution.re, -1.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, solution.im, 0.0, epsilon = 0.0001));

    poly = polynomial![5.0, 2.0, -2.0, -1.0];
    let solution = muller_polynomial((0.0, 1.0, 2.0), &poly, 0.0001, 1000).unwrap();
    assert!(approx_eq!(f64, solution.re, 0.66157, epsilon = 0.00001));
    assert!(approx_eq!(f64, solution.im, 0.0, epsilon = 0.00001));
}

#[test]
fn polynomial_roots() {
    let poly = polynomial![1.0, -1.0];

    let roots = poly.roots(&[0.0], 1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 1);
    assert!(approx_eq!(f64, roots[0].re, 1.0, epsilon = 1e-10));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 1e-10));

    let poly = polynomial![0.0];
    let roots = poly.roots(&[], 1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 1);
    assert!(approx_eq!(f64, roots[0].re, 0.0, epsilon = 1e-10));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 1e-10));

    let poly = polynomial![1.0];
    let roots = poly.roots(&[0.0], 1e-10, 1000);
    if let Ok(_) = roots {
        panic!("Should have not found a root");
    }

    let poly = polynomial![1.0, 0.0, -1.0];
    let roots = poly.roots(&[2.0, -2.0], 1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 2);
    assert!(approx_eq!(f64, roots[0].re, 1.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[1].re, -1.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[1].im, 0.0, epsilon = 0.0000001));

    let poly = polynomial![1.0, -2.0, -2.0, 1.0];
    let roots = poly.roots(&[-0.5, 0.3, 3.0], 1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 3);
    assert!(approx_eq!(f64, roots[0].re, -1.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[1].re, 0.38197, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[1].im, 0.0, epsilon = 0.0000001));
    assert!(approx_eq!(f64, roots[2].re, 2.6180, epsilon = 0.0001));
    assert!(approx_eq!(f64, roots[2].im, 0.0, epsilon = 0.0000001));

    let poly = polynomial![
        Complex64::new(5.0, 0.0),
        Complex64::new(-3.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0)
    ];
    let roots = poly
        .roots(
            &[
                Complex64::new(-0.4, -0.25),
                Complex64::new(-0.4, 0.25),
                Complex64::new(1.0, -0.6),
                Complex64::new(1.0, 0.6),
            ],
            1e-10,
            1000,
        )
        .unwrap();
    assert_eq!(roots.len(), 4);
    assert!(approx_eq!(f64, roots[0].re, -0.39932, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[0].im, -0.25396, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[1].re, -0.39932, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[1].im, 0.25396, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[2].re, 0.69932, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[2].im, -0.63562, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[3].re, 0.69932, epsilon = 0.00001));
    assert!(approx_eq!(f64, roots[3].im, 0.63562, epsilon = 0.00001));
}
