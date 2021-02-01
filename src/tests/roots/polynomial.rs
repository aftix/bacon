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

    let roots = poly.roots(1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 1);
    assert!(approx_eq!(f64, roots[0].re, 1.0, epsilon = 1e-10));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 1e-10));

    let poly = polynomial![0.0];
    let roots = poly.roots(1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 1);
    assert!(approx_eq!(f64, roots[0].re, 0.0, epsilon = 1e-10));
    assert!(approx_eq!(f64, roots[0].im, 0.0, epsilon = 1e-10));

    let poly = polynomial![1.0];
    let roots = poly.roots(1e-10, 1000);
    assert!(roots.is_err(), "Should have not found a root");

    let poly = polynomial![1.0, 0.0, -1.0];
    let cmplx = poly.make_complex();
    let roots = poly.roots(1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 2);
    for (ind, root) in roots.iter().enumerate() {
        let val = cmplx.evaluate(*root);
        assert!(approx_eq!(f64, val.re, 0.0, epsilon = 1e-10));
        assert!(approx_eq!(f64, val.im, 0.0, epsilon = 1e-10));
        for (j, r) in roots.iter().enumerate() {
            if j == ind {
                continue;
            }
            assert!(
                !(approx_eq!(f64, root.re, r.re, epsilon = 0.0001)
                    && approx_eq!(f64, root.im, r.im, epsilon = 0.0001))
            );
        }
    }

    let poly = polynomial![1.0, -2.0, -2.0, 1.0];
    let cmplx = poly.make_complex();
    let roots = poly.roots(1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 3);
    for (ind, root) in roots.iter().enumerate() {
        let val = cmplx.evaluate(*root);
        assert!(approx_eq!(f64, val.re, 0.0, epsilon = 1e-10));
        assert!(approx_eq!(f64, val.im, 0.0, epsilon = 1e-10));
        for (j, r) in roots.iter().enumerate() {
            if j == ind {
                continue;
            }
            assert!(
                !(approx_eq!(f64, root.re, r.re, epsilon = 0.0001)
                    && approx_eq!(f64, root.im, r.im, epsilon = 0.0001))
            );
        }
    }

    let poly = polynomial![
        Complex64::new(5.0, 0.0),
        Complex64::new(-3.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0)
    ];
    let roots = poly.roots(1e-10, 1000).unwrap();
    assert_eq!(roots.len(), 4);
    for (ind, root) in roots.iter().enumerate() {
        let val = poly.evaluate(*root);
        assert!(approx_eq!(f64, val.re, 0.0, epsilon = 1e-10));
        assert!(approx_eq!(f64, val.im, 0.0, epsilon = 1e-10));
        for (j, r) in roots.iter().enumerate() {
            if j == ind {
                continue;
            }
            assert!(
                !(approx_eq!(f64, root.re, r.re, epsilon = 0.0001)
                    && approx_eq!(f64, root.im, r.im, epsilon = 0.0001))
            );
        }
    }
}
