use crate::interp::{hermite, lagrange};

#[test]
fn lagrange_interp() {
    let xs: Vec<_> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.cos()).collect();

    let poly = lagrange(&xs, &ys, 1e-6).unwrap();

    for x in xs {
        println!("{} {}", poly.evaluate(x), x.cos());
        assert!(approx_eq!(
            f64,
            poly.evaluate(x),
            x.cos(),
            epsilon = 0.00001
        ));
    }

    for x in 0..=100 {
        let x = x as f64 * 0.1;
        assert!(approx_eq!(f64, poly.evaluate(x), x.cos(), epsilon = 0.5));
    }
}

#[test]
fn hermite_interp() {
    let xs: Vec<_> = (0..2).map(|i| i as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.cos()).collect();
    let derivs: Vec<_> = xs.iter().map(|x| -x.sin()).collect();

    let poly = hermite(&xs, &ys, &derivs, 1e-6).unwrap();
    println!("{:?}", poly);

    for x in xs {
        let (p, deriv) = poly.evaluate_derivative(x);
        println!("({} {}) ({} {})", p, deriv, x.cos(), -x.sin());
        assert!(approx_eq!(f64, p, x.cos(), epsilon = 1.0));
        assert!(approx_eq!(f64, deriv, -x.sin(), epsilon = 1.0));
    }
}
