mod operations;

#[test]
fn polynomial_evaluation() {
    let mut polynomial = polynomial![4.0, -3.0, 5.0];

    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            4.0 * i.powi(2) - 3.0 * i + 5.0,
            polynomial.evaluate(i),
            epsilon = 0.00001
        ));
    }

    polynomial = polynomial![5.0];
    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            5.0,
            polynomial.evaluate(i),
            epsilon = 0.00001
        ));
    }

    polynomial.set_coefficient(10, 127.0);
    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            127.0 * i.powi(10) + 5.0,
            polynomial.evaluate(i),
            epsilon = 0.00001
        ));
    }
}

#[test]
fn polynomial_derivative_evaluation() {
    let mut polynomial = polynomial![4.0, -3.0, 5.0];

    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        let (eval, deriv) = polynomial.evaluate_derivative(i);
        assert!(approx_eq!(
            f64,
            4.0 * i.powi(2) - 3.0 * i + 5.0,
            eval,
            epsilon = 0.00001
        ));
        assert!(approx_eq!(f64, 8.0 * i - 3.0, deriv));
    }

    polynomial = polynomial![5.0];
    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let (eval, deriv) = polynomial.evaluate_derivative(i);
        println!("{} {}", eval, deriv);
        assert!(approx_eq!(f64, 5.0, eval, epsilon = 0.00001));
        assert!(approx_eq!(f64, 0.0, deriv, epsilon = 0.00001));
    }

    polynomial.set_coefficient(10, 127.0);
    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        let (eval, deriv) = polynomial.evaluate_derivative(i);
        assert!(approx_eq!(
            f64,
            127.0 * i.powi(10) + 5.0,
            eval,
            epsilon = 0.00001
        ));
        assert!(approx_eq!(
            f64,
            1270.0 * i.powi(9),
            deriv,
            epsilon = 0.00001
        ));
    }
}

#[test]
fn polynomial_derivative() {
    let mut polynomial = polynomial![4.0, -3.0, 5.0];
    polynomial = polynomial.derivative();

    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(f64, 8.0 * i - 3.0, polynomial.evaluate(i)));
    }

    polynomial = polynomial![5.0];
    polynomial = polynomial.derivative();
    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            0.0,
            polynomial.evaluate(i),
            epsilon = 0.00001
        ));
    }

    polynomial.set_coefficient(10, 127.0);
    polynomial.set_coefficient(0, 5.0);
    polynomial = polynomial.derivative();
    for i in (-1000..1000).step_by(1) {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            1270.0 * i.powi(9),
            polynomial.evaluate(i),
            epsilon = 0.00001
        ));
    }
}

#[test]
fn polynomial_integrate() {
    let mut polynomial = polynomial![1.0, 0.0, 0.0];

    let area = polynomial.integrate(0.0, 10.0);
    assert!(approx_eq!(f64, area, 333.3333333, epsilon = 0.0000001));

    polynomial = polynomial![1.0, 0.0, 0.0, 0.0];
    let area = polynomial.integrate(-100.0, 100.0);
    assert!(approx_eq!(f64, area, 0.0, epsilon = 0.000001));

    polynomial = polynomial![1.0, 0.0, -3.0, 0.0];
    let area = polynomial.integrate(-2.0, 8.0);
    assert!(approx_eq!(f64, area, 930.0, epsilon = 0.00001))
}
