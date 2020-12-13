use crate::special::legendre;

#[test]
fn legendre_test() {
    let p = legendre::<f64>(0);
    assert_eq!(p.order(), 0);
    assert!(approx_eq!(f64, p.get_coefficient(0), 1.0, epsilon = 0.0001));

    let p = legendre::<f64>(1);
    assert_eq!(p.order(), 1);
    assert!(approx_eq!(f64, p.get_coefficient(0), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, p.get_coefficient(1), 1.0, epsilon = 0.0001));

    let p = legendre::<f64>(2);
    assert_eq!(p.order(), 2);
    assert!(approx_eq!(
        f64,
        p.get_coefficient(0),
        -0.5,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, p.get_coefficient(2), 1.5, epsilon = 0.0001));

    let p = legendre::<f64>(10);
    assert_eq!(p.order(), 10);
    assert!(approx_eq!(
        f64,
        p.get_coefficient(0),
        -63.0 / 256.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        p.get_coefficient(2),
        3465.0 / 256.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(3), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        p.get_coefficient(4),
        -30030.0 / 256.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(5), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        p.get_coefficient(6),
        90090.0 / 256.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(7), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        p.get_coefficient(8),
        -109395.0 / 256.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(9), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        p.get_coefficient(10),
        46189.0 / 256.0,
        epsilon = 0.0001
    ));
}
