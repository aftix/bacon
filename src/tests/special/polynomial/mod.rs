use crate::special::{hermite, laguerre, legendre};

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

#[test]
fn hermite_test() {
    let h = hermite::<f64>(0);
    assert_eq!(h.order(), 0);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        1.0,
        epsilon = 0.0000001
    ));

    let h = hermite::<f64>(1);
    assert_eq!(h.order(), 1);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(1),
        2.0,
        epsilon = 0.000001
    ));

    let h = hermite::<f64>(2);
    assert_eq!(h.order(), 2);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        -2.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, h.get_coefficient(2), 4.0, epsilon = 0.0001));

    let h = hermite::<f64>(10);
    assert_eq!(h.order(), 10);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        -30240.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(2),
        302400.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(3), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(4),
        -403200.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(5), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(6),
        161280.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(7), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(8),
        -23040.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(9), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        h.get_coefficient(10),
        1024.0,
        epsilon = 0.0001
    ));
}

#[test]
fn laguerre_test() {
    let l = laguerre::<f64>(0);
    assert_eq!(l.order(), 0);
    assert!(approx_eq!(f64, l.get_coefficient(0), 1.0, epsilon = 0.0001));

    let l = laguerre::<f64>(1);
    assert_eq!(l.order(), 1);
    assert!(approx_eq!(f64, l.get_coefficient(0), 1.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(1),
        -1.0,
        epsilon = 0.0001
    ));

    let l = laguerre::<f64>(6);
    assert_eq!(l.order(), 6);
    assert!(approx_eq!(
        f64,
        l.get_coefficient(0),
        720.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(1),
        -4320.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(2),
        5400.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(3),
        -2400.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(4),
        450.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(5),
        -36.0 / 720.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(6),
        1.0 / 720.0,
        epsilon = 0.0001
    ));
}
