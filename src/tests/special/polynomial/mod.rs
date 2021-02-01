use crate::special::{
    chebyshev, chebyshev_second, hermite, hermite_zeros, laguerre, laguerre_zeros, legendre,
    legendre_zeros,
};

#[test]
fn legendre_test() {
    let p = legendre::<f64>(0, 1e-8).unwrap();
    assert_eq!(p.order(), 0);
    assert!(approx_eq!(f64, p.get_coefficient(0), 1.0, epsilon = 0.0001));

    let p = legendre::<f64>(1, 1e-8).unwrap();
    assert_eq!(p.order(), 1);
    assert!(approx_eq!(f64, p.get_coefficient(0), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, p.get_coefficient(1), 1.0, epsilon = 0.0001));

    let p = legendre::<f64>(2, 1e-8).unwrap();
    assert_eq!(p.order(), 2);
    assert!(approx_eq!(
        f64,
        p.get_coefficient(0),
        -0.5,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, p.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, p.get_coefficient(2), 1.5, epsilon = 0.0001));

    let p = legendre::<f64>(10, 1e-8).unwrap();
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
fn legendre_zero_test() {
    for i in 1..20 {
        let poly = legendre::<f64>(i, 1e-8).unwrap();
        let zeros = legendre_zeros::<f64>(i, 1e-8, 1e-8, 100).unwrap();
        for (ind, zero) in zeros.iter().enumerate() {
            assert!(approx_eq!(f64, poly.evaluate(*zero), 0.0, epsilon = 0.0001));
            for (j, other) in zeros.iter().enumerate() {
                if j == ind {
                    continue;
                }
                assert!(!approx_eq!(f64, *zero, *other, epsilon = 0.0001));
            }
        }
    }
}

#[test]
fn hermite_test() {
    let h = hermite::<f64>(0, 1e-8).unwrap();
    assert_eq!(h.order(), 0);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        1.0,
        epsilon = 0.0000001
    ));

    let h = hermite::<f64>(1, 1e-8).unwrap();
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

    let h = hermite::<f64>(2, 1e-8).unwrap();
    assert_eq!(h.order(), 2);
    assert!(approx_eq!(
        f64,
        h.get_coefficient(0),
        -2.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, h.get_coefficient(1), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(f64, h.get_coefficient(2), 4.0, epsilon = 0.0001));

    let h = hermite::<f64>(10, 1e-8).unwrap();
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
fn hermite_zero_test() {
    for i in 1..15 {
        let poly = hermite::<f64>(i, 1e-8).unwrap();
        let zeros = hermite_zeros::<f64>(i, 1e-10, 1e-8, 10000).unwrap();
        for (ind, zero) in zeros.iter().enumerate() {
            assert!(approx_eq!(f64, poly.evaluate(*zero), 0.0, epsilon = 0.005));
            for (j, other) in zeros.iter().enumerate() {
                if j == ind {
                    continue;
                }
                assert!(!approx_eq!(f64, *zero, *other, epsilon = 0.0001));
            }
        }
    }
}

#[allow(clippy::eq_op)]
#[test]
fn laguerre_test() {
    let l = laguerre::<f64>(0, 1e-8).unwrap();
    assert_eq!(l.order(), 0);
    assert!(approx_eq!(f64, l.get_coefficient(0), 1.0, epsilon = 0.0001));

    let l = laguerre::<f64>(1, 1e-8).unwrap();
    assert_eq!(l.order(), 1);
    assert!(approx_eq!(f64, l.get_coefficient(0), 1.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        l.get_coefficient(1),
        -1.0,
        epsilon = 0.0001
    ));

    let l = laguerre::<f64>(6, 1e-8).unwrap();
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

#[test]
fn laguerre_zero_test() {
    for i in 1..10 {
        let poly = laguerre::<f64>(i, 1e-20).unwrap();
        let zeros = laguerre_zeros::<f64>(i, 1e-8, 1e-40, 100).unwrap();
        for (ind, zero) in zeros.iter().enumerate() {
            assert!(approx_eq!(f64, poly.evaluate(*zero), 0.0, epsilon = 0.0001));
            for (j, other) in zeros.iter().enumerate() {
                if j == ind {
                    continue;
                }
                assert!(!approx_eq!(f64, *zero, *other, epsilon = 0.0001));
            }
        }
    }
}

#[test]
fn chebyshev_test() {
    let t = chebyshev::<f64>(0, 1e-8).unwrap();
    assert_eq!(t.order(), 0);
    assert!(approx_eq!(f64, t.get_coefficient(0), 1.0, epsilon = 0.0001),);

    let t = chebyshev::<f64>(1, 1e-8).unwrap();
    assert_eq!(t.order(), 1);
    assert!(approx_eq!(f64, t.get_coefficient(0), 0.0, epsilon = 0.0001),);
    assert!(approx_eq!(f64, t.get_coefficient(1), 1.0, epsilon = 0.0001),);

    let t = chebyshev::<f64>(11, 1e-8).unwrap();
    assert_eq!(t.order(), 11);
    assert!(approx_eq!(f64, t.get_coefficient(0), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(1),
        -11.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(2), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(3),
        220.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(4), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(5),
        -1232.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(6), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(7),
        2816.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(8), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(9),
        -2816.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(10),
        0.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(11),
        1024.0,
        epsilon = 0.0001
    ));
}

#[test]
fn chebyshev_second_test() {
    let t = chebyshev_second::<f64>(0, 1e-8).unwrap();
    assert_eq!(t.order(), 0);
    assert!(approx_eq!(f64, t.get_coefficient(0), 1.0, epsilon = 0.0001),);

    let t = chebyshev_second::<f64>(1, 1e-8).unwrap();
    assert_eq!(t.order(), 1);
    assert!(approx_eq!(f64, t.get_coefficient(0), 0.0, epsilon = 0.0001),);
    assert!(approx_eq!(f64, t.get_coefficient(1), 2.0, epsilon = 0.0001),);

    let t = chebyshev_second::<f64>(9, 1e-8).unwrap();
    assert_eq!(t.order(), 9);
    assert!(approx_eq!(f64, t.get_coefficient(0), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(1),
        10.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(2), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(3),
        -160.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(4), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(5),
        672.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(6), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(7),
        -1024.0,
        epsilon = 0.0001
    ));
    assert!(approx_eq!(f64, t.get_coefficient(8), 0.0, epsilon = 0.0001));
    assert!(approx_eq!(
        f64,
        t.get_coefficient(9),
        512.0,
        epsilon = 0.0001
    ));
}
