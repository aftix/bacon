#[test]
fn polynomial_addition() {
    let mut poly = polynomial![4.0, 6.0, 5.0];

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            (&poly + i).evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 + i,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly = poly + i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 + i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly += i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 + i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        assert!(approx_eq!(
            f64,
            (&poly + addend).evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        assert!(approx_eq!(
            f64,
            (&poly + &addend).evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly + addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly + &addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly + addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly += addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly += &addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 + 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }
}

#[test]
fn polynomial_subtraction() {
    let mut poly = polynomial![4.0, 6.0, 5.0];

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            (&poly - i).evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 - i,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly = poly - i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 - i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly -= i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + 6.0 * i + 5.0 - i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        assert!(approx_eq!(
            f64,
            (&poly - addend).evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        assert!(approx_eq!(
            f64,
            (&poly - &addend).evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly - addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly - &addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly = poly - addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly -= addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        let addend = polynomial![5.0 * i, 0.0];
        poly -= &addend;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            4.0 * i.powi(2) + (6.0 - 5.0 * i) * i + 5.0,
            epsilon = 0.0001
        ));
        poly.set_coefficient(1, 6.0);
    }
}

#[test]
fn polynomial_multiplication() {
    let mut poly = polynomial![4.0, 6.0, 5.0];

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        assert!(approx_eq!(
            f64,
            (&poly * i).evaluate(i),
            i * (4.0 * i.powi(2) + 6.0 * i + 5.0),
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly = poly * i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            (4.0 * i.powi(2) + 6.0 * i + 5.0) * i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
        poly.set_coefficient(1, 6.0);
        poly.set_coefficient(2, 4.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        poly *= i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            (4.0 * i.powi(2) + 6.0 * i + 5.0) * i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
        poly.set_coefficient(1, 6.0);
        poly.set_coefficient(2, 4.0);
    }

    let poly = polynomial![1.0, 0.0, -1.0];
    let poly_2 = polynomial![1.0, 0.0];
    let product = &poly * &poly_2;
    assert_eq!(product.order(), 3);
    assert!(approx_eq!(
        f64,
        product.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(1),
        -1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(2),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(3),
        1.0,
        epsilon = 0.000001
    ));
    let poly = polynomial![1.0, 0.0, -1.0];
    let poly_2 = polynomial![1.0, 0.0];
    let product = poly * &poly_2;
    assert_eq!(product.order(), 3);
    assert!(approx_eq!(
        f64,
        product.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(1),
        -1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(2),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(3),
        1.0,
        epsilon = 0.000001
    ));
    let poly = polynomial![1.0, 0.0, -1.0];
    let poly_2 = polynomial![1.0, 0.0];
    let product = &poly * poly_2;
    assert_eq!(product.order(), 3);
    assert!(approx_eq!(
        f64,
        product.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(1),
        -1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(2),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(3),
        1.0,
        epsilon = 0.000001
    ));
    let poly = polynomial![1.0, 0.0, -1.0];
    let poly_2 = polynomial![1.0, 0.0];
    let product = poly * poly_2;
    assert_eq!(product.order(), 3);
    assert!(approx_eq!(
        f64,
        product.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(1),
        -1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(2),
        0.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        product.get_coefficient(3),
        1.0,
        epsilon = 0.000001
    ));
}

#[test]
fn polynomial_scalar_divsion() {
    let mut poly = polynomial![4.0, 6.0, 5.0];

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        if approx_eq!(f64, i, 0.0, epsilon = 0.001) {
            continue;
        }
        assert!(approx_eq!(
            f64,
            (&poly / i).evaluate(i),
            (4.0 * i.powi(2) + 6.0 * i + 5.0) / i,
            epsilon = 0.0001
        ));
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        if approx_eq!(f64, i, 0.0, epsilon = 0.001) {
            continue;
        }
        poly = poly / i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            (4.0 * i.powi(2) + 6.0 * i + 5.0) / i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
        poly.set_coefficient(1, 6.0);
        poly.set_coefficient(2, 4.0);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.001;
        if approx_eq!(f64, i, 0.0, epsilon = 0.001) {
            continue;
        }
        poly /= i;
        assert!(approx_eq!(
            f64,
            poly.evaluate(i),
            (4.0 * i.powi(2) + 6.0 * i + 5.0) / i,
            epsilon = 0.0001
        ));
        poly.set_coefficient(0, 5.0);
        poly.set_coefficient(1, 6.0);
        poly.set_coefficient(2, 4.0);
    }
}

#[test]
fn polynomial_long_division() {
    let dividend = polynomial![1.0, 0.0, -1.0];
    let divisor = polynomial![1.0, 1.0];
    let (quotient, remainder) = dividend.divide(&divisor, 1e-5).unwrap();
    assert_eq!(remainder.order(), 0);
    assert!(approx_eq!(
        f64,
        remainder.get_coefficient(0),
        0.0,
        epsilon = 0.000001
    ));
    assert_eq!(quotient.order(), 1);
    assert!(approx_eq!(
        f64,
        quotient.get_coefficient(0),
        -1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        quotient.get_coefficient(1),
        1.0,
        epsilon = 0.000001
    ));

    let dividend = polynomial![1.0, -2.0, 1.0, 1.0];
    let divisor = polynomial![1.0, 0.0, 1.0];
    let (quotient, remainder) = dividend.divide(&divisor, 1e-5).unwrap();
    assert_eq!(remainder.order(), 0);
    assert!(approx_eq!(
        f64,
        remainder.get_coefficient(0),
        3.0,
        epsilon = 0.000001
    ));
    assert_eq!(quotient.order(), 1);
    assert!(approx_eq!(
        f64,
        quotient.get_coefficient(1),
        1.0,
        epsilon = 0.000001
    ));
    assert!(approx_eq!(
        f64,
        quotient.get_coefficient(0),
        -2.0,
        epsilon = 0.0000001
    ));
}
