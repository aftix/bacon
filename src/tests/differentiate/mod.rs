use crate::differentiate::*;

fn exp(f: f64) -> f64 {
    f.exp()
}

fn quadratic(f: f64) -> f64 {
    f.powi(2)
}

fn sin(f: f64) -> f64 {
    f.sin()
}

#[test]
fn deriv() {
    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = derivative(exp, i, 0.1);
        assert!(((d - i.exp()) / i.exp()).abs() < 0.01);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = derivative(quadratic, i, 0.1);
        if 2.0 * i.abs() > 0.05 {
            assert!(((d - 2.0 * i) / (2.0 * i)).abs() < 0.05);
        } else {
            assert!((d - 2.0 * i).abs() < 0.05);
        }
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = derivative(sin, i, 0.1);
        if i.cos().abs() > 0.05 {
            assert!(((d - i.cos()) / i.cos()).abs() < 0.01);
        } else {
            assert!((d - i.cos()).abs() < 0.05);
        }
    }
}

#[test]
fn second_deriv() {
    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = second_derivative(exp, i, 0.1);
        assert!(((d - i.exp()) / i.exp()).abs() < 0.01);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = second_derivative(quadratic, i, 0.1);
        assert!(((d - 2.0) / 2.0).abs() < 0.05);
    }

    for i in -1000..1000 {
        let i = i as f64 * 0.01;
        let d = second_derivative(sin, i, 0.1);
        if i.sin().abs() > 0.05 {
            assert!(((d + i.sin()) / i.sin()).abs() < 0.01);
        } else {
            assert!((d + i.sin()).abs() < 0.05);
        }
    }
}
