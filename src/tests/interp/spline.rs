/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::interp::{spline_clamped, spline_free};

#[test]
fn free_spline() {
    let xs: Vec<_> = (0..=10).map(|x| x as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.exp()).collect();

    let spline = spline_free(&xs, &ys, 1e-10).unwrap();

    for i in 0..1000 {
        let i = i as f64 * 0.01;
        assert!((spline.evaluate(i).unwrap() - i.exp()).abs() / i.exp() < 0.1);
    }
}

#[test]
fn clamped_spline() {
    let xs: Vec<_> = (0..=10).map(|x| x as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.exp()).collect();

    let spline = spline_clamped(&xs, &ys, (1.0, (10.0f64).exp()), 1e-10).unwrap();

    for i in 0..1000 {
        let i = i as f64 * 0.01;
        assert!((spline.evaluate(i).unwrap() - i.exp()).abs() / i.exp() < 0.05);
    }
}
