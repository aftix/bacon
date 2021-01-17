use crate::optimize::linear_fit;
use rand::prelude::*;

#[test]
fn test_linear_fit() {
    let line = polynomial![4.0, -1.0];
    let mut xs: Vec<f64> = vec![];
    let mut ys: Vec<f64> = vec![];

    for i in -5..=5 {
        xs.push(i as f64);
        ys.push(line.evaluate(i as f64));
    }

    let fit = linear_fit(&xs, &ys).unwrap();

    assert!(approx_eq!(
        f64,
        line.get_coefficient(0),
        fit.get_coefficient(0),
        epsilon = 1e-5
    ));
    assert!(approx_eq!(
        f64,
        line.get_coefficient(1),
        fit.get_coefficient(1),
        epsilon = 1e-5
    ));

    xs.clear();
    ys.clear();

    let distr = rand::distributions::Uniform::new_inclusive(-0.1, 0.1);
    let mut rng = thread_rng();
    for i in -5..=5 {
        xs.push(i as f64);
        let wiggle: f64 = rng.sample(distr);
        ys.push(wiggle + line.evaluate(i as f64));
    }

    let fit = linear_fit(&xs, &ys).unwrap();

    assert!(approx_eq!(
        f64,
        line.get_coefficient(0),
        fit.get_coefficient(0),
        epsilon = 0.1
    ));
    assert!(approx_eq!(
        f64,
        line.get_coefficient(1),
        fit.get_coefficient(1),
        epsilon = 0.1
    ));

    xs.clear();
    ys.clear();

    for i in -5..=5 {
        xs.push(i as f64);
        ys.push(((i as f64).exp() * 2.0).ln());
    }

    let fit = linear_fit(&xs, &ys).unwrap();

    assert!(approx_eq!(
        f64,
        2.0_f64.ln(),
        fit.get_coefficient(0),
        epsilon = 0.1
    ));
    assert!(approx_eq!(f64, 1.0, fit.get_coefficient(1), epsilon = 0.1));
}
