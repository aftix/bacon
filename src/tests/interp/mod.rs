use crate::interp::lagrange;

#[test]
fn lagrange_interp() {
    let xs: Vec<_> = (0..10).map(|i| i as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.cos()).collect();

    let poly = lagrange(&xs, &ys).unwrap();

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
