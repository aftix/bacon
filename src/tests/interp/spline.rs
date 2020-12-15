use crate::interp::spline_free;

#[test]
fn free_spline() {
    let xs: Vec<_> = (0..=10).map(|x| x as f64).collect();
    let ys: Vec<_> = xs.iter().map(|x| x.exp()).collect();

    let spline = spline_free(&xs, &ys, 1e-10).unwrap();

    for i in 0..500 {
        let i = i as f64 * 0.01;
        assert!((spline.evaluate(i).unwrap() - i.exp()).abs() / i.exp() < 0.25);
    }
}
