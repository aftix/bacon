use crate::polynomial::Polynomial;
use alga::general::ComplexField;

/// Linear least-squares regression
pub fn linear_fit<N: ComplexField>(xs: &[N], ys: &[N]) -> Result<Polynomial<N>, String> {
    if xs.len() != ys.len() {
        return Err("linear_fit: xs length does not match ys length".to_owned());
    }

    let mut sum_x = N::zero();
    let mut sum_y = N::zero();
    let mut sum_x_sq = N::zero();
    let mut sum_y_sq = N::zero();
    let mut sum_xy = N::zero();

    for (ind, x) in xs.iter().enumerate() {
        sum_x += *x;
        sum_y += ys[ind];
        sum_x_sq += x.powi(2);
        sum_y_sq += ys[ind].powi(2);
        sum_xy += ys[ind] * *x;
    }

    let m = N::from_usize(xs.len()).unwrap();
    let denom = m * sum_x_sq - sum_x.powi(2);
    let a = (m * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_x_sq * sum_y - sum_xy * sum_x) / denom;

    Ok(polynomial![a, b])
}
