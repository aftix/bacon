use super::polynomial::Polynomial;
use alga::general::ComplexField;

/// Create a Lagrange interpolating polynomial.
///
/// Create an nth degree polynomial matching the n points (xs[i], ys[i])
/// using Neville's iterated method for Lagrange polynomials. The result will
/// match no derivatives.
///
/// # Examples
/// ```
/// use bacon_sci::interp::lagrange;
/// use bacon_sci::polynomial::Polynomial;
/// fn example() {
///     let xs: Vec<_> = (0..10).map(|i| i as f64).collect();
///     let ys: Vec<_> = xs.iter().map(|x| x.cos()).collect();
///     let poly = lagrange(&xs, &ys).unwrap();
///     for x in xs {
///         assert!((x.cos() - poly.evaluate(x)).abs() < 0.00001);
///     }
/// }
/// ```
pub fn lagrange<N: ComplexField>(xs: &[N], ys: &[N]) -> Result<Polynomial<N>, String> {
    if xs.len() != ys.len() {
        return Err("lagrange: slices have mismatched dimension".to_owned());
    }

    let mut qs = vec![Polynomial::new(); xs.len() * xs.len()];
    for (ind, y) in ys.iter().enumerate() {
        qs[ind] = polynomial![*y];
    }

    for i in 1..xs.len() {
        let poly_2 = polynomial![N::one(), -xs[i]];
        for j in 1..=i {
            let poly_1 = polynomial![N::one(), -xs[i - j]];
            let idenom = N::one() / (xs[i] - xs[i - j]);
            let numer =
                &poly_1 * &qs[i + xs.len() * (j - 1)] - &poly_2 * &qs[(i - 1) + xs.len() * (j - 1)];
            qs[i + xs.len() * j] = numer * idenom;
        }
    }

    Ok(qs[xs.len() * xs.len() - 1].clone())
}
