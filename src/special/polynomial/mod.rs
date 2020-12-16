use crate::polynomial::Polynomial;
use alga::general::ComplexField;

/// Get the nth legendre polynomial.
///
/// Gets the nth legendre polynomial over a specified field. This is
/// done using the recurrence relation and is properly normalized.
///
/// # Examples
/// ```
/// use bacon_sci::special::legendre;
/// fn example() {
///     let p_3 = legendre::<f64>(3);
///     assert_eq!(p_3.order(), 3);
///     assert!(p_3.get_coefficient(0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(1) + 1.5).abs() < 0.00001);
///     assert!(p_3.get_coefficient(2).abs() < 0.00001);
///     assert!((p_3.get_coefficient(3) - 2.5).abs() < 0.00001);
/// }
///
pub fn legendre<N: ComplexField>(n: u32) -> Polynomial<N> {
    if n == 0 {
        return polynomial![N::one()];
    }
    if n == 1 {
        return polynomial![N::one(), N::zero()];
    }

    let mut p_0 = polynomial![N::one()];
    let mut p_1 = polynomial![N::one(), N::zero()];

    for i in 1..n {
        // Get p_i+1 from p_i and p_i-1
        let mut p_next = polynomial![N::from_u32(2 * i + 1).unwrap(), N::zero()] * &p_1;
        p_next -= &p_0 * N::from_u32(i).unwrap();
        p_next /= N::from_u32(i + 1).unwrap();
        p_0 = p_1;
        p_1 = p_next;
    }

    p_1
}

/// Get the nth hermite polynomial.
///
/// Gets the nth physicist's hermite polynomial over a specified field. This is
/// done using the recurrance relation so the normalization is standard for the
/// physicist's hermite polynomial.
///
/// # Examples
/// ```
/// use bacon_sci::special::hermite;
/// fn example() {
///     let h_3 = hermite::<f64>(3);
///     assert_eq!(h_3.order(), 3);
///     assert!(h_3.get_coefficient(0).abs() < 0.0001);
///     assert!((h_3.get_coefficient(1) - 12.0).abs() < 0.0001);
///     assert!(h_3.get_coefficient(2).abs() < 0.0001);
///     assert!((h_3.get_coefficient(3) - 8.0).abs() < 0.0001);
/// }
/// ```
pub fn hermite<N: ComplexField>(n: u32) -> Polynomial<N> {
    if n == 0 {
        return polynomial![N::one()];
    }
    if n == 1 {
        return polynomial![N::from_f64(2.0).unwrap(), N::zero()];
    }

    let mut h_0 = polynomial![N::one()];
    let mut h_1 = polynomial![N::from_f64(2.0).unwrap(), N::zero()];
    let x_2 = h_1.clone();

    for i in 1..n {
        let next = &x_2 * &h_1 - (&h_0 * N::from_f64(2.0 * i as f64).unwrap());
        h_0 = h_1;
        h_1 = next;
    }

    h_1
}
