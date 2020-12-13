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
