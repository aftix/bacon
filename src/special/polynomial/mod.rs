use crate::polynomial::Polynomial;
use alga::general::ComplexField;
use std::iter::FromIterator;

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

/// Get the zeros of the nth legendre polynomial.
pub fn legendre_zeros<N: ComplexField>(
    n: u32,
    tol: N::RealField,
    n_max: usize,
) -> Result<Vec<N>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![N::zero()]);
    }

    let mut p_0 = polynomial![N::one()];
    let mut p_1 = polynomial![N::one(), N::zero()];
    let mut zeros = vec![N::zero()];

    for i in 1..n {
        let mut p_next = polynomial![N::from_u32(2 * i + 1).unwrap(), N::zero()] * &p_1;
        p_next -= &p_0 * N::from_u32(i).unwrap();
        p_next /= N::from_u32(i + 1).unwrap();

        let mut guesses = Vec::with_capacity(i as usize + 1);
        guesses.push(N::from_f64(0.5).unwrap() * (zeros[0] - N::one()));
        for j in 1..zeros.len() {
            guesses.push(N::from_f64(0.5).unwrap() * (zeros[j] + zeros[j - 1]));
        }
        guesses.push(N::from_f64(0.5).unwrap() * (N::one() + zeros[zeros.len() - 1]));

        p_0 = p_1;
        p_1 = p_next;
        zeros = Vec::from_iter(
            p_1.roots(&guesses, tol, n_max)?
                .iter()
                .map(|c| N::from_real(c.re)),
        );
    }

    Ok(zeros)
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

fn factorial(k: u32) -> u32 {
    let mut acc = 1;
    for i in 2..=k {
        acc *= i;
    }
    acc
}

fn choose(n: u32, k: u32) -> u32 {
    let mut acc = 1;
    for i in n - k + 1..=n {
        acc *= i;
    }
    for i in 2..=k {
        acc /= i;
    }
    acc
}

/// Get the nth (positive) laguerre polynomial.
///
/// Gets the nth (positive) laguerre polynomial over a specified field. This is
/// done using the direct formula and is properly normalized.
///
/// # Examples
/// ```
/// use bacon_sci::special::laguerre;
/// fn example() {
///     let p_3 = laguerre::<f64>(3);
///     assert_eq!(p_3.order(), 3);
///     assert!((p_3.get_coefficient(0) - 1.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(1) + 3.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(2) - 9.0/6.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(3) + 1.0/6.0).abs() < 0.00001);
/// }
///
pub fn laguerre<N: ComplexField>(n: u32) -> Polynomial<N> {
    let mut coefficients = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        coefficients.push(
            N::from_u32(choose(n, k)).unwrap() / N::from_u32(factorial(k)).unwrap()
                * if k % 2 == 0 { N::one() } else { -N::one() },
        );
    }

    Polynomial::from_iter(coefficients.iter().copied())
}

/// Get the nth chebyshev polynomial.
///
/// Gets the nth chebyshev polynomial over a specified field. This is
/// done using the recursive formula and is properly normalized.
///
/// # Examples
/// ```
/// use bacon_sci::special::chebyshev;
/// fn example() {
///     let t_3 = chebyshev::<f64>(3);
///     assert_eq!(t_3.order(), 3);
///     assert!(t_3.get_coefficient(0).abs() < 0.00001);
///     assert!((t_3.get_coefficient(1) + 3.0).abs() < 0.00001);
///     assert!(t_3.get_coefficient(2).abs() < 0.00001);
///     assert!((t_3.get_coefficient(3) - 4.0).abs() < 0.00001);
/// }
///
pub fn chebyshev<N: ComplexField>(n: u32) -> Polynomial<N> {
    if n == 0 {
        return polynomial![N::one()];
    }
    if n == 1 {
        return polynomial![N::one(), N::zero()];
    }

    let mut t_0 = polynomial![N::one()];
    let mut t_1 = polynomial![N::one(), N::zero()];
    let double = polynomial![N::from_i32(2).unwrap(), N::zero()];

    for _ in 1..n {
        let next = &double * &t_1 - &t_0;
        t_0 = t_1;
        t_1 = next;
    }
    t_1
}

/// Get the nth chebyshev polynomial of the second kind.
///
/// Gets the nth chebyshev polynomial of the second kind over a specified field. This is
/// done using the recursive formula and is properly normalized.
///
/// # Examples
/// ```
/// use bacon_sci::special::chebyshev_second;
/// fn example() {
///     let u_3 = chebyshev_second::<f64>(3);
///     assert_eq!(u_3.order(), 3);
///     assert!(u_3.get_coefficient(0).abs() < 0.00001);
///     assert!((u_3.get_coefficient(1) + 4.0).abs() < 0.00001);
///     assert!(u_3.get_coefficient(2).abs() < 0.00001);
///     assert!((u_3.get_coefficient(3) - 8.0).abs() < 0.00001);
/// }
///
pub fn chebyshev_second<N: ComplexField>(n: u32) -> Polynomial<N> {
    if n == 0 {
        return polynomial![N::one()];
    }
    if n == 1 {
        return polynomial![N::from_i32(2).unwrap(), N::zero()];
    }

    let mut t_0 = polynomial![N::one()];
    let mut t_1 = polynomial![N::from_i32(2).unwrap(), N::zero()];
    let double = t_1.clone();

    for _ in 1..n {
        let next = &double * &t_1 - &t_0;
        t_0 = t_1;
        t_1 = next;
    }
    t_1
}
