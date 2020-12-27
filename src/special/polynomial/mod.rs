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
///     let p_3 = legendre::<f64>(3, 1e-8).unwrap();
///     assert_eq!(p_3.order(), 3);
///     assert!(p_3.get_coefficient(0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(1) + 1.5).abs() < 0.00001);
///     assert!(p_3.get_coefficient(2).abs() < 0.00001);
///     assert!((p_3.get_coefficient(3) - 2.5).abs() < 0.00001);
/// }
///
pub fn legendre<N: ComplexField>(n: u32, tol: N::RealField) -> Result<Polynomial<N>, String> {
    if n == 0 {
        let mut poly = polynomial![N::one()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }
    if n == 1 {
        let mut poly = polynomial![N::one(), N::zero()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }

    let mut p_0 = polynomial![N::one()];
    p_0.set_tolerance(tol)?;
    let mut p_1 = polynomial![N::one(), N::zero()];
    p_1.set_tolerance(tol)?;

    for i in 1..n {
        // Get p_i+1 from p_i and p_i-1
        let mut p_next = polynomial![N::from_u32(2 * i + 1).unwrap(), N::zero()] * &p_1;
        p_next.set_tolerance(tol)?;
        p_next -= &p_0 * N::from_u32(i).unwrap();
        p_next /= N::from_u32(i + 1).unwrap();
        p_0 = p_1;
        p_1 = p_next;
    }

    Ok(p_1)
}

/// Get the zeros of the nth legendre polynomial.
/// Calculate zeros to tolerance `tol`, have polynomials
/// with tolerance `poly_tol`.
pub fn legendre_zeros<N: ComplexField>(
    n: u32,
    tol: N::RealField,
    poly_tol: N::RealField,
    n_max: usize,
) -> Result<Vec<N>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![N::zero()]);
    }

    let poly: Polynomial<N> = legendre(n, poly_tol)?;
    Ok(Vec::from_iter(poly.roots(tol, n_max)?.iter().map(|c| {
        if c.re.abs() < tol {
            N::zero()
        } else {
            N::from_real(c.re)
        }
    })))
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
///     let h_3 = hermite::<f64>(3, 1e-8).unwrap();
///     assert_eq!(h_3.order(), 3);
///     assert!(h_3.get_coefficient(0).abs() < 0.0001);
///     assert!((h_3.get_coefficient(1) - 12.0).abs() < 0.0001);
///     assert!(h_3.get_coefficient(2).abs() < 0.0001);
///     assert!((h_3.get_coefficient(3) - 8.0).abs() < 0.0001);
/// }
/// ```
pub fn hermite<N: ComplexField>(n: u32, tol: N::RealField) -> Result<Polynomial<N>, String> {
    if n == 0 {
        let mut poly = polynomial![N::one()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }
    if n == 1 {
        let mut poly = polynomial![N::from_f64(2.0).unwrap(), N::zero()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }

    let mut h_0 = polynomial![N::one()];
    h_0.set_tolerance(tol)?;
    let mut h_1 = polynomial![N::from_f64(2.0).unwrap(), N::zero()];
    h_1.set_tolerance(tol)?;
    let x_2 = h_1.clone();

    for i in 1..n {
        let next = &x_2 * &h_1 - (&h_0 * N::from_f64(2.0 * i as f64).unwrap());
        h_0 = h_1;
        h_1 = next;
    }

    Ok(h_1)
}

/// Get the zeros of the nth Hermite polynomial within tolerance `tol` with polynomial
/// tolerance `poly_tol`
pub fn hermite_zeros<N: ComplexField>(
    n: u32,
    tol: N::RealField,
    poly_tol: N::RealField,
    n_max: usize,
) -> Result<Vec<N>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![N::zero()]);
    }

    let poly: Polynomial<N> = hermite(n, poly_tol)?;

    Ok(Vec::from_iter(poly.roots(tol, n_max)?.iter().map(|c| {
        if c.re.abs() < tol {
            N::zero()
        } else {
            N::from_real(c.re)
        }
    })))
}

fn factorial<N: ComplexField>(k: u32) -> N {
    let mut acc = N::one();
    for i in 2..=k {
        acc *= N::from_u32(i).unwrap();
    }
    acc
}

fn choose<N: ComplexField>(n: u32, k: u32) -> N {
    let mut acc = N::one();
    for i in n - k + 1..=n {
        acc *= N::from_u32(i).unwrap();
    }
    for i in 2..=k {
        acc /= N::from_u32(i).unwrap();
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
///     let p_3 = laguerre::<f64>(3, 1e-8).unwrap();
///     assert_eq!(p_3.order(), 3);
///     assert!((p_3.get_coefficient(0) - 1.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(1) + 3.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(2) - 9.0/6.0).abs() < 0.00001);
///     assert!((p_3.get_coefficient(3) + 1.0/6.0).abs() < 0.00001);
/// }
///
pub fn laguerre<N: ComplexField>(n: u32, tol: N::RealField) -> Result<Polynomial<N>, String> {
    let mut coefficients = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        coefficients.push(
            choose::<N>(n, k) / factorial::<N>(k) * if k % 2 == 0 { N::one() } else { -N::one() },
        );
    }

    let mut poly = Polynomial::from_iter(coefficients.iter().copied());
    poly.set_tolerance(tol)?;
    Ok(poly)
}

/// Get the zeros of the nth Laguerre polynomial
pub fn laguerre_zeros<N: ComplexField>(
    n: u32,
    tol: N::RealField,
    poly_tol: N::RealField,
    n_max: usize,
) -> Result<Vec<N>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![N::one()]);
    }

    let poly: Polynomial<N> = laguerre(n, poly_tol)?;

    Ok(Vec::from_iter(poly.roots(tol, n_max)?.iter().map(|c| {
        if c.re.abs() < tol {
            N::zero()
        } else {
            N::from_real(c.re)
        }
    })))
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
///     let t_3 = chebyshev::<f64>(3, 1e-8).unwrap();
///     assert_eq!(t_3.order(), 3);
///     assert!(t_3.get_coefficient(0).abs() < 0.00001);
///     assert!((t_3.get_coefficient(1) + 3.0).abs() < 0.00001);
///     assert!(t_3.get_coefficient(2).abs() < 0.00001);
///     assert!((t_3.get_coefficient(3) - 4.0).abs() < 0.00001);
/// }
///
pub fn chebyshev<N: ComplexField>(n: u32, tol: N::RealField) -> Result<Polynomial<N>, String> {
    if n == 0 {
        let mut poly = polynomial![N::one()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }
    if n == 1 {
        let mut poly = polynomial![N::one(), N::zero()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }

    if n % 2 == 0 {
        let half = chebyshev(n / 2, tol)?;
        Ok(&half * &half * N::from_i32(2).unwrap() - polynomial![N::one()])
    } else {
        let half = chebyshev(n / 2, tol)?;
        let other_half = chebyshev(n / 2 + 1, tol)?;
        Ok(&half * &other_half * N::from_i32(2).unwrap() - polynomial![N::one(), N::zero()])
    }
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
///     let u_3 = chebyshev_second::<f64>(3, 1e-8).unwrap();
///     assert_eq!(u_3.order(), 3);
///     assert!(u_3.get_coefficient(0).abs() < 0.00001);
///     assert!((u_3.get_coefficient(1) + 4.0).abs() < 0.00001);
///     assert!(u_3.get_coefficient(2).abs() < 0.00001);
///     assert!((u_3.get_coefficient(3) - 8.0).abs() < 0.00001);
/// }
///
pub fn chebyshev_second<N: ComplexField>(
    n: u32,
    tol: N::RealField,
) -> Result<Polynomial<N>, String> {
    if n == 0 {
        let mut poly = polynomial![N::one()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }
    if n == 1 {
        let mut poly = polynomial![N::from_i32(2).unwrap(), N::zero()];
        poly.set_tolerance(tol)?;
        return Ok(poly);
    }

    let mut t_0 = polynomial![N::one()];
    t_0.set_tolerance(tol)?;
    let mut t_1 = polynomial![N::from_i32(2).unwrap(), N::zero()];
    t_1.set_tolerance(tol)?;
    let double = t_1.clone();

    for _ in 1..n {
        let next = &double * &t_1 - &t_0;
        t_0 = t_1;
        t_1 = next;
    }
    Ok(t_1)
}
