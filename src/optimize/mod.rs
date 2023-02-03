use crate::polynomial::Polynomial;
use nalgebra::{ComplexField, DMatrix, DVector, RealField, SVector};
use num_traits::{FromPrimitive, One, Zero};

/// Linear least-squares regression
///
/// # Errors
/// Returns an error if the linear fit fails. (`xs.len() != ys.len()`)
///
/// # Panics
/// Panics if a `usize` can not be transformed into the generic type.
pub fn linear_fit<N>(xs: &[N], ys: &[N]) -> Result<Polynomial<N>, String>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
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

// Compute the J matrix for LM using finite differences, 3 point formula
fn jac_finite_differences<N, F, const V: usize>(
    mut f: F,
    xs: &[N],
    params: &mut SVector<N, V>,
    mat: &mut DMatrix<N>,
    h: N::RealField,
) where
    N: ComplexField + FromPrimitive + Copy,
    F: FnMut(N, &SVector<N, V>) -> N,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    let h = N::from_real(h);
    let denom = N::one() / (N::from_i32(2).unwrap() * h);
    for row in 0..mat.column(0).len() {
        for col in 0..mat.row(0).len() {
            params[col] += h;
            let above = f(xs[row], params);
            params[col] -= h;
            params[col] -= h;
            let below = f(xs[row], params);
            mat[(row, col)] = denom * (above + below);
            params[col] += h;
        }
    }
}

// Compute the J matrix for LM using analytic formula
fn jac_analytic<N, F, const V: usize>(
    mut jac: F,
    xs: &[N],
    params: &mut SVector<N, V>,
    mat: &mut DMatrix<N>,
) where
    N: ComplexField + Copy,
    F: FnMut(N, &SVector<N, V>) -> SVector<N, V>,
{
    for row in 0..mat.column(0).len() {
        let deriv = jac(xs[row], params);
        for col in 0..mat.row(0).len() {
            mat[(row, col)] = deriv[col];
        }
    }
}

#[derive(Debug, Clone)]
pub struct CurveFitParams<N: ComplexField> {
    pub damping: N::RealField,
    pub tolerance: N::RealField,
    pub h: N::RealField,
    pub damping_mult: N::RealField,
}

impl<N: ComplexField + FromPrimitive> Default for CurveFitParams<N> {
    fn default() -> Self {
        CurveFitParams {
            damping: N::from_f64(2.0).unwrap().real(),
            tolerance: N::from_f64(1e-5).unwrap().real(),
            h: N::from_f64(0.1).unwrap().real(),
            damping_mult: N::from_f64(1.5).unwrap().real(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn initial_residuals<N, F, const V: usize>(
    xs: &[N],
    ys: &DVector<N>,
    damping: &mut N::RealField,
    damping_mult: N::RealField,
    h: N::RealField,
    mut f: F,
    jac: &mut DMatrix<N>,
    jac_transpose: &mut DMatrix<N>,
    mut params: SVector<N, V>,
) -> Result<(N::RealField, DVector<N>), String>
where
    N: ComplexField + Copy + FromPrimitive,
    <N as ComplexField>::RealField: Copy + FromPrimitive,
    F: FnMut(N, &SVector<N, V>) -> N,
{
    let mut resid = Vec::with_capacity(xs.len());
    for (ind, &x) in xs.iter().enumerate() {
        resid.push(ys[ind] - f(x, &params));
    }
    let sum_sq_initial: N::RealField = resid
        .iter()
        .map(|&r| r.modulus_squared())
        .fold(N::RealField::zero(), |acc, r| acc + r);

    // Get initial factor
    let mut sum_sq = sum_sq_initial + N::RealField::one();
    let mut damping_tmp = *damping / damping_mult;
    let mut j = 0;
    let mut evaluation: DVector<N> =
        DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
    while sum_sq > sum_sq_initial && j < 1000 {
        damping_tmp *= damping_mult;
        let diff = ys - &evaluation;
        let mut b = jac_transpose as &DMatrix<N> * &diff;
        // Always square
        let mut multiplied = jac_transpose as &DMatrix<N> * jac as &DMatrix<N>;
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping_tmp);
        }
        let lu = multiplied.clone().lu();
        let solved = lu.solve_mut(&mut b);
        if !solved {
            return Err("curve_fit: unable to solve linear equation".to_owned());
        }
        params += &b;
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
        let diff = ys - &evaluation;
        sum_sq = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        j += 1;
        jac_finite_differences(&mut f, xs, &mut params, jac, h);
        *jac_transpose = jac.transpose();
    }
    if j != 1000 {
        *damping = damping_tmp;
    }
    Ok((sum_sq, evaluation))
}

#[allow(clippy::too_many_arguments)]
fn initial_residuals_exact<N, F, G, const V: usize>(
    xs: &[N],
    ys: &DVector<N>,
    damping: &mut N::RealField,
    damping_mult: N::RealField,
    mut f: F,
    mut jacobian: G,
    jac: &mut DMatrix<N>,
    jac_transpose: &mut DMatrix<N>,
    mut params: SVector<N, V>,
) -> Result<(N::RealField, DVector<N>), String>
where
    N: ComplexField + Copy + FromPrimitive,
    <N as ComplexField>::RealField: Copy + FromPrimitive,
    F: FnMut(N, &SVector<N, V>) -> N,
    G: FnMut(N, &SVector<N, V>) -> SVector<N, V>,
{
    // Get the initial sum of square residuals
    let mut resid = Vec::with_capacity(xs.len());
    for (ind, &x) in xs.iter().enumerate() {
        resid.push(ys[ind] - f(x, &params));
    }
    let sum_sq_initial: N::RealField = resid
        .iter()
        .map(|&r| r.modulus_squared())
        .fold(N::RealField::zero(), |acc, r| acc + r);

    // Get initial factor
    let mut sum_sq = sum_sq_initial + N::RealField::one();
    let mut damping_tmp = *damping / damping_mult;
    let mut j = 0;
    let mut evaluation: DVector<N> =
        DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
    while sum_sq > sum_sq_initial && j < 1000 {
        damping_tmp *= damping_mult;
        let diff = ys - &evaluation;
        let mut b = jac_transpose as &DMatrix<N> * &diff;
        // Always square
        let mut multiplied = jac_transpose as &DMatrix<N> * jac as &DMatrix<N>;
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping_tmp);
        }
        let lu = multiplied.clone().lu();
        let solved = lu.solve_mut(&mut b);
        if !solved {
            let lu = multiplied.clone().full_piv_lu();
            let full_lu_solved = lu.solve_mut(&mut b);
            if !full_lu_solved {
                let qr = multiplied.qr();
                let qr_solved = qr.solve_mut(&mut b);
                if !qr_solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        params += &b;
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
        let diff = ys - &evaluation;
        sum_sq = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        j += 1;
        jac_analytic(&mut jacobian, xs, &mut params, jac);
        *jac_transpose = jac.transpose();
    }
    if j != 1000 {
        *damping = damping_tmp;
    }

    Ok((sum_sq, evaluation))
}

/// Fit a curve using the Levenberg-Marquardt algorithm.
///
/// Uses finite differences of h to calculate the jacobian. If jacobian
/// can be found analytically, then use `curve_fit_jac`. Keeps iterating until
/// the differences between the sum of the square residuals of two iterations
/// is under tol.
///
/// # Errors
/// Returns an error if curve fitting fails.
///
/// # Panics
/// Panics if a u8 can not be converted to the generic type.
pub fn curve_fit<N, F, const V: usize>(
    mut f: F,
    xs: &[N],
    ys: &[N],
    initial: &[N],
    params: &CurveFitParams<N>,
) -> Result<SVector<N, V>, String>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    F: FnMut(N, &SVector<N, V>) -> N,
{
    let tol = params.tolerance;
    let mut damping = params.damping;
    let h = params.h;
    let damping_mult = params.damping_mult;

    if !tol.is_sign_positive() {
        return Err("curve_fit: tol must be positive".to_owned());
    }

    if !h.is_sign_positive() {
        return Err("curve_fit: h must be positive".to_owned());
    }

    if !damping.is_sign_positive() {
        return Err("curve_fit: damping must be positive".to_owned());
    }

    if xs.len() != ys.len() {
        return Err("curve_fit: xs length must match ys length".to_owned());
    }

    let mut params = SVector::<N, V>::from_column_slice(initial);
    let ys = DVector::<N>::from_column_slice(ys);
    let mut jac: DMatrix<N> = DMatrix::identity(xs.len(), params.len());
    jac_finite_differences(&mut f, xs, &mut params, &mut jac, h);
    let mut jac_transpose = jac.transpose();

    // Get the initial sum of square residuals
    let (mut sum_sq, mut evaluation) = initial_residuals(
        xs,
        &ys,
        &mut damping,
        damping_mult,
        h,
        &mut f,
        &mut jac,
        &mut jac_transpose,
        params,
    )?;

    let mut last_sum_sq = sum_sq;
    sum_sq += N::from_u8(2).unwrap().real() * tol;
    while (last_sum_sq - sum_sq).abs() > tol {
        last_sum_sq = sum_sq;
        // Get right side of iteration equation
        let diff = &ys - &evaluation;
        let mut b = &jac_transpose * &diff;
        let mut b_div = b.clone();
        // Get left side of equation
        let mut multiplied = &jac_transpose * &jac;
        let mut multiplied_div = multiplied.clone();
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping);
        }
        // Solve equation with LU w/ partial pivoting first
        let lu = multiplied.clone().lu();
        let lu_solved = lu.solve_mut(&mut b);
        if !lu_solved {
            return Err("curve_fit: unable to solve linear equation".to_owned());
        }
        let new_params = params + &b;

        // Now solve for damping / damping_mult
        for i in 0..multiplied_div.row(0).len() {
            multiplied_div[(i, i)] *= N::one() + N::from_real(damping / damping_mult);
        }
        let lu = multiplied_div.clone().lu();
        let solved = lu.solve_mut(&mut b_div);
        if !solved {
            let lu = multiplied_div.clone().full_piv_lu();
            let full_lu_solved = lu.solve_mut(&mut b_div);
            if !full_lu_solved {
                let qr = multiplied_div.qr();
                let qr_solved = qr.solve_mut(&mut b_div);
                if !qr_solved {
                    return Err("curve_fit: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params_div = params + &b_div;

        // get residuals for each of the new solutions
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &new_params)));
        let evaluation_div =
            DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &new_params_div)));
        let diff = &ys - &evaluation;
        let diff_div = &ys - &evaluation_div;

        let resid: N::RealField = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        let resid_div: N::RealField = diff_div
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);

        if resid_div < resid {
            damping /= damping_mult;
            evaluation = evaluation_div;
            params = new_params_div;
            sum_sq = resid_div;
        } else {
            params = new_params;
            sum_sq = resid;
        }

        jac_finite_differences(&mut f, xs, &mut params, &mut jac, h);
        jac_transpose = jac.transpose();
    }

    Ok(params)
}

/// Fit a curve using the Levenberg-Marquardt algorithm.
///
/// Uses an analytic jacobian.Keeps iterating until
/// the differences between the sum of the square residuals of two iterations
/// is under tol. Jacobian should be a function that returns a column vector
/// where jacobian[i] is the partial derivative of f with respect to param[i].
///
/// # Errors
/// Returns an error if curve fit fails.
///
/// # Panics
/// Panics if a u8 can not be converted to the generic type.
pub fn curve_fit_jac<N, F, G, const V: usize>(
    mut f: F,
    xs: &[N],
    ys: &[N],
    initial: &[N],
    mut jacobian: G,
    params: &CurveFitParams<N>,
) -> Result<SVector<N, V>, String>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    F: FnMut(N, &SVector<N, V>) -> N,
    G: FnMut(N, &SVector<N, V>) -> SVector<N, V>,
{
    let tol = params.tolerance;
    let mut damping = params.damping;
    let damping_mult = params.damping_mult;

    if !tol.is_sign_positive() {
        return Err("curve_fit_jac: tol must be positive".to_owned());
    }

    if !damping.is_sign_positive() {
        return Err("curve_fit_jac: damping must be positive".to_owned());
    }

    if xs.len() != ys.len() {
        return Err("curve_fit_jac: xs length must match ys length".to_owned());
    }

    let mut params = SVector::<N, V>::from_column_slice(initial);
    let ys = DVector::<N>::from_column_slice(ys);
    let mut jac: DMatrix<N> = DMatrix::identity(xs.len(), params.len());
    jac_analytic(&mut jacobian, xs, &mut params, &mut jac);
    let mut jac_transpose = jac.transpose();

    let (mut sum_sq, mut evaluation) = initial_residuals_exact(
        xs,
        &ys,
        &mut damping,
        damping_mult,
        &mut f,
        &mut jacobian,
        &mut jac,
        &mut jac_transpose,
        params,
    )?;

    let mut last_sum_sq = sum_sq;
    sum_sq += N::from_u8(2).unwrap().real() * tol;
    while (last_sum_sq - sum_sq).abs() > tol {
        last_sum_sq = sum_sq;
        // Get right side of iteration equation
        let diff = &ys - &evaluation;
        let mut b = &jac_transpose * &diff;
        let mut b_div = b.clone();
        // Get left side of equation
        let mut multiplied = &jac_transpose * &jac;
        let mut multiplied_div = multiplied.clone();
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping);
        }
        // Solve equation with LU w/ partial pivoting first
        // Then try LU w/ Full pivoting, QR
        let lu = multiplied.clone().lu();
        let lu_solved = lu.solve_mut(&mut b);
        if !lu_solved {
            let lu = multiplied.clone().full_piv_lu();
            let full_lu_solved = lu.solve_mut(&mut b);
            if !full_lu_solved {
                let qr = multiplied.qr();
                let qr_solved = qr.solve_mut(&mut b);
                if !qr_solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params = params + &b;

        // Now solve for damping / damping_mult
        for i in 0..multiplied_div.row(0).len() {
            multiplied_div[(i, i)] *= N::one() + N::from_real(damping / damping_mult);
        }
        let lu = multiplied_div.clone().lu();
        let solved = lu.solve_mut(&mut b_div);
        if !solved {
            let lu = multiplied_div.clone().full_piv_lu();
            let full_lu_solved = lu.solve_mut(&mut b_div);
            if !full_lu_solved {
                let qr = multiplied_div.qr();
                let qr_solved = qr.solve_mut(&mut b_div);
                if !qr_solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params_div = params + &b_div;

        // get residuals for each of the new solutions
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &new_params)));
        let evaluation_div =
            DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &new_params_div)));
        let diff = &ys - &evaluation;
        let diff_div = &ys - &evaluation_div;

        let resid: N::RealField = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        let resid_div: N::RealField = diff_div
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);

        if resid_div < resid {
            damping /= damping_mult;
            evaluation = evaluation_div;
            params = new_params_div;
            sum_sq = resid_div;
        } else {
            params = new_params;
            sum_sq = resid;
        }

        jac_analytic(&mut jacobian, xs, &mut params, &mut jac);
        jac_transpose = jac.transpose();
    }

    Ok(params)
}
