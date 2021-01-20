use crate::polynomial::Polynomial;
use nalgebra::{
    allocator::Allocator, ComplexField, DMatrix, DVector, DefaultAllocator, DimName, RealField,
    VectorN,
};
use num_traits::{FromPrimitive, One, Zero};

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

// Compute the J matrix for LM using finite differences, 3 point formula
fn jac_finite_differences<N: ComplexField, V: DimName, F: FnMut(N, &VectorN<N, V>) -> N>(
    mut f: F,
    xs: &[N],
    params: &mut VectorN<N, V>,
    mat: &mut DMatrix<N>,
    h: N::RealField,
) where
    DefaultAllocator: Allocator<N, V>,
{
    let h = N::from_real(h);
    let denom = N::one() / (N::from_i32(2).unwrap() * h);
    for row in 0..mat.column(0).len() {
        for col in 0..mat.row(0).len() {
            params[col] += h;
            let above = f(xs[row], &params);
            params[col] -= h;
            params[col] -= h;
            let below = f(xs[row], &params);
            mat[(row, col)] = denom * (above + below);
            params[col] += h;
        }
    }
}

// Compute the J matrix for LM using analytic formula
fn jac_analytic<N: ComplexField, V: DimName, F: FnMut(N, &VectorN<N, V>) -> VectorN<N, V>>(
    mut jac: F,
    xs: &[N],
    params: &mut VectorN<N, V>,
    mat: &mut DMatrix<N>,
) where
    DefaultAllocator: Allocator<N, V>,
{
    for row in 0..mat.column(0).len() {
        let deriv = jac(xs[row], &params);
        for col in 0..mat.row(0).len() {
            mat[(row, col)] = deriv[col];
        }
    }
}

/// Fit a curve using the Levenberg-Marquardt algorithm.
///
/// Uses finite differences of h to calculate the jacobian. If jacobian
/// can be found analytically, then use curve_fit_jac. Keeps iterating until
/// the differences between the sum of the square residuals of two iterations
/// is under tol.
pub fn curve_fit<N: ComplexField, V: DimName, F: FnMut(N, &VectorN<N, V>) -> N>(
    mut f: F,
    xs: &[N],
    ys: &[N],
    initial: &[N],
    tol: N::RealField,
    h: N::RealField,
    mut damping: N::RealField,
) -> Result<VectorN<N, V>, String>
where
    DefaultAllocator: Allocator<N, V>,
{
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

    let mut params = VectorN::<N, V>::from_column_slice(initial);
    let ys = DVector::<N>::from_column_slice(ys);
    let mut jac: DMatrix<N> = DMatrix::identity(xs.len(), params.len());
    jac_finite_differences(&mut f, xs, &mut params, &mut jac, h);
    let mut jac_transpose = jac.transpose();

    let damping_mult = N::RealField::from_f64(1.5).unwrap();

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
    let mut damping_tmp = damping / damping_mult;
    let mut j = 0;
    let mut evaluation: DVector<N> =
        DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
    while sum_sq > sum_sq_initial && j < 1000 {
        damping_tmp *= damping_mult;
        let diff = &ys - &evaluation;
        let mut b = &jac_transpose * &diff;
        // Always square
        let mut multiplied = &jac_transpose * &jac;
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping_tmp);
        }
        let lu = multiplied.clone().lu();
        let solved = lu.solve_mut(&mut b);
        if !solved {
            let lu = multiplied.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b);
            if !solved {
                let qr = multiplied.qr();
                let solved = qr.solve_mut(&mut b);
                if !solved {
                    return Err("curve_fit: unable to solve linear equation".to_owned());
                }
            }
        }
        params += &b;
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
        let diff = &ys - &evaluation;
        sum_sq = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        j += 1;
        jac_finite_differences(&mut f, xs, &mut params, &mut jac, h);
        jac_transpose = jac.transpose();
    }
    if j != 1000 {
        damping = damping_tmp;
    }

    let mut last_sum_sq = sum_sq;
    sum_sq += N::RealField::from_i32(2).unwrap() * tol;
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
        let solved = lu.solve_mut(&mut b);
        if !solved {
            let lu = multiplied.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b);
            if !solved {
                let qr = multiplied.qr();
                let solved = qr.solve_mut(&mut b);
                if !solved {
                    return Err("curve_fit: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params = &params + &b;

        // Now solve for damping / damping_mult
        for i in 0..multiplied_div.row(0).len() {
            multiplied_div[(i, i)] *= N::one() + N::from_real(damping / damping_mult);
        }
        let lu = multiplied_div.clone().lu();
        let solved = lu.solve_mut(&mut b_div);
        if !solved {
            let lu = multiplied_div.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b_div);
            if !solved {
                let qr = multiplied_div.qr();
                let solved = qr.solve_mut(&mut b_div);
                if !solved {
                    return Err("curve_fit: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params_div = &params + &b_div;

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
pub fn curve_fit_jac<
    N: ComplexField,
    V: DimName,
    F: FnMut(N, &VectorN<N, V>) -> N,
    G: FnMut(N, &VectorN<N, V>) -> VectorN<N, V>,
>(
    mut f: F,
    xs: &[N],
    ys: &[N],
    initial: &[N],
    tol: N::RealField,
    mut jacobian: G,
    mut damping: N::RealField,
) -> Result<VectorN<N, V>, String>
where
    DefaultAllocator: Allocator<N, V>,
{
    if !tol.is_sign_positive() {
        return Err("curve_fit_jac: tol must be positive".to_owned());
    }

    if !damping.is_sign_positive() {
        return Err("curve_fit_jac: damping must be positive".to_owned());
    }

    if xs.len() != ys.len() {
        return Err("curve_fit_jac: xs length must match ys length".to_owned());
    }

    let mut params = VectorN::<N, V>::from_column_slice(initial);
    let ys = DVector::<N>::from_column_slice(ys);
    let mut jac: DMatrix<N> = DMatrix::identity(xs.len(), params.len());
    jac_analytic(&mut jacobian, xs, &mut params, &mut jac);
    let mut jac_transpose = jac.transpose();

    let damping_mult = N::RealField::from_f64(1.5).unwrap();

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
    let mut damping_tmp = damping / damping_mult;
    let mut j = 0;
    let mut evaluation: DVector<N> =
        DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
    while sum_sq > sum_sq_initial && j < 1000 {
        damping_tmp *= damping_mult;
        let diff = &ys - &evaluation;
        let mut b = &jac_transpose * &diff;
        // Always square
        let mut multiplied = &jac_transpose * &jac;
        for i in 0..multiplied.row(0).len() {
            multiplied[(i, i)] *= N::one() + N::from_real(damping_tmp);
        }
        let lu = multiplied.clone().lu();
        let solved = lu.solve_mut(&mut b);
        if !solved {
            let lu = multiplied.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b);
            if !solved {
                let qr = multiplied.qr();
                let solved = qr.solve_mut(&mut b);
                if !solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        params += &b;
        evaluation = DVector::from_iterator(xs.len(), xs.iter().map(|&x| f(x, &params)));
        let diff = &ys - &evaluation;
        sum_sq = diff
            .iter()
            .map(|&r| r.modulus_squared())
            .fold(N::RealField::zero(), |acc, r| acc + r);
        j += 1;
        jac_analytic(&mut jacobian, xs, &mut params, &mut jac);
        jac_transpose = jac.transpose();
    }
    if j != 1000 {
        damping = damping_tmp;
    }

    let mut last_sum_sq = sum_sq;
    sum_sq += N::RealField::from_i32(2).unwrap() * tol;
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
        let solved = lu.solve_mut(&mut b);
        if !solved {
            let lu = multiplied.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b);
            if !solved {
                let qr = multiplied.qr();
                let solved = qr.solve_mut(&mut b);
                if !solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params = &params + &b;

        // Now solve for damping / damping_mult
        for i in 0..multiplied_div.row(0).len() {
            multiplied_div[(i, i)] *= N::one() + N::from_real(damping / damping_mult);
        }
        let lu = multiplied_div.clone().lu();
        let solved = lu.solve_mut(&mut b_div);
        if !solved {
            let lu = multiplied_div.clone().full_piv_lu();
            let solved = lu.solve_mut(&mut b_div);
            if !solved {
                let qr = multiplied_div.qr();
                let solved = qr.solve_mut(&mut b_div);
                if !solved {
                    return Err("curve_fit_jac: unable to solve linear equation".to_owned());
                }
            }
        }
        let new_params_div = &params + &b_div;

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
