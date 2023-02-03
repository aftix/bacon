/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{IVPSolver, IVPStatus};
use nalgebra::{ComplexField, RealField, SMatrix, SVector};
use num_traits::{FromPrimitive, Zero};

/// This trait allows a struct to be used in the Runge-Kutta solver.
///
/// Things implementing RungeKuttaSolver should have an RKInfo to handle
/// the actual IVP solving. It should also provide the with_* helper functions
/// for convience.
///
/// # Examples
/// See `struct RK45` for an example of implementing this trait
pub trait RungeKuttaSolver<N, const S: usize, const O: usize>: Sized
where
    N: ComplexField,
{
    /// Returns a vec of coeffecients to multiply the time step by when getting
    /// intermediate results. Upper-left portion of Butch Tableaux
    fn t_coefficients() -> SVector<N::RealField, O>;

    /// Returns the coefficients to use on the k_i's when finding another
    /// k_i. Upper-right portion of the Butch Tableax. Should be
    /// an NxN-1 matrix, where N is the order of the Runge-Kutta Method (Or order+1 for
    /// adaptive methods)
    fn k_coefficients() -> SMatrix<N::RealField, O, O>;

    /// Coefficients to use when calculating the final step to take.
    /// These are the weights of the weighted average of k_i's. Bottom
    /// portion of the Butch Tableaux. For adaptive methods, this is the first
    /// row of the bottom portion.
    fn avg_coefficients() -> SVector<N::RealField, O>;

    /// Coefficients to use on
    /// the k_i's to find the error between the two orders
    /// of Runge-Kutta methods. In the Butch Tableaux, this is
    /// the first row of the bottom portion minus the second row.
    fn error_coefficients() -> SVector<N::RealField, O>;

    /// Ideally, call RKInfo.solve_ivp
    fn solve_ivp<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>>(
        self,
        f: F,
        params: &mut T,
    ) -> super::Path<N, N::RealField, S>;

    /// Set the error tolerance for this solver.
    fn with_tolerance(self, tol: N::RealField) -> Result<Self, String>;
    /// Set the maximum time step for this solver.
    fn with_dt_max(self, max: N::RealField) -> Result<Self, String>;
    /// Set the minimum time step for this solver.
    fn with_dt_min(self, min: N::RealField) -> Result<Self, String>;
    /// Set the initial time for this solver.
    fn with_start(self, t_initial: N::RealField) -> Result<Self, String>;
    /// Set the end time for this solver.
    fn with_end(self, t_final: N::RealField) -> Result<Self, String>;
    /// Set the initial conditions for this solver.
    fn with_initial_conditions(self, start: &[N]) -> Result<Self, String>;
    /// Build this solver.
    fn build(self) -> Self;
}

/// Provides an IVPSolver implementation for RungeKuttaSolver,
/// based entirely on the Butch Tableaux coefficients. It is up
/// to the RungeKuttaSolver to set up RKInfo. See RK45 for an example.
#[derive(Debug, Clone)]
pub struct RKInfo<N: ComplexField + FromPrimitive, const S: usize, const O: usize>
where
    N: ComplexField + FromPrimitive,
    <N as ComplexField>::RealField: FromPrimitive,
{
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<SVector<N, S>>,
    dt_max: Option<N::RealField>,
    dt_min: Option<N::RealField>,
    tolerance: Option<N::RealField>,
    a_coefficients: SVector<N, O>,
    k_coefficients: SMatrix<N, O, O>,
    avg_coefficients: SVector<N, O>,
    error_coefficients: SVector<N, O>,
}

impl<N: ComplexField + FromPrimitive, const S: usize, const O: usize> RKInfo<N, S, O>
where
    N: ComplexField + FromPrimitive,
    <N as ComplexField>::RealField: FromPrimitive,
{
    fn new() -> Self {
        RKInfo {
            dt: None,
            time: None,
            end: None,
            state: None,
            dt_max: None,
            dt_min: None,
            tolerance: None,
            a_coefficients: SVector::zero(),
            k_coefficients: SMatrix::zero(),
            avg_coefficients: SVector::zero(),
            error_coefficients: SVector::zero(),
        }
    }
}

impl<N, const S: usize, const O: usize> IVPSolver<N, S> for RKInfo<N, S, O>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    fn step<T, F>(&mut self, mut f: F, params: &mut T) -> Result<IVPStatus<N, S>, String>
    where
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    {
        if self.time.unwrap() >= self.end.unwrap() {
            return Ok(IVPStatus::Done);
        }

        let tenth_real = N::RealField::from_f64(0.1).unwrap();
        let quater_real = N::RealField::from_f64(0.25).unwrap();
        let eighty_fourths_real = N::RealField::from_f64(0.84).unwrap();
        let four_real = N::RealField::from_i32(4).unwrap();

        let mut set_dt = false;
        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            set_dt = true;
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
        }

        let num_k = self.k_coefficients.row(0).len();

        let mut half_steps: SMatrix<N, S, O> = SMatrix::zero();
        let num_s = half_steps.column(0).len();
        for i in 0..num_k {
            let state = SVector::<N, S>::from_iterator(
                self.state
                    .as_ref()
                    .unwrap()
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(|(ind, y)| {
                        let mut acc = *y;
                        for j in 0..i {
                            acc += half_steps[(ind, j)] * self.k_coefficients[(i, j)];
                        }
                        acc
                    }),
            );
            let step = f(
                self.time.unwrap() + self.a_coefficients.column(0)[i].real() * self.dt.unwrap(),
                state.as_slice(),
                &mut params.clone(),
            )? * N::from_real(self.dt.unwrap());
            for j in 0..num_s {
                half_steps[(j, i)] = step[j];
            }
        }

        let error_vec = SVector::<N, S>::from_iterator(
            half_steps.column(0).iter().enumerate().map(|(ind, y)| {
                let mut acc = *y * self.error_coefficients[0];
                for j in 1..num_k {
                    acc += half_steps[(ind, j)] * self.error_coefficients[j];
                }
                acc
            }),
        );
        let error = error_vec.dot(&error_vec).real() / self.dt.unwrap();

        let mut output = false;
        if error <= self.tolerance.unwrap() {
            output = true;
            *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
            /*let mut state = &half_steps.column(0) * self.avg_coefficients[0];
            for ind in 1..half_steps.row(0).len() {
                state += &half_steps.column(ind) * self.avg_coefficients[ind];
            }*/
            let state = SVector::<N, S>::from_iterator(
                self.state
                    .as_ref()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(ind, y)| {
                        let mut acc = *y;
                        for j in 0..num_k {
                            acc += half_steps[(ind, j)] * self.avg_coefficients[j];
                        }
                        acc
                    }),
            );
            //state += self.state.as_ref().unwrap();
            self.state = Some(state);
        }

        let delta = eighty_fourths_real * (self.tolerance.unwrap() / error).powf(quater_real);
        if delta <= tenth_real {
            *self.dt.get_or_insert(N::RealField::zero()) *= tenth_real;
        } else if delta >= four_real {
            *self.dt.get_or_insert(N::RealField::zero()) *= four_real;
        } else {
            *self.dt.get_or_insert(N::RealField::zero()) *= delta;
        }

        if self.dt.unwrap() > self.dt_max.unwrap() {
            self.dt = Some(self.dt_max.unwrap());
        }

        if !set_dt && self.dt.unwrap() < self.dt_min.unwrap() {
            return Err("RKInfo step: minimum dt exceeded".to_owned());
        }

        if output {
            Ok(IVPStatus::Ok(vec![(
                self.time.unwrap(),
                *self.state.as_ref().unwrap(),
            )]))
        } else {
            Ok(IVPStatus::Redo)
        }
    }

    fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        if !tol.is_sign_positive() {
            return Err("RKInfo with_tolerance: tolerance must be postive".to_owned());
        }
        self.tolerance = Some(tol);
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        if !max.is_sign_positive() {
            return Err("RKInfo with_dt_max: dt_max must be positive".to_owned());
        }
        if let Some(min) = self.dt_min {
            if max <= min {
                return Err("RKInfo with_dt_max: dt_max must be greater than dt_min".to_owned());
            }
        }
        self.dt_max = Some(max);
        self.dt = Some(max);
        Ok(self)
    }

    fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        if !min.is_sign_positive() {
            return Err("RKInfo with_dt_min: dt_min must be positive".to_owned());
        }
        if let Some(max) = self.dt_max {
            if min >= max {
                return Err("RKInfo with_dt_min: dt_min must be less than dt_max".to_owned());
            }
        }
        self.dt_min = Some(min);
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        if let Some(end) = self.end {
            if end <= t_initial {
                return Err("RKInfo with_start: Start must be before end".to_owned());
            }
        }
        self.time = Some(t_initial);
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        if let Some(start) = self.time {
            if t_final <= start {
                return Err("RKInfo with_end: Start must be before end".to_owned());
            }
        }
        self.end = Some(t_final);
        Ok(self)
    }

    fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.state = Some(SVector::<N, S>::from_column_slice(start));
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }

    fn get_initial_conditions(&self) -> Option<SVector<N, S>> {
        self.state.as_ref().copied()
    }

    fn get_time(&self) -> Option<N::RealField> {
        self.time.as_ref().copied()
    }

    fn check_start(&self) -> Result<(), String> {
        if self.time == None {
            Err("RKInfo check_start: No initial time".to_owned())
        } else if self.end == None {
            Err("RKInfo check_start: No end time".to_owned())
        } else if self.tolerance == None {
            Err("RKInfo check_start: No tolerance".to_owned())
        } else if self.state == None {
            Err("RKInfo check_start: No initial conditions".to_owned())
        } else if self.dt_max == None {
            Err("RKInfo check_start: No dt_max".to_owned())
        } else if self.dt_min == None {
            Err("RKInfo check_start: No dt_min".to_owned())
        } else {
            Ok(())
        }
    }
}

/// Runge-Kutta-Fehlberg method for solving an IVP.
///
/// Defines the Butch Tableaux for a 5(4) order adaptive
/// runge-kutta method. Uses RKInfo to do the actual solving.
/// Provides an interface for setting the conditions on RKInfo.
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use bacon_sci::ivp::{RK45, RungeKuttaSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<SVector<f64, 1>, String> {
///     Ok(SVector::<f64, 1>::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), String> {
///     let rk45 = RK45::new()
///         .with_dt_max(0.1)?
///         .with_dt_min(0.001)?
///         .with_start(0.0)?
///         .with_end(10.0)?
///         .with_tolerance(0.0001)?
///         .with_initial_conditions(&[1.0])?
///         .build();
///     let path = rk45.solve_ivp(derivatives, &mut ())?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RK45<N, const S: usize>
where
    N: ComplexField + FromPrimitive,
    <N as ComplexField>::RealField: FromPrimitive,
{
    info: RKInfo<N, S, 6>,
}

// Derivatives for tests
#[cfg(test)]
fn exp_deriv(_: f32, y: &[f32], _: &mut ()) -> Result<SVector<f32, 1>, String> {
    Ok(SVector::from_column_slice(y))
}

// Tests for RK45
#[test]
fn rk45_exp() -> Result<(), String> {
    let t_initial = 0.0;
    let t_final = 2.0;

    let solver = RK45::new()
        .with_dt_min(0.0001)?
        .with_dt_max(0.001)?
        .with_start(t_initial)?
        .with_end(t_final)?
        .with_tolerance(0.000001)?
        .with_initial_conditions(&[1.0])?;

    let path = solver.solve_ivp(&exp_deriv, &mut ());

    match path {
        Ok(path) => {
            for step in &path {
                println!("{} {}", step.1.column(0)[0], step.0.exp());
                assert!(approx_eq!(
                    f32,
                    step.1.column(0)[0],
                    step.0.exp(),
                    epsilon = 0.01
                ));
            }
        }
        Err(s) => panic!("Result not Ok: {}", s),
    }

    Ok(())
}

impl<N, const S: usize> RK45<N, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut info: RKInfo<N, S, 6> = RKInfo::new();
        info.a_coefficients = SVector::<N, 6>::from_iterator(
            Self::t_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.k_coefficients = SMatrix::<N, 6, 6>::from_iterator(
            Self::k_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.avg_coefficients = SVector::<N, 6>::from_iterator(
            Self::avg_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.error_coefficients = SVector::<N, 6>::from_iterator(
            Self::error_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        RK45 { info }
    }
}

impl<N, const S: usize> RungeKuttaSolver<N, S, 6> for RK45<N, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    fn t_coefficients() -> SVector<N::RealField, 6> {
        SVector::<N::RealField, 6>::from_column_slice(&[
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(0.25).unwrap(),
            N::RealField::from_f64(3.0 / 8.0).unwrap(),
            N::RealField::from_f64(12.0 / 13.0).unwrap(),
            N::RealField::from_f64(1.0).unwrap(),
            N::RealField::from_f64(0.5).unwrap(),
        ])
    }

    fn k_coefficients() -> SMatrix<N::RealField, 6, 6> {
        SMatrix::<N::RealField, 6, 6>::from_vec(vec![
            // Row 0
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 1
            N::RealField::from_f64(0.25).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 2
            N::RealField::from_f64(3.0 / 32.0).unwrap(),
            N::RealField::from_f64(9.0 / 32.0).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 3
            N::RealField::from_f64(1932.0 / 2197.0).unwrap(),
            N::RealField::from_f64(-7200.0 / 2197.0).unwrap(),
            N::RealField::from_f64(7296.0 / 2197.0).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 4
            N::RealField::from_f64(439.0 / 216.0).unwrap(),
            N::RealField::from_f64(-8.0).unwrap(),
            N::RealField::from_f64(3680.0 / 513.0).unwrap(),
            N::RealField::from_f64(-845.0 / 4104.0).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 5
            N::RealField::from_f64(-8.0 / 27.0).unwrap(),
            N::RealField::from_f64(2.0).unwrap(),
            N::RealField::from_f64(-3544.0 / 2565.0).unwrap(),
            N::RealField::from_f64(1859.0 / 4104.0).unwrap(),
            N::RealField::from_f64(-11.0 / 40.0).unwrap(),
            N::RealField::zero(),
        ])
    }

    fn avg_coefficients() -> SVector<N::RealField, 6> {
        SVector::<N::RealField, 6>::from_column_slice(&[
            N::RealField::from_f64(25.0 / 216.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(1408.0 / 2565.0).unwrap(),
            N::RealField::from_f64(2197.0 / 4104.0).unwrap(),
            N::RealField::from_f64(-1.0 / 5.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
        ])
    }

    fn error_coefficients() -> SVector<N::RealField, 6> {
        SVector::<N::RealField, 6>::from_column_slice(&[
            N::RealField::from_f64(1.0 / 360.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(-128.0 / 4275.0).unwrap(),
            N::RealField::from_f64(-2197.0 / 75240.0).unwrap(),
            N::RealField::from_f64(1.0 / 50.0).unwrap(),
            N::RealField::from_f64(2.0 / 55.0).unwrap(),
        ])
    }

    fn solve_ivp<T, F>(self, f: F, params: &mut T) -> super::Path<N, N::RealField, S>
    where
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    {
        self.info.solve_ivp(f, params)
    }

    fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_tolerance(tol)?;
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_max(max)?;
        Ok(self)
    }

    fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_min(min)?;
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_start(t_initial)?;
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_end(t_final)?;
        Ok(self)
    }

    fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.info = self.info.with_initial_conditions(start)?;
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }
}

impl<N, const S: usize> From<RK45<N, S>> for RKInfo<N, S, 6>
where
    N: ComplexField + FromPrimitive,
    <N as ComplexField>::RealField: FromPrimitive,
{
    fn from(rk: RK45<N, S>) -> RKInfo<N, S, 6> {
        rk.info
    }
}

/// Bogacki-Shampine method for solving an IVP.
///
/// Defines the Butch Tableaux for a 5(4) order adaptive
/// runge-kutta method. Uses RKInfo to do the actual solving.
/// Provides an interface for setting the conditions on RKInfo.
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use bacon_sci::ivp::{RK23, RungeKuttaSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<SVector<f64, 1>, String> {
///     Ok(SVector::<f64, 1>::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), String> {
///     let rk45 = RK23::new()
///         .with_dt_max(0.1)?
///         .with_dt_min(0.001)?
///         .with_start(0.0)?
///         .with_end(10.0)?
///         .with_tolerance(0.0001)?
///         .with_initial_conditions(&[1.0])?
///         .build();
///     let path = rk45.solve_ivp(derivatives, &mut ())?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RK23<N, const S: usize>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    info: RKInfo<N, S, 4>,
}

impl<N, const S: usize> RK23<N, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    pub fn new() -> Self {
        let mut info: RKInfo<N, S, 4> = RKInfo::new();
        info.a_coefficients = SVector::<N, 4>::from_iterator(
            Self::t_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.k_coefficients = SMatrix::<N, 4, 4>::from_iterator(
            Self::k_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.avg_coefficients = SVector::<N, 4>::from_iterator(
            Self::avg_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        info.error_coefficients = SVector::<N, 4>::from_iterator(
            Self::error_coefficients()
                .as_slice()
                .iter()
                .map(|x| N::from_real(*x)),
        );
        RK23 { info }
    }
}

impl<N, const S: usize> Default for RK23<N, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N, const S: usize> RungeKuttaSolver<N, S, 4> for RK23<N, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    fn t_coefficients() -> SVector<N::RealField, 4> {
        SVector::<N::RealField, 4>::from_column_slice(&[
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(0.5).unwrap(),
            N::RealField::from_f64(0.75).unwrap(),
            N::RealField::from_f64(1.0).unwrap(),
        ])
    }

    fn k_coefficients() -> SMatrix<N::RealField, 4, 4> {
        SMatrix::<N::RealField, 4, 4>::from_vec(vec![
            // Row 0
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 1
            N::RealField::from_f64(0.5).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 2
            N::RealField::zero(),
            N::RealField::from_f64(0.75).unwrap(),
            N::RealField::zero(),
            N::RealField::zero(),
            // Row 3
            N::RealField::from_f64(2.0 / 9.0).unwrap(),
            N::RealField::from_f64(1.0 / 3.0).unwrap(),
            N::RealField::from_f64(4.0 / 9.0).unwrap(),
            N::RealField::zero(),
        ])
    }

    fn avg_coefficients() -> SVector<N::RealField, 4> {
        SVector::<N::RealField, 4>::from_column_slice(&[
            N::RealField::from_f64(2.0 / 9.0).unwrap(),
            N::RealField::from_f64(1.0 / 3.0).unwrap(),
            N::RealField::from_f64(4.0 / 9.0).unwrap(),
            N::RealField::zero(),
        ])
    }

    fn error_coefficients() -> SVector<N::RealField, 4> {
        SVector::<N::RealField, 4>::from_column_slice(&[
            N::RealField::from_f64(2.0 / 9.0 - 7.0 / 24.0).unwrap(),
            N::RealField::from_f64(1.0 / 3.0 - 0.25).unwrap(),
            N::RealField::from_f64(4.0 / 9.0 - 1.0 / 3.0).unwrap(),
            N::RealField::from_f64(-1.0 / 8.0).unwrap(),
        ])
    }

    fn solve_ivp<
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    >(
        self,
        f: F,
        params: &mut T,
    ) -> super::Path<N, N::RealField, S> {
        self.info.solve_ivp(f, params)
    }

    fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_tolerance(tol)?;
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_max(max)?;
        Ok(self)
    }

    fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_min(min)?;
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_start(t_initial)?;
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_end(t_final)?;
        Ok(self)
    }

    fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.info = self.info.with_initial_conditions(start)?;
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }
}

impl<N, const S: usize> From<RK23<N, S>> for RKInfo<N, S, 4>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
{
    fn from(rk: RK23<N, S>) -> RKInfo<N, S, 4> {
        rk.info
    }
}
