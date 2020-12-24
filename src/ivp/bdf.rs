/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{IVPSolver, IVPStatus};
use crate::roots::secant;
use alga::general::{ComplexField, RealField};
use nalgebra::DVector;
use num_traits::{FromPrimitive, Zero};
use std::collections::VecDeque;

/// This trait allows a struct to be used in the BDF
///
/// Types implementing BDFSolver should have a BDFInfo to
/// handle the actual IVP solving.
///
/// # Examples
/// See `struct BDF6` for an example of implementing this trait.
pub trait BDFSolver<N: ComplexField>: Sized {
    /// The polynomial interpolation coefficients for the higher-order
    /// method. Should start
    /// with the coefficient for the derivative
    /// function without h, then n - 1. The
    /// coefficients for the previous terms
    /// should have the sign as if they're on the
    /// same side of the = as the next state.
    fn higher_coefficients() -> Vec<N::RealField>;
    /// The polynomial interpolation coefficients for the lower-order
    /// method. Must be
    /// one less in length than higher_coefficients.
    /// Should start with the coefficient for the
    /// derivative function without h, then n-1. The
    /// coefficients for the previous terms
    /// should have the sign as if they're on the
    /// same side of the = as the next state.
    fn lower_coefficients() -> Vec<N::RealField>;

    /// Use BDFInfo to solve an initial value problem
    fn solve_ivp<T: Clone, F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        self,
        f: &F,
        params: &mut T,
    ) -> super::Path<N, N::RealField>;

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

/// Provides an IVPSolver implementation for BDFSolver, based
/// on the higher and lower order coefficients. It is up to the
/// BDFSolver to correctly implement the coefficients.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BDFInfo<N: ComplexField> {
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<DVector<N>>,
    dt_max: Option<N::RealField>,
    dt_min: Option<N::RealField>,
    tolerance: Option<N::RealField>,
    higher_coffecients: Vec<N::RealField>,
    lower_coefficients: Vec<N::RealField>,
    memory: VecDeque<(N::RealField, DVector<N>)>,
    nflag: bool,
    last: bool,
}

impl<N: ComplexField> BDFInfo<N> {
    pub fn new() -> Self {
        BDFInfo {
            dt: None,
            time: None,
            end: None,
            state: None,
            dt_max: None,
            dt_min: None,
            tolerance: None,
            higher_coffecients: vec![],
            lower_coefficients: vec![],
            memory: VecDeque::new(),
            nflag: false,
            last: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rk4<
    N: ComplexField,
    T: Clone,
    F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>,
>(
    time: N::RealField,
    dt: N::RealField,
    initial: &[N],
    states: &mut VecDeque<(N::RealField, DVector<N>)>,
    f: F,
    params: &mut T,
    num: usize,
) -> Result<(), String> {
    let mut state = DVector::from_column_slice(initial);
    let mut time = time;
    for i in 0..num {
        let k1 = f(time, state.column(0).as_slice(), &mut params.clone())? * N::from_real(dt);
        let intermediate = &state + &k1 * N::from_f64(0.5).unwrap();
        let k2 = f(
            time + N::RealField::from_f64(0.5).unwrap() * dt,
            intermediate.column(0).as_slice(),
            &mut params.clone(),
        )? * N::from_real(dt);
        let intermediate = &state + &k2 * N::from_f64(0.5).unwrap();
        let k3 = f(
            time + N::RealField::from_f64(0.5).unwrap() * dt,
            intermediate.column(0).as_slice(),
            &mut params.clone(),
        )? * N::from_real(dt);
        let intermediate = &state + &k3;
        let k4 = f(
            time + dt,
            intermediate.column(0).as_slice(),
            &mut params.clone(),
        )? * N::from_real(dt);
        if i != 0 {
            states.push_back((time, state.clone()));
        }
        state += (k1 + k2 * N::from_f64(2.0).unwrap() + k3 * N::from_f64(2.0).unwrap() + k4)
            * N::from_f64(1.0 / 6.0).unwrap();
        time += dt;
    }
    states.push_back((time, state));

    Ok(())
}

impl<N: ComplexField> Default for BDFInfo<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> IVPSolver<N> for BDFInfo<N> {
    fn step<T: Clone, F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        &mut self,
        f: &F,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String> {
        if self.time.unwrap() >= self.end.unwrap() {
            return Ok(IVPStatus::Done);
        }

        let mut output = vec![];

        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
            rk4(
                self.time.unwrap(),
                self.dt.unwrap(),
                self.state.as_ref().unwrap().column(0).as_slice(),
                &mut self.memory,
                f,
                params,
                1,
            )?;
            *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
            return Ok(IVPStatus::Ok(vec![(
                self.time.unwrap(),
                self.memory.back().unwrap().1.clone(),
            )]));
        }

        if self.memory.is_empty() {
            rk4(
                self.time.unwrap(),
                self.dt.unwrap(),
                self.state.as_ref().unwrap().column(0).as_slice(),
                &mut self.memory,
                f,
                params,
                self.higher_coffecients.len(),
            )?;
            self.time = Some(
                self.time.unwrap()
                    + N::RealField::from_usize(self.higher_coffecients.len()).unwrap()
                        * self.dt.unwrap(),
            );
            self.state = Some(self.memory.back().unwrap().1.clone());
        }

        let tenth_real = N::RealField::from_f64(0.1).unwrap();
        let half_real = N::RealField::from_f64(0.5).unwrap();
        let two_real = N::RealField::from_i32(2).unwrap();

        let higher_func = |y: &[N]| -> DVector<N> {
            let y = DVector::from_column_slice(y);
            let mut state = -f(
                self.time.unwrap() + self.dt.unwrap(),
                y.column(0).as_slice(),
                &mut params.clone(),
            )
            .unwrap()
                * N::from_real(self.dt.unwrap() * self.higher_coffecients[0]);
            for (ind, coeff) in self.higher_coffecients.iter().enumerate().skip(1) {
                state += &self.memory[self.memory.len() - ind].1 * N::from_real(*coeff);
            }
            state + y
        };
        let lower_func = |y: &[N]| -> DVector<N> {
            let y = DVector::from_column_slice(y);
            let mut state = -f(
                self.time.unwrap() + self.dt.unwrap(),
                y.column(0).as_slice(),
                &mut params.clone(),
            )
            .unwrap()
                * N::from_real(self.dt.unwrap() * self.lower_coefficients[0]);
            for (ind, coeff) in self.lower_coefficients.iter().enumerate().skip(1) {
                state += &self.memory[self.memory.len() - ind].1 * N::from_real(*coeff);
            }
            state + y
        };

        let higher = secant(
            (
                self.memory[self.memory.len() - 1].1.column(0).as_slice(),
                self.memory[self.memory.len() - 2].1.column(0).as_slice(),
            ),
            higher_func,
            self.tolerance.unwrap(),
            1000,
        )?;
        let lower = secant(
            (
                self.memory[self.memory.len() - 1].1.column(0).as_slice(),
                self.memory[self.memory.len() - 2].1.column(0).as_slice(),
            ),
            lower_func,
            self.tolerance.unwrap(),
            1000,
        )?;

        let diff = &higher - &lower;
        let error = diff.dot(&diff).sqrt().abs();

        if error <= self.tolerance.unwrap() {
            self.state = Some(higher.clone());
            self.time = Some(self.time.unwrap() + self.dt.unwrap());
            if self.nflag {
                for state in self.memory.iter() {
                    output.push((state.0, state.1.clone()));
                }
                self.nflag = false;
            }
            output.push((self.time.unwrap(), self.state.as_ref().unwrap().clone()));

            self.memory.push_back((self.time.unwrap(), higher));
            self.memory.pop_front();

            if self.last {
                return Ok(IVPStatus::Ok(output));
            }

            if error < tenth_real * self.tolerance.unwrap()
                || self.time.unwrap() > self.end.unwrap()
            {
                self.dt = Some(self.dt.unwrap() * two_real);

                if self.dt.unwrap() > self.dt_max.unwrap() {
                    self.dt = Some(self.dt_max.unwrap());
                }

                if self.time.unwrap()
                    + N::RealField::from_usize(self.higher_coffecients.len()).unwrap()
                        * self.dt.unwrap()
                    > self.end.unwrap()
                {
                    self.dt = Some(
                        (self.end.unwrap() - self.time.unwrap())
                            / N::RealField::from_usize(self.higher_coffecients.len()).unwrap(),
                    );
                    self.last = true;
                }

                self.memory.clear();
            }

            return Ok(IVPStatus::Ok(output));
        }

        self.dt = Some(self.dt.unwrap() * half_real);
        if self.dt.unwrap() < self.dt_min.unwrap() {
            return Err("BDFInfo step: minimum dt exceeded".to_owned());
        }

        self.memory.clear();
        Ok(IVPStatus::Redo)
    }

    fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        if !tol.is_sign_positive() {
            return Err("BDFInfo with_tolerance: tolerance must be postive".to_owned());
        }
        self.tolerance = Some(tol);
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        if !max.is_sign_positive() {
            return Err("BDFInfo with_dt_max: dt_max must be positive".to_owned());
        }
        if let Some(min) = self.dt_min {
            if max <= min {
                return Err("BDFInfo with_dt_max: dt_max must be greater than dt_min".to_owned());
            }
        }
        self.dt_max = Some(max);
        self.dt = Some(max);
        Ok(self)
    }

    fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        if !min.is_sign_positive() {
            return Err("BDFInfo with_dt_min: dt_min must be positive".to_owned());
        }
        if let Some(max) = self.dt_max {
            if min >= max {
                return Err("BDFInfo with_dt_min: dt_min must be less than dt_max".to_owned());
            }
        }
        self.dt_min = Some(min);
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        if let Some(end) = self.end {
            if end <= t_initial {
                return Err("BDFInfo with_start: Start must be before end".to_owned());
            }
        }
        self.time = Some(t_initial);
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        if let Some(start) = self.time {
            if t_final <= start {
                return Err("BDFInfo with_end: Start must be before end".to_owned());
            }
        }
        self.end = Some(t_final);
        Ok(self)
    }

    fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.state = Some(DVector::from_column_slice(start));
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }

    fn get_initial_conditions(&self) -> Option<DVector<N>> {
        if let Some(state) = &self.state {
            Some(state.clone())
        } else {
            None
        }
    }

    fn get_time(&self) -> Option<N::RealField> {
        if let Some(time) = &self.time {
            Some(*time)
        } else {
            None
        }
    }

    fn check_start(&self) -> Result<(), String> {
        if self.time == None {
            Err("BDFInfo check_start: No initial time".to_owned())
        } else if self.end == None {
            Err("BDFInfo check_start: No end time".to_owned())
        } else if self.tolerance == None {
            Err("BDFInfo check_start: No tolerance".to_owned())
        } else if self.state == None {
            Err("BDFInfo check_start: No initial conditions".to_owned())
        } else if self.dt_max == None {
            Err("BDFInfo check_start: No dt_max".to_owned())
        } else if self.dt_min == None {
            Err("BDFInfo check_start: No dt_min".to_owned())
        } else {
            Ok(())
        }
    }
}

/// 6th order backwards differentiation formula method for
/// solving an initial value problem.
///
/// Defines the higher and lower order coefficients. Uses
/// BDFInfo for the actual solving.
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon_sci::ivp::{IVPSolver, BDF6, BDFSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<DVector<f64>, String> {
///     Ok(-DVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), String> {
///     let bdf = BDF6::new()
///         .with_dt_max(0.1)?
///         .with_dt_min(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_start(0.0)?
///         .with_end(10.0)?
///         .with_initial_conditions(&[1.0])?
///         .build();
///     let path = bdf.solve_ivp(&derivatives, &mut ())?;
///     for (time, state) in &path {
///         assert!(((-time).exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BDF6<N: ComplexField> {
    info: BDFInfo<N>,
}

impl<N: ComplexField> BDF6<N> {
    pub fn new() -> Self {
        let mut info = BDFInfo::new();
        info.higher_coffecients = Self::higher_coefficients();
        info.lower_coefficients = Self::lower_coefficients();

        BDF6 { info }
    }
}

impl<N: ComplexField> Default for BDF6<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> BDFSolver<N> for BDF6<N> {
    fn higher_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(60.0 / 147.0).unwrap(),
            N::RealField::from_f64(-360.0 / 147.0).unwrap(),
            N::RealField::from_f64(450.0 / 147.0).unwrap(),
            N::RealField::from_f64(-400.0 / 147.0).unwrap(),
            N::RealField::from_f64(225.0 / 147.0).unwrap(),
            N::RealField::from_f64(-72.0 / 147.0).unwrap(),
            N::RealField::from_f64(10.0 / 147.0).unwrap(),
        ]
    }

    fn lower_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(60.0 / 137.0).unwrap(),
            N::RealField::from_f64(-300.0 / 137.0).unwrap(),
            N::RealField::from_f64(300.0 / 137.0).unwrap(),
            N::RealField::from_f64(-200.0 / 137.0).unwrap(),
            N::RealField::from_f64(75.0 / 137.0).unwrap(),
            N::RealField::from_f64(-12.0 / 137.0).unwrap(),
        ]
    }

    fn solve_ivp<T: Clone, F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        self,
        f: &F,
        params: &mut T,
    ) -> super::Path<N, N::RealField> {
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

    fn build(mut self) -> Self {
        self.info = self.info.build();
        self
    }
}

impl<N: ComplexField> From<BDF6<N>> for BDFInfo<N> {
    fn from(bdf: BDF6<N>) -> BDFInfo<N> {
        bdf.info
    }
}
