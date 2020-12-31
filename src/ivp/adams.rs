/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{IVPSolver, IVPStatus};
use alga::general::{ComplexField, RealField};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, VectorN, U5};
use num_traits::{FromPrimitive, Zero};
use std::collections::VecDeque;

/// This trait allows a struct to be used in the Adams Predictor-Corrector solver
///
/// Things implementing AdamsSolver should have an AdamsInfo to handle the actual
/// IVP solving.
///
/// # Examples
/// See `struct Adams` for an example of implementing this trait
pub trait AdamsSolver<N: ComplexField, S: DimName, O: DimName>: Sized
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N::RealField, O>,
{
    /// The polynomial interpolation coefficients for the predictor. Should start
    /// with the coefficient for n - 1
    fn predictor_coefficients() -> VectorN<N::RealField, O>;
    /// The polynomial interpolation coefficients for the corrector. Must be
    /// the same length as predictor_coefficients. Should start with the
    /// implicit coefficient.
    fn corrector_coefficients() -> VectorN<N::RealField, O>;
    /// Coefficient for multiplying error by.
    fn error_coefficient() -> N::RealField;

    /// Use AdamsInfo to solve an initial value problem
    fn solve_ivp<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<VectorN<N, S>, String>>(
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

/// Provides an IVPSolver implementation for AdamsSolver, based on
/// the predictor and correctorr coefficients. It is up to the AdamsSolver
/// to set up AdamsInfo with the correct coefficients.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AdamsInfo<N: ComplexField, S: DimName, O: DimName>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, O>,
{
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<VectorN<N, S>>,
    dt_max: Option<N::RealField>,
    dt_min: Option<N::RealField>,
    tolerance: Option<N::RealField>,
    predictor_coefficients: VectorN<N, O>,
    corrector_coefficients: VectorN<N, O>,
    error_coefficient: N::RealField,
    memory: VecDeque<VectorN<N, S>>,
    states: VecDeque<(N::RealField, VectorN<N, S>)>,
    nflag: bool,
    last: bool,
}

impl<N: ComplexField, S: DimName, O: DimName> AdamsInfo<N, S, O>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, O>,
{
    pub fn new() -> Self {
        AdamsInfo {
            dt: None,
            time: None,
            end: None,
            state: None,
            dt_max: None,
            dt_min: None,
            tolerance: None,
            predictor_coefficients: VectorN::<N, O>::zero(),
            corrector_coefficients: VectorN::<N, O>::zero(),
            error_coefficient: N::RealField::zero(),
            memory: VecDeque::new(),
            states: VecDeque::new(),
            nflag: false,
            last: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rk4<
    N: ComplexField,
    S: DimName,
    T: Clone,
    F: FnMut(N::RealField, &[N], &mut T) -> Result<VectorN<N, S>, String>,
>(
    time: N::RealField,
    dt: N::RealField,
    initial: &[N],
    states: &mut VecDeque<(N::RealField, VectorN<N, S>)>,
    derivs: &mut VecDeque<VectorN<N, S>>,
    mut f: F,
    params: &mut T,
    num: usize,
) -> Result<(), String>
where
    DefaultAllocator: Allocator<N, S>,
{
    let mut state = VectorN::from_column_slice(initial);
    let mut time = time;
    for i in 0..num {
        let k1 = f(time, state.as_slice(), &mut params.clone())? * N::from_real(dt);
        let intermediate = &state + &k1 * N::from_f64(0.5).unwrap();
        let k2 = f(
            time + N::RealField::from_f64(0.5).unwrap() * dt,
            intermediate.as_slice(),
            &mut params.clone(),
        )? * N::from_real(dt);
        let intermediate = &state + &k2 * N::from_f64(0.5).unwrap();
        let k3 = f(
            time + N::RealField::from_f64(0.5).unwrap() * dt,
            intermediate.as_slice(),
            &mut params.clone(),
        )? * N::from_real(dt);
        let intermediate = &state + &k3;
        let k4 = f(time + dt, intermediate.as_slice(), &mut params.clone())? * N::from_real(dt);
        if i != 0 {
            derivs.push_back(f(time, state.as_slice(), params)?);
            states.push_back((time, state.clone()));
        }
        state += (k1 + k2 * N::from_f64(2.0).unwrap() + k3 * N::from_f64(2.0).unwrap() + k4)
            * N::from_f64(1.0 / 6.0).unwrap();
        time += dt;
    }
    derivs.push_back(f(time, state.as_slice(), params)?);
    states.push_back((time, state));

    Ok(())
}

impl<N: ComplexField, S: DimName, O: DimName> Default for AdamsInfo<N, S, O>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, O>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField, S: DimName, O: DimName> IVPSolver<N, S> for AdamsInfo<N, S, O>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, O>,
{
    fn step<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<VectorN<N, S>, String>>(
        &mut self,
        mut f: F,
        params: &mut T,
    ) -> Result<IVPStatus<N, S>, String> {
        if self.time.unwrap() >= self.end.unwrap() {
            return Ok(IVPStatus::Done);
        }

        let mut output = vec![];

        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
            rk4(
                self.time.unwrap(),
                self.dt.unwrap(),
                self.state.as_ref().unwrap().as_slice(),
                &mut self.states,
                &mut self.memory,
                &mut f,
                params,
                1,
            )?;
            *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
            return Ok(IVPStatus::Ok(vec![(
                self.time.unwrap(),
                self.states.back().unwrap().1.clone(),
            )]));
        }

        if self.memory.is_empty() {
            rk4(
                self.time.unwrap(),
                self.dt.unwrap(),
                self.state.as_ref().unwrap().as_slice(),
                &mut self.states,
                &mut self.memory,
                &mut f,
                params,
                self.predictor_coefficients.len(),
            )?;
            self.time = Some(
                self.time.unwrap()
                    + N::RealField::from_usize(self.predictor_coefficients.len()).unwrap()
                        * self.dt.unwrap(),
            );
            self.state = Some(self.states.back().unwrap().1.clone());
        }

        let tenth_real = N::RealField::from_f64(0.1).unwrap();
        let two_real = N::RealField::from_i32(2).unwrap();
        let four_real = N::RealField::from_i32(4).unwrap();

        let wp = &self.state.as_ref().unwrap();
        let wp = VectorN::<N, S>::from_iterator(wp.column(0).iter().enumerate().map(|(ind, y)| {
            let mut acc = *y;
            let dt = N::from_real(self.dt.unwrap());
            for (j, coef) in self.predictor_coefficients.iter().enumerate() {
                acc += self.memory[self.memory.len() - j - 1].column(0)[ind] * *coef * dt;
            }
            acc
        }));

        let implicit = f(
            self.time.unwrap() + self.dt.unwrap(),
            self.state.as_ref().unwrap().as_slice(),
            params,
        )?;
        let wc = &self.state.as_ref().unwrap();
        let wc = VectorN::<N, S>::from_iterator(wc.column(0).iter().enumerate().map(|(ind, y)| {
            let dt = N::from_real(self.dt.unwrap());
            let mut acc = implicit.column(0)[ind] * self.corrector_coefficients[0] * dt;
            for (j, coef) in self.corrector_coefficients.iter().enumerate().skip(1) {
                acc += self.memory[self.memory.len() - j - 1].column(0)[ind] * *coef * dt;
            }
            *y + acc
        }));

        let diff = &wc - &wp;
        let error = self.error_coefficient / self.dt.unwrap() * diff.dot(&diff).sqrt().abs();

        if error <= self.tolerance.unwrap() {
            self.state = Some(wc);
            self.time = Some(self.time.unwrap() + self.dt.unwrap());
            if self.nflag {
                for state in self.states.iter() {
                    output.push((state.0, state.1.clone()));
                }
                self.nflag = false;
            }
            output.push((self.time.unwrap(), self.state.as_ref().unwrap().clone()));

            self.memory.push_back(implicit);
            self.states
                .push_back((self.time.unwrap(), self.state.as_ref().unwrap().clone()));
            self.memory.pop_front();
            self.states.pop_front();

            if self.last {
                return Ok(IVPStatus::Ok(output));
            }

            if error < tenth_real * self.tolerance.unwrap()
                || self.time.unwrap() > self.end.unwrap()
            {
                let q = (self.tolerance.unwrap() / (two_real * error)).powf(
                    N::RealField::from_f64(1.0 / self.predictor_coefficients.len() as f64).unwrap(),
                );
                if q > four_real {
                    self.dt = Some(self.dt.unwrap() * four_real);
                } else {
                    self.dt = Some(self.dt.unwrap() * q);
                }

                if self.dt.unwrap() > self.dt_max.unwrap() {
                    self.dt = Some(self.dt_max.unwrap());
                }

                if self.time.unwrap()
                    + N::RealField::from_usize(self.predictor_coefficients.len()).unwrap()
                        * self.dt.unwrap()
                    > self.end.unwrap()
                {
                    self.dt = Some(
                        (self.end.unwrap() - self.time.unwrap())
                            / N::RealField::from_usize(self.predictor_coefficients.len()).unwrap(),
                    );
                    self.last = true;
                }

                self.memory.clear();
                self.states.clear();
            }

            return Ok(IVPStatus::Ok(output));
        }

        let q = (self.tolerance.unwrap() / (N::RealField::from_f64(2.0).unwrap() * error)).powf(
            N::RealField::from_f64(1.0 / (self.predictor_coefficients.len() as f64)).unwrap(),
        );

        if q < tenth_real {
            self.dt = Some(self.dt.unwrap() * tenth_real);
        } else {
            self.dt = Some(self.dt.unwrap() * q);
        }

        if self.dt.unwrap() < self.dt_min.unwrap() {
            return Err("AdamsInfo step: minimum dt exceeded".to_owned());
        }

        self.memory.clear();
        self.states.clear();
        Ok(IVPStatus::Redo)
    }

    fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        if !tol.is_sign_positive() {
            return Err("AdamsInfo with_tolerance: tolerance must be postive".to_owned());
        }
        self.tolerance = Some(tol);
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        if !max.is_sign_positive() {
            return Err("AdamsInfo with_dt_max: dt_max must be positive".to_owned());
        }
        if let Some(min) = self.dt_min {
            if max <= min {
                return Err("AdamsInfo with_dt_max: dt_max must be greater than dt_min".to_owned());
            }
        }
        self.dt_max = Some(max);
        self.dt = Some(max);
        Ok(self)
    }

    fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        if !min.is_sign_positive() {
            return Err("AdamsInfo with_dt_min: dt_min must be positive".to_owned());
        }
        if let Some(max) = self.dt_max {
            if min >= max {
                return Err("AdamsInfo with_dt_min: dt_min must be less than dt_max".to_owned());
            }
        }
        self.dt_min = Some(min);
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        if let Some(end) = self.end {
            if end <= t_initial {
                return Err("AdamsInfo with_start: Start must be before end".to_owned());
            }
        }
        self.time = Some(t_initial);
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        if let Some(start) = self.time {
            if t_final <= start {
                return Err("AdamsInfo with_end: Start must be before end".to_owned());
            }
        }
        self.end = Some(t_final);
        Ok(self)
    }

    fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.state = Some(VectorN::<N, S>::from_column_slice(start));
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }

    fn get_initial_conditions(&self) -> Option<VectorN<N, S>> {
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
            Err("AdamsInfo check_start: No initial time".to_owned())
        } else if self.end == None {
            Err("AdamsInfo check_start: No end time".to_owned())
        } else if self.tolerance == None {
            Err("AdamsInfo check_start: No tolerance".to_owned())
        } else if self.state == None {
            Err("AdamsInfo check_start: No initial conditions".to_owned())
        } else if self.dt_max == None {
            Err("AdamsInfo check_start: No dt_max".to_owned())
        } else if self.dt_min == None {
            Err("AdamsInfo check_start: No dt_min".to_owned())
        } else {
            Ok(())
        }
    }
}

/// 5th order Adams predictor-corrector method for solving an IVP.
///
/// Defines the predictor and corrector coefficients, as well as
/// the error coefficient. Uses AdamsInfo for the actual solving.
///
/// # Examples
/// ```
/// use nalgebra::{VectorN, U1};
/// use bacon_sci::ivp::{Adams, AdamsSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<VectorN<f64, U1>, String> {
///     Ok(VectorN::<f64, U1>::from_column_slice(state))
/// }
///
///
/// fn example() -> Result<(), String> {
///     let adams = Adams::new()
///         .with_dt_max(0.1)?
///         .with_dt_min(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_start(0.0)?
///         .with_end(1.0)?
///         .with_initial_conditions(&[1.0])?
///         .build();
///     let path = adams.solve_ivp(derivatives, &mut ())?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Adams<N: ComplexField, S: DimName>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N::RealField, U5>,
    DefaultAllocator: Allocator<N, U5>,
{
    info: AdamsInfo<N, S, U5>,
}

impl<N: ComplexField, S: DimName> Adams<N, S>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, U5>,
    DefaultAllocator: Allocator<N::RealField, U5>,
{
    pub fn new() -> Self {
        let mut info = AdamsInfo::new();
        info.corrector_coefficients = VectorN::<N, U5>::from_iterator(
            Self::corrector_coefficients()
                .iter()
                .map(|&x| N::from_real(x)),
        );
        info.predictor_coefficients = VectorN::<N, U5>::from_iterator(
            Self::predictor_coefficients()
                .iter()
                .map(|&x| N::from_real(x)),
        );
        info.error_coefficient = Self::error_coefficient();

        Adams { info }
    }
}

impl<N: ComplexField, S: DimName> Default for Adams<N, S>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N::RealField, U5>,
    DefaultAllocator: Allocator<N, U5>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField, S: DimName> AdamsSolver<N, S, U5> for Adams<N, S>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N::RealField, U5>,
    DefaultAllocator: Allocator<N, U5>,
{
    fn predictor_coefficients() -> VectorN<N::RealField, U5> {
        VectorN::<N::RealField, U5>::from_column_slice(&[
            N::RealField::from_f64(1901.0 / 720.0).unwrap(),
            N::RealField::from_f64(-2774.0 / 720.0).unwrap(),
            N::RealField::from_f64(2616.0 / 720.0).unwrap(),
            N::RealField::from_f64(-1274.0 / 720.0).unwrap(),
            N::RealField::from_f64(251.0 / 720.0).unwrap(),
        ])
    }

    fn corrector_coefficients() -> VectorN<N::RealField, U5> {
        VectorN::<N::RealField, U5>::from_column_slice(&[
            N::RealField::from_f64(251.0 / 720.0).unwrap(),
            N::RealField::from_f64(646.0 / 720.0).unwrap(),
            N::RealField::from_f64(-264.0 / 720.0).unwrap(),
            N::RealField::from_f64(106.0 / 720.0).unwrap(),
            N::RealField::from_f64(-19.0 / 720.0).unwrap(),
        ])
    }

    fn error_coefficient() -> N::RealField {
        N::RealField::from_f64(19.0 / 270.0).unwrap()
    }

    fn solve_ivp<
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<VectorN<N, S>, String>,
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

    fn build(mut self) -> Self {
        self.info = self.info.build();
        self
    }
}

impl<N: ComplexField, S: DimName> From<Adams<N, S>> for AdamsInfo<N, S, U5>
where
    DefaultAllocator: Allocator<N, S>,
    DefaultAllocator: Allocator<N, U5>,
    DefaultAllocator: Allocator<N::RealField, U5>,
{
    fn from(adams: Adams<N, S>) -> AdamsInfo<N, S, U5> {
        adams.info
    }
}
