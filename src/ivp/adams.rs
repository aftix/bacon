/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{IVPSolver, IVPStatus};
use alga::general::{ComplexField, RealField};
use nalgebra::DVector;
use num_traits::{FromPrimitive, Zero};
use std::collections::VecDeque;

/// This trait allows a struct to be used in the Adams Predictor-Corrector solver
///
/// Things implementing AdamsSolver should have an AdamsInfo to handle the actual
/// IVP solving.
///
/// # Examples
/// See `struct Adams` for an example of implementing this trait
pub trait AdamsSolver<N: ComplexField>: Sized {
    /// The polynomial interpolation coefficients for the predictor. Should start
    /// with the coefficient for n - 1
    fn predictor_coefficients() -> Vec<N::RealField>;
    /// The polynomial interpolation coefficients for the corrector. Must be
    /// the same length as predictor_coefficients. Should start with the
    /// implicit coefficient.
    fn corrector_coefficients() -> Vec<N::RealField>;
    /// Coefficient for multiplying error by.
    fn error_coefficient() -> N::RealField;

    /// Use AdamsInfo to solve an initial value problem
    fn solve_ivp<T: Clone, F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        self,
        f: F,
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

/// Provides an IVPSolver implementation for AdamsSolver, based on
/// the predictor and correctorr coefficients. It is up to the AdamsSolver
/// to set up AdamsInfo with the correct coefficients.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AdamsInfo<N: ComplexField> {
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<DVector<N>>,
    dt_max: Option<N::RealField>,
    dt_min: Option<N::RealField>,
    tolerance: Option<N::RealField>,
    predictor_coefficients: Vec<N::RealField>,
    corrector_coefficients: Vec<N::RealField>,
    error_coefficient: N::RealField,
    memory: VecDeque<DVector<N>>,
    states: VecDeque<(N::RealField, DVector<N>)>,
    nflag: bool,
    last: bool,
}

impl<N: ComplexField> AdamsInfo<N> {
    pub fn new() -> Self {
        AdamsInfo {
            dt: None,
            time: None,
            end: None,
            state: None,
            dt_max: None,
            dt_min: None,
            tolerance: None,
            predictor_coefficients: vec![],
            corrector_coefficients: vec![],
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
    T: Clone,
    F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>,
>(
    time: N::RealField,
    dt: N::RealField,
    initial: &[N],
    states: &mut VecDeque<(N::RealField, DVector<N>)>,
    derivs: &mut VecDeque<DVector<N>>,
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
            derivs.push_back(f(time, state.column(0).as_slice(), params)?);
            states.push_back((time, state.clone()));
        }
        state += (k1 + k2 * N::from_f64(2.0).unwrap() + k3 * N::from_f64(2.0).unwrap() + k4)
            * N::from_f64(1.0 / 6.0).unwrap();
        time += dt;
    }
    derivs.push_back(f(time, state.column(0).as_slice(), params)?);
    states.push_back((time, state));

    Ok(())
}

impl<N: ComplexField> Default for AdamsInfo<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> IVPSolver<N> for AdamsInfo<N> {
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
                &mut self.states,
                &mut self.memory,
                f,
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
                self.state.as_ref().unwrap().column(0).as_slice(),
                &mut self.states,
                &mut self.memory,
                f,
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

        let mut wp = self.state.as_ref().unwrap().clone();
        for (ind, coef) in self.predictor_coefficients.iter().rev().enumerate() {
            wp += &self.memory[ind] * N::from_real(*coef) * N::from_real(self.dt.unwrap());
        }
        let implicit = f(
            self.time.unwrap() + self.dt.unwrap(),
            self.state.as_ref().unwrap().column(0).as_slice(),
            params,
        )?;
        let mut wc = self.state.as_ref().unwrap().clone();
        wc += &implicit
            * N::from_real(self.dt.unwrap())
            * N::from_real(self.corrector_coefficients[0]);
        for (ind, coef) in self.corrector_coefficients.iter().enumerate().skip(1) {
            wc += &self.memory[self.memory.len() - ind - 1]
                * N::from_real(*coef)
                * N::from_real(self.dt.unwrap());
        }

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
/// use nalgebra::DVector;
/// use bacon_sci::ivp::{IVPSolver, Adams, AdamsSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<DVector<f64>, String> {
///     Ok(DVector::from_column_slice(state))
/// }
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
pub struct Adams<N: ComplexField> {
    info: AdamsInfo<N>,
}

impl<N: ComplexField> Adams<N> {
    pub fn new() -> Self {
        let mut info = AdamsInfo::new();
        info.corrector_coefficients = Self::corrector_coefficients();
        info.predictor_coefficients = Self::predictor_coefficients();
        info.error_coefficient = Self::error_coefficient();

        Adams { info }
    }
}

impl<N: ComplexField> Default for Adams<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> AdamsSolver<N> for Adams<N> {
    fn predictor_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(1901.0 / 720.0).unwrap(),
            N::RealField::from_f64(-2774.0 / 720.0).unwrap(),
            N::RealField::from_f64(2616.0 / 720.0).unwrap(),
            N::RealField::from_f64(-1274.0 / 720.0).unwrap(),
            N::RealField::from_f64(251.0 / 720.0).unwrap(),
        ]
    }

    fn corrector_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(251.0 / 720.0).unwrap(),
            N::RealField::from_f64(646.0 / 720.0).unwrap(),
            N::RealField::from_f64(-264.0 / 720.0).unwrap(),
            N::RealField::from_f64(106.0 / 720.0).unwrap(),
            N::RealField::from_f64(-19.0 / 720.0).unwrap(),
        ]
    }

    fn error_coefficient() -> N::RealField {
        N::RealField::from_f64(19.0 / 270.0).unwrap()
    }

    fn solve_ivp<T: Clone, F: Fn(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        self,
        f: F,
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

impl<N: ComplexField> From<Adams<N>> for AdamsInfo<N> {
    fn from(adams: Adams<N>) -> AdamsInfo<N> {
        adams.info
    }
}
