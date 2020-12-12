/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{IVPSolver, IVPStatus};
use alga::general::{ComplexField, RealField};
use nalgebra::DVector;
use num_traits::{FromPrimitive, Zero};

/// This trait allows a struct to be used in the Runge-Kutta solver.
///
/// # Examples
/// See `struct RungeKutta` and `struct RungeKuttaFehlberg` for examples of implementing
/// this trait.
pub trait RungeKuttaSolver<N: ComplexField>: Sized {
    /// Returns a slice of coeffecients to multiply the time step by when getting
    /// intermediate results. Upper-left portion of Butch Tableaux
    fn t_coefficients() -> Vec<N::RealField>;

    /// Returns the coefficients to use on the k_i's when finding another
    /// k_i. Upper-right portion of the Butch Tableax. Should be
    /// an NxN-1 matrix, where N is the order of the Runge-Kutta Method (Or order+1 for
    /// adaptive methods)
    fn k_coefficients() -> Vec<Vec<N::RealField>>;

    /// Coefficients to use when calculating the final step to take.
    /// These are the weights of the weighted average of k_i's. Bottom
    /// portion of the Butch Tableaux. For adaptive methods, this is the first
    /// row of the bottom portion.
    fn avg_coefficients() -> Vec<N::RealField>;

    /// Used for adaptive methods only. Coefficients to use on
    /// the k_i's to find the error between the two orders
    /// of Runge-Kutta methods. In the Butch Tableaux, this is
    /// the first row of the bottom portion minus the second row.
    fn error_coefficients() -> Vec<N::RealField>;

    fn solve_ivp<T: Clone>(
        &mut self,
        f: super::DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> super::Path<N, N::RealField>;
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(serialize, derive(Serialize, Deserialize))]
pub struct RKInfo<N: ComplexField> {
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<DVector<N>>,
    dt_max: Option<N::RealField>,
    dt_min: Option<N::RealField>,
    tolerance: Option<N::RealField>,
    a_coefficients: Vec<N::RealField>,
    k_coefficients: Vec<Vec<N::RealField>>,
    avg_coefficients: Vec<N::RealField>,
    error_coefficients: Vec<N::RealField>,
}

impl<N: ComplexField> RKInfo<N> {
    fn new() -> Self {
        RKInfo {
            dt: None,
            time: None,
            end: None,
            state: None,
            dt_max: None,
            dt_min: None,
            tolerance: None,
            a_coefficients: vec![],
            k_coefficients: vec![],
            avg_coefficients: vec![],
            error_coefficients: vec![],
        }
    }
}

impl<N: ComplexField> IVPSolver<N> for RKInfo<N> {
    fn step<T: Clone>(
        &mut self,
        f: super::DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String> {
        if self.time.unwrap() >= self.end.unwrap() {
            return Ok(IVPStatus::Done);
        }

        let mut set_dt = false;
        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            set_dt = true;
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
        }

        let num_k = self.k_coefficients.len();

        let mut half_steps: Vec<DVector<N>> = Vec::with_capacity(num_k);
        for i in 0..num_k {
            let mut state = self.state.as_ref().unwrap().clone();
            for (j, k) in half_steps.iter().enumerate() {
                state += k * N::from_real(self.k_coefficients[i][j]);
            }
            half_steps.push(
                f(
                    self.time.unwrap() + self.a_coefficients[i] * self.dt.unwrap(),
                    state.column(0).as_slice(),
                    &mut params.clone(),
                )? * N::from_real(self.dt.unwrap()),
            );
        }

        let mut error_vec = half_steps[0].clone() * N::from_real(self.error_coefficients[0]);
        for (ind, k) in half_steps.iter().enumerate().skip(1) {
            error_vec += k * N::from_real(self.error_coefficients[ind]);
        }
        let error = error_vec.dot(&error_vec).real() / self.dt.unwrap();

        let mut output = false;
        if error <= self.tolerance.unwrap() {
            output = true;
            *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
            for (ind, k) in half_steps.iter().enumerate() {
                *self
                    .state
                    .get_or_insert(DVector::from_column_slice(&[N::zero()])) +=
                    k * N::from_real(self.avg_coefficients[ind]);
            }
        }

        let delta = N::RealField::from_f64(0.84).unwrap()
            * (self.tolerance.unwrap() / error).powf(N::RealField::from_f64(0.25).unwrap());
        if delta <= N::RealField::from_f64(0.1).unwrap() {
            *self.dt.get_or_insert(N::RealField::zero()) *= N::RealField::from_f64(0.1).unwrap();
        } else if delta >= N::RealField::from_f64(4.0).unwrap() {
            *self.dt.get_or_insert(N::RealField::zero()) *= N::RealField::from_f64(4.0).unwrap();
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
            Ok(IVPStatus::Ok((
                self.time.unwrap(),
                self.state.as_ref().unwrap().clone(),
            )))
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

#[derive(Debug, Clone)]
#[cfg_attr(serialize, derive(Serialize, Deserialize))]
pub struct RK45<N: ComplexField> {
    info: RKInfo<N>,
}

impl<N: ComplexField> RK45<N> {
    pub fn new() -> Self {
        let mut info = RKInfo::<N>::new();
        info.a_coefficients = Self::t_coefficients();
        info.k_coefficients = Self::k_coefficients();
        info.avg_coefficients = Self::avg_coefficients();
        info.error_coefficients = Self::error_coefficients();
        RK45 { info }
    }

    pub fn with_tolerance(mut self, tol: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_tolerance(tol)?;
        Ok(self)
    }

    pub fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_max(max)?;
        Ok(self)
    }

    pub fn with_dt_min(mut self, min: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_dt_min(min)?;
        Ok(self)
    }

    pub fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_start(t_initial)?;
        Ok(self)
    }

    pub fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        self.info = self.info.with_end(t_final)?;
        Ok(self)
    }

    pub fn with_initial_conditions(mut self, start: &[N]) -> Result<Self, String> {
        self.info = self.info.with_initial_conditions(start)?;
        Ok(self)
    }
}

impl<N: ComplexField> RungeKuttaSolver<N> for RK45<N> {
    fn t_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(0.25).unwrap(),
            N::RealField::from_f64(3.0 / 8.0).unwrap(),
            N::RealField::from_f64(12.0 / 13.0).unwrap(),
            N::RealField::from_f64(1.0).unwrap(),
            N::RealField::from_f64(0.5).unwrap(),
        ]
    }

    fn k_coefficients() -> Vec<Vec<N::RealField>> {
        vec![
            vec![],
            vec![N::RealField::from_f64(0.25).unwrap()],
            vec![
                N::RealField::from_f64(3.0 / 32.0).unwrap(),
                N::RealField::from_f64(9.0 / 32.0).unwrap(),
            ],
            vec![
                N::RealField::from_f64(1932.0 / 2197.0).unwrap(),
                N::RealField::from_f64(-7200.0 / 2197.0).unwrap(),
                N::RealField::from_f64(7296.0 / 2197.0).unwrap(),
            ],
            vec![
                N::RealField::from_f64(439.0 / 216.0).unwrap(),
                N::RealField::from_f64(-8.0).unwrap(),
                N::RealField::from_f64(3680.0 / 513.0).unwrap(),
                N::RealField::from_f64(-845.0 / 4104.0).unwrap(),
            ],
            vec![
                N::RealField::from_f64(-8.0 / 27.0).unwrap(),
                N::RealField::from_f64(2.0).unwrap(),
                N::RealField::from_f64(-3544.0 / 2565.0).unwrap(),
                N::RealField::from_f64(1859.0 / 4104.0).unwrap(),
                N::RealField::from_f64(-11.0 / 40.0).unwrap(),
            ],
        ]
    }

    fn avg_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(25.0 / 216.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(1408.0 / 2565.0).unwrap(),
            N::RealField::from_f64(2197.0 / 4104.0).unwrap(),
            N::RealField::from_f64(-1.0 / 5.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
        ]
    }

    fn error_coefficients() -> Vec<N::RealField> {
        vec![
            N::RealField::from_f64(1.0 / 360.0).unwrap(),
            N::RealField::from_f64(0.0).unwrap(),
            N::RealField::from_f64(-128.0 / 4275.0).unwrap(),
            N::RealField::from_f64(-2197.0 / 75240.0).unwrap(),
            N::RealField::from_f64(1.0 / 50.0).unwrap(),
            N::RealField::from_f64(2.0 / 55.0).unwrap(),
        ]
    }

    fn solve_ivp<T: Clone>(
        &mut self,
        f: super::DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> super::Path<N, N::RealField> {
        self.info.solve_ivp(f, params)
    }
}
