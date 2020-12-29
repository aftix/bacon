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
/// Things implementing RungeKuttaSolver should have an RKInfo to handle
/// the actual IVP solving. It should also provide the with_* helper functions
/// for convience.
///
/// # Examples
/// See `struct RK45` for an example of implementing this trait
pub trait RungeKuttaSolver<N: ComplexField>: Sized {
    /// Returns a vec of coeffecients to multiply the time step by when getting
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

    /// Coefficients to use on
    /// the k_i's to find the error between the two orders
    /// of Runge-Kutta methods. In the Butch Tableaux, this is
    /// the first row of the bottom portion minus the second row.
    fn error_coefficients() -> Vec<N::RealField>;

    /// Ideally, call RKInfo.solve_ivp
    fn solve_ivp<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
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

/// Provides an IVPSolver implementation for RungeKuttaSolver,
/// based entirely on the Butch Tableaux coefficients. It is up
/// to the RungeKuttaSolver to set up RKInfo. See RK45 for an example.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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

impl<N: ComplexField> Default for RKInfo<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: ComplexField> IVPSolver<N> for RKInfo<N> {
    fn step<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
        &mut self,
        mut f: F,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String> {
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

        let num_k = self.k_coefficients.len();

        let mut half_steps: Vec<DVector<N>> = Vec::with_capacity(num_k);
        for i in 0..num_k {
            let state = &self.state.as_ref().unwrap();
            let state: Vec<_> = state
                .column(0)
                .as_slice()
                .iter()
                .enumerate()
                .map(|(ind, y)| {
                    let mut acc = N::zero();
                    for (j, k) in half_steps.iter().enumerate() {
                        acc += k[ind] * N::from_real(self.k_coefficients[i][j]);
                    }
                    *y + acc
                })
                .collect();
            half_steps.push(
                f(
                    self.time.unwrap() + self.a_coefficients[i] * self.dt.unwrap(),
                    &state,
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
                self.state.as_ref().unwrap().clone(),
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

/// Runge-Kutta-Fehlberg method for solving an IVP.
///
/// Defines the Butch Tableaux for a 5(4) order adaptive
/// runge-kutta method. Uses RKInfo to do the actual solving.
/// Provides an interface for setting the conditions on RKInfo.
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon_sci::ivp::{IVPSolver, RK45, RungeKuttaSolver};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<DVector<f64>, String> {
///     Ok(DVector::from_column_slice(state))
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
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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
}

impl<N: ComplexField> Default for RK45<N> {
    fn default() -> Self {
        Self::new()
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

    fn solve_ivp<T: Clone, F: FnMut(N::RealField, &[N], &mut T) -> Result<DVector<N>, String>>(
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

    fn build(self) -> Self {
        self
    }
}

impl<N: ComplexField> From<RK45<N>> for RKInfo<N> {
    fn from(rk: RK45<N>) -> RKInfo<N> {
        rk.info
    }
}
