/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use alga::general::ComplexField;
use nalgebra::DVector;
use num_traits::Zero;

pub mod adams;
pub mod rk;
pub use adams::*;
pub use rk::*;

/// Status of an IVP Solver after a step
pub enum IVPStatus<N: ComplexField> {
    Redo,
    Ok(Vec<(N::RealField, DVector<N>)>),
    Done,
}

type DerivativeFunc<Complex, Real, T> =
    fn(Real, &[Complex], &mut T) -> Result<DVector<Complex>, String>;
type Path<Complex, Real> = Result<Vec<(Real, DVector<Complex>)>, String>;

/// Trait defining what it means to be an IVP solver.
/// solve_ivp is automatically implemented based on your step implementation.
pub trait IVPSolver<N: ComplexField>: Sized {
    /// Step forward in the solver. Returns if the solver is finished, produced
    /// an acceptable state, or needs to be redone.
    fn step<T: Clone>(
        &mut self,
        f: DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String>;
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

    /// Return the initial conditions. Called once at the very start
    /// of solving.
    fn get_initial_conditions(&self) -> Option<DVector<N>>;
    /// Get the current time of the solver.
    fn get_time(&self) -> Option<N::RealField>;
    /// Make sure that every value that needs to be set
    /// is set before the solver starts
    fn check_start(&self) -> Result<(), String>;

    /// Solve an initial value problem, consuming the solver
    fn solve_ivp<T: Clone>(
        mut self,
        f: DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> Path<N, N::RealField> {
        self.check_start()?;
        let mut path = vec![];
        let init_conditions = self.get_initial_conditions();
        let time = self.get_time();
        path.push((time.unwrap(), init_conditions.unwrap()));

        'out: loop {
            let step = self.step(f, params)?;
            match step {
                IVPStatus::Done => break 'out,
                IVPStatus::Redo => {}
                IVPStatus::Ok(mut state) => path.append(&mut state),
            }
        }

        Ok(path)
    }
}

/// Euler solver for an IVP.
///
/// Solves an initial value problem using Euler's method.
///
/// # Examples
/// ```
/// use nalgebra::DVector;
/// use bacon_sci::ivp::{Euler, IVPSolver};
/// fn derivative(_t: f64, state: &[f64], _p: &mut ()) -> Result<DVector<f64>, String> {
///     Ok(DVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), String> {
///     let solver = Euler::new()
///         .with_dt_max(0.001)?
///         .with_initial_conditions(&[1.0])?
///         .with_start(0.0)?
///         .with_end(1.0)?
///         .build();
///     let path = solver.solve_ivp(derivative, &mut ())?;
///
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() <= 0.001);
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(serialize, derive(Serialize, Deserialize))]
pub struct Euler<N: ComplexField> {
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<DVector<N>>,
}

impl<N: ComplexField> Euler<N> {
    pub fn new() -> Self {
        Euler {
            dt: None,
            time: None,
            end: None,
            state: None,
        }
    }
}

impl<N: ComplexField> IVPSolver<N> for Euler<N> {
    fn step<T: Clone>(
        &mut self,
        f: DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String> {
        if self.time >= self.end {
            return Ok(IVPStatus::Done);
        }
        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
        }

        let deriv = f(
            self.time.unwrap(),
            self.state.as_ref().unwrap().column(0).as_slice(),
            params,
        )?;

        *self
            .state
            .get_or_insert(DVector::from_column_slice(&[N::zero()])) +=
            deriv * N::from_real(self.dt.unwrap());
        *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
        Ok(IVPStatus::Ok(vec![(
            self.time.unwrap(),
            self.state.clone().unwrap(),
        )]))
    }

    fn with_tolerance(self, _tol: N::RealField) -> Result<Self, String> {
        Ok(self)
    }

    fn with_dt_max(mut self, max: N::RealField) -> Result<Self, String> {
        self.dt = Some(max);
        Ok(self)
    }

    fn with_dt_min(self, _min: N::RealField) -> Result<Self, String> {
        Ok(self)
    }

    fn with_start(mut self, t_initial: N::RealField) -> Result<Self, String> {
        if let Some(end) = self.end {
            if end <= t_initial {
                return Err("Euler with_end: Start must be after end".to_owned());
            }
        }
        self.time = Some(t_initial);
        Ok(self)
    }

    fn with_end(mut self, t_final: N::RealField) -> Result<Self, String> {
        if let Some(start) = self.time {
            if start >= t_final {
                return Err("Euler with_end: Start must be after end".to_owned());
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
        self.time
    }

    fn check_start(&self) -> Result<(), String> {
        if self.time == None {
            Err("Euler check_start: No initial time".to_owned())
        } else if self.end == None {
            Err("Euler check_start: No end time".to_owned())
        } else if self.state == None {
            Err("Euler check_start: No initial conditions".to_owned())
        } else if self.dt == None {
            Err("Euler check_start: No dt".to_owned())
        } else {
            Ok(())
        }
    }
}
