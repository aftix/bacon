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

pub enum IVPStatus<N: ComplexField> {
    Redo,
    Ok((N::RealField, DVector<N>)),
    Done,
}

type DerivativeFunc<Complex, Real, T> =
    fn(Real, &[Complex], &mut T) -> Result<DVector<Complex>, String>;
type Path<Complex, Real> = Result<Vec<(Real, DVector<Complex>)>, String>;

pub trait IVPSolver<N: ComplexField>: Sized {
    fn step<T: Clone>(
        &mut self,
        f: DerivativeFunc<N, N::RealField, T>,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String>;
    fn with_tolerance(self, tol: N::RealField) -> Result<Self, String>;
    fn with_dt_max(self, max: N::RealField) -> Result<Self, String>;
    fn with_dt_min(self, min: N::RealField) -> Result<Self, String>;
    fn with_start(self, t_initial: N::RealField) -> Result<Self, String>;
    fn with_end(self, t_final: N::RealField) -> Result<Self, String>;
    fn with_initial_conditions(self, start: &[N]) -> Result<Self, String>;
    fn build(self) -> Self;

    fn get_initial_conditions(&self) -> Option<DVector<N>>;
    fn get_time(&self) -> Option<N::RealField>;
    // Make sure we have t_initial, t_final, dt, initial conditions
    fn check_start(&self) -> Result<(), String>;

    fn solve_ivp<T: Clone>(
        &mut self,
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
                IVPStatus::Ok(state) => path.push(state),
            }
        }

        Ok(path)
    }
}

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
        Ok(IVPStatus::Ok((
            self.time.unwrap(),
            self.state.clone().unwrap(),
        )))
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
