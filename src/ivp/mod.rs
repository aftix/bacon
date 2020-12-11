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
    Ok((<N as ComplexField>::RealField, DVector<N>)),
    Done,
}

pub trait IVPSolver<N: ComplexField> {
    fn step<T: Clone>(
        &mut self,
        f: fn(<N as ComplexField>::RealField, &[N], &mut T) -> Result<DVector<N>, String>,
        params: &mut T,
    ) -> Result<IVPStatus<N>, String>;
    fn with_tolerance(&mut self, tol: <N as ComplexField>::RealField) -> Result<&mut Self, String>;
    fn with_dt_max(&mut self, max: <N as ComplexField>::RealField) -> Result<&mut Self, String>;
    fn with_dt_min(&mut self, min: <N as ComplexField>::RealField) -> Result<&mut Self, String>;
    fn with_start(
        &mut self,
        t_initial: <N as ComplexField>::RealField,
    ) -> Result<&mut Self, String>;
    fn with_end(&mut self, t_final: <N as ComplexField>::RealField) -> Result<&mut Self, String>;
    fn build(&self) -> Self;

    fn with_initial_conditions(&mut self, start: &[N]) -> Result<&mut Self, String>;
    fn get_initial_conditions(&self) -> Option<DVector<N>>;
    fn get_time(&self) -> Option<<N as ComplexField>::RealField>;
    // Make sure we have t_initial, t_final, dt, initial conditions
    fn check_start(&self) -> Result<(), String>;

    fn solve_ivp<T: Clone>(
        &mut self,
        f: fn(<N as ComplexField>::RealField, &[N], &mut T) -> Result<DVector<N>, String>,
        params: &mut T,
    ) -> Result<Vec<(<N as ComplexField>::RealField, DVector<N>)>, String> {
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
    dt: Option<<N as ComplexField>::RealField>,
    time: Option<<N as ComplexField>::RealField>,
    end: Option<<N as ComplexField>::RealField>,
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
        f: fn(<N as ComplexField>::RealField, &[N], &mut T) -> Result<DVector<N>, String>,
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

    fn with_tolerance(
        &mut self,
        _tol: <N as ComplexField>::RealField,
    ) -> Result<&mut Self, String> {
        Ok(self)
    }

    fn with_dt_max(&mut self, max: <N as ComplexField>::RealField) -> Result<&mut Self, String> {
        self.dt = Some(max);
        Ok(self)
    }

    fn with_dt_min(&mut self, _min: <N as ComplexField>::RealField) -> Result<&mut Self, String> {
        Ok(self)
    }

    fn with_start(
        &mut self,
        t_initial: <N as ComplexField>::RealField,
    ) -> Result<&mut Self, String> {
        self.time = Some(t_initial);
        Ok(self)
    }

    fn with_end(&mut self, t_final: <N as ComplexField>::RealField) -> Result<&mut Self, String> {
        self.end = Some(t_final);
        Ok(self)
    }

    fn with_initial_conditions(&mut self, start: &[N]) -> Result<&mut Self, String> {
        self.state = Some(DVector::from_column_slice(start));
        Ok(self)
    }

    fn build(&self) -> Self {
        self.clone()
    }

    fn get_initial_conditions(&self) -> Option<DVector<N>> {
        if let Some(state) = &self.state {
            Some(state.clone())
        } else {
            None
        }
    }

    fn get_time(&self) -> Option<<N as ComplexField>::RealField> {
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
