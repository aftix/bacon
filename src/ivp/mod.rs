/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use nalgebra::{ComplexField, Const, DimMin, SVector};
use num_traits::{FromPrimitive, Zero};

mod adams;
mod bdf;
mod rk;
pub use adams::*;
pub use bdf::*;
pub use rk::*;

/// Status of an IVP Solver after a step
pub enum IVPStatus<N, const S: usize>
where
    N: ComplexField,
{
    Redo,
    Ok(Vec<(N::RealField, SVector<N, S>)>),
    Done,
}

type Path<Complex, Real, const S: usize> = Result<Vec<(Real, SVector<Complex, S>)>, String>;

/// Trait defining what it means to be an IVP solver.
/// solve_ivp is automatically implemented based on your step implementation.
pub trait IVPSolver<N, const S: usize>: Sized
where
    N: ComplexField,
{
    /// Step forward in the solver. Returns if the solver is finished, produced
    /// an acceptable state, or needs to be redone.
    fn step<T, F>(&mut self, f: F, params: &mut T) -> Result<IVPStatus<N, S>, String>
    where
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>;
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
    fn get_initial_conditions(&self) -> Option<SVector<N, S>>;
    /// Get the current time of the solver.
    fn get_time(&self) -> Option<N::RealField>;
    /// Make sure that every value that needs to be set
    /// is set before the solver starts
    fn check_start(&self) -> Result<(), String>;

    /// Solve an initial value problem, consuming the solver
    fn solve_ivp<
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    >(
        mut self,
        mut f: F,
        params: &mut T,
    ) -> Path<N, N::RealField, S> {
        self.check_start()?;
        let mut path = vec![];
        let init_conditions = self.get_initial_conditions();
        let time = self.get_time();
        path.push((time.unwrap(), init_conditions.unwrap()));

        'out: loop {
            let step = self.step(&mut f, params)?;
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
/// use nalgebra::SVector;
/// use bacon_sci::ivp::{Euler, IVPSolver};
/// fn derivative(_t: f64, state: &[f64], _p: &mut ()) -> Result<SVector<f64, 1>, String> {
///     Ok(SVector::<f64, 1>::from_column_slice(state))
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
pub struct Euler<N, const S: usize>
where
    N: ComplexField,
{
    dt: Option<N::RealField>,
    time: Option<N::RealField>,
    end: Option<N::RealField>,
    state: Option<SVector<N, S>>,
}

impl<N, const S: usize> Euler<N, S>
where
    N: ComplexField,
{
    pub fn new() -> Self {
        Euler {
            dt: None,
            time: None,
            end: None,
            state: None,
        }
    }
}

impl<N, const S: usize> IVPSolver<N, S> for Euler<N, S>
where
    N: ComplexField + Copy,
    <N as ComplexField>::RealField: Copy,
{
    fn step<T, F>(&mut self, mut f: F, params: &mut T) -> Result<IVPStatus<N, S>, String>
    where
        T: Clone,
        F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    {
        if self.time >= self.end {
            return Ok(IVPStatus::Done);
        }
        if self.time.unwrap() + self.dt.unwrap() >= self.end.unwrap() {
            self.dt = Some(self.end.unwrap() - self.time.unwrap());
        }

        let deriv = f(
            self.time.unwrap(),
            self.state.as_ref().unwrap().as_slice(),
            params,
        )?;

        *self.state.get_or_insert(SVector::from_iterator(
            [N::zero()].repeat(self.state.as_ref().unwrap().as_slice().len()),
        )) += deriv * N::from_real(self.dt.unwrap());
        *self.time.get_or_insert(N::RealField::zero()) += self.dt.unwrap();
        Ok(IVPStatus::Ok(vec![(
            self.time.unwrap(),
            self.state.unwrap(),
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
        self.state = Some(SVector::from_column_slice(start));
        Ok(self)
    }

    fn build(self) -> Self {
        self
    }

    fn get_initial_conditions(&self) -> Option<SVector<N, S>> {
        self.state.as_ref().copied()
    }

    fn get_time(&self) -> Option<N::RealField> {
        self.time
    }

    fn check_start(&self) -> Result<(), String> {
        if self.time.is_none() {
            Err("Euler check_start: No initial time".to_owned())
        } else if self.end.is_none() {
            Err("Euler check_start: No end time".to_owned())
        } else if self.state.is_none() {
            Err("Euler check_start: No initial conditions".to_owned())
        } else if self.dt.is_none() {
            Err("Euler check_start: No dt".to_owned())
        } else {
            Ok(())
        }
    }
}

/// Solve an initial value problem of y'(t) = f(t, y) numerically.
///
/// Tries to solve an initial value problem with an Adams predictor-corrector,
/// the Runge-Kutta-Fehlberg method, and finally a backwards differentiation formula.
/// This is probably what you want to use.
///
/// # Params
/// `(start, end)` The start and end times for the IVP
///
/// `(dt_max, dt_min)` The maximum and minimum time step for solving
///
/// `y_0` The initial conditions at `start`
///
/// `f` the derivative function
///
/// `tol` acceptable error between steps.
///
/// `params` parameters to pass to the derivative function
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use bacon_sci::ivp::solve_ivp;
/// fn derivatives(_: f64, y: &[f64], _: &mut ()) -> Result<SVector<f64, 1>, String> {
///     Ok(-SVector::<f64, 1>::from_column_slice(y))
/// }
///
/// fn example() -> Result<(), String> {
///     let path = solve_ivp((0.0, 10.0), (0.1, 0.001), &[1.0], derivatives, 0.00001, &mut ())?;
///
///     for step in path {
///         assert!(((-step.0).exp() - step.1.column(0)[0]).abs() < 0.001);
///     }
///
///     Ok(())
/// }
/// ```
pub fn solve_ivp<N, T, F, const S: usize>(
    (start, end): (N::RealField, N::RealField),
    (dt_max, dt_min): (N::RealField, N::RealField),
    y_0: &[N],
    mut f: F,
    tol: N::RealField,
    params: &mut T,
) -> Path<N, N::RealField, S>
where
    N: ComplexField + FromPrimitive + Copy,
    <N as ComplexField>::RealField: FromPrimitive + Copy,
    T: Clone,
    F: FnMut(N::RealField, &[N], &mut T) -> Result<SVector<N, S>, String>,
    Const<S>: DimMin<Const<S>, Output = Const<S>>,
{
    let solver = Adams::new()
        .with_start(start)?
        .with_end(end)?
        .with_dt_max(dt_max)?
        .with_dt_min(dt_min)?
        .with_tolerance(tol)?
        .with_initial_conditions(y_0)?
        .build();

    let path = solver.solve_ivp(&mut f, &mut params.clone());

    if let Ok(path) = path {
        return Ok(path);
    }

    let solver: RK45<N, S> = RK45::new()
        .with_initial_conditions(y_0)?
        .with_start(start)?
        .with_end(end)?
        .with_dt_max(dt_max)?
        .with_dt_min(dt_min)?
        .with_tolerance(tol)?
        .build();

    let path = solver.solve_ivp(&mut f, &mut params.clone());

    if let Ok(path) = path {
        return Ok(path);
    }

    let solver = BDF6::new()
        .with_start(start)?
        .with_end(end)?
        .with_dt_max(dt_max)?
        .with_dt_min(dt_min)?
        .with_tolerance(tol)?
        .with_initial_conditions(y_0)?
        .build();

    solver.solve_ivp(&mut f, params)
}
