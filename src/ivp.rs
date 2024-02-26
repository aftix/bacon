/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use crate::{BVector, Dimension, DimensionError};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, RealField, U1};
use num_traits::{FromPrimitive, Zero};
use std::{error::Error, marker::PhantomData};
use thiserror::Error;

pub mod adams;
pub mod bdf;
pub mod rk;

/// Status returned from the IVPStepper
/// Used by the IVPIterator struct to correctly step through
/// the IVP solution.
#[derive(Error, Clone, Debug)]
pub enum IVPStatus<T: Error> {
    #[error("the solver needs the step to be re-done")]
    Redo,
    #[error("the solver is complete")]
    Done,
    #[error("unspecified solver error: {0}")]
    Failure(#[from] T),
}

/// An error generated in a derivative function
pub type UserError = Box<dyn Error>;

/// A function that can be used as a derivative for the solver
pub trait Derivative<N: ComplexField + Copy, D: Dim, T: Clone>:
    FnMut(N::RealField, &[N], &mut T) -> Result<BVector<N, D>, UserError>
where
    DefaultAllocator: Allocator<N, D>,
{
}

impl<N, D: Dim, T, F> Derivative<N, D, T> for F
where
    N: ComplexField + Copy,
    T: Clone,
    F: FnMut(N::RealField, &[N], &mut T) -> Result<BVector<N, D>, UserError>,
    DefaultAllocator: Allocator<N, D>,
{
}

#[derive(Error, Debug)]
pub enum IVPError {
    #[error("the solver does not have all required parameters set")]
    MissingParameters,
    #[error("the solver hit an error from the user-provided derivative function: {0}")]
    UserError(#[from] UserError),
    #[error("the provided tolerance was out-of-bounds")]
    ToleranceOOB,
    #[error("the provided time delta was out-of-bounds")]
    TimeDeltaOOB,
    #[error("the provided ending time was before the provided starting time")]
    TimeEndOOB,
    #[error("the provided starting time was after the provided ending time")]
    TimeStartOOB,
    #[error("a conversion from a necessary primitive failed")]
    FromPrimitiveFailure,
    #[error("the time step fell below the paramater minimum allowed value")]
    MinimumTimeDeltaExceeded,
    #[error("the number of iterations exceeded the maximum allowable")]
    MaximumIterationsExceeded,
    #[error("a matrix was unable to be inverted")]
    SingularMatrix,
    #[error("attempted to build a dynamic solver with static dimension")]
    DynamicOnStatic,
    #[error("attempted to build a static solver with dynamic dimension")]
    StaticOnDynamic,
}

impl From<UserError> for IVPStatus<IVPError> {
    fn from(value: UserError) -> Self {
        Self::Failure(IVPError::UserError(value))
    }
}

impl From<DimensionError> for IVPError {
    fn from(value: DimensionError) -> Self {
        match value {
            DimensionError::DynamicOnStatic => Self::DynamicOnStatic,
            DimensionError::StaticOnDynamic => Self::StaticOnDynamic,
        }
    }
}

/// A type alias for a Result of a IVPStepper step
/// Ok is a tuple of the time and solution at that time
/// Err is an IVPError
pub type Step<R, C, D, E> = Result<(R, BVector<C, D>), IVPStatus<E>>;

/// Implementing this trait is providing the main functionality of
/// an initial value problem solver. This should be used only when
/// implementing an IVPSolver, users should use the solver via the IVPSolver
/// trait's interface.
pub trait IVPStepper<D: Dimension>: Sized
where
    DefaultAllocator: Allocator<Self::Field, D>,
{
    /// Error type. IVPError must be able to convert to the error type.
    type Error: Error + From<IVPError>;
    /// The field, complex or real, that the solver is operating on.
    type Field: ComplexField + Copy;
    /// The real field associated with the solver's Field.
    type RealField: RealField;
    /// Arbitrary data provided by the user for the derivative function
    /// It must be clone because for any intermediate time steps (e.g. in runge-kutta)
    /// gives the derivative function a clone of the params: only normal time steps get to update
    /// the internal UserData state
    type UserData: Clone;

    /// Step forward in the solver.
    /// The solver may request a step be redone, signal that the
    /// solution is finished, or give the value of the next solution value.
    fn step(&mut self) -> Step<Self::RealField, Self::Field, D, Self::Error>;

    /// Get the current time of the solver.
    fn time(&self) -> Self::RealField;
}

/// Trait covering all initial value problem solvers.
/// Build up the solver using the parameter builder functions and then use solve.
///
/// This is used as a builder pattern, setting parameters of the solver.
/// IVPSolver implementations should implement a step function that
/// returns an IVPStatus, then a blanket impl will allow it to be used as an
/// IntoIterator for the user to iterate over the results.
pub trait IVPSolver<'a, D: Dimension>: Sized
where
    DefaultAllocator: Allocator<Self::Field, D>,
{
    /// Error type. IVPError must be able to convert to the error type.
    type Error: Error + From<IVPError>;
    /// The field, complex or real, that the solver is operating on.
    type Field: ComplexField + Copy;
    /// The real field associated with the solver's Field.
    type RealField: RealField;
    /// Arbitrary data provided by the user for the derivative function
    type UserData: Clone;
    /// The type signature of the derivative function to use
    type Derivative: Derivative<Self::Field, D, Self::UserData> + 'a;
    /// The type that actually does the solving.
    type Solver: IVPStepper<
        D,
        Error = Self::Error,
        Field = Self::Field,
        RealField = Self::RealField,
        UserData = Self::UserData,
    >;

    /// Create the solver.
    /// Will fail for dynamically sized solvers
    fn new() -> Result<Self, Self::Error>;

    /// Create the solver with a run-time dimension.
    /// Will fail for statically sized solvers
    fn new_dyn(size: usize) -> Result<Self, Self::Error>;

    /// Gets the dimension of the solver
    fn dim(&self) -> D;

    /// Set the error tolerance for any condition needing needing a float epsilon
    fn with_tolerance(self, tol: Self::RealField) -> Result<Self, Self::Error>;

    fn with_maximum_dt(self, max: Self::RealField) -> Result<Self, Self::Error>;
    fn with_minimum_dt(self, min: Self::RealField) -> Result<Self, Self::Error>;
    fn with_initial_time(self, initial: Self::RealField) -> Result<Self, Self::Error>;
    fn with_ending_time(self, ending: Self::RealField) -> Result<Self, Self::Error>;

    /// The initial conditions of the problem, should reset any previous values.
    fn with_initial_conditions_slice(self, start: &[Self::Field]) -> Result<Self, Self::Error> {
        let svector = BVector::from_column_slice_generic(self.dim(), U1::from_usize(1), start);
        self.with_initial_conditions(svector)
    }

    /// The initial conditions of the problem, in a BVector. Should reset any previous values.
    fn with_initial_conditions(self, start: BVector<Self::Field, D>) -> Result<Self, Self::Error>;

    /// Sets the derivative function to use during the solve
    fn with_derivative(self, derivative: Self::Derivative) -> Self;

    /// Turns the solver into an iterator over the solution, using IVPStep::step
    fn solve(self, data: Self::UserData) -> Result<IVPIterator<D, Self::Solver>, Self::Error>;
}

pub struct IVPIterator<D: Dimension, T: IVPStepper<D>>
where
    DefaultAllocator: Allocator<T::Field, D>,
{
    solver: T,
    finished: bool,
    _dim: PhantomData<D>,
}

/// A type alias for collecting all Steps into a Result
/// of a Vec of the solution ((time, system state))
pub type Path<R, C, D, E> = Result<Vec<(R, BVector<C, D>)>, E>;

impl<D: Dimension, T: IVPStepper<D>> IVPIterator<D, T>
where
    DefaultAllocator: Allocator<T::Field, D>,
{
    pub fn collect_vec(self) -> Path<T::RealField, T::Field, D, T::Error> {
        self.collect::<Result<Vec<_>, _>>()
    }
}

impl<D: Dimension, T: IVPStepper<D>> Iterator for IVPIterator<D, T>
where
    DefaultAllocator: Allocator<T::Field, D>,
{
    type Item = Result<(T::RealField, BVector<T::Field, D>), T::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        use IVPStatus as IE;

        if self.finished {
            return None;
        }

        loop {
            match self.solver.step() {
                Ok(vec) => break Some(Ok(vec)),
                Err(IE::Done) => break None,
                Err(IE::Redo) => continue,
                Err(IE::Failure(e)) => {
                    self.finished = true;
                    break Some(Err(e));
                }
            }
        }
    }
}

/// Euler solver for an IVP.
///
/// Solves an initial value problem using Euler's method.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{Euler, IVPSolver, IVPError}};
/// fn derivative(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(BSVector::<f64, 1>::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let solver = Euler::new()?
///         .with_maximum_dt(0.001)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_initial_time(0.0)?
///         .with_ending_time(1.0)?
///         .with_derivative(derivative)
///         .solve(())?;
///     let path = solver.collect_vec()?;
///
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() <= 0.001);
///     }
///     Ok(())
/// }
/// ```
pub struct Euler<'a, N, D, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    init_dt: Option<N::RealField>,
    init_time: Option<N::RealField>,
    init_end: Option<N::RealField>,
    init_state: Option<BVector<N, D>>,
    init_derivative: Option<F>,
    dim: D,
    _data: PhantomData<&'a T>,
}

/// The struct that actually solves an IVP with Euler's method
/// Is the associated IVPStepper for Euler (the IVPSolver)
/// You should use Euler and not this type directly
pub struct EulerSolver<'a, N, D, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    dt: N,
    time: N,
    end: N,
    state: BVector<N, D>,
    derivative: F,
    data: T,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, N, D, T, F> IVPStepper<D> for EulerSolver<'a, N, D, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type UserData = T;

    fn step(
        &mut self,
    ) -> Result<(Self::RealField, BVector<Self::Field, D>), IVPStatus<Self::Error>> {
        if self.time.real() >= self.end.real() {
            return Err(IVPStatus::Done);
        }
        if (self.time + self.dt).real() >= self.end.real() {
            self.dt = self.end - self.time;
        }

        let derivative = (self.derivative)(self.time.real(), self.state.as_slice(), &mut self.data)
            .map_err(IVPError::UserError)?;

        let old_time = self.time.real();
        let old_state = self.state.clone();

        self.state += derivative * self.dt;
        self.time += self.dt;

        Ok((old_time, old_state))
    }

    fn time(&self) -> Self::RealField {
        self.time.real()
    }
}

impl<'a, N, D, T, F> IVPSolver<'a, D> for Euler<'a, N, D, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type Derivative = F;
    type UserData = T;
    type Solver = EulerSolver<'a, N, D, T, F>;

    fn new() -> Result<Self, Self::Error> {
        Ok(Self {
            init_dt: None,
            init_time: None,
            init_end: None,
            init_state: None,
            init_derivative: None,
            dim: D::dim()?,
            _data: PhantomData,
        })
    }

    fn new_dyn(size: usize) -> Result<Self, Self::Error> {
        Ok(Self {
            init_dt: None,
            init_time: None,
            init_end: None,
            init_state: None,
            init_derivative: None,
            dim: D::dim_dyn(size)?,
            _data: PhantomData,
        })
    }

    fn dim(&self) -> D {
        self.dim
    }

    /// Unused for Euler, call is a no-op
    fn with_tolerance(self, _tol: Self::RealField) -> Result<Self, Self::Error> {
        Ok(self)
    }

    /// If there is not time step already, set, then set the time step.
    /// If there is, set the time step to the average of that and the max passed in.
    fn with_maximum_dt(mut self, max: Self::RealField) -> Result<Self, Self::Error> {
        if max <= <Self::RealField as Zero>::zero() {
            return Err(IVPError::TimeDeltaOOB);
        }

        self.init_dt = if let Some(dt) = self.init_dt {
            Some((dt + max) / Self::RealField::from_u8(2).ok_or(IVPError::FromPrimitiveFailure)?)
        } else {
            Some(max)
        };
        Ok(self)
    }

    /// If there is not time step already, set, then set the time step.
    /// If there is, set the time step to the average of that and the max passed in.
    fn with_minimum_dt(mut self, min: Self::RealField) -> Result<Self, Self::Error> {
        if min <= <Self::RealField as Zero>::zero() {
            return Err(IVPError::TimeDeltaOOB);
        }

        self.init_dt = if let Some(dt) = self.init_dt {
            Some((dt + min) / Self::RealField::from_u8(2).ok_or(IVPError::FromPrimitiveFailure)?)
        } else {
            Some(min)
        };
        Ok(self)
    }

    fn with_initial_time(mut self, initial: Self::RealField) -> Result<Self, Self::Error> {
        self.init_time = Some(initial.clone());

        if let Some(end) = self.init_end.as_ref() {
            if *end <= initial {
                return Err(IVPError::TimeStartOOB);
            }
        }

        Ok(self)
    }

    fn with_ending_time(mut self, ending: Self::RealField) -> Result<Self, Self::Error> {
        self.init_end = Some(ending.clone());

        if let Some(initial) = self.init_time.as_ref() {
            if *initial >= ending {
                return Err(IVPError::TimeEndOOB);
            }
        }

        Ok(self)
    }

    fn with_initial_conditions(
        mut self,
        start: BVector<Self::Field, D>,
    ) -> Result<Self, Self::Error> {
        self.init_state = Some(start);
        Ok(self)
    }

    fn with_derivative(mut self, derivative: Self::Derivative) -> Self {
        self.init_derivative = Some(derivative);
        self
    }

    fn solve(mut self, data: Self::UserData) -> Result<IVPIterator<D, Self::Solver>, Self::Error> {
        let dt = self.init_dt.ok_or(IVPError::MissingParameters)?;
        let time = self.init_time.ok_or(IVPError::MissingParameters)?;
        let end = self.init_end.ok_or(IVPError::MissingParameters)?;
        let state = self.init_state.take().ok_or(IVPError::MissingParameters)?;
        let derivative = self
            .init_derivative
            .take()
            .ok_or(IVPError::MissingParameters)?;

        Ok(IVPIterator {
            solver: EulerSolver {
                dt: N::from_real(dt),
                time: N::from_real(time),
                end: N::from_real(end),
                state,
                derivative,
                data,
                _lifetime: PhantomData,
            },
            finished: false,
            _dim: PhantomData,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::BSVector;
    use nalgebra::{DimName, Dyn};

    type Path<D> = Vec<(f64, BVector<f64, D>)>;

    fn solve_ivp<'a, D, F>(
        (initial, end): (f64, f64),
        dt: f64,
        initial_conds: &[f64],
        derivative: F,
    ) -> Result<Path<D>, IVPError>
    where
        D: Dimension,
        F: Derivative<f64, D, ()> + 'a,
        DefaultAllocator: Allocator<f64, D>,
    {
        let ivp = Euler::new()?
            .with_initial_time(initial)?
            .with_ending_time(end)?
            .with_maximum_dt(dt)?
            .with_initial_conditions_slice(initial_conds)?
            .with_derivative(derivative);
        ivp.solve(())?.collect()
    }

    fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(y))
    }

    fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(&[-2.0 * t]))
    }

    fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_iterator(y.iter().map(|_| t.cos())))
    }

    fn cos_deriv(_t: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 2>, UserError> {
        Ok(BSVector::from_column_slice(&[y[1], -y[0]]))
    }

    fn dynamic_cos_deriv(_t: f64, y: &[f64], _: &mut ()) -> Result<BVector<f64, Dyn>, UserError> {
        Ok(BVector::from_column_slice_generic(
            Dyn::from_usize(y.len()),
            U1::name(),
            &[y[1], -y[0]],
        ))
    }

    #[test]
    #[should_panic]
    fn euler_dynamic_cos_panics() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let path = solve_ivp((t_initial, t_final), 0.01, &[1.0, 0.0], dynamic_cos_deriv).unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.cos(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn euler_dynamic_cos() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let ivp = Euler::new_dyn(2)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_maximum_dt(0.01)
            .unwrap()
            .with_initial_conditions_slice(&[1.0, 0.0])
            .unwrap()
            .with_derivative(dynamic_cos_deriv)
            .solve(())
            .unwrap();
        let path = ivp.collect_vec().unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.cos(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn euler_cos() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let path = solve_ivp((t_initial, t_final), 0.01, &[1.0, 0.0], cos_deriv).unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.cos(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn euler_exp() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let path = solve_ivp((t_initial, t_final), 0.005, &[1.0], exp_deriv).unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.exp(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn euler_quadratic() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let path = solve_ivp((t_initial, t_final), 0.01, &[1.0], quadratic_deriv).unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                1.0 - step.0.powi(2),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn euler_sin() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let path = solve_ivp((t_initial, t_final), 0.01, &[0.0], sine_deriv).unwrap();

        for step in path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.sin(),
                epsilon = 0.01
            ));
        }
    }
}
