/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{Derivative, IVPError, IVPIterator, IVPSolver, IVPStatus, IVPStepper, Step};
use crate::{BSVector, BVector, Dimension};
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, DimName, RealField, U1,
};
use num_traits::{FromPrimitive, One, Zero};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// This trait defines an Adams predictor-corrector solver
/// The Adams struct takes an implemetation of this trait
/// as a type argument since the algorithm is the same for
/// all the predictor correctors, just the order and these functions
/// need to be different.
pub trait AdamsCoefficients<const O: usize> {
    /// The real field associated with the solver's Field.
    type RealField: RealField;

    /// The polynomial interpolation coefficients for the predictor. Should start
    /// with the coefficient for n - 1
    fn predictor_coefficients() -> Option<BSVector<Self::RealField, O>>;

    /// The polynomial interpolation coefficients for the corrector. Must be
    /// the same length as predictor_coefficients. Should start with the
    /// implicit coefficient.
    fn corrector_coefficients() -> Option<BSVector<Self::RealField, O>>;

    /// Coefficient for multiplying error by.
    fn error_coefficient() -> Option<Self::RealField>;
}

/// The nuts and bolts Adams solver
/// Users won't use this directly if they aren't defining their own Adams predictor-corrector
/// Used as a common struct for the specific implementations
pub struct Adams<'a, N, D, const O: usize, T, F, A>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    A: AdamsCoefficients<O, RealField = N::RealField>,
    DefaultAllocator: Allocator<N, D>,
{
    init_dt_max: Option<N::RealField>,
    init_dt_min: Option<N::RealField>,
    init_time: Option<N::RealField>,
    init_end: Option<N::RealField>,
    init_tolerance: Option<N::RealField>,
    init_state: Option<BVector<N, D>>,
    init_derivative: Option<F>,
    dim: D,
    _data: PhantomData<&'a (T, A)>,
}

/// The solver for any Adams predictor-corrector
/// Users should not use this type directly, and should
/// instead get it from a specific Adams method struct
/// (wrapped in an IVPIterator)
pub struct AdamsSolver<'a, N, D, const O: usize, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    // Parameters set by the user
    dt_max: N,
    dt_min: N,
    time: N,
    end: N,
    tolerance: N,
    derivative: F,
    data: T,

    // Current solution at t = self.time
    dt: N,
    state: BVector<N, D>,

    // Per-order constants set by an AdamsCoefficients
    predictor_coefficients: BSVector<N, O>,
    corrector_coefficients: BSVector<N, O>,
    error_coefficient: N,

    // Previous steps to interpolate with
    prev_values: VecDeque<(N::RealField, BVector<N, D>)>,
    prev_derivatives: VecDeque<BVector<N, D>>,

    // A scratch vector to use during the algorithm (to avoid allocating & de-allocating every step)
    scratch_pad: BVector<N, D>,
    // Another scratch vector, used to store values for the implicit step
    implicit_derivs: BVector<N, D>,
    // A place to store solver state while taking speculative steps trying to find a good timestep
    save_state: BVector<N, D>,

    // Constants for the particular field
    one_tenth: N,
    one_sixth: N,
    half: N,
    two: N,
    four: N,

    // generic parameter O in the type N
    order: N,

    // The number of items in prev_values that need to be yielded to the iterator
    // due to a previous runge-kutta step
    yield_memory: usize,

    _lifetime: PhantomData<&'a ()>,
}

impl<'a, N, D, const O: usize, T, F, A> IVPSolver<'a, D> for Adams<'a, N, D, O, T, F, A>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    A: AdamsCoefficients<O, RealField = N::RealField>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, Const<O>>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type Derivative = F;
    type UserData = T;
    type Solver = AdamsSolver<'a, N, D, O, T, F>;

    fn new() -> Result<Self, IVPError> {
        Ok(Self {
            init_dt_max: None,
            init_dt_min: None,
            init_time: None,
            init_end: None,
            init_tolerance: None,
            init_state: None,
            init_derivative: None,
            dim: D::dim()?,
            _data: PhantomData,
        })
    }

    fn new_dyn(size: usize) -> Result<Self, Self::Error> {
        Ok(Self {
            init_dt_max: None,
            init_dt_min: None,
            init_time: None,
            init_end: None,
            init_tolerance: None,
            init_state: None,
            init_derivative: None,
            dim: D::dim_dyn(size)?,
            _data: PhantomData,
        })
    }

    fn dim(&self) -> D {
        self.dim
    }

    fn with_tolerance(mut self, tol: Self::RealField) -> Result<Self, Self::Error> {
        if tol <= <Self::RealField as Zero>::zero() {
            return Err(IVPError::ToleranceOOB);
        }
        self.init_tolerance = Some(tol);
        Ok(self)
    }

    /// Will overwrite any previously set value
    /// If the provided maximum is less than a previously set minimum, then the minimum
    /// is set to this value as well.
    fn with_maximum_dt(mut self, max: Self::RealField) -> Result<Self, Self::Error> {
        if max <= <Self::RealField as Zero>::zero() {
            return Err(IVPError::TimeDeltaOOB);
        }

        self.init_dt_max = Some(max.clone());
        if let Some(dt_min) = self.init_dt_min.as_mut() {
            if *dt_min > max {
                *dt_min = max;
            }
        }

        Ok(self)
    }

    /// Will overwrite any previously set value
    /// If the provided minimum is greatear than a previously set maximum, then the maximum
    /// is set to this value as well.
    fn with_minimum_dt(mut self, min: Self::RealField) -> Result<Self, Self::Error> {
        if min <= <Self::RealField as Zero>::zero() {
            return Err(IVPError::TimeDeltaOOB);
        }

        self.init_dt_min = Some(min.clone());
        if let Some(dt_max) = self.init_dt_max.as_mut() {
            if *dt_max < min {
                *dt_max = min;
            }
        }

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

    fn solve(self, data: Self::UserData) -> Result<IVPIterator<D, Self::Solver>, Self::Error> {
        let dt_max = self.init_dt_max.ok_or(IVPError::MissingParameters)?;
        let dt_min = self.init_dt_min.ok_or(IVPError::MissingParameters)?;
        let tolerance = self.init_tolerance.ok_or(IVPError::MissingParameters)?;
        let time = self.init_time.ok_or(IVPError::MissingParameters)?;
        let end = self.init_end.ok_or(IVPError::MissingParameters)?;
        let state = self.init_state.ok_or(IVPError::MissingParameters)?;
        let derivative = self.init_derivative.ok_or(IVPError::MissingParameters)?;

        let two = Self::Field::from_u8(2).ok_or(IVPError::FromPrimitiveFailure)?;
        let half = two.recip();
        let one_sixth = Self::Field::from_u8(6)
            .ok_or(IVPError::FromPrimitiveFailure)?
            .recip();
        let one_tenth = Self::Field::from_u8(10)
            .ok_or(IVPError::FromPrimitiveFailure)?
            .recip();
        let four = two * two;

        let predictor_coefficients = BSVector::from_iterator(
            A::predictor_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let corrector_coefficients = BSVector::from_iterator(
            A::corrector_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let order = Self::Field::from_usize(O).ok_or(IVPError::FromPrimitiveFailure)?;

        Ok(IVPIterator {
            solver: AdamsSolver {
                dt_max: Self::Field::from_real(dt_max.clone()),
                dt_min: Self::Field::from_real(dt_min.clone()),
                time: Self::Field::from_real(time),
                end: Self::Field::from_real(end),
                tolerance: Self::Field::from_real(tolerance),
                dt: Self::Field::from_real(dt_max + dt_min) * half,
                state,
                derivative,
                data,
                predictor_coefficients,
                corrector_coefficients,
                error_coefficient: Self::Field::from_real(
                    A::error_coefficient().ok_or(IVPError::FromPrimitiveFailure)?,
                ),
                prev_values: VecDeque::new(),
                prev_derivatives: VecDeque::new(),
                scratch_pad: BVector::from_element_generic(
                    self.dim,
                    U1::name(),
                    Self::Field::zero(),
                ),
                implicit_derivs: BVector::from_element_generic(
                    self.dim,
                    U1::name(),
                    Self::Field::zero(),
                ),
                save_state: BVector::from_element_generic(
                    self.dim,
                    U1::name(),
                    Self::Field::zero(),
                ),
                one_tenth,
                one_sixth,
                half,
                two,
                four,
                order,
                yield_memory: 0,
                _lifetime: PhantomData,
            },
            finished: false,
            _dim: PhantomData,
        })
    }
}

impl<'a, N, D, const O: usize, T, F> AdamsSolver<'a, N, D, O, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    fn runge_kutta(&mut self, iterations: usize) -> Result<(), IVPError> {
        for i in 0..iterations {
            let k1 = (self.derivative)(
                self.time.real(),
                self.state.as_slice(),
                &mut self.data.clone(),
            )? * self.dt;
            let intermediate = &self.state + &k1 * self.half;

            let k2 = (self.derivative)(
                (self.time + self.half * self.dt).real(),
                intermediate.as_slice(),
                &mut self.data.clone(),
            )? * self.dt;
            let intermediate = &self.state + &k2 * self.half;

            let k3 = (self.derivative)(
                (self.time + self.half * self.dt).real(),
                intermediate.as_slice(),
                &mut self.data.clone(),
            )? * self.dt;
            let intermediate = &self.state + &k3;

            let k4 = (self.derivative)(
                (self.time + self.dt).real(),
                intermediate.as_slice(),
                &mut self.data.clone(),
            )? * self.dt;

            if i != 0 {
                self.prev_derivatives.push_back((self.derivative)(
                    self.time.real(),
                    self.state.as_slice(),
                    &mut self.data,
                )?);
                self.prev_values
                    .push_back((self.time.real(), self.state.clone()));
            }

            self.state += (k1 + k2 * self.two + k3 * self.two + k4) * self.one_sixth;
            self.time += self.dt;
        }
        self.prev_derivatives.push_back((self.derivative)(
            self.time.real(),
            self.state.as_slice(),
            &mut self.data,
        )?);
        self.prev_values
            .push_back((self.time.real(), self.state.clone()));

        Ok(())
    }
}

impl<'a, N, D, const O: usize, T, F> IVPStepper<D> for AdamsSolver<'a, N, D, O, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type UserData = T;

    fn step(&mut self) -> Step<Self::RealField, Self::Field, D, Self::Error> {
        // If yield_memory is in [1, Order) then we have taken a runge-kutta step
        // and committed to it (i.e. determined that we are within error bounds)
        // If yield_memory is Order then we have taken a runge-kutta step but haven't
        // checked if it is correct, so we don't want to yield the steps to the Iterator yet
        if self.yield_memory > 0 && self.yield_memory < O {
            let get_item = O - self.yield_memory - 1;
            self.yield_memory -= 1;

            // If this is the last runge-kutta step to be yielded,
            // set yield_memory to the sentinel value O+1 so that the next step() call
            // will yield the value in self.state (the adams step that was within
            // tolerance after these runge-kutta steps)
            if self.yield_memory == 0 {
                self.yield_memory = O + 1;
            }
            return Ok(self.prev_values[get_item].clone());
        }

        // Sentinel value to signify that the runge-kutta steps are yielded
        // and the solver can yield the adams step and continue as normal.
        // The current state needs to be returned and pushed onto the memory deque.
        // The derivatives memory deque already has the derivatives for this step,
        // since the derivatives deque is unused while yielding runge-kutta steps
        if self.yield_memory == O + 1 {
            self.yield_memory = 0;
            self.prev_values
                .push_back((self.time.real(), self.state.clone()));
            self.prev_values.pop_front();
            return Ok((self.time.real(), self.state.clone()));
        }

        if self.time.real() >= self.end.real() {
            return Err(IVPStatus::Done);
        }

        if self.time.real() + self.dt.real() >= self.end.real() {
            self.dt = self.end - self.time;
            self.runge_kutta(1)?;
            return Ok((self.time.real(), self.prev_values.back().unwrap().1.clone()));
        }

        if self.prev_values.is_empty() {
            self.save_state = self.state.clone();
            if self.time.real() + self.dt.real() * (self.order - Self::Field::one()).real()
                >= self.end.real()
            {
                self.dt = (self.end - self.time) / (self.order - Self::Field::one());
            }
            self.runge_kutta(O - 1)?;
            self.yield_memory = O;

            return Err(IVPStatus::Redo);
        }

        self.scratch_pad = &self.prev_derivatives[0] * self.predictor_coefficients[O - 2];
        for i in 1..O - 1 {
            let coefficient = self.predictor_coefficients[O - i - 2];
            self.scratch_pad += &self.prev_derivatives[i] * coefficient;
        }
        let predictor = &self.state + &self.scratch_pad * self.dt;

        self.implicit_derivs = (self.derivative)(
            self.time.real() + self.dt.real(),
            predictor.as_slice(),
            &mut self.data.clone(),
        )?;
        self.scratch_pad = &self.implicit_derivs * self.corrector_coefficients[0];

        for i in 0..O - 1 {
            let coefficient = self.corrector_coefficients[O - i - 1];
            self.scratch_pad += &self.prev_derivatives[i] * coefficient;
        }
        let corrector = &self.state + &self.scratch_pad * self.dt;

        let difference = &corrector - &predictor;
        let error = self.error_coefficient.real() / self.dt.real() * difference.norm();

        if error <= self.tolerance.real() {
            self.state = corrector;
            self.time += self.dt;

            // We have determined that this step passes the tolerance bounds.
            // If yield_memory is non-zero, then we still need to yield the runge-kutta
            // steps to the Iterator. We store the successful adams step in self.state,
            // and self.time, decrement yield memory, and return (we never want to adjust the dt
            // the step after adjusting it down). We return IVPStatus::Redo so IVPIterator
            // calls again, yielding the runge-kutta steps.
            if self.yield_memory == O {
                self.yield_memory -= 1;
                return Err(IVPStatus::Redo);
            }

            self.prev_derivatives
                .push_back(self.implicit_derivs.clone());
            self.prev_values
                .push_back((self.time.real(), self.state.clone()));

            self.prev_values.pop_front();
            self.prev_derivatives.pop_front();

            if error < self.one_tenth.real() * self.tolerance.real() {
                let q = (self.tolerance.real() / (self.two.real() * error))
                    .powf(self.order.recip().real());

                if q > self.four.real() {
                    self.dt *= self.four;
                } else {
                    self.dt *= Self::Field::from_real(q);
                }

                if self.dt.real() > self.dt_max.real() {
                    self.dt = self.dt_max;
                }

                // Clear the saved steps since we have changed the timestep
                // so we can no longer use linear interpolation.
                self.prev_values.clear();
                self.prev_derivatives.clear();
            }

            return Ok((self.time.real(), self.state.clone()));
        }

        // yield_memory can be Order here, meaning we speculatively tried a timestep and the lower timestep
        // still didn't pass the tolerances.
        // In this case, we need to return the state to what it was previously, before the runge-kutta steps,
        // and reset the time to what it was previously.
        if self.yield_memory == O {
            // We took Order - 1 runge kutta steps at this dt
            self.time -= self.dt * (self.order - Self::Field::one());
            self.state = self.save_state.clone();
        }

        let q = (self.tolerance.real() / (self.two.real() * error.real()))
            .powf(self.order.recip().real());

        if q < self.one_tenth.real() {
            self.dt *= self.one_tenth;
        } else {
            self.dt *= Self::Field::from_real(q);
        }

        if self.dt.real() < self.dt_min.real() {
            return Err(IVPStatus::Failure(IVPError::MinimumTimeDeltaExceeded));
        }

        self.prev_values.clear();
        self.prev_derivatives.clear();
        Err(IVPStatus::Redo)
    }

    fn time(&self) -> Self::RealField {
        self.time.real()
    }
}

pub struct AdamsCoefficients5<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField> AdamsCoefficients<5> for AdamsCoefficients5<N> {
    type RealField = N::RealField;

    fn predictor_coefficients() -> Option<BSVector<Self::RealField, 5>> {
        let twenty_four = Self::RealField::from_u8(24)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(55)? / twenty_four.clone(),
            -Self::RealField::from_u8(59)? / twenty_four.clone(),
            Self::RealField::from_u8(37)? / twenty_four.clone(),
            -Self::RealField::from_u8(9)? / twenty_four,
            Self::RealField::zero(),
        ]))
    }

    fn corrector_coefficients() -> Option<BSVector<Self::RealField, 5>> {
        let seven_hundred_twenty = Self::RealField::from_u16(720)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(251)? / seven_hundred_twenty.clone(),
            Self::RealField::from_u16(646)? / seven_hundred_twenty.clone(),
            -Self::RealField::from_u16(264)? / seven_hundred_twenty.clone(),
            Self::RealField::from_u8(106)? / seven_hundred_twenty.clone(),
            -Self::RealField::from_u8(19)? / seven_hundred_twenty,
        ]))
    }

    fn error_coefficient() -> Option<Self::RealField> {
        Some(Self::RealField::from_u8(19)? / Self::RealField::from_u16(270)?)
    }
}

/// 5th order Adams predictor-corrector method for solving an IVP.
///
/// Defines the predictor and corrector coefficients, as well as
/// the error coefficient. Uses Adams for the actual solving.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, adams::Adams5}};
///
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(BSVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let adams = Adams5::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(1.0)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///     let path = adams.collect_vec()?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
pub type Adams5<'a, N, D, T, F> = Adams<'a, N, D, 5, T, F, AdamsCoefficients5<N>>;

pub struct AdamsCoefficients3<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField + Copy> AdamsCoefficients<3> for AdamsCoefficients3<N> {
    type RealField = N::RealField;

    fn predictor_coefficients() -> Option<BSVector<Self::RealField, 3>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::one() + Self::RealField::from_u8(2)?.recip(),
            -Self::RealField::from_u8(2)?.recip(),
            Self::RealField::zero(),
        ]))
    }

    fn corrector_coefficients() -> Option<BSVector<Self::RealField, 3>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(5)? / Self::RealField::from_u8(12)?,
            Self::RealField::from_u8(2)? / Self::RealField::from_u8(3)?,
            -Self::RealField::from_u8(12)?.recip(),
        ]))
    }

    fn error_coefficient() -> Option<Self::RealField> {
        Some(Self::RealField::from_u8(19)? / Self::RealField::from_u16(270)?)
    }
}

/// 3rd order Adams predictor-corrector method for solving an IVP.
///
/// Defines the predictor and corrector coefficients, as well as
/// the error coefficient. Uses Adams for the actual solving.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, adams::Adams3}};
///
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(BSVector::from_column_slice(state))
/// }
///
///
/// fn example() -> Result<(), IVPError> {
///     let adams = Adams3::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(1.0)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///     let path = adams.collect_vec()?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
pub type Adams3<'a, N, D, T, F> = Adams<'a, N, D, 3, T, F, AdamsCoefficients3<N>>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{ivp::IVPSolver, BSVector};
    use std::error::Error;

    fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
        Ok(BSVector::from_column_slice(y))
    }

    fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
        Ok(BSVector::from_column_slice(&[-2.0 * t]))
    }

    fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
        Ok(BSVector::from_iterator(y.iter().map(|_| t.cos())))
    }

    // Test predictor-corrector for y=exp(t)
    #[test]
    fn adams5_exp() {
        let t_initial = 0.0;
        let t_final = 2.0;

        let solver = Adams5::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.0005)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(exp_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.exp(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn adams5_quadratic() {
        let t_initial = 0.0;
        let t_final = 5.0;

        let solver = Adams5::new()
            .unwrap()
            .with_minimum_dt(1e-7)
            .unwrap()
            .with_maximum_dt(0.001)
            .unwrap()
            .with_tolerance(0.01)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(quadratic_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                1.0 - step.0.powi(2),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn adams5_sine() {
        let t_initial = 0.0;
        let t_final = std::f64::consts::TAU;

        let solver = Adams5::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.001)
            .unwrap()
            .with_tolerance(0.01)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[0.0])
            .unwrap()
            .with_derivative(sine_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.sin(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn adams3_exp() {
        let t_initial = 0.0;
        let t_final = 2.0;

        let solver = Adams3::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.001)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(exp_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.exp(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn adams3_quadratic() {
        let t_initial = 0.0;
        let t_final = 5.0;

        let solver = Adams3::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.001)
            .unwrap()
            .with_tolerance(0.1)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(quadratic_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                1.0 - step.0.powi(2),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn adams3_sine() {
        let t_initial = 0.0;
        let t_final = std::f64::consts::TAU;

        let solver = Adams3::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.001)
            .unwrap()
            .with_tolerance(0.01)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[0.0])
            .unwrap()
            .with_derivative(sine_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                step.0.sin(),
                epsilon = 0.01
            ));
        }
    }
}
