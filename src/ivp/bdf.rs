/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{Derivative, IVPError, IVPIterator, IVPSolver, IVPStatus, IVPStepper, Step, UserError};
use crate::{BMatrix, BSVector, BVector, Dimension};
use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, DimMin, DimName, RealField, U1,
};
use num_traits::{FromPrimitive, One, Zero};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// This trait defines an BDF solver
/// The BDF struct takes an implemetation of this trait
/// as a type argument since the algorithm is the same for
/// all the orders, just the constants are different.
pub trait BDFCoefficients<const O: usize> {
    type RealField: RealField;

    /// The polynomial interpolation coefficients for the higher-order
    /// method. Should start
    /// with the coefficient for the derivative
    /// function without h, then n - 1. The
    /// coefficients for the previous terms
    /// should have the sign as if they're on the
    /// same side of the = as the next state.
    fn higher_coefficients() -> Option<BSVector<Self::RealField, O>>;

    /// The polynomial interpolation coefficients for the lower-order
    /// method. Must be
    /// one less in length than higher_coefficients.
    /// Should start with the coefficient for the
    /// derivative function without h, then n-1. The
    /// coefficients for the previous terms
    /// should have the sign as if they're on the
    /// same side of the = as the next state.
    fn lower_coefficients() -> Option<BSVector<Self::RealField, O>>;
}

/// The nuts and bolts BDF solver
/// Users won't use this directly if they aren't defining their own BDF
/// Used as a common struct for the specific implementations
pub struct BDF<'a, N, D, const O: usize, T, F, B>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    B: BDFCoefficients<O, RealField = N::RealField>,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, D, D>,
{
    init_dt_max: Option<N::RealField>,
    init_dt_min: Option<N::RealField>,
    init_time: Option<N::RealField>,
    init_end: Option<N::RealField>,
    init_tolerance: Option<N::RealField>,
    init_state: Option<BVector<N, D>>,
    init_derivative: Option<F>,
    dim: D,
    _data: PhantomData<&'a (T, B)>,
}

/// The solver for any BDF predictor-corrector
/// Users should not use this type directly, and should
/// instead get it from a specific BDF method struct
/// (wrapped in an IVPIterator)
pub struct BDFSolver<'a, N, D, const O: usize, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, D, D>,
{
    // Parameters set by the user
    dt_max: N,
    dt_min: N,
    time: N,
    end: N,
    tolerance: N,
    derivative: F,
    dim: D,
    data: T,

    // Current solution at t = self.time
    dt: N,
    state: BVector<N, D>,

    // Per-order constants set by an BDFCoefficients
    higher_coefficients: BSVector<N, O>,
    lower_coefficients: BSVector<N, O>,

    // Previous steps to interpolate with
    prev_values: VecDeque<(N::RealField, BVector<N, D>)>,

    // A scratch vector to use during the algorithm (to avoid allocating & de-allocating every step)
    scratch_pad: BVector<N, D>,
    // A place to store solver state while taking speculative steps trying to find a good timestep
    save_state: BVector<N, D>,

    // Constants for the particular field
    one_tenth: N,
    one_sixth: N,
    half: N,
    two: N,

    // generic parameter O in the type N
    order: N,

    // The number of items in prev_values that need to be yielded to the iterator
    // due to a previous runge-kutta step
    yield_memory: usize,

    _lifetime: PhantomData<&'a ()>,
}

impl<'a, N, D, const O: usize, T, F, B> IVPSolver<'a, D> for BDF<'a, N, D, O, T, F, B>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    B: BDFCoefficients<O, RealField = N::RealField>,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, U1, D>,
    DefaultAllocator: Allocator<N, D, D>,
    DefaultAllocator: Allocator<(usize, usize), D>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type Derivative = F;
    type UserData = T;
    type Solver = BDFSolver<'a, N, D, O, T, F>;

    fn new() -> Result<Self, Self::Error> {
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

        let higher_coefficients = BSVector::from_iterator(
            B::higher_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let lower_coefficients = BSVector::from_iterator(
            B::lower_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let order = Self::Field::from_usize(O).ok_or(IVPError::FromPrimitiveFailure)?;

        Ok(IVPIterator {
            solver: BDFSolver {
                dt_max: Self::Field::from_real(dt_max.clone()),
                dt_min: Self::Field::from_real(dt_min.clone()),
                time: Self::Field::from_real(time),
                end: Self::Field::from_real(end),
                tolerance: Self::Field::from_real(tolerance),
                dt: Self::Field::from_real(dt_max + dt_min) * half,
                state,
                derivative,
                dim: self.dim,
                data,
                higher_coefficients,
                lower_coefficients,
                prev_values: VecDeque::new(),
                scratch_pad: BVector::from_element_generic(
                    self.dim,
                    U1::name(),
                    Self::Field::zero(),
                ),
                save_state: BVector::from_element_generic(
                    self.dim,
                    U1::from_usize(1),
                    Self::Field::zero(),
                ),
                one_tenth,
                one_sixth,
                half,
                two,
                order,
                yield_memory: 0,
                _lifetime: PhantomData,
            },
            finished: false,
            _dim: PhantomData,
        })
    }
}

impl<'a, N, D, const O: usize, T, F> BDFSolver<'a, N, D, O, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, U1, D>,
    DefaultAllocator: Allocator<N, D, D>,
    DefaultAllocator: Allocator<(usize, usize), D>,
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
                self.prev_values
                    .push_back((self.time.real(), self.state.clone()));
            }

            self.state += (k1 + k2 * self.two + k3 * self.two + k4) * self.one_sixth;
            self.time += self.dt;
        }
        self.prev_values
            .push_back((self.time.real(), self.state.clone()));

        Ok(())
    }

    // Used for secant method
    fn jac_finite_diff<G>(
        &mut self,
        x: &mut BVector<N, D>,
        g: &mut G,
    ) -> Result<BMatrix<N, D, D>, IVPError>
    where
        G: FnMut(&mut Self, N::RealField, &[N], &mut T) -> Result<BVector<N, D>, UserError>,
    {
        let mut mat = BMatrix::from_element_generic(self.dim, self.dim, N::zero());
        let denom = (self.two * self.dt).recip();

        for (ind, mut col) in mat.column_iter_mut().enumerate() {
            x[ind] += self.dt;
            let above = g(self, self.time.real(), x.as_slice(), &mut self.data.clone())?;
            x[ind] -= self.two * self.dt;
            let below = g(self, self.time.real(), x.as_slice(), &mut self.data.clone())?;
            x[ind] += self.dt;
            col.set_column(0, &((above + below) * denom));
        }

        Ok(mat)
    }

    // Secant method for performing the BDF
    fn secant<G>(&mut self, g: &mut G) -> Result<BVector<N, D>, IVPError>
    where
        G: FnMut(&mut Self, N::RealField, &[N], &mut T) -> Result<BVector<N, D>, UserError>,
    {
        let mut n = 2;

        let mut guess = self.state.clone();
        let mut derivative = g(
            self,
            self.time.real(),
            guess.as_slice(),
            &mut self.data.clone(),
        )?;

        let jac = self.jac_finite_diff(&mut guess, g)?;
        let lu = jac.clone().lu();
        let mut jac_inv = if let Some(inv) = lu.try_inverse() {
            inv
        } else {
            let lu = jac.clone().full_piv_lu();
            if let Some(inv) = lu.try_inverse() {
                inv
            } else {
                let qr = jac.qr();
                if let Some(inv) = qr.try_inverse() {
                    inv
                } else {
                    return Err(IVPError::SingularMatrix);
                }
            }
        };

        let mut shift = -&jac_inv * &derivative;
        guess += &shift;

        while n < 1000 {
            let derivative_last = derivative;
            derivative = g(
                self,
                self.time.real(),
                guess.as_slice(),
                &mut self.data.clone(),
            )?;

            let difference = &derivative - &derivative_last;
            let adjustment = -&jac_inv * difference;
            let s_transpose = shift.clone().transpose();
            let p = (-&s_transpose * &adjustment)[(0, 0)];
            let u = s_transpose * &jac_inv;

            jac_inv += (shift + adjustment) * u / p;
            shift = -&jac_inv * &derivative;
            guess += &shift;

            if shift.norm() <= self.tolerance.real() {
                return Ok(guess);
            }
            n += 1;
        }

        Err(IVPError::MaximumIterationsExceeded)
    }
}

impl<'a, N, D, const O: usize, T, F> IVPStepper<D> for BDFSolver<'a, N, D, O, T, F>
where
    N: ComplexField + Copy,
    D: Dimension,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, U1, D>,
    DefaultAllocator: Allocator<N, D, D>,
    DefaultAllocator: Allocator<(usize, usize), D>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type UserData = T;

    fn step(&mut self) -> Step<Self::RealField, Self::Field, D, Self::Error> {
        // If yield_memory is in [1, Order] then we have taken a runge-kutta step
        // and committed to it (i.e. determined that we are within error bounds)
        // If yield_memory is Order+1 then we have taken a runge-kutta step but haven't
        // checked if it is correct, so we don't want to yield the steps to the Iterator yet
        if self.yield_memory > 0 && self.yield_memory <= O {
            let get_item = O - self.yield_memory;
            self.yield_memory -= 1;

            // If this is the last runge-kutta step to be yielded,
            // set yield_memory to the sentinel value O+2 so that the next step() call
            // will yield the value in self.state (the bdf step that was within
            // tolerance after these runge-kutta steps)
            if self.yield_memory == 0 {
                self.yield_memory = O + 2;
            }
            return Ok(self.prev_values[get_item].clone());
        }

        // Sentinel value to signify that the runge-kutta steps are yielded
        // and the solver can yield the bdf step and continue as normal.
        // The current state needs to be returned and pushed onto the memory deque.
        // The derivatives memory deque already has the derivatives for this step,
        // since the derivatives deque is unused while yielding runge-kutta steps
        if self.yield_memory == O + 2 {
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
            if self.time.real() + self.dt.real() * self.order.real() >= self.end.real() {
                self.dt = (self.end - self.time) / self.order;
            }
            self.runge_kutta(O)?;
            self.yield_memory = O + 1;

            return Err(IVPStatus::Redo);
        }

        let mut higher_func = |bdf: &mut Self,
                               t: Self::RealField,
                               y: &[N],
                               data: &mut T|
         -> Result<BVector<N, D>, UserError> {
            bdf.scratch_pad = -(bdf.derivative)(t, y, data)? * bdf.dt * bdf.higher_coefficients[0];
            for (ind, &coeff) in bdf.higher_coefficients.column(0).iter().enumerate().skip(1) {
                bdf.scratch_pad += &bdf.prev_values[O - ind].1 * coeff;
            }
            Ok(
                bdf.scratch_pad.clone()
                    + BVector::from_column_slice_generic(bdf.dim, U1::name(), y),
            )
        };
        let mut lower_func = |bdf: &mut Self,
                              t: Self::RealField,
                              y: &[N],
                              data: &mut T|
         -> Result<BVector<N, D>, UserError> {
            bdf.scratch_pad = -(bdf.derivative)(t, y, data)? * bdf.dt * bdf.lower_coefficients[0];
            for (ind, &coeff) in bdf.higher_coefficients.column(0).iter().enumerate().skip(1) {
                bdf.scratch_pad += &bdf.prev_values[O - ind].1 * coeff;
            }
            Ok(
                bdf.scratch_pad.clone()
                    + BVector::from_column_slice_generic(bdf.dim, U1::name(), y),
            )
        };

        let higher_step = self.secant(&mut higher_func)?;
        let lower_step = self.secant(&mut lower_func)?;

        let difference = &higher_step - &lower_step;
        let error = difference.norm();

        if error <= self.tolerance.real() {
            self.state = higher_step;
            self.time += self.dt;

            // We have determined that this step passes the tolerance bounds.
            // If yield_memory is non-zero, then we still need to yield the runge-kutta
            // steps to the Iterator. We store the successful bdf step in self.state,
            // and self.time, decrement yield memory, and return (we never want to adjust the dt
            // the step after adjusting it down). We return IVPStatus::Redo so IVPIterator
            // calls again, yielding the runge-kutta steps.
            if self.yield_memory == O + 1 {
                self.yield_memory -= 1;
                return Err(IVPStatus::Redo);
            }

            self.prev_values
                .push_back((self.time.real(), self.state.clone()));
            self.prev_values.pop_front();

            if error < self.one_tenth.real() * self.tolerance.real() {
                self.dt *= self.two;
                if self.dt.real() > self.dt_max.real() {
                    self.dt = self.dt_max;
                }

                // Clear the saved steps since we have changed the timestep
                // so we can no longer use linear interpolation.
                self.prev_values.clear();
            }

            return Ok((self.time.real(), self.state.clone()));
        }

        // yield_memory can be Order+1 here, meaning we speculatively tried a timestep and the lower timestep
        // still didn't pass the tolerances.
        // In this case, we need to return the state to what it was previously, before the runge-kutta steps,
        // and reset the time to what it was previously.
        if self.yield_memory == O + 1 {
            // We took Order - 1 runge kutta steps at this dt
            self.time -= self.dt - self.order;
            self.state = self.save_state.clone();
        }

        self.dt *= self.half;

        if self.dt.real() < self.dt_min.real() {
            return Err(IVPStatus::Failure(IVPError::MinimumTimeDeltaExceeded));
        }

        self.prev_values.clear();
        Err(IVPStatus::Redo)
    }

    fn time(&self) -> Self::RealField {
        self.time.real()
    }
}

pub struct BDF6Coefficients<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField> BDFCoefficients<7> for BDF6Coefficients<N> {
    type RealField = N::RealField;

    fn higher_coefficients() -> Option<BSVector<Self::RealField, 7>> {
        let one_hundred_forty_seven = Self::RealField::from_u8(147)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(60)? / one_hundred_forty_seven.clone(),
            -Self::RealField::from_u16(360)? / one_hundred_forty_seven.clone(),
            Self::RealField::from_u16(450)? / one_hundred_forty_seven.clone(),
            -Self::RealField::from_u16(400)? / one_hundred_forty_seven.clone(),
            Self::RealField::from_u8(225)? / one_hundred_forty_seven.clone(),
            -Self::RealField::from_u8(72)? / one_hundred_forty_seven.clone(),
            Self::RealField::from_u8(10)? / one_hundred_forty_seven,
        ]))
    }

    fn lower_coefficients() -> Option<BSVector<Self::RealField, 7>> {
        let one_hundred_thirty_seven = Self::RealField::from_u8(137)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(60)? / one_hundred_thirty_seven.clone(),
            -Self::RealField::from_u16(300)? / one_hundred_thirty_seven.clone(),
            Self::RealField::from_u16(300)? / one_hundred_thirty_seven.clone(),
            -Self::RealField::from_u8(200)? / one_hundred_thirty_seven.clone(),
            Self::RealField::from_u8(75)? / one_hundred_thirty_seven.clone(),
            -Self::RealField::from_u8(12)? / one_hundred_thirty_seven,
            Self::RealField::zero(),
        ]))
    }
}

/// 6th order backwards differentiation formula method for
/// solving an initial value problem.
///
/// Defines the higher and lower order coefficients. Uses
/// BDFInfo for the actual solving.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, bdf::BDF6}};
///
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(-BSVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let bdf = BDF6::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(10.0)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///     let path = bdf.collect_vec()?;
///     for (time, state) in &path {
///         assert!(((-time).exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
pub type BDF6<'a, N, D, T, F> = BDF<'a, N, D, 7, T, F, BDF6Coefficients<N>>;

pub struct BDF2Coefficients<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField> BDFCoefficients<3> for BDF2Coefficients<N> {
    type RealField = N::RealField;

    fn higher_coefficients() -> Option<BSVector<Self::RealField, 3>> {
        let three = Self::RealField::from_u8(3)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(2)? / three.clone(),
            -Self::RealField::from_u8(4)? / three.clone(),
            three.recip(),
        ]))
    }

    fn lower_coefficients() -> Option<BSVector<Self::RealField, 3>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::one(),
            -Self::RealField::one(),
            Self::RealField::zero(),
        ]))
    }
}

/// 2nd order backwards differentiation formula method for
/// solving an initial value problem.
///
/// Defines the higher and lower order coefficients. Uses
/// BDFInfo for the actual solving.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, bdf::BDF2}};
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(-BSVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let bdf = BDF2::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.00001)?
///         .with_tolerance(0.00001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(10.0)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///     let path = bdf.collect_vec()?;
///     for (time, state) in &path {
///         assert!(((-time).exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
pub type BDF2<'a, N, D, T, F> = BDF<'a, N, D, 3, T, F, BDF2Coefficients<N>>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::BSVector;

    fn exp_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(y))
    }

    fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(&[-2.0 * t]))
    }

    fn sine_deriv(t: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_iterator(y.iter().map(|_| t.cos())))
    }

    fn unstable_deriv(_: f64, y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(-BSVector::from_column_slice(y))
    }

    #[test]
    fn bdf6_exp() {
        let t_initial = 0.0;
        let t_final = 7.0;

        let solver = BDF6::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
    fn bdf6_unstable() {
        let t_initial = 0.0;
        let t_final = 10.0;

        let solver = BDF6::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(unstable_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                (-step.0).exp(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn bdf6_quadratic() {
        let t_initial = 0.0;
        let t_final = 2.0;

        let solver = BDF6::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
    fn bdf6_sin() {
        let t_initial = 0.0;
        let t_final = 6.0;

        let solver = BDF6::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
    fn bdf2_exp() {
        let t_initial = 0.0;
        let t_final = 7.0;

        let solver = BDF2::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
    fn bdf2_unstable() {
        let t_initial = 0.0;
        let t_final = 10.0;

        let solver = BDF2::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_initial_conditions_slice(&[1.0])
            .unwrap()
            .with_derivative(unstable_deriv)
            .solve(())
            .unwrap();

        let path = solver.collect_vec().unwrap();

        for step in &path {
            assert!(approx_eq!(
                f64,
                step.1.column(0)[0],
                (-step.0).exp(),
                epsilon = 0.01
            ));
        }
    }

    #[test]
    fn bdf2_quadratic() {
        let t_initial = 0.0;
        let t_final = 1.0;

        let solver = BDF2::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
    fn bdf2_sin() {
        let t_initial = 0.0;
        let t_final = 6.0;

        let solver = BDF2::new()
            .unwrap()
            .with_minimum_dt(1e-5)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_tolerance(0.00001)
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
