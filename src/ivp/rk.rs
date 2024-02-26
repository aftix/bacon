/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

use super::{Derivative, IVPError, IVPIterator, IVPSolver, IVPStatus, IVPStepper, Step};
use crate::{BMatrix, BSMatrix, BSVector, BVector, Dimension};
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, Dim, DimName, RealField, U1,
};
use num_traits::{FromPrimitive, One, Zero};
use std::marker::PhantomData;

/// This trait defines a Runge-Kutta solver
/// The RungeKutta struct takes an implemetation of this trait
/// as a type argument since the algorithm is the same for
/// all the methods, just the order and these functions
/// need to be different.
pub trait RungeKuttaCoefficients<const O: usize> {
    /// The real field associated with the solver's Field.
    type RealField: RealField;

    /// Returns a vec of coeffecients to multiply the time step by when getting
    /// intermediate results. Upper-left portion of Butch Tableaux
    fn t_coefficients() -> Option<BSVector<Self::RealField, O>>;

    /// Returns the coefficients to use on the k_i's when finding another
    /// k_i. Upper-right portion of the Butch Tableax. Should be
    /// an NxN-1 matrix, where N is the order of the Runge-Kutta Method (Or order+1 for
    /// adaptive methods)
    fn k_coefficients() -> Option<BSMatrix<Self::RealField, O, O>>;

    /// Coefficients to use when calculating the final step to take.
    /// These are the weights of the weighted average of k_i's. Bottom
    /// portion of the Butch Tableaux. For adaptive methods, this is the first
    /// row of the bottom portion.
    fn avg_coefficients() -> Option<BSVector<Self::RealField, O>>;

    /// Coefficients to use on
    /// the k_i's to find the error between the two orders
    /// of Runge-Kutta methods. In the Butch Tableaux, this is
    /// the first row of the bottom portion minus the second row.
    fn error_coefficients() -> Option<BSVector<Self::RealField, O>>;
}

/// The nuts and bolts Runge-Kutta solver
/// Users won't use this directly if they aren't defining their own Runge-Kutta solver
/// Used as a common struct for the specific implementations
pub struct RungeKutta<'a, N, D, const O: usize, T, F, R>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    R: RungeKuttaCoefficients<O, RealField = N::RealField>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, Const<O>>,
{
    init_dt_max: Option<N::RealField>,
    init_dt_min: Option<N::RealField>,
    init_time: Option<N::RealField>,
    init_end: Option<N::RealField>,
    init_tolerance: Option<N::RealField>,
    init_state: Option<BVector<N, D>>,
    init_derivative: Option<F>,
    dim: D,
    _data: PhantomData<&'a (T, R)>,
}

/// The solver for any Runge-Kutta method
/// Users should not use this type directly, and should
/// instead get it from a specific RungeKutta struct
/// (wrapped in an IVPIterator)
pub struct RungeKuttaSolver<'a, N, D, const O: usize, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, Const<O>>,
    DefaultAllocator: Allocator<N, D, Const<O>>,
{
    // Parameters set by the user
    dt_max: N,
    dt_min: N,
    time: N,
    end: N,
    tolerance: N,
    derivative: F,
    data: T,

    // The current state of the solver
    dt: N,
    state: BVector<N, D>,

    // Per-order constants set by RungeKuttaCoefficients
    t_coefficients: BSVector<N, O>,
    k_coefficients: BSMatrix<N, O, O>,
    avg_coefficients: BSVector<N, O>,
    error_coefficients: BSVector<N, O>,

    // Scratch space to store the partial steps needed for the algorithm
    half_steps: BMatrix<N, D, Const<O>>,
    step: BVector<N, D>,
    scratch_pad: BVector<N, D>,

    // Constants needed for algorithm
    one_tenth: N,
    one_fourth: N,
    point_eighty_four: N,
    four: N,

    _lifetime: PhantomData<&'a ()>,
}

impl<'a, N, D, const O: usize, T, F, R> IVPSolver<'a, D> for RungeKutta<'a, N, D, O, T, F, R>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    R: RungeKuttaCoefficients<O, RealField = N::RealField>,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, Const<O>>,
    DefaultAllocator: Allocator<N, D, Const<O>>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type Derivative = F;
    type UserData = T;
    type Solver = RungeKuttaSolver<'a, N, D, O, T, F>;

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

    fn new_dyn(size: usize) -> Result<Self, IVPError> {
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
        let half = Self::Field::one() / two;

        let one_tenth =
            Self::Field::one() / Self::Field::from_u8(10).ok_or(IVPError::FromPrimitiveFailure)?;
        let four = Self::Field::from_u8(4).ok_or(IVPError::FromPrimitiveFailure)?;
        let one_fourth = Self::Field::one() / four;

        let one_hundred = Self::Field::from_u8(100).ok_or(IVPError::FromPrimitiveFailure)?;
        let eighty_four = Self::Field::from_u8(100).ok_or(IVPError::FromPrimitiveFailure)?;
        let point_eighty_four = eighty_four / one_hundred;

        let t_coefficients = BSVector::from_iterator(
            R::t_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let k_coefficients = BSMatrix::<N, O, O>::from_iterator_generic(
            <Const<O> as Dim>::from_usize(O),
            <Const<O> as Dim>::from_usize(O),
            R::k_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let avg_coefficients = BSVector::from_iterator(
            R::avg_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        let error_coefficients = BSVector::from_iterator(
            R::error_coefficients()
                .ok_or(IVPError::FromPrimitiveFailure)?
                .as_slice()
                .iter()
                .cloned()
                .map(Self::Field::from_real),
        );

        Ok(IVPIterator {
            solver: RungeKuttaSolver {
                dt_max: Self::Field::from_real(dt_max.clone()),
                dt_min: Self::Field::from_real(dt_min.clone()),
                time: Self::Field::from_real(time),
                end: Self::Field::from_real(end),
                tolerance: Self::Field::from_real(tolerance),
                dt: Self::Field::from_real(dt_max + dt_min) * half,
                state,
                derivative,
                data,
                t_coefficients,
                k_coefficients,
                avg_coefficients,
                error_coefficients,
                half_steps: BMatrix::from_element_generic(
                    self.dim,
                    <Const<O> as DimName>::name(),
                    Self::Field::zero(),
                ),
                scratch_pad: BVector::from_element_generic(
                    self.dim,
                    U1::name(),
                    Self::Field::zero(),
                ),
                step: BVector::from_element_generic(self.dim, U1::name(), Self::Field::zero()),
                one_tenth,
                one_fourth,
                point_eighty_four,
                four,
                _lifetime: PhantomData,
            },
            finished: false,
            _dim: PhantomData,
        })
    }
}

impl<'a, N, D, const O: usize, T, F> IVPStepper<D> for RungeKuttaSolver<'a, N, D, O, T, F>
where
    D: Dimension,
    N: ComplexField + Copy,
    T: Clone,
    F: Derivative<N, D, T> + 'a,
    DefaultAllocator: Allocator<N, D>,
    DefaultAllocator: Allocator<N, Const<O>>,
    DefaultAllocator: Allocator<N, D, Const<O>>,
{
    type Error = IVPError;
    type Field = N;
    type RealField = N::RealField;
    type UserData = T;

    fn step(&mut self) -> Step<Self::RealField, Self::Field, D, Self::Error> {
        if self.time.real() >= self.end.real() {
            return Err(IVPStatus::Done);
        }

        if self.time.real() + self.dt.real() >= self.end.real() {
            self.dt = self.end - self.time;
        }

        for (i, k_row) in self.k_coefficients.row_iter().enumerate() {
            self.scratch_pad = self.state.clone();
            for (j, &k_coeff) in k_row.iter().enumerate() {
                self.scratch_pad += self.half_steps.column(j) * k_coeff;
            }

            let step_time = self.time + self.t_coefficients[i] * self.dt;
            self.step = (self.derivative)(
                step_time.real(),
                self.scratch_pad.as_slice(),
                &mut self.data.clone(),
            )? * self.dt;

            self.half_steps.set_column(i, &self.step);
        }

        self.scratch_pad = self.half_steps.column(0) * self.error_coefficients[0];
        for (ind, &e_coeff) in self.error_coefficients.iter().enumerate().skip(1) {
            self.scratch_pad += self.half_steps.column(ind) * e_coeff;
        }
        let error = self.scratch_pad.norm() / self.dt.real();

        if error <= self.tolerance.real() {
            self.time += self.dt;

            for (ind, &avg_coeff) in self.avg_coefficients.iter().enumerate() {
                self.state += self.half_steps.column(ind) * avg_coeff;
            }
        }

        let delta = self.point_eighty_four.real()
            * (self.tolerance.real() / error.clone()).powf(self.one_fourth.real());
        if delta <= self.one_tenth.real() {
            self.dt *= self.one_tenth;
        } else if delta >= self.four.real() {
            self.dt *= self.four;
        } else {
            self.dt *= Self::Field::from_real(delta);
        }

        if self.dt.real() > self.dt_max.real() {
            self.dt = self.dt_max;
        }

        if self.dt.real() < self.dt_min.real() && self.time.real() < self.end.real() {
            return Err(IVPStatus::Failure(IVPError::MinimumTimeDeltaExceeded));
        }

        if error <= self.tolerance.real() {
            Ok((self.time.real(), self.state.clone()))
        } else {
            Err(IVPStatus::Redo)
        }
    }

    fn time(&self) -> Self::RealField {
        self.time.real()
    }
}

pub struct RKCoefficients45<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField> RungeKuttaCoefficients<6> for RKCoefficients45<N> {
    type RealField = N::RealField;

    fn t_coefficients() -> Option<BSVector<Self::RealField, 6>> {
        let one_fourth = Self::RealField::from_u8(4)?.recip();
        let one_half = Self::RealField::from_u8(2)?.recip();
        let three = Self::RealField::from_u8(3)?;
        let eight = Self::RealField::from_u8(8)?;
        let twelve = Self::RealField::from_u8(12)?;
        let thirteen = Self::RealField::from_u8(13)?;

        Some(BSVector::from_column_slice(&[
            Self::RealField::zero(),
            one_fourth,
            three / eight,
            twelve / thirteen,
            Self::RealField::one(),
            one_half,
        ]))
    }

    fn k_coefficients() -> Option<BSMatrix<Self::RealField, 6, 6>> {
        let zero = Self::RealField::zero();
        let one_fourth = Self::RealField::from_u8(4)?.recip();
        let thirty_two = Self::RealField::from_u8(32)?;
        let two_one_nine_seven = Self::RealField::from_u16(2197)?;

        Some(BSMatrix::from_vec(vec![
            // Row 0
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 1
            one_fourth,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 2
            Self::RealField::from_u8(3)? / thirty_two.clone(),
            Self::RealField::from_u8(9)? / thirty_two.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 3
            Self::RealField::from_u16(1932)? / two_one_nine_seven.clone(),
            -Self::RealField::from_u16(7200)? / two_one_nine_seven.clone(),
            Self::RealField::from_u16(7296)? / two_one_nine_seven,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 4
            Self::RealField::from_u16(439)? / Self::RealField::from_u8(216)?,
            -Self::RealField::from_u8(8)?,
            Self::RealField::from_u16(3680)? / Self::RealField::from_u16(513)?,
            -Self::RealField::from_u16(845)? / Self::RealField::from_u16(4104)?,
            zero.clone(),
            zero.clone(),
            // Row 5
            -Self::RealField::from_u8(8)? / Self::RealField::from_u8(27)?,
            Self::RealField::from_u8(2)?,
            -Self::RealField::from_u16(3544)? / Self::RealField::from_u16(2565)?,
            Self::RealField::from_u16(1859)? / Self::RealField::from_u16(4014)?,
            -Self::RealField::from_u8(11)? / Self::RealField::from_u8(40)?,
            zero,
        ]))
    }

    fn avg_coefficients() -> Option<BSVector<Self::RealField, 6>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(25)? / Self::RealField::from_u8(216)?,
            Self::RealField::zero(),
            Self::RealField::from_u16(1408)? / Self::RealField::from_u16(2565)?,
            Self::RealField::from_u16(2197)? / Self::RealField::from_u16(4104)?,
            -Self::RealField::from_u8(5)?.recip(),
            Self::RealField::zero(),
        ]))
    }

    fn error_coefficients() -> Option<BSVector<Self::RealField, 6>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u16(360)?.recip(),
            Self::RealField::from_f64(0.0).unwrap(),
            Self::RealField::from_f64(-128.0 / 4275.0).unwrap(),
            Self::RealField::from_f64(-2197.0 / 75240.0).unwrap(),
            Self::RealField::from_f64(1.0 / 50.0).unwrap(),
            Self::RealField::from_f64(2.0 / 55.0).unwrap(),
        ]))
    }
}

/// Runge-Kutta-Fehlberg method for solving an IVP.
///
/// Defines the Butch Tableaux for a 5(4) order adaptive
/// runge-kutta method. Uses RungeKutta to do the actual solving.
/// Provides an implementation of the IVPSolver trait.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, rk::RungeKutta45}};
///
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(BSVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let rk45 = RungeKutta45::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(10.0)?
///         .with_tolerance(0.0001)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///
///     let path = rk45.collect_vec()?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
pub type RungeKutta45<'a, N, D, T, F> = RungeKutta<'a, N, D, 6, T, F, RKCoefficients45<N>>;

pub struct RK23Coefficients<N: ComplexField>(PhantomData<N>);

impl<N: ComplexField> RungeKuttaCoefficients<4> for RK23Coefficients<N> {
    type RealField = N::RealField;

    fn t_coefficients() -> Option<BSVector<Self::RealField, 4>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::zero(),
            Self::RealField::from_u8(2)?.recip(),
            Self::RealField::from_u8(3)? / Self::RealField::from_u8(4)?,
            Self::RealField::one(),
        ]))
    }

    fn k_coefficients() -> Option<BSMatrix<Self::RealField, 4, 4>> {
        let zero = Self::RealField::zero();

        Some(BSMatrix::from_vec(vec![
            // Row 0
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 1
            Self::RealField::from_u8(2)?.recip(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            // Row 2
            zero.clone(),
            Self::RealField::from_u8(3)? / Self::RealField::from_u8(4)?,
            zero.clone(),
            zero.clone(),
            // Row 3
            Self::RealField::from_u8(2)? / Self::RealField::from_u8(9)?,
            Self::RealField::from_u8(3)?.recip(),
            Self::RealField::from_u8(4)? / Self::RealField::from_u8(9)?,
            zero,
        ]))
    }

    fn avg_coefficients() -> Option<BSVector<Self::RealField, 4>> {
        Some(BSVector::from_column_slice(&[
            Self::RealField::from_u8(2)? / Self::RealField::from_u8(9)?,
            Self::RealField::from_u8(3)?.recip(),
            Self::RealField::from_u8(4)? / Self::RealField::from_u8(9)?,
            Self::RealField::zero(),
        ]))
    }

    fn error_coefficients() -> Option<BSVector<Self::RealField, 4>> {
        Some(BSVector::from_column_slice(&[
            -Self::RealField::from_u8(5)? / Self::RealField::from_u8(72)?,
            Self::RealField::from_u8(12)?.recip(),
            Self::RealField::from_u8(9)?.recip(),
            Self::RealField::from_u8(8)?.recip(),
        ]))
    }
}

/// Bogacki-Shampine method for solving an IVP.
///
/// Defines the Butch Tableaux for a 5(4) order adaptive
/// runge-kutta method. Uses RungeKutta to do the actual solving.
/// Provides an implementation of the IVPSolver trait.
///
/// # Examples
/// ```
/// use std::error::Error;
/// use bacon_sci::{BSVector, ivp::{IVPSolver, IVPError, rk::RungeKutta23}};
///
/// fn derivatives(_t: f64, state: &[f64], _p: &mut ()) -> Result<BSVector<f64, 1>, Box<dyn Error>> {
///     Ok(BSVector::from_column_slice(state))
/// }
///
/// fn example() -> Result<(), IVPError> {
///     let rk23 = RungeKutta23::new()?
///         .with_maximum_dt(0.1)?
///         .with_minimum_dt(0.001)?
///         .with_initial_time(0.0)?
///         .with_ending_time(10.0)?
///         .with_tolerance(0.0001)?
///         .with_initial_conditions_slice(&[1.0])?
///         .with_derivative(derivatives)
///         .solve(())?;
///
///     let path = rk23.collect_vec()?;
///     for (time, state) in &path {
///         assert!((time.exp() - state.column(0)[0]).abs() < 0.001);
///     }
///     Ok(())
/// }
/// ```
pub type RungeKutta23<'a, N, D, T, F> = RungeKutta<'a, N, D, 4, T, F, RK23Coefficients<N>>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{ivp::UserError, BSVector};

    fn quadratic_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(&[-2.0 * t]))
    }

    fn sine_deriv(t: f64, _y: &[f64], _: &mut ()) -> Result<BSVector<f64, 1>, UserError> {
        Ok(BSVector::from_column_slice(&[t.cos()]))
    }

    #[test]
    fn rungekutta45_quadratic() {
        let t_initial = 0.0;
        let t_final = 10.0;

        let solver = RungeKutta45::new()
            .unwrap()
            .with_minimum_dt(0.0001)
            .unwrap()
            .with_maximum_dt(0.1)
            .unwrap()
            .with_initial_time(t_initial)
            .unwrap()
            .with_ending_time(t_final)
            .unwrap()
            .with_tolerance(1e-5)
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
                epsilon = 0.0001
            ));
        }
    }

    #[test]
    fn rungekutta45_sine() {
        let t_initial = 0.0;
        let t_final = 10.0;

        let solver = RungeKutta45::new()
            .unwrap()
            .with_minimum_dt(0.001)
            .unwrap()
            .with_maximum_dt(0.01)
            .unwrap()
            .with_tolerance(0.0001)
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
