/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */
use nalgebra::{Const, Dim, DimName, Dyn, Matrix, Owned, Vector};
use thiserror::Error;

#[cfg(test)]
#[macro_use]
extern crate float_cmp;

/// A possibly dynamically sized vector
pub type BVector<T, D> = Vector<T, D, Owned<T, D, Const<1>>>;
/// A statically sized vector
pub type BSVector<T, const D: usize> = Vector<T, Const<D>, Owned<T, Const<D>, Const<1>>>;
/// A possibly dynamically sized matrix
pub type BMatrix<T, R, C> = Matrix<T, R, C, Owned<T, R, C>>;
/// A statically sized vector
pub type BSMatrix<T, const R: usize, const C: usize> =
    Matrix<T, Const<R>, Const<C>, Owned<T, Const<R>, Const<C>>>;

pub mod prelude {
    pub use crate::{BSVector, BVector};
    pub use nalgebra;
    pub use nalgebra::{Const, DimName, Dyn, U1};
    pub use num_complex;
    pub use num_traits;
}

pub mod constants;
pub mod ivp;
pub mod roots;
#[macro_use]
pub mod polynomial;
pub mod differentiate;
pub mod integrate;
pub mod interp;
pub mod optimize;
pub mod special;

#[cfg(test)]
mod tests;

#[derive(Debug, Error)]
pub enum DimensionError {
    #[error("attempted to build a dynamic solver with static dimension")]
    DynamicOnStatic,
    #[error("attempted to build a static solver with dynamic dimension")]
    StaticOnDynamic,
}

pub trait Dimension: Dim {
    fn dim() -> Result<Self, DimensionError>;
    fn dim_dyn(size: usize) -> Result<Self, DimensionError>;
}

impl<const C: usize> Dimension for Const<C> {
    fn dim() -> Result<Self, DimensionError> {
        Ok(Self::name())
    }

    fn dim_dyn(_size: usize) -> Result<Self, DimensionError> {
        Err(DimensionError::DynamicOnStatic)
    }
}

impl Dimension for Dyn {
    fn dim() -> Result<Self, DimensionError> {
        Err(DimensionError::StaticOnDynamic)
    }

    fn dim_dyn(size: usize) -> Result<Self, DimensionError> {
        Ok(Self::from_usize(size))
    }
}
