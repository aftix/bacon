/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

#[allow(unused_imports)]
extern crate num_complex;
extern crate num_traits;

#[cfg(test)]
#[macro_use]
extern crate float_cmp;

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
