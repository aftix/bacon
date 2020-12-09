/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

#[macro_use]
extern crate lazy_static;
#[allow(unused_imports)]
#[macro_use]
extern crate float_cmp;

extern crate num_complex;
extern crate num_traits;

pub mod constants;
pub mod ivp;
pub mod roots;
#[macro_use]
pub mod polynomial;

#[cfg(test)]
mod tests;
