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

pub mod constants;
pub mod ivp;
pub mod roots;
pub mod polynomial;

#[cfg(test)]
mod tests;
