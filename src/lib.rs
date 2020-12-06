#[macro_use]
extern crate lazy_static;
#[allow(unused_imports)]
#[macro_use]
extern crate float_cmp;

pub mod constants;
pub mod functions;
pub mod ivp;
pub mod roots;

#[cfg(test)]
mod tests;
