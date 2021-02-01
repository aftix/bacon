/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

#![allow(non_upper_case_globals)]
// Important values of note taken from SciPy
// Values taken from CODATA
// https://physics.nist.gov/cuu/Constants/index.html

/// Speed of light in a vacuum, m/s
pub const c: f64 = 299792458.0;

/// Permittivity of free space, ε₀, F/
pub const permittivity: f64 = 8.8541878128e-12;
pub const permittivity_uncertainty: f64 = 0.0000000013e-12;

/// Permeability of free space, μ₀, N/A^2
pub const permeability: f64 = 1.25663706212e-6;
pub const permeability_uncertainty: f64 = 0.00000000019e-6;

/// Plank constant, h, J
pub const h: f64 = 6.62607015e-34;

/// Reduced plank constant, ħ, J s
pub const h_bar: f64 = 1.054571817e-34;

/// Newtonian gravitational constant, G, m^3 / kg s^2
pub const G: f64 = 6.67430e-11;
pub const G_uncertainty: f64 = 0.00015e-11;

/// Gravitational acceleration on Earth's surface, m / s^2
pub const g: f64 = 9.80665;

/// Elementary charge, e, C
pub const e_charge: f64 = 1.602176634e-19;

/// Molar gas constant, R, J / mol K
pub const R: f64 = 8.314462618;

/// Fine structure constant, α, unitless
pub const fine_structure: f64 = 7.2973525693e-3;
pub const fine_structure_uncertainty: f64 = 0.0000000011e-3;

/// Avogadro constant, N_A, 1/ mol
pub const avogadro: f64 = 6.02214076e23;

/// Boltzmann constant, k, J/K
pub const boltzmann: f64 = 1.380649e-23;

/// Stefan-Boltzmann constant, σ, W / m^2 K^4
pub const stefan_boltzmann: f64 = 5.670374419e-8;

/// Wien displacement law constant, b, wavelength, m K
pub const wien: f64 = 2.897771955e-3;

/// Wien displacement law constant, b', frequency, Hz /K
pub const wien_frequency: f64 = 5.878925757e10;

/// Rydberg constant, R_inf, 1/m
pub const rydberg: f64 = 10973731.56816;
pub const rydberg_uncertainty: f64 = 0.000021;

/// Mass of an electron, kg
pub const electron_mass: f64 = 9.1093837015e-31;
pub const electron_mass_uncertainty: f64 = 0.0000000028e-31;

/// Mass of an proton kg
pub const proton_mass: f64 = 1.67262192369e-27;
pub const proton_mass_uncertainty: f64 = 0.00000000051e-27;

/// Mass of an neutron kg
pub const neutron_mass: f64 = 1.67492749804e-27;
pub const neutron_mass_uncertainty: f64 = 0.00000000095e-27;

/// NIST CODATA. Maps a string of a value name to a triplet
/// containing an f64 of its value, an f64 of its uncertainty,
/// and a string of its units
#[allow(clippy::inconsistent_digit_grouping)]
pub static CODATA: phf::Map<&'static str, (f64, f64, &'static str)> =
    include!(concat!(env!("OUT_DIR"), "/codata.rs"));
