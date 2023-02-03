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
pub const c: f64 = 299_792_458.0;

/// Permittivity of free space, ε₀, F/
pub const permittivity: f64 = 8.854_187_812_8e-12;
pub const permittivity_uncertainty: f64 = 0.000_000_001_3e-12;

/// Permeability of free space, μ₀, N/A^2
pub const permeability: f64 = 1.256_637_062_12e-6;
pub const permeability_uncertainty: f64 = 0.000_000_000_19e-6;

/// Plank constant, h, J
pub const h: f64 = 6.626_070_15e-34;

/// Reduced plank constant, ħ, J s
pub const h_bar: f64 = 1.054_571_817e-34;

/// Newtonian gravitational constant, G, m^3 / kg s^2
pub const G: f64 = 6.674_30e-11;
pub const G_uncertainty: f64 = 0.000_15e-11;

/// Gravitational acceleration on Earth's surface, m / s^2
pub const g: f64 = 9.806_65;

/// Elementary charge, e, C
pub const e_charge: f64 = 1.602_176_634e-19;

/// Molar gas constant, R, J / mol K
pub const R: f64 = 8.314_462_618;

/// Fine structure constant, α, unitless
pub const fine_structure: f64 = 7.297_352_569_3e-3;
pub const fine_structure_uncertainty: f64 = 0.000_000_001_1e-3;

/// Avogadro constant, N_A, 1/ mol
pub const avogadro: f64 = 6.022_140_76e23;

/// Boltzmann constant, k, J/K
pub const boltzmann: f64 = 1.380_649e-23;

/// Stefan-Boltzmann constant, σ, W / m^2 K^4
pub const stefan_boltzmann: f64 = 5.670_374_419e-8;

/// Wien displacement law constant, b, wavelength, m K
pub const wien: f64 = 2.897_771_955e-3;

/// Wien displacement law constant, b', frequency, Hz /K
pub const wien_frequency: f64 = 5.878_925_757e10;

/// Rydberg constant, R_inf, 1/m
pub const rydberg: f64 = 10_973_731.568_16;
pub const rydberg_uncertainty: f64 = 0.000_021;

/// Mass of an electron, kg
pub const electron_mass: f64 = 9.109_383_701_5e-31;
pub const electron_mass_uncertainty: f64 = 0.000_000_002_8e-31;

/// Mass of an proton kg
pub const proton_mass: f64 = 1.672_621_923_69e-27;
pub const proton_mass_uncertainty: f64 = 0.000_000_000_51e-27;

/// Mass of an neutron kg
pub const neutron_mass: f64 = 1.674_927_498_04e-27;
pub const neutron_mass_uncertainty: f64 = 0.000_000_000_95e-27;

/// NIST CODATA. Maps a string of a value name to a triplet
/// containing an f64 of its value, an f64 of its uncertainty,
/// and a string of its units
#[allow(clippy::inconsistent_digit_grouping)]
pub static CODATA: phf::Map<&'static str, (f64, f64, &'static str)> =
    include!(concat!(env!("OUT_DIR"), "/codata.rs"));
