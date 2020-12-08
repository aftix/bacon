/* This file is part of bacon.
 * Copyright (c) Wyatt Campbell.
 *
 * See repository LICENSE for information.
 */

#![allow(non_upper_case_globals)]
// Important values of note taken from SciPy
// Values taken from CODATA
// https://physics.nist.gov/cuu/Constants/index.html

// Speed of light in a vacuum, m/s
pub const c: f64 = 299792458.0;

// Permittivity of free space, ε₀, F/
pub const permittivity: f64 = 8.8541878128e-12;
pub const permittivity_uncertainty: f64 = 0.0000000013e-12;

// Permeability of free space, μ₀, N/A^2
pub const permeability: f64 =  1.25663706212e-6;
pub const permeability_uncertainty: f64 = 0.00000000019e-6;

// Plank constant, h, J
pub const h: f64 =  6.62607015e-34;

// Reduced plank constant, ħ, J s
pub const h_bar: f64 = 1.054571817e-34;

// Newtonian gravitational constant, G, m^3 / kg s^2
pub const G: f64 = 6.67430e-11;
pub const G_uncertainty: f64 = 0.00015e-11;

// Gravitational acceleration on Earth's surface, m / s^2
pub const g: f64 = 9.80665;

// Elementary charge, e, C
pub const e_charge: f64 = 1.602176634e-19;

// Molar gas constant, R, J / mol K
pub const R: f64 = 8.314462618;

// Fine structure constant, α, unitless
pub const fine_structure: f64 = 7.2973525693e-3;
pub const fine_structure_uncertainty: f64 = 0.0000000011e-3;

// Avogadro constant, N_A, 1/ mol
pub const avogadro: f64 = 6.02214076e23;

// Boltzmann constant, k, J/K
pub const boltzmann: f64 = 1.380649e-23;

// Stefan-Boltzmann constant, σ, W / m^2 K^4
pub const stefan_boltzmann: f64 = 5.670374419e-8;

// Wien displacement law constant, b, wavelength, m K
pub const wien: f64 = 2.897771955e-3;

// Wien displacement law constant, b', frequency, Hz /K
pub const wien_frequency: f64 = 5.878925757e10;

// Rydberg constant, R_inf, 1/m
pub const rydberg: f64 = 10973731.56816;
pub const rydberg_uncertainty: f64 = 0.000021;

// Mass of an electron, kg
pub const electron_mass: f64 = 9.1093837015e-31;
pub const electron_mass_uncertainty: f64 = 0.0000000028e-31;

// Mass of an proton kg
pub const proton_mass: f64 = 1.67262192369e-27;
pub const proton_mass_uncertainty: f64 = 0.00000000051e-27;

// Mass of an neutron kg
pub const neutron_mass: f64 = 1.67492749804e-27;
pub const neutron_mass_uncertainty: f64 = 0.00000000095e-27;

use std::collections::HashMap;

/* Fundamental Physical Constants --- Complete Listing
   2018 CODATA adjustment

  From:  http://physics.nist.gov/constants
*/
lazy_static! {
  pub static ref CODATA: HashMap<&'static str, (f64, f64, &'static str)> = {
    let mut m = HashMap::new();

    m.insert("alpha particle-electron mass ratio", (7294.29954142, 0.00000024, ""));
    m.insert("alpha particle mass", (6.6446573357e-27, 0.0000000020e-27, "kg"));
    m.insert("alpha particle mass energy equivalent", (5.9719201914e-10, 0.0000000018e-10, "J"));
    m.insert("alpha particle mass energy equivalent in MeV", (3727.3794066, 0.0000011, "MeV"));
    m.insert("alpha particle mass in u", (4.001506179127, 0.000000000063, "u"));
    m.insert("alpha particle molar mass", (4.0015061777e-3, 0.0000000012e-3, "kg mol^-1"));
    m.insert("alpha particle-proton mass ratio", (3.97259969009, 0.00000000022, ""));
    m.insert("alpha particle relative atomic mass", (4.001506179127, 0.000000000063, ""));
    m.insert("Angstrom star", (1.00001495e-10, 0.00000090e-10, "m"));
    m.insert("atomic mass constant", (1.66053906660e-27, 0.00000000050e-27, "kg"));
    m.insert("atomic mass constant energy equivalent", (1.49241808560e-10, 0.00000000045e-10, "J"));
    m.insert("atomic mass constant energy equivalent in MeV", (931.49410242, 0.00000028, "MeV"));
    m.insert("atomic mass unit-electron volt relationship", (9.3149410242e8, 0.0000000028e8, "eV"));
    m.insert("atomic mass unit-hartree relationship", (3.4231776874e7, 0.0000000010e7, "E_h"));
    m.insert("atomic mass unit-hertz relationship", (2.25234271871e23, 0.00000000068e23, "Hz"));
    m.insert("atomic mass unit-inverse meter relationship", (7.5130066104e14, 0.0000000023e14, "m^-1"));
    m.insert("atomic mass unit-joule relationship", (1.49241808560e-10, 0.00000000045e-10, "J"));
    m.insert("atomic mass unit-kelvin relationship", (1.08095401916e13, 0.00000000033e13, "K"));
    m.insert("atomic mass unit-kilogram relationship", (1.66053906660e-27, 0.00000000050e-27, "kg"));
    m.insert("atomic unit of 1st hyperpolarizability", (3.2063613061e-53, 0.0000000015e-53, "C^3 m^3 J^-2"));
    m.insert("atomic unit of 2nd hyperpolarizability", (6.2353799905e-65, 0.0000000038e-65, "C^4 m^4 J^-3"));
    m.insert("atomic unit of action", (1.054571817e-34, 0.0, "J s"));
    m.insert("atomic unit of charge", (1.602176634e-19, 0.0, "C"));
    m.insert("atomic unit of charge density", (1.08120238457e12, 0.00000000049e12, "C m^-3"));
    m.insert("atomic unit of current", (6.623618237510e-3, 0.000000000013e-3, "A"));
    m.insert("atomic unit of electric dipole mom.", (8.4783536255e-30, 0.0000000013e-30, "C m"));
    m.insert("atomic unit of electric field", (5.14220674763e11, 0.00000000078e11, "V m^-1"));
    m.insert("atomic unit of electric field gradient", (9.7173624292e21, 0.0000000029e21, "V m^-2"));
    m.insert("atomic unit of electric polarizability", (1.64877727436e-41, 0.00000000050e-41, "C^2 m^2 J^-1"));
    m.insert("atomic unit of electric potential", (27.211386245988, 0.000000000053, "V"));
    m.insert("atomic unit of electric quadrupole mom.", (4.4865515246e-40, 0.0000000014e-40, "C m^2"));
    m.insert("atomic unit of energy", (4.3597447222071e-18, 0.0000000000085e-18, "J"));
    m.insert("atomic unit of force", (8.2387234983e-8, 0.0000000012e-8, "N"));
    m.insert("atomic unit of length", (5.29177210903e-11, 0.00000000080e-11, "m"));
    m.insert("atomic unit of mag. dipole mom.", (1.85480201566e-23, 0.00000000056e-23, "J T^-1"));
    m.insert("atomic unit of mag. flux density", (2.35051756758e5, 0.00000000071e5, "T"));
    m.insert("atomic unit of magnetizability", (7.8910366008e-29, 0.0000000048e-29, "J T^-2"));
    m.insert("atomic unit of mass", (9.1093837015e-31, 0.0000000028e-31, "kg"));
    m.insert("atomic unit of momentum", (1.99285191410e-24, 0.00000000030e-24, "kg m s^-1"));
    m.insert("atomic unit of permittivity", (1.11265005545e-10, 0.00000000017e-10, "F m^-1"));
    m.insert("atomic unit of time", (2.4188843265857e-17, 0.0000000000047e-17, "s"));
    m.insert("atomic unit of velocity", (2.18769126364e6, 0.00000000033e6, "m s^-1"));
    m.insert("Avogadro constant", (6.02214076e23, 0.0, "mol^-1"));
    m.insert("Bohr magneton", (9.2740100783e-24, 0.0000000028e-24, "J T^-1"));
    m.insert("Bohr magneton in eV/T", (5.7883818060e-5, 0.0000000017e-5, "eV T^-1"));
    m.insert("Bohr magneton in Hz/T", (1.39962449361e10, 0.00000000042e10, "Hz T^-1"));
    m.insert("Bohr magneton in inverse meter per tesla", (46.686447783, 0.000000014, "m^-1 T^-1"));
    m.insert("Bohr magneton in K/T", (0.67171381563, 0.00000000020, "K T^-1"));
    m.insert("Bohr radius", (5.29177210903e-11, 0.00000000080e-11, "m"));
    m.insert("Boltzmann constant", (1.380649e-23, 0.0, "J K^-1"));
    m.insert("Boltzmann constant in eV/K", (8.617333262e-5, 0.0, "eV K^-1"));
    m.insert("Boltzmann constant in Hz/K", (2.083661912e10, 0.0, "Hz K^-1"));
    m.insert("Boltzmann constant in inverse meter per kelvin", (69.50348004, 0.0, "m^-1 K^-1"));
    m.insert("characteristic impedance of vacuum", (376.730313668, 0.000000057, "ohm"));
    m.insert("classical electron radius", (2.8179403262e-15, 0.0000000013e-15, "m"));
    m.insert("Compton wavelength", (2.42631023867e-12, 0.00000000073e-12, "m"));
    m.insert("conductance quantum", (7.748091729e-5, 0.0, "S"));
    m.insert("conventional value of ampere-90", (1.00000008887, 0.0, "A"));
    m.insert("conventional value of coulomb-90", (1.00000008887, 0.0, "C"));
    m.insert("conventional value of farad-90", (0.99999998220, 0.0, "F"));
    m.insert("conventional value of henry-90", (1.00000001779, 0.0, "H"));
    m.insert("conventional value of Josephson constant", (483597.9e9, 0.0, "Hz V^-1"));
    m.insert("conventional value of ohm-90", (1.00000001779, 0.0, "ohm"));
    m.insert("conventional value of volt-90", (1.00000010666, 0.0, "V"));
    m.insert("conventional value of von Klitzing constant", (25812.807, 0.0, "ohm"));
    m.insert("conventional value of watt-90", (1.00000019553, 0.0, "W"));
    m.insert("Copper x unit", (1.00207697e-13, 0.00000028e-13, "m"));
    m.insert("deuteron-electron mag. mom. ratio", (-4.664345551e-4, 0.000000012e-4, ""));
    m.insert("deuteron-electron mass ratio", (3670.48296788, 0.00000013, ""));
    m.insert("deuteron g factor", (0.8574382338, 0.0000000022, ""));
    m.insert("deuteron mag. mom.", (4.330735094e-27, 0.000000011e-27, "J T^-1"));
    m.insert("deuteron mag. mom. to Bohr magneton ratio", (4.669754570e-4, 0.000000012e-4, ""));
    m.insert("deuteron mag. mom. to nuclear magneton ratio", (0.8574382338, 0.0000000022, ""));
    m.insert("deuteron mass", (3.3435837724e-27, 0.0000000010e-27, "kg"));
    m.insert("deuteron mass energy equivalent", (3.00506323102e-10, 0.00000000091e-10, "J"));
    m.insert("deuteron mass energy equivalent in MeV", (1875.61294257, 0.00000057, "MeV"));
    m.insert("deuteron mass in u", (2.013553212745, 0.000000000040, "u"));
    m.insert("deuteron molar mass", (2.01355321205e-3, 0.00000000061e-3, "kg mol^-1"));
    m.insert("deuteron-neutron mag. mom. ratio", (-0.44820653, 0.00000011, ""));
    m.insert("deuteron-proton mag. mom. ratio", (0.30701220939, 0.00000000079, ""));
    m.insert("deuteron-proton mass ratio", (1.99900750139, 0.00000000011, ""));
    m.insert("deuteron relative atomic mass", (2.013553212745, 0.000000000040, ""));
    m.insert("deuteron rms charge radius", (2.12799e-15, 0.00074e-15, "m"));
    m.insert("electron charge to mass quotient", (-1.75882001076e11, 0.00000000053e11, "C kg^-1"));
    m.insert("electron-deuteron mag. mom. ratio", (-2143.9234915, 0.0000056, ""));
    m.insert("electron-deuteron mass ratio", (2.724437107462e-4, 0.000000000096e-4, ""));
    m.insert("electron g factor", (-2.00231930436256, 0.00000000000035, ""));
    m.insert("electron gyromag. ratio", (1.76085963023e11, 0.00000000053e11, "s^-1 T^-1"));
    m.insert("electron gyromag. ratio in MHz/T", (28024.9514242, 0.0000085, "MHz T^-1"));
    m.insert("electron-helion mass ratio", (1.819543074573e-4, 0.000000000079e-4, ""));
    m.insert("electron mag. mom.", (-9.2847647043e-24, 0.0000000028e-24, "J T^-1"));
    m.insert("electron mag. mom. anomaly", (1.15965218128e-3, 0.00000000018e-3, ""));
    m.insert("electron mag. mom. to Bohr magneton ratio", (-1.00115965218128, 0.00000000000018, ""));
    m.insert("electron mag. mom. to nuclear magneton ratio", (-1838.28197188, 0.00000011, ""));
    m.insert("electron mass", (9.1093837015e-31, 0.0000000028e-31, "kg"));
    m.insert("electron mass energy equivalent", (8.1871057769e-14, 0.0000000025e-14, "J"));
    m.insert("electron mass energy equivalent in MeV", (0.51099895000, 0.00000000015, "MeV"));
    m.insert("electron mass in u", (5.48579909065e-4, 0.00000000016e-4, "u"));
    m.insert("electron molar mass", (5.4857990888e-7, 0.0000000017e-7, "kg mol^-1"));
    m.insert("electron-muon mag. mom. ratio", (206.7669883, 0.0000046, ""));
    m.insert("electron-muon mass ratio", (4.83633169e-3, 0.00000011e-3, ""));
    m.insert("electron-neutron mag. mom. ratio", (960.92050, 0.00023, ""));
    m.insert("electron-neutron mass ratio", (5.4386734424e-4, 0.0000000026e-4, ""));
    m.insert("electron-proton mag. mom. ratio", (-658.21068789, 0.00000020, ""));
    m.insert("electron-proton mass ratio", (5.44617021487e-4, 0.00000000033e-4, ""));
    m.insert("electron relative atomic mass", (5.48579909065e-4, 0.00000000016e-4, ""));
    m.insert("electron-tau mass ratio", (2.87585e-4, 0.00019e-4, ""));
    m.insert("electron to alpha particle mass ratio", (1.370933554787e-4, 0.000000000045e-4, ""));
    m.insert("electron to shielded helion mag. mom. ratio", (864.058257, 0.000010, ""));
    m.insert("electron to shielded proton mag. mom. ratio", (-658.2275971, 0.0000072, ""));
    m.insert("electron-triton mass ratio", (1.819200062251e-4, 0.000000000090e-4, ""));
    m.insert("electron volt", (1.602176634e-19, 0.0, "J"));
    m.insert("electron volt-atomic mass unit relationship", (1.07354410233e-9, 0.00000000032e-9, "u"));
    m.insert("electron volt-hartree relationship", (3.6749322175655e-2, 0.0000000000071e-2, "E_h"));
    m.insert("electron volt-hertz relationship", (2.417989242e14, 0.0, "Hz"));
    m.insert("electron volt-inverse meter relationship", (8.065543937e5, 0.0, "m^-1"));
    m.insert("electron volt-joule relationship", (1.602176634e-19, 0.0, "J"));
    m.insert("electron volt-kelvin relationship", (1.160451812e4, 0.0, "K"));
    m.insert("electron volt-kilogram relationship", (1.782661921e-36, 0.0, "kg"));
    m.insert("elementary charge", (1.602176634e-19, 0.0, "C"));
    m.insert("elementary charge over h-bar", (1.519267447e15, 0.0, "A J^-1"));
    m.insert("Faraday constant", (96485.33212, 0.0, "C mol^-1"));
    m.insert("Fermi coupling constant", (1.1663787e-5, 0.0000006e-5, "GeV^-2"));
    m.insert("fine-structure constant", (7.2973525693e-3, 0.0000000011e-3, ""));
    m.insert("first radiation constant", (3.741771852e-16, 0.0, "W m^2"));
    m.insert("first radiation constant for spectral radiance", (1.191042972e-16, 0.0, "W m^2 sr^-1"));
    m.insert("hartree-atomic mass unit relationship", (2.92126232205e-8, 0.00000000088e-8, "u"));
    m.insert("hartree-electron volt relationship", (27.211386245988, 0.000000000053, "eV"));
    m.insert("Hartree energy", (4.3597447222071e-18, 0.0000000000085e-18, "J"));
    m.insert("Hartree energy in eV", (27.211386245988, 0.000000000053, "eV"));
    m.insert("hartree-hertz relationship", (6.579683920502e15, 0.000000000013e15, "Hz"));
    m.insert("hartree-inverse meter relationship", (2.1947463136320e7, 0.0000000000043e7, "m^-1"));
    m.insert("hartree-joule relationship", (4.3597447222071e-18, 0.0000000000085e-18, "J"));
    m.insert("hartree-kelvin relationship", (3.1577502480407e5, 0.0000000000061e5, "K"));
    m.insert("hartree-kilogram relationship", (4.8508702095432e-35, 0.0000000000094e-35, "kg"));
    m.insert("helion-electron mass ratio", (5495.88528007, 0.00000024, ""));
    m.insert("helion g factor", (-4.255250615, 0.000000050, ""));
    m.insert("helion mag. mom.", (-1.074617532e-26, 0.000000013e-26, "J T^-1"));
    m.insert("helion mag. mom. to Bohr magneton ratio", (-1.158740958e-3, 0.000000014e-3, ""));
    m.insert("helion mag. mom. to nuclear magneton ratio", (-2.127625307, 0.000000025, ""));
    m.insert("helion mass", (5.0064127796e-27, 0.0000000015e-27, "kg"));
    m.insert("helion mass energy equivalent", (4.4995394125e-10, 0.0000000014e-10, "J"));
    m.insert("helion mass energy equivalent in MeV", (2808.39160743, 0.00000085, "MeV"));
    m.insert("helion mass in u", (3.014932247175, 0.000000000097, "u"));
    m.insert("helion molar mass", (3.01493224613e-3, 0.00000000091e-3, "kg mol^-1"));
    m.insert("helion-proton mass ratio", (2.99315267167, 0.00000000013, ""));
    m.insert("helion relative atomic mass", (3.014932247175, 0.000000000097, ""));
    m.insert("helion shielding shift", (5.996743e-5, 0.000010e-5, ""));
    m.insert("hertz-atomic mass unit relationship", (4.4398216652e-24, 0.0000000013e-24, "u"));
    m.insert("hertz-electron volt relationship", (4.135667696e-15, 0.0, "eV"));
    m.insert("hertz-hartree relationship", (1.5198298460570e-16, 0.0000000000029e-16, "E_h"));
    m.insert("hertz-inverse meter relationship", (3.335640951e-9, 0.0, "m^-1"));
    m.insert("hertz-joule relationship", (6.62607015e-34, 0.0, "J"));
    m.insert("hertz-kelvin relationship", (4.799243073e-11, 0.0, "K"));
    m.insert("hertz-kilogram relationship", (7.372497323e-51, 0.0, "kg"));
    m.insert("hyperfine transition frequency of Cs-133", (9192631770.0, 0.0, "Hz"));
    m.insert("inverse fine-structure constant", (137.035999084, 0.000000021, ""));
    m.insert("inverse meter-atomic mass unit relationship", (1.33102505010e-15, 0.00000000040e-15, "u"));
    m.insert("inverse meter-electron volt relationship", (1.239841984e-6, 0.0, "eV"));
    m.insert("inverse meter-hartree relationship", (4.5563352529120e-8, 0.0000000000088e-8, "E_h"));
    m.insert("inverse meter-hertz relationship", (299792458.0, 0.0, "Hz"));
    m.insert("inverse meter-joule relationship", (1.986445857e-25, 0.0, "J"));
    m.insert("inverse meter-kelvin relationship", (1.438776877e-2, 0.0, "K"));
    m.insert("inverse meter-kilogram relationship", (2.210219094e-42, 0.0, "kg"));
    m.insert("inverse of conductance quantum", (12906.40372, 0.0, "ohm"));
    m.insert("Josephson constant", (483597.8484e9, 0.0, "Hz V^-1"));
    m.insert("joule-atomic mass unit relationship", (6.7005352565e9, 0.0000000020e9, "u"));
    m.insert("joule-electron volt relationship", (6.241509074e18, 0.0, "eV"));
    m.insert("joule-hartree relationship", (2.2937122783963e17, 0.0000000000045e17, "E_h"));
    m.insert("joule-hertz relationship", (1.509190179e33, 0.0, "Hz"));
    m.insert("joule-inverse meter relationship", (5.034116567e24, 0.0, "m^-1"));
    m.insert("joule-kelvin relationship", (7.242970516e22, 0.0, "K"));
    m.insert("joule-kilogram relationship", (1.112650056e-17, 0.0, "kg"));
    m.insert("kelvin-atomic mass unit relationship", (9.2510873014e-14, 0.0000000028e-14, "u"));
    m.insert("kelvin-electron volt relationship", (8.617333262e-5, 0.0, "eV"));
    m.insert("kelvin-hartree relationship", (3.1668115634556e-6, 0.0000000000061e-6, "E_h"));
    m.insert("kelvin-hertz relationship", (2.083661912e10, 0.0, "Hz"));
    m.insert("kelvin-inverse meter relationship", (69.50348004, 0.0, "m^-1"));
    m.insert("kelvin-joule relationship", (1.380649e-23, 0.0, "J"));
    m.insert("kelvin-kilogram relationship", (1.536179187e-40, 0.0, "kg"));
    m.insert("kilogram-atomic mass unit relationship", (6.0221407621e26, 0.0000000018e26, "u"));
    m.insert("kilogram-electron volt relationship", (5.609588603e35, 0.0, "eV"));
    m.insert("kilogram-hartree relationship", (2.0614857887409e34, 0.0000000000040e34, "E_h"));
    m.insert("kilogram-hertz relationship", (1.356392489e50, 0.0, "Hz"));
    m.insert("kilogram-inverse meter relationship", (4.524438335e41, 0.0, "m^-1"));
    m.insert("kilogram-joule relationship", (8.987551787e16, 0.0, "J"));
    m.insert("kilogram-kelvin relationship", (6.509657260e39, 0.0, "K"));
    m.insert("lattice parameter of silicon", (5.431020511e-10, 0.000000089e-10, "m"));
    m.insert("lattice spacing of ideal Si (220)", (1.920155716e-10, 0.000000032e-10, "m"));
    m.insert("Loschmidt constant (273.15 K,  100 kPa)", (2.651645804e25, 0.0, "m^-3"));
    m.insert("Loschmidt constant (273.15 K,  101.325 kPa)", (2.686780111e25, 0.0, "m^-3"));
    m.insert("luminous efficacy", (683.0, 0.0, "lm W^-1"));
    m.insert("mag. flux quantum", (2.067833848e-15, 0.0, "Wb"));
    m.insert("molar gas constant", (8.314462618, 0.0, "J mol^-1 K^-1"));
    m.insert("molar mass constant", (0.99999999965e-3, 0.00000000030e-3, "kg mol^-1"));
    m.insert("molar mass of carbon-12", (11.9999999958e-3, 0.0000000036e-3, "kg mol^-1"));
    m.insert("molar Planck constant", (3.990312712e-10, 0.0, "J Hz^-1 mol^-1"));
    m.insert("molar volume of ideal gas (273.15 K,  100 kPa)", (22.71095464e-3, 0.0, "m^3 mol^-1"));
    m.insert("molar volume of ideal gas (273.15 K,  101.325 kPa)", (22.41396954e-3, 0.0, "m^3 mol^-1"));
    m.insert("molar volume of silicon", (1.205883199e-5, 0.000000060e-5, "m^3 mol^-1"));
    m.insert("Molybdenum x unit", (1.00209952e-13, 0.00000053e-13, "m"));
    m.insert("muon Compton wavelength", (1.173444110e-14, 0.000000026e-14, "m"));
    m.insert("muon-electron mass ratio", (206.7682830, 0.0000046, ""));
    m.insert("muon g factor", (-2.0023318418, 0.0000000013, ""));
    m.insert("muon mag. mom.", (-4.49044830e-26, 0.00000010e-26, "J T^-1"));
    m.insert("muon mag. mom. anomaly", (1.16592089e-3, 0.00000063e-3, ""));
    m.insert("muon mag. mom. to Bohr magneton ratio", (-4.84197047e-3, 0.00000011e-3, ""));
    m.insert("muon mag. mom. to nuclear magneton ratio", (-8.89059703, 0.00000020, ""));
    m.insert("muon mass", (1.883531627e-28, 0.000000042e-28, "kg"));
    m.insert("muon mass energy equivalent", (1.692833804e-11, 0.000000038e-11, "J"));
    m.insert("muon mass energy equivalent in MeV", (105.6583755, 0.0000023, "MeV"));
    m.insert("muon mass in u", (0.1134289259, 0.0000000025, "u"));
    m.insert("muon molar mass", (1.134289259e-4, 0.000000025e-4, "kg mol^-1"));
    m.insert("muon-neutron mass ratio", (0.1124545170, 0.0000000025, ""));
    m.insert("muon-proton mag. mom. ratio", (-3.183345142, 0.000000071, ""));
    m.insert("muon-proton mass ratio", (0.1126095264, 0.0000000025, ""));
    m.insert("muon-tau mass ratio", (5.94635e-2, 0.00040e-2, ""));
    m.insert("natural unit of action", (1.054571817e-34, 0.0, "J s"));
    m.insert("natural unit of action in eV s", (6.582119569e-16, 0.0, "eV s"));
    m.insert("natural unit of energy", (8.1871057769e-14, 0.0000000025e-14, "J"));
    m.insert("natural unit of energy in MeV", (0.51099895000, 0.00000000015, "MeV"));
    m.insert("natural unit of length", (3.8615926796e-13, 0.0000000012e-13, "m"));
    m.insert("natural unit of mass", (9.1093837015e-31, 0.0000000028e-31, "kg"));
    m.insert("natural unit of momentum", (2.73092453075e-22, 0.00000000082e-22, "kg m s^-1"));
    m.insert("natural unit of momentum in MeV/c", (0.51099895000, 0.00000000015, "MeV/c"));
    m.insert("natural unit of time", (1.28808866819e-21, 0.00000000039e-21, "s"));
    m.insert("natural unit of velocity", (299792458.0, 0.0, "m s^-1"));
    m.insert("neutron Compton wavelength", (1.31959090581e-15, 0.00000000075e-15, "m"));
    m.insert("neutron-electron mag. mom. ratio", (1.04066882e-3, 0.00000025e-3, ""));
    m.insert("neutron-electron mass ratio", (1838.68366173, 0.00000089, ""));
    m.insert("neutron g factor", (-3.82608545, 0.00000090, ""));
    m.insert("neutron gyromag. ratio", (1.83247171e8, 0.00000043e8, "s^-1 T^-1"));
    m.insert("neutron gyromag. ratio in MHz/T", (29.1646931, 0.0000069, "MHz T^-1"));
    m.insert("neutron mag. mom.", (-9.6623651e-27, 0.0000023e-27, "J T^-1"));
    m.insert("neutron mag. mom. to Bohr magneton ratio", (-1.04187563e-3, 0.00000025e-3, ""));
    m.insert("neutron mag. mom. to nuclear magneton ratio", (-1.91304273, 0.00000045, ""));
    m.insert("neutron mass", (1.67492749804e-27, 0.00000000095e-27, "kg"));
    m.insert("neutron mass energy equivalent", (1.50534976287e-10, 0.00000000086e-10, "J"));
    m.insert("neutron mass energy equivalent in MeV", (939.56542052, 0.00000054, "MeV"));
    m.insert("neutron mass in u", (1.00866491595, 0.00000000049, "u"));
    m.insert("neutron molar mass", (1.00866491560e-3, 0.00000000057e-3, "kg mol^-1"));
    m.insert("neutron-muon mass ratio", (8.89248406, 0.00000020, ""));
    m.insert("neutron-proton mag. mom. ratio", (-0.68497934, 0.00000016, ""));
    m.insert("neutron-proton mass difference", (2.30557435e-30, 0.00000082e-30, "kg"));
    m.insert("neutron-proton mass difference energy equivalent", (2.07214689e-13, 0.00000074e-13, "J"));
    m.insert("neutron-proton mass difference energy equivalent in MeV", (1.29333236, 0.00000046, "MeV"));
    m.insert("neutron-proton mass difference in u", (1.38844933e-3, 0.00000049e-3, "u"));
    m.insert("neutron-proton mass ratio", (1.00137841931, 0.00000000049, ""));
    m.insert("neutron relative atomic mass", (1.00866491595, 0.00000000049, ""));
    m.insert("neutron-tau mass ratio", (0.528779, 0.000036, ""));
    m.insert("neutron to shielded proton mag. mom. ratio", (-0.68499694, 0.00000016, ""));
    m.insert("Newtonian constant of gravitation", (6.67430e-11, 0.00015e-11, "m^3 kg^-1 s^-2"));
    m.insert("Newtonian constant of gravitation over h-bar c", (6.70883e-39, 0.00015e-39, "(GeV/c^2)^-2"));
    m.insert("nuclear magneton", (5.0507837461e-27, 0.0000000015e-27, "J T^-1"));
    m.insert("nuclear magneton in eV/T", (3.15245125844e-8, 0.00000000096e-8, "eV T^-1"));
    m.insert("nuclear magneton in inverse meter per tesla", (2.54262341353e-2, 0.00000000078e-2, "m^-1 T^-1"));
    m.insert("nuclear magneton in K/T", (3.6582677756e-4, 0.0000000011e-4, "K T^-1"));
    m.insert("nuclear magneton in MHz/T", (7.6225932291, 0.0000000023, "MHz T^-1"));
    m.insert("Planck constant", (6.62607015e-34, 0.0, "J Hz^-1"));
    m.insert("Planck constant in eV/Hz", (4.135667696e-15, 0.0, "eV Hz^-1"));
    m.insert("Planck length", (1.616255e-35, 0.000018e-35, "m"));
    m.insert("Planck mass", (2.176434e-8, 0.000024e-8, "kg"));
    m.insert("Planck mass energy equivalent in GeV", (1.220890e19, 0.000014e19, "GeV"));
    m.insert("Planck temperature", (1.416784e32, 0.000016e32, "K"));
    m.insert("Planck time", (5.391247e-44, 0.000060e-44, "s"));
    m.insert("proton charge to mass quotient", (9.5788331560e7, 0.0000000029e7, "C kg^-1"));
    m.insert("proton Compton wavelength", (1.32140985539e-15, 0.00000000040e-15, "m"));
    m.insert("proton-electron mass ratio", (1836.15267343, 0.00000011, ""));
    m.insert("proton g factor", (5.5856946893, 0.0000000016, ""));
    m.insert("proton gyromag. ratio", (2.6752218744e8, 0.0000000011e8, "s^-1 T^-1"));
    m.insert("proton gyromag. ratio in MHz/T", (42.577478518, 0.000000018, "MHz T^-1"));
    m.insert("proton mag. mom.", (1.41060679736e-26, 0.00000000060e-26, "J T^-1"));
    m.insert("proton mag. mom. to Bohr magneton ratio", (1.52103220230e-3, 0.00000000046e-3, ""));
    m.insert("proton mag. mom. to nuclear magneton ratio", (2.79284734463, 0.00000000082, ""));
    m.insert("proton mag. shielding correction", (2.5689e-5, 0.0011e-5, ""));
    m.insert("proton mass", (1.67262192369e-27, 0.00000000051e-27, "kg"));
    m.insert("proton mass energy equivalent", (1.50327761598e-10, 0.00000000046e-10, "J"));
    m.insert("proton mass energy equivalent in MeV", (938.27208816, 0.00000029, "MeV"));
    m.insert("proton mass in u", (1.007276466621, 0.000000000053, "u"));
    m.insert("proton molar mass", (1.00727646627e-3, 0.00000000031e-3, "kg mol^-1"));
    m.insert("proton-muon mass ratio", (8.88024337, 0.00000020, ""));
    m.insert("proton-neutron mag. mom. ratio", (-1.45989805, 0.00000034, ""));
    m.insert("proton-neutron mass ratio", (0.99862347812, 0.00000000049, ""));
    m.insert("proton relative atomic mass", (1.007276466621, 0.000000000053, ""));
    m.insert("proton rms charge radius", (8.414e-16, 0.019e-16, "m"));
    m.insert("proton-tau mass ratio", (0.528051, 0.000036, ""));
    m.insert("quantum of circulation", (3.6369475516e-4, 0.0000000011e-4, "m^2 s^-1"));
    m.insert("quantum of circulation times 2", (7.2738951032e-4, 0.0000000022e-4, "m^2 s^-1"));
    m.insert("reduced Compton wavelength", (3.8615926796e-13, 0.0000000012e-13, "m"));
    m.insert("reduced muon Compton wavelength", (1.867594306e-15, 0.000000042e-15, "m"));
    m.insert("reduced neutron Compton wavelength", (2.1001941552e-16, 0.0000000012e-16, "m"));
    m.insert("reduced Planck constant", (1.054571817e-34, 0.0, "J s"));
    m.insert("reduced Planck constant in eV s", (6.582119569e-16, 0.0, "eV s"));
    m.insert("reduced Planck constant times c in MeV fm", (197.3269804, 0.0, "MeV fm"));
    m.insert("reduced proton Compton wavelength", (2.10308910336e-16, 0.00000000064e-16, "m"));
    m.insert("reduced tau Compton wavelength", (1.110538e-16, 0.000075e-16, "m"));
    m.insert("Rydberg constant", (10973731.568160, 0.000021, "m^-1"));
    m.insert("Rydberg constant times c in Hz", (3.2898419602508e15, 0.0000000000064e15, "Hz"));
    m.insert("Rydberg constant times hc in eV", (13.605693122994, 0.000000000026, "eV"));
    m.insert("Rydberg constant times hc in J", (2.1798723611035e-18, 0.0000000000042e-18, "J"));
    m.insert("Sackur-Tetrode constant (1 K,  100 kPa)", (-1.15170753706, 0.00000000045, ""));
    m.insert("Sackur-Tetrode constant (1 K,  101.325 kPa)", (-1.16487052358, 0.00000000045, ""));
    m.insert("second radiation constant", (1.438776877e-2, 0.0, "m K"));
    m.insert("shielded helion gyromag. ratio", (2.037894569e8, 0.000000024e8, "s^-1 T^-1"));
    m.insert("shielded helion gyromag. ratio in MHz/T", (32.43409942, 0.00000038, "MHz T^-1"));
    m.insert("shielded helion mag. mom.", (-1.074553090e-26, 0.000000013e-26, "J T^-1"));
    m.insert("shielded helion mag. mom. to Bohr magneton ratio", (-1.158671471e-3, 0.000000014e-3, ""));
    m.insert("shielded helion mag. mom. to nuclear magneton ratio", (-2.127497719, 0.000000025, ""));
    m.insert("shielded helion to proton mag. mom. ratio", (-0.7617665618, 0.0000000089, ""));
    m.insert("shielded helion to shielded proton mag. mom. ratio", (-0.7617861313, 0.0000000033, ""));
    m.insert("shielded proton gyromag. ratio", (2.675153151e8, 0.000000029e8, "s^-1 T^-1"));
    m.insert("shielded proton gyromag. ratio in MHz/T", (42.57638474, 0.00000046, "MHz T^-1"));
    m.insert("shielded proton mag. mom.", (1.410570560e-26, 0.000000015e-26, "J T^-1"));
    m.insert("shielded proton mag. mom. to Bohr magneton ratio", (1.520993128e-3, 0.000000017e-3, ""));
    m.insert("shielded proton mag. mom. to nuclear magneton ratio", (2.792775599, 0.000000030, ""));
    m.insert("shielding difference of d and p in HD", (2.0200e-8, 0.0020e-8, ""));
    m.insert("shielding difference of t and p in HT", (2.4140e-8, 0.0020e-8, ""));
    m.insert("speed of light in vacuum", (299792458.0, 0.0, "m s^-1"));
    m.insert("standard acceleration of gravity", (9.80665, 0.0, "m s^-2"));
    m.insert("standard atmosphere", (101325.0, 0.0, "Pa"));
    m.insert("standard-state pressure", (100000.0, 0.0, "Pa"));
    m.insert("Stefan-Boltzmann constant", (5.670374419e-8, 0.0, "W m^-2 K^-4"));
    m.insert("tau Compton wavelength", (6.97771e-16, 0.00047e-16, "m"));
    m.insert("tau-electron mass ratio", (3477.23, 0.23, ""));
    m.insert("tau energy equivalent", (1776.86, 0.12, "MeV"));
    m.insert("tau mass", (3.16754e-27, 0.00021e-27, "kg"));
    m.insert("tau mass energy equivalent", (2.84684e-10, 0.00019e-10, "J"));
    m.insert("tau mass in u", (1.90754, 0.00013, "u"));
    m.insert("tau molar mass", (1.90754e-3, 0.00013e-3, "kg mol^-1"));
    m.insert("tau-muon mass ratio", (16.8170, 0.0011, ""));
    m.insert("tau-neutron mass ratio", (1.89115, 0.00013, ""));
    m.insert("tau-proton mass ratio", (1.89376, 0.00013, ""));
    m.insert("Thomson cross section", (6.6524587321e-29, 0.0000000060e-29, "m^2"));
    m.insert("triton-electron mass ratio", (5496.92153573, 0.00000027, ""));
    m.insert("triton g factor", (5.957924931, 0.000000012, ""));
    m.insert("triton mag. mom.", (1.5046095202e-26, 0.0000000030e-26, "J T^-1"));
    m.insert("triton mag. mom. to Bohr magneton ratio", (1.6223936651e-3, 0.0000000032e-3, ""));
    m.insert("triton mag. mom. to nuclear magneton ratio", (2.9789624656, 0.0000000059, ""));
    m.insert("triton mass", (5.0073567446e-27, 0.0000000015e-27, "kg"));
    m.insert("triton mass energy equivalent", (4.5003878060e-10, 0.0000000014e-10, "J"));
    m.insert("triton mass energy equivalent in MeV", (2808.92113298, 0.00000085, "MeV"));
    m.insert("triton mass in u", (3.01550071621, 0.00000000012, "u"));
    m.insert("triton molar mass", (3.01550071517e-3, 0.00000000092e-3, "kg mol^-1"));
    m.insert("triton-proton mass ratio", (2.99371703414, 0.00000000015, ""));
    m.insert("triton relative atomic mass", (3.01550071621, 0.00000000012, ""));
    m.insert("triton to proton mag. mom. ratio", (1.0666399191, 0.0000000021, ""));
    m.insert("unified atomic mass unit", (1.66053906660e-27, 0.00000000050e-27, "kg"));
    m.insert("vacuum electric permittivity", (8.8541878128e-12, 0.0000000013e-12, "F m^-1"));
    m.insert("vacuum mag. permeability", (1.25663706212e-6, 0.00000000019e-6, "N A^-2"));
    m.insert("von Klitzing constant", (25812.80745, 0.0, "ohm"));
    m.insert("weak mixing angle", (0.22290, 0.00030, ""));
    m.insert("Wien frequency displacement law constant", (5.878925757e10, 0.0, "Hz K^-1"));
    m.insert("Wien wavelength displacement law constant", (2.897771955e-3, 0.0, "m K"));
    m.insert("W to Z mass ratio", (0.88153, 0.00017, ""));

    m
  };
}
