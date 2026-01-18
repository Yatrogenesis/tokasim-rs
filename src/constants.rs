//! # Physical Constants
//!
//! All constants in SI units for plasma physics and fusion.

use std::f64::consts::PI;

// ============================================================================
// FUNDAMENTAL CONSTANTS
// ============================================================================

/// Speed of light (m/s)
pub const C: f64 = 299_792_458.0;

/// Vacuum permittivity ε₀ (F/m)
pub const EPSILON_0: f64 = 8.854_187_817e-12;

/// Vacuum permeability μ₀ (H/m)
pub const MU_0: f64 = 4.0 * PI * 1e-7;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602_176_634e-19;

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380_649e-23;

/// Planck constant (J·s)
pub const H_PLANCK: f64 = 6.626_070_15e-34;

/// Avogadro number (1/mol)
pub const N_A: f64 = 6.022_140_76e23;

// ============================================================================
// PARTICLE MASSES
// ============================================================================

/// Electron mass (kg)
pub const M_ELECTRON: f64 = 9.109_383_56e-31;

/// Proton mass (kg)
pub const M_PROTON: f64 = 1.672_621_898e-27;

/// Neutron mass (kg)
pub const M_NEUTRON: f64 = 1.674_927_471e-27;

/// Deuterium mass (kg) - approximately 2 * proton mass
pub const M_DEUTERIUM: f64 = 3.343_583_72e-27;

/// Tritium mass (kg) - approximately 3 * proton mass
pub const M_TRITIUM: f64 = 5.008_267_16e-27;

/// Helium-4 mass (kg)
pub const M_HELIUM4: f64 = 6.644_657_23e-27;

/// Alpha particle mass (same as He-4)
pub const M_ALPHA: f64 = M_HELIUM4;

// ============================================================================
// FUSION REACTION ENERGIES
// ============================================================================

/// D-T fusion total energy release (MeV)
pub const DT_FUSION_ENERGY_MEV: f64 = 17.6;

/// D-T fusion total energy release (J)
pub const DT_FUSION_ENERGY_J: f64 = DT_FUSION_ENERGY_MEV * 1.602_176_634e-13;

/// D-T alpha particle energy (MeV)
pub const DT_ALPHA_ENERGY_MEV: f64 = 3.5;

/// D-T neutron energy (MeV)
pub const DT_NEUTRON_ENERGY_MEV: f64 = 14.1;

/// D-D fusion energy (branch 1: He-3 + n) (MeV)
pub const DD_HE3_ENERGY_MEV: f64 = 3.27;

/// D-D fusion energy (branch 2: T + p) (MeV)
pub const DD_T_ENERGY_MEV: f64 = 4.03;

// ============================================================================
// PLASMA PHYSICS CONSTANTS
// ============================================================================

/// Electron volt in Joules
pub const EV_TO_J: f64 = 1.602_176_634e-19;

/// Kelvin to eV conversion (1 eV = 11604.5 K)
pub const EV_TO_KELVIN: f64 = 11604.5;

/// keV to Kelvin
pub const KEV_TO_KELVIN: f64 = EV_TO_KELVIN * 1000.0;

/// Debye length normalization constant
/// λ_D = sqrt(ε₀ * k_B * T_e / (n_e * e²))
pub const DEBYE_CONST: f64 = 69.0; // meters * sqrt(eV / m^-3)

/// Plasma frequency normalization
/// ω_pe = sqrt(n_e * e² / (ε₀ * m_e))
pub const PLASMA_FREQ_CONST: f64 = 56.4; // rad/s * sqrt(m^-3)

/// Cyclotron frequency for electrons
/// ω_ce = e * B / m_e
pub const ELECTRON_CYCLOTRON_CONST: f64 = E_CHARGE / M_ELECTRON;

/// Cyclotron frequency for protons
/// ω_ci = e * B / m_p
pub const ION_CYCLOTRON_CONST: f64 = E_CHARGE / M_PROTON;

/// Larmor radius normalization
/// r_L = m * v_perp / (|q| * B)
pub const LARMOR_CONST: f64 = M_PROTON / E_CHARGE;

// ============================================================================
// TOKAMAK TS-1 DESIGN PARAMETERS (Our optimized design)
// ============================================================================

/// Major radius R₀ (m)
pub const TS1_MAJOR_RADIUS: f64 = 1.5;

/// Minor radius a (m)
pub const TS1_MINOR_RADIUS: f64 = 0.6;

/// Aspect ratio A = R/a
pub const TS1_ASPECT_RATIO: f64 = TS1_MAJOR_RADIUS / TS1_MINOR_RADIUS;

/// Elongation κ
pub const TS1_ELONGATION: f64 = 1.97;

/// Triangularity δ
pub const TS1_TRIANGULARITY: f64 = 0.54;

/// Toroidal magnetic field B_t (T)
pub const TS1_TOROIDAL_FIELD: f64 = 25.0;

/// Plasma current I_p (MA)
pub const TS1_PLASMA_CURRENT_MA: f64 = 12.0;

/// Plasma current (A)
pub const TS1_PLASMA_CURRENT: f64 = TS1_PLASMA_CURRENT_MA * 1e6;

/// Target plasma density (m⁻³)
pub const TS1_PLASMA_DENSITY: f64 = 3e20;

/// Target plasma temperature (keV)
pub const TS1_TEMPERATURE_KEV: f64 = 15.0;

/// Target plasma temperature (Kelvin)
pub const TS1_TEMPERATURE_K: f64 = TS1_TEMPERATURE_KEV * KEV_TO_KELVIN;

/// Target fusion power (MW)
pub const TS1_FUSION_POWER_MW: f64 = 500.0;

/// Target Q factor (P_fusion / P_input)
pub const TS1_Q_FACTOR: f64 = 10.0;

/// Auxiliary heating power (MW)
pub const TS1_HEATING_POWER_MW: f64 = 50.0;

/// Plasma volume (m³) - approximate for elongated D-shape
pub const TS1_PLASMA_VOLUME: f64 = 2.0 * PI * PI * TS1_MAJOR_RADIUS
    * TS1_MINOR_RADIUS * TS1_MINOR_RADIUS * TS1_ELONGATION;

// ============================================================================
// SPARC REFERENCE PARAMETERS (for comparison)
// ============================================================================

/// SPARC major radius (m)
pub const SPARC_MAJOR_RADIUS: f64 = 1.85;

/// SPARC minor radius (m)
pub const SPARC_MINOR_RADIUS: f64 = 0.57;

/// SPARC toroidal field (T)
pub const SPARC_TOROIDAL_FIELD: f64 = 12.2;

/// SPARC plasma current (MA)
pub const SPARC_PLASMA_CURRENT_MA: f64 = 8.7;

/// SPARC target fusion power (MW)
pub const SPARC_FUSION_POWER_MW: f64 = 140.0;

// ============================================================================
// ITER REFERENCE PARAMETERS (for comparison)
// ============================================================================

/// ITER major radius (m)
pub const ITER_MAJOR_RADIUS: f64 = 6.2;

/// ITER minor radius (m)
pub const ITER_MINOR_RADIUS: f64 = 2.0;

/// ITER toroidal field (T)
pub const ITER_TOROIDAL_FIELD: f64 = 5.3;

/// ITER plasma current (MA)
pub const ITER_PLASMA_CURRENT_MA: f64 = 15.0;

/// ITER target fusion power (MW)
pub const ITER_FUSION_POWER_MW: f64 = 500.0;

// ============================================================================
// SIMULATION PARAMETERS
// ============================================================================

/// Default timestep for PIC simulation (s) - picosecond scale
pub const DEFAULT_TIMESTEP: f64 = 1e-12;

/// Default grid resolution (cells per minor radius)
pub const DEFAULT_GRID_RESOLUTION: usize = 100;

/// Default macro-particle weight (real particles per macro-particle)
pub const DEFAULT_MACRO_WEIGHT: f64 = 1e6;

/// CFL condition safety factor
pub const CFL_SAFETY: f64 = 0.5;

/// Maximum iterations for Poisson solver
pub const POISSON_MAX_ITER: usize = 1000;

/// Convergence tolerance for Poisson solver
pub const POISSON_TOLERANCE: f64 = 1e-8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fundamental_constants() {
        // Verify speed of light relation: c = 1/sqrt(ε₀μ₀)
        let c_computed = 1.0 / (EPSILON_0 * MU_0).sqrt();
        assert!((C - c_computed).abs() / C < 1e-6);
    }

    #[test]
    fn test_dt_fusion_energy() {
        // Verify total energy = alpha + neutron
        let total = DT_ALPHA_ENERGY_MEV + DT_NEUTRON_ENERGY_MEV;
        assert!((total - DT_FUSION_ENERGY_MEV).abs() < 0.1);
    }

    #[test]
    fn test_ts1_aspect_ratio() {
        assert!((TS1_ASPECT_RATIO - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_ts1_vs_sparc() {
        // TS-1 should have higher field
        assert!(TS1_TOROIDAL_FIELD > SPARC_TOROIDAL_FIELD);
        // TS-1 should be more compact
        assert!(TS1_MAJOR_RADIUS < SPARC_MAJOR_RADIUS);
    }
}
