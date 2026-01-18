//! # Core Types
//!
//! Fundamental types for the TOKASIM simulator.

use std::fmt;

// ============================================================================
// VECTOR TYPES (3D)
// ============================================================================

/// 3D vector for positions, velocities, fields
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create new vector
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Unit vectors
    pub const fn unit_x() -> Self { Self::new(1.0, 0.0, 0.0) }
    pub const fn unit_y() -> Self { Self::new(0.0, 1.0, 0.0) }
    pub const fn unit_z() -> Self { Self::new(0.0, 0.0, 1.0) }

    /// Magnitude squared
    pub fn mag_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Magnitude
    pub fn mag(&self) -> f64 {
        self.mag_squared().sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let m = self.mag();
        if m > 1e-15 {
            Self::new(self.x / m, self.y / m, self.z / m)
        } else {
            Self::zero()
        }
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Scale by scalar
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Add vectors
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Subtract vectors
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new(self * v.x, self * v.y, self * v.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.6e}, {:.6e}, {:.6e})", self.x, self.y, self.z)
    }
}

// ============================================================================
// PARTICLE SPECIES
// ============================================================================

/// Particle species in the plasma
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Species {
    /// Electron
    Electron,
    /// Deuterium ion (D+)
    Deuterium,
    /// Tritium ion (T+)
    Tritium,
    /// Helium-4 ion / Alpha particle (He++)
    Alpha,
    /// Helium-3 ion (He3++)
    Helium3,
    /// Proton (H+)
    Proton,
    /// Neutron (for tracking)
    Neutron,
    /// Impurity (generic)
    Impurity { z: i32, a: u32 },
}

impl Species {
    /// Get mass in kg
    pub fn mass(&self) -> f64 {
        use crate::constants::*;
        match self {
            Species::Electron => M_ELECTRON,
            Species::Deuterium => M_DEUTERIUM,
            Species::Tritium => M_TRITIUM,
            Species::Alpha => M_ALPHA,
            Species::Helium3 => 5.008e-27, // ~3 * proton
            Species::Proton => M_PROTON,
            Species::Neutron => M_NEUTRON,
            Species::Impurity { a, .. } => *a as f64 * M_PROTON,
        }
    }

    /// Get charge in Coulombs
    pub fn charge(&self) -> f64 {
        use crate::constants::E_CHARGE;
        match self {
            Species::Electron => -E_CHARGE,
            Species::Deuterium => E_CHARGE,
            Species::Tritium => E_CHARGE,
            Species::Alpha => 2.0 * E_CHARGE,
            Species::Helium3 => 2.0 * E_CHARGE,
            Species::Proton => E_CHARGE,
            Species::Neutron => 0.0,
            Species::Impurity { z, .. } => *z as f64 * E_CHARGE,
        }
    }

    /// Get charge number (Z)
    pub fn z(&self) -> i32 {
        match self {
            Species::Electron => -1,
            Species::Deuterium => 1,
            Species::Tritium => 1,
            Species::Alpha => 2,
            Species::Helium3 => 2,
            Species::Proton => 1,
            Species::Neutron => 0,
            Species::Impurity { z, .. } => *z,
        }
    }

    /// Get mass number (A)
    pub fn a(&self) -> u32 {
        match self {
            Species::Electron => 0,
            Species::Deuterium => 2,
            Species::Tritium => 3,
            Species::Alpha => 4,
            Species::Helium3 => 3,
            Species::Proton => 1,
            Species::Neutron => 1,
            Species::Impurity { a, .. } => *a,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Species::Electron => "electron",
            Species::Deuterium => "deuterium",
            Species::Tritium => "tritium",
            Species::Alpha => "alpha",
            Species::Helium3 => "helium-3",
            Species::Proton => "proton",
            Species::Neutron => "neutron",
            Species::Impurity { .. } => "impurity",
        }
    }
}

// ============================================================================
// TOKAMAK CONFIGURATION
// ============================================================================

/// Configuration for a tokamak device
#[derive(Debug, Clone)]
pub struct TokamakConfig {
    /// Name of the device
    pub name: String,

    // Geometry
    /// Major radius R₀ (m)
    pub major_radius: f64,
    /// Minor radius a (m)
    pub minor_radius: f64,
    /// Plasma elongation κ
    pub elongation: f64,
    /// Plasma triangularity δ
    pub triangularity: f64,

    // Magnetic field
    /// Toroidal field at R₀ (T)
    pub toroidal_field: f64,
    /// Plasma current (A)
    pub plasma_current: f64,

    // Plasma parameters
    /// Target electron density (m⁻³)
    pub density: f64,
    /// Target ion temperature (keV)
    pub ion_temperature_kev: f64,
    /// Target electron temperature (keV)
    pub electron_temperature_kev: f64,

    // Heating
    /// ICRF heating power (MW)
    pub icrf_power_mw: f64,
    /// ECRH heating power (MW)
    pub ecrh_power_mw: f64,
    /// NBI heating power (MW)
    pub nbi_power_mw: f64,

    // Fuel mix
    /// Deuterium fraction (0-1)
    pub deuterium_fraction: f64,
    /// Tritium fraction (0-1)
    pub tritium_fraction: f64,
}

impl TokamakConfig {
    /// Create TS-1 configuration (our optimized design)
    pub fn ts1() -> Self {
        use crate::constants::*;
        Self {
            name: "TS-1".to_string(),
            major_radius: TS1_MAJOR_RADIUS,
            minor_radius: TS1_MINOR_RADIUS,
            elongation: TS1_ELONGATION,
            triangularity: TS1_TRIANGULARITY,
            toroidal_field: TS1_TOROIDAL_FIELD,
            plasma_current: TS1_PLASMA_CURRENT,
            density: TS1_PLASMA_DENSITY,
            ion_temperature_kev: TS1_TEMPERATURE_KEV,
            electron_temperature_kev: TS1_TEMPERATURE_KEV,
            icrf_power_mw: 25.0,
            ecrh_power_mw: 10.0,
            nbi_power_mw: 15.0,
            deuterium_fraction: 0.5,
            tritium_fraction: 0.5,
        }
    }

    /// Create SPARC-like configuration
    pub fn sparc() -> Self {
        use crate::constants::*;
        Self {
            name: "SPARC".to_string(),
            major_radius: SPARC_MAJOR_RADIUS,
            minor_radius: SPARC_MINOR_RADIUS,
            elongation: 1.8,
            triangularity: 0.4,
            toroidal_field: SPARC_TOROIDAL_FIELD,
            plasma_current: SPARC_PLASMA_CURRENT_MA * 1e6,
            density: 1.8e20,
            ion_temperature_kev: 10.0,
            electron_temperature_kev: 10.0,
            icrf_power_mw: 25.0,
            ecrh_power_mw: 0.0,
            nbi_power_mw: 0.0,
            deuterium_fraction: 0.5,
            tritium_fraction: 0.5,
        }
    }

    /// Create ITER-like configuration
    pub fn iter() -> Self {
        use crate::constants::*;
        Self {
            name: "ITER".to_string(),
            major_radius: ITER_MAJOR_RADIUS,
            minor_radius: ITER_MINOR_RADIUS,
            elongation: 1.7,
            triangularity: 0.33,
            toroidal_field: ITER_TOROIDAL_FIELD,
            plasma_current: ITER_PLASMA_CURRENT_MA * 1e6,
            density: 1.0e20,
            ion_temperature_kev: 8.0,
            electron_temperature_kev: 8.5,
            icrf_power_mw: 20.0,
            ecrh_power_mw: 20.0,
            nbi_power_mw: 33.0,
            deuterium_fraction: 0.5,
            tritium_fraction: 0.5,
        }
    }

    /// Calculate aspect ratio A = R/a
    pub fn aspect_ratio(&self) -> f64 {
        self.major_radius / self.minor_radius
    }

    /// Calculate plasma volume (m³) - approximate for D-shaped plasma
    pub fn plasma_volume(&self) -> f64 {
        use std::f64::consts::PI;
        2.0 * PI * PI * self.major_radius
            * self.minor_radius * self.minor_radius
            * self.elongation
    }

    /// Calculate total heating power (MW)
    pub fn total_heating_mw(&self) -> f64 {
        self.icrf_power_mw + self.ecrh_power_mw + self.nbi_power_mw
    }
}

// ============================================================================
// SIMULATION STATE
// ============================================================================

/// Current state of the simulation
#[derive(Debug, Clone)]
pub struct SimulationState {
    /// Current simulation time (s)
    pub time: f64,
    /// Timestep number
    pub step: u64,
    /// Total kinetic energy (J)
    pub kinetic_energy: f64,
    /// Total field energy (J)
    pub field_energy: f64,
    /// Total fusion power (W)
    pub fusion_power: f64,
    /// Fusion events this timestep
    pub fusion_count: u64,
    /// Number of particles
    pub particle_count: usize,
    /// Status
    pub status: SimulationStatus,
}

/// Simulation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationStatus {
    /// Initializing
    Initializing,
    /// Running normally
    Running,
    /// Paused
    Paused,
    /// Disruption detected
    Disruption,
    /// Completed
    Completed,
    /// Error
    Error,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            time: 0.0,
            step: 0,
            kinetic_energy: 0.0,
            field_energy: 0.0,
            fusion_power: 0.0,
            fusion_count: 0,
            particle_count: 0,
            status: SimulationStatus::Initializing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        // Addition
        let c = a + b;
        assert!((c.x - 5.0).abs() < 1e-10);

        // Dot product
        let dot = a.dot(&b);
        assert!((dot - 32.0).abs() < 1e-10);

        // Cross product
        let cross = a.cross(&b);
        assert!((cross.x - (-3.0)).abs() < 1e-10);
        assert!((cross.y - 6.0).abs() < 1e-10);
        assert!((cross.z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_species() {
        assert!(Species::Electron.charge() < 0.0);
        assert!(Species::Deuterium.charge() > 0.0);
        assert_eq!(Species::Alpha.z(), 2);
        assert_eq!(Species::Tritium.a(), 3);
    }

    #[test]
    fn test_tokamak_config() {
        let ts1 = TokamakConfig::ts1();
        assert!((ts1.aspect_ratio() - 2.5).abs() < 0.01);
        assert!(ts1.plasma_volume() > 0.0);
    }
}
