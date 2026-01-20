//! # Plasma-Wall Interaction (PWI) Module
//!
//! Detailed modeling of plasma-surface interactions including:
//! - Physical sputtering (momentum transfer)
//! - Chemical sputtering/erosion
//! - Surface layer formation (deposition)
//! - Hydrogen isotope retention
//! - Material mixing
//!
//! ## Physical Processes
//!
//! ### Sputtering
//! ```text
//! Incident Ion → Surface → Ejected Atoms
//!     E_i, θ       W/Be/C    Y(E,θ) atoms/ion
//! ```
//!
//! ### Chemical Erosion (Carbon)
//! ```text
//! D⁺ + C → CD₄ (methane)
//! T⁺ + C → CT₄
//! ```
//!
//! ### Surface Layer Formation
//! ```text
//!       Incident flux
//!           ↓
//!   ┌───────────────────┐
//!   │ Mixed layer (nm)  │ ← Co-deposited species
//!   ├───────────────────┤
//!   │ Implantation zone │ ← H/D/T retention
//!   ├───────────────────┤
//!   │ Bulk material     │
//!   └───────────────────┘
//! ```
//!
//! ## References
//!
//! - Eckstein, "Computer Simulation of Ion-Solid Interactions" (1991)
//! - Roth et al., "Chemical erosion of carbon" (2004)
//! - TRIM/SRIM (Stopping and Range of Ions in Matter)
//! - Behrisch & Eckstein, "Sputtering by Particle Bombardment" (2007)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use crate::types::Vec3;
use std::collections::HashMap;

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// Atomic mass unit (kg)
pub const AMU: f64 = 1.66054e-27;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.60218e-19;

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.38065e-23;

/// Avogadro's number
pub const N_A: f64 = 6.02214e23;

// ============================================================================
// MATERIAL DEFINITIONS
// ============================================================================

/// Wall material types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallMaterial {
    Tungsten,
    Carbon,       // CFC (Carbon Fiber Composite)
    Beryllium,
    Lithium,      // Liquid Li coating
    Molybdenum,
    Steel,        // EUROFER/SS316
}

/// Material properties for sputtering calculations
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub material: WallMaterial,
    /// Atomic number Z
    pub z: f64,
    /// Atomic mass (amu)
    pub mass_amu: f64,
    /// Surface binding energy (eV)
    pub surface_binding_energy: f64,
    /// Lattice binding energy (eV)
    pub lattice_binding_energy: f64,
    /// Displacement energy (eV)
    pub displacement_energy: f64,
    /// Density (atoms/m³)
    pub density: f64,
    /// Melting point (K)
    pub melting_point: f64,
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
}

impl MaterialProperties {
    pub fn tungsten() -> Self {
        Self {
            material: WallMaterial::Tungsten,
            z: 74.0,
            mass_amu: 183.84,
            surface_binding_energy: 8.68,  // eV
            lattice_binding_energy: 3.0,
            displacement_energy: 90.0,
            density: 6.3e28,  // atoms/m³
            melting_point: 3695.0,  // K
            thermal_conductivity: 173.0,
        }
    }

    pub fn carbon() -> Self {
        Self {
            material: WallMaterial::Carbon,
            z: 6.0,
            mass_amu: 12.01,
            surface_binding_energy: 7.41,
            lattice_binding_energy: 3.0,
            displacement_energy: 28.0,
            density: 1.13e29,
            melting_point: 3823.0,  // Sublimation
            thermal_conductivity: 200.0,  // CFC parallel to fibers
        }
    }

    pub fn beryllium() -> Self {
        Self {
            material: WallMaterial::Beryllium,
            z: 4.0,
            mass_amu: 9.01,
            surface_binding_energy: 3.32,
            lattice_binding_energy: 3.0,
            displacement_energy: 15.0,
            density: 1.24e29,
            melting_point: 1560.0,
            thermal_conductivity: 200.0,
        }
    }

    pub fn lithium() -> Self {
        Self {
            material: WallMaterial::Lithium,
            z: 3.0,
            mass_amu: 6.94,
            surface_binding_energy: 1.63,
            lattice_binding_energy: 1.0,
            displacement_energy: 10.0,
            density: 4.6e28,
            melting_point: 453.7,
            thermal_conductivity: 85.0,
        }
    }

    /// Get Thomas-Fermi screening length (m)
    pub fn screening_length(&self, projectile_z: f64) -> f64 {
        let a_0 = 5.29e-11;  // Bohr radius
        0.8854 * a_0 / (projectile_z.powf(0.23) + self.z.powf(0.23))
    }

    /// Get threshold energy for sputtering (eV)
    ///
    /// Uses the Bohdansky threshold formula with corrections for:
    /// - Light projectiles on heavy targets
    /// - Heavy projectiles on light targets
    /// - Self-sputtering (equal masses)
    pub fn sputtering_threshold(&self, projectile_mass: f64) -> f64 {
        let gamma = 4.0 * projectile_mass * self.mass_amu /
            (projectile_mass + self.mass_amu).powi(2);

        // Handle near-equal mass case (self-sputtering)
        // When gamma → 1, the formula diverges; use empirical threshold
        if (gamma - 1.0).abs() < 0.01 || (1.0 - gamma) < 0.01 {
            // Self-sputtering: threshold ≈ surface binding energy
            return self.surface_binding_energy;
        }

        if projectile_mass <= self.mass_amu {
            self.surface_binding_energy / (gamma * (1.0 - gamma))
        } else {
            8.0 * self.surface_binding_energy / gamma
        }
    }
}

// ============================================================================
// INCIDENT SPECIES
// ============================================================================

/// Incident particle species
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IncidentSpecies {
    Hydrogen,
    Deuterium,
    Tritium,
    Helium,
    Argon,
    Neon,
    Nitrogen,
    Oxygen,
    Carbon,
    Tungsten,  // Self-sputtering
}

impl IncidentSpecies {
    pub fn mass_amu(&self) -> f64 {
        match self {
            Self::Hydrogen => 1.008,
            Self::Deuterium => 2.014,
            Self::Tritium => 3.016,
            Self::Helium => 4.003,
            Self::Argon => 39.95,
            Self::Neon => 20.18,
            Self::Nitrogen => 14.01,
            Self::Oxygen => 16.00,
            Self::Carbon => 12.01,
            Self::Tungsten => 183.84,
        }
    }

    pub fn atomic_number(&self) -> f64 {
        match self {
            Self::Hydrogen | Self::Deuterium | Self::Tritium => 1.0,
            Self::Helium => 2.0,
            Self::Argon => 18.0,
            Self::Neon => 10.0,
            Self::Nitrogen => 7.0,
            Self::Oxygen => 8.0,
            Self::Carbon => 6.0,
            Self::Tungsten => 74.0,
        }
    }
}

// ============================================================================
// SPUTTERING YIELD MODELS
// ============================================================================

/// Sputtering yield calculator
pub struct SputteringYield {
    /// Target material
    pub material: MaterialProperties,
}

impl SputteringYield {
    pub fn new(material: MaterialProperties) -> Self {
        Self { material }
    }

    /// Calculate physical sputtering yield using Bohdansky formula
    ///
    /// Y = Q × S_n(E) × g(E/E_th)
    ///
    /// where:
    /// - Q is a material-dependent factor
    /// - S_n(E) is the nuclear stopping cross-section
    /// - g(x) = (1 - (1/x)^(2/3)) × (1 - 1/x)²
    ///
    /// Returns atoms/ion
    pub fn bohdansky_yield(&self, energy_ev: f64, projectile: IncidentSpecies, angle_deg: f64) -> f64 {
        let m1 = projectile.mass_amu();
        let m2 = self.material.mass_amu;
        let z1 = projectile.atomic_number();
        let z2 = self.material.z;

        // Threshold energy
        let e_th = self.material.sputtering_threshold(m1);

        if energy_ev <= e_th {
            return 0.0;
        }

        // Reduced energy
        let epsilon = self.reduced_energy(energy_ev, z1, z2, m1, m2);

        // Nuclear stopping cross-section (Lindhard)
        let s_n = self.nuclear_stopping(epsilon);

        // Mass transfer factor (used in extended models)
        let _alpha = 0.3 * (m2 / m1).powf(2.0 / 3.0);

        // Q factor (empirical)
        let q = self.q_factor(z1, z2, m1, m2);

        // Threshold function
        let x = energy_ev / e_th;
        let g = (1.0 - (1.0 / x).powf(2.0 / 3.0)) * (1.0 - 1.0 / x).powi(2);

        // Angular dependence (Yamamura)
        let f_angle = self.angular_factor(angle_deg);

        // Final yield
        (0.042 / self.material.surface_binding_energy) * q * s_n * g * f_angle
    }

    /// Yamamura-Tawara formula (more accurate at low energies)
    pub fn yamamura_yield(&self, energy_ev: f64, projectile: IncidentSpecies, angle_deg: f64) -> f64 {
        let m1 = projectile.mass_amu();
        let m2 = self.material.mass_amu;
        let z1 = projectile.atomic_number();
        let z2 = self.material.z;

        let e_th = self.material.sputtering_threshold(m1);

        if energy_ev <= e_th {
            return 0.0;
        }

        // Reduced energy
        let epsilon = self.reduced_energy(energy_ev, z1, z2, m1, m2);

        // Nuclear stopping
        let s_n = self.nuclear_stopping(epsilon);

        // Yamamura Q parameter
        let q = self.yamamura_q(z1, z2, m1, m2);

        // Fitting parameter W (used in extended models)
        let _w = self.yamamura_w(m1, m2);

        // Sputtering function
        let sqrt_term = 1.0 - (e_th / energy_ev).sqrt();
        let s_term = s_n / (1.0 + (m1 / m2) * (self.material.lattice_binding_energy / energy_ev));

        // Angular factor
        let f_angle = self.angular_factor(angle_deg);

        // Yield
        0.042 * q * s_term * sqrt_term.powf(2.8) * f_angle / self.material.surface_binding_energy
    }

    /// Reduced energy (dimensionless)
    fn reduced_energy(&self, e: f64, z1: f64, z2: f64, m1: f64, m2: f64) -> f64 {
        let a = 0.8854 * 5.29e-11 / (z1.powf(0.23) + z2.powf(0.23));  // Screening length
        m2 * e / ((m1 + m2) * z1 * z2 * 14.4 * 1e9 * a)  // 14.4 eV·nm ≈ e²/(4πε₀)
    }

    /// Nuclear stopping cross-section (Lindhard-Scharff-Schiøtt)
    fn nuclear_stopping(&self, epsilon: f64) -> f64 {
        if epsilon < 0.01 {
            1.212 * epsilon.sqrt()
        } else if epsilon < 10.0 {
            0.5 * (3.441 * epsilon.sqrt() * (1.0 + 6.35 * epsilon).ln() +
                   epsilon / (1.0 + 6.882 * epsilon.sqrt() + 1.708 * epsilon))
        } else {
            epsilon.ln() / (2.0 * epsilon)
        }
    }

    /// Q factor for Bohdansky formula
    fn q_factor(&self, z1: f64, z2: f64, m1: f64, m2: f64) -> f64 {
        // Empirical fit
        let z_ratio = z1 / z2;
        let m_ratio = m1 / m2;

        0.5 * (z_ratio.powf(2.0/3.0) + z_ratio.powf(-2.0/3.0)).powf(1.5) *
            (1.0 + 5.0 * m_ratio.powf(-1.0/3.0))
    }

    /// Yamamura Q parameter
    fn yamamura_q(&self, z1: f64, z2: f64, m1: f64, m2: f64) -> f64 {
        let m_ratio = m2 / m1;
        let z_factor = z1.powf(2.0/3.0) * z2.powf(1.0/2.0);

        0.035 * m_ratio.powf(1.0/3.0) * z_factor
    }

    /// Yamamura W parameter
    fn yamamura_w(&self, m1: f64, m2: f64) -> f64 {
        let ratio = m2 / m1;
        ratio + 0.5 * ratio.ln()
    }

    /// Angular dependence factor
    fn angular_factor(&self, angle_deg: f64) -> f64 {
        let theta = angle_deg.to_radians();
        let cos_theta = theta.cos();

        if cos_theta < 0.01 {
            return 0.0;  // Grazing incidence
        }

        // Yamamura angular formula
        let f = 1.0;  // Fitting parameter (material dependent)
        cos_theta.powf(-f) * (-(1.0 / cos_theta - 1.0).powi(2)).exp()
    }

    /// Calculate yield with energy and angle distribution
    pub fn average_yield(&self, energies: &[(f64, f64)], projectile: IncidentSpecies) -> f64 {
        // energies: Vec of (energy_eV, weight)
        let mut total_yield = 0.0;
        let mut total_weight = 0.0;

        for &(energy, weight) in energies {
            total_yield += weight * self.yamamura_yield(energy, projectile, 0.0);
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_yield / total_weight
        } else {
            0.0
        }
    }
}

// ============================================================================
// CHEMICAL SPUTTERING (CARBON)
// ============================================================================

/// Chemical erosion model for carbon
pub struct ChemicalErosion {
    /// Surface temperature (K)
    pub temperature: f64,
    /// Ion flux (ions/m²/s)
    pub ion_flux: f64,
    /// Maximum erosion yield at optimal temperature
    pub y_max: f64,
    /// Optimal temperature (K)
    pub t_max: f64,
    /// Activation energy for chemical erosion (eV)
    pub e_act: f64,
}

impl ChemicalErosion {
    /// Create for carbon with deuterium bombardment
    pub fn carbon_deuterium() -> Self {
        Self {
            temperature: 600.0,  // K
            ion_flux: 1e22,      // ions/m²/s (typical)
            y_max: 0.02,         // atoms/ion at T_max
            t_max: 850.0,        // K
            e_act: 0.27,         // eV
        }
    }

    /// Calculate chemical erosion yield
    ///
    /// Based on Roth's model (1999):
    /// Y_chem = Y_therm × f(E) × g(T)
    ///
    /// Returns atoms/ion
    pub fn chemical_yield(&self, energy_ev: f64, temperature: f64) -> f64 {
        // Energy dependence (threshold ~2 eV for D on C)
        let e_th = 2.0;
        if energy_ev < e_th {
            return 0.0;
        }

        let f_energy = if energy_ev < 100.0 {
            (energy_ev / 30.0).sqrt().min(1.0)
        } else {
            1.0 / (1.0 + (energy_ev / 200.0).sqrt())  // Decreases at high energy
        };

        // Temperature dependence (bell-shaped curve)
        let t_norm = temperature / self.t_max;
        let sigma = 0.3;  // Width parameter
        let g_temp = (-((t_norm - 1.0) / sigma).powi(2) / 2.0).exp();

        // Flux dependence (weak)
        let phi_ref = 1e22;  // Reference flux
        let f_flux = 1.0 - 0.1 * (self.ion_flux / phi_ref).ln().max(-10.0).min(10.0);

        self.y_max * f_energy * g_temp * f_flux.max(0.1)
    }

    /// Swift chemical sputtering (synergistic effect with physical)
    pub fn swift_chemical_yield(&self, physical_yield: f64, energy_ev: f64, temperature: f64) -> f64 {
        // Swift chemical sputtering occurs when physical sputtering
        // creates reactive sites
        let chem_yield = self.chemical_yield(energy_ev, temperature);

        // Synergistic enhancement
        let synergy_factor = 1.0 + 0.5 * physical_yield / 0.01;

        chem_yield * synergy_factor.min(3.0)
    }

    /// Total carbon erosion (physical + chemical)
    pub fn total_carbon_yield(
        &self,
        energy_ev: f64,
        temperature: f64,
        physical_yield: f64,
    ) -> f64 {
        let chem = self.swift_chemical_yield(physical_yield, energy_ev, temperature);
        physical_yield + chem
    }
}

// ============================================================================
// SURFACE LAYER MODEL
// ============================================================================

/// Surface layer representing deposited/implanted material
#[derive(Debug, Clone)]
pub struct SurfaceLayer {
    /// Layer thickness (m)
    pub thickness: f64,
    /// Composition: species → concentration (atoms/m³)
    pub composition: HashMap<String, f64>,
    /// Temperature (K)
    pub temperature: f64,
    /// Implanted hydrogen isotopes (D, T) concentration
    pub hydrogen_content: f64,
    /// Maximum hydrogen retention (Sottlinger limit)
    pub max_hydrogen: f64,
}

impl SurfaceLayer {
    pub fn new() -> Self {
        Self {
            thickness: 0.0,
            composition: HashMap::new(),
            temperature: 300.0,
            hydrogen_content: 0.0,
            max_hydrogen: 1e27,  // atoms/m³ typical saturation
        }
    }

    /// Add deposited material
    pub fn deposit(&mut self, species: &str, amount: f64, layer_density: f64) {
        *self.composition.entry(species.to_string()).or_insert(0.0) += amount;

        // Update thickness based on deposited atoms
        if layer_density > 0.0 {
            self.thickness += amount / layer_density;
        }
    }

    /// Erode layer
    pub fn erode(&mut self, amount: f64, layer_density: f64) {
        if layer_density > 0.0 {
            let eroded_thickness = amount / layer_density;
            self.thickness = (self.thickness - eroded_thickness).max(0.0);

            // Proportionally reduce all species
            if self.thickness > 0.0 {
                let ratio = self.thickness / (self.thickness + eroded_thickness);
                for (_, conc) in self.composition.iter_mut() {
                    *conc *= ratio;
                }
            } else {
                self.composition.clear();
            }
        }
    }

    /// Implant hydrogen isotope
    pub fn implant_hydrogen(&mut self, amount: f64) {
        self.hydrogen_content = (self.hydrogen_content + amount).min(self.max_hydrogen);
    }

    /// Release hydrogen (thermal desorption)
    pub fn thermal_release(&mut self, temperature: f64, dt: f64) -> f64 {
        // Arrhenius-type release
        let e_act = 1.0;  // eV (binding energy)
        let nu = 1e13;    // Attempt frequency (Hz)
        let k_b_ev = 8.617e-5;  // eV/K

        let rate = nu * (-e_act / (k_b_ev * temperature)).exp();
        let released = self.hydrogen_content * (1.0 - (-rate * dt).exp());

        self.hydrogen_content -= released;
        released
    }

    /// Get total areal density (atoms/m²)
    pub fn areal_density(&self) -> f64 {
        self.composition.values().sum::<f64>() * self.thickness
    }

    /// Check if layer is "clean" (no significant deposits)
    pub fn is_clean(&self, threshold: f64) -> bool {
        self.thickness < threshold
    }
}

impl Default for SurfaceLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PLASMA-WALL INTERACTION CELL
// ============================================================================

/// Represents one surface cell for PWI calculations
#[derive(Debug, Clone)]
pub struct PWICell {
    /// Position on wall (m)
    pub position: Vec3,
    /// Surface normal
    pub normal: Vec3,
    /// Surface area (m²)
    pub area: f64,
    /// Base material
    pub material: MaterialProperties,
    /// Surface layer (deposits, implantation)
    pub surface_layer: SurfaceLayer,
    /// Current temperature (K)
    pub temperature: f64,
    /// Incident ion flux (ions/m²/s) by species
    pub ion_flux: HashMap<IncidentSpecies, f64>,
    /// Incident ion energy distribution (eV, weight)
    pub energy_distribution: Vec<(f64, f64)>,
    /// Net erosion rate (m/s)
    pub erosion_rate: f64,
    /// Net deposition rate (m/s)
    pub deposition_rate: f64,
    /// Cumulative erosion (m)
    pub total_erosion: f64,
    /// Cumulative deposition (m)
    pub total_deposition: f64,
    /// Sputtered flux (atoms/m²/s)
    pub sputtered_flux: f64,
}

impl PWICell {
    pub fn new(position: Vec3, normal: Vec3, area: f64, material: MaterialProperties) -> Self {
        Self {
            position,
            normal,
            area,
            material,
            surface_layer: SurfaceLayer::new(),
            temperature: 300.0,
            ion_flux: HashMap::new(),
            energy_distribution: vec![(100.0, 1.0)],  // Default 100 eV
            erosion_rate: 0.0,
            deposition_rate: 0.0,
            total_erosion: 0.0,
            total_deposition: 0.0,
            sputtered_flux: 0.0,
        }
    }

    /// Set incident ion flux
    pub fn set_flux(&mut self, species: IncidentSpecies, flux: f64) {
        self.ion_flux.insert(species, flux);
    }

    /// Set energy distribution (energy_eV, weight pairs)
    pub fn set_energy_distribution(&mut self, distribution: Vec<(f64, f64)>) {
        self.energy_distribution = distribution;
    }

    /// Calculate sputtering for one time step
    pub fn calculate_sputtering(&mut self, dt: f64) {
        let sputtering = SputteringYield::new(self.material.clone());
        let chemical = ChemicalErosion::carbon_deuterium();

        let mut total_sputtered = 0.0;

        for (&species, &flux) in &self.ion_flux {
            // Average yield over energy distribution
            let phys_yield = sputtering.average_yield(&self.energy_distribution, species);

            // Chemical erosion (only for carbon with H isotopes)
            let chem_yield = if self.material.material == WallMaterial::Carbon &&
                matches!(species, IncidentSpecies::Deuterium | IncidentSpecies::Tritium | IncidentSpecies::Hydrogen) {
                let avg_energy: f64 = self.energy_distribution.iter()
                    .map(|(e, w)| e * w)
                    .sum::<f64>() / self.energy_distribution.iter()
                    .map(|(_, w)| w)
                    .sum::<f64>();
                chemical.total_carbon_yield(avg_energy, self.temperature, phys_yield) - phys_yield
            } else {
                0.0
            };

            let total_yield = phys_yield + chem_yield;
            total_sputtered += flux * total_yield;

            // Hydrogen implantation
            if matches!(species, IncidentSpecies::Deuterium | IncidentSpecies::Tritium | IncidentSpecies::Hydrogen) {
                let implanted = flux * (1.0 - total_yield.min(1.0)) * dt;
                self.surface_layer.implant_hydrogen(implanted);
            }
        }

        self.sputtered_flux = total_sputtered;

        // Erosion rate (convert atoms/m²/s to m/s)
        self.erosion_rate = total_sputtered / self.material.density;
        self.total_erosion += self.erosion_rate * dt;

        // Thermal release of hydrogen
        let _released = self.surface_layer.thermal_release(self.temperature, dt);
    }

    /// Apply deposition from sputtered material (from other cells)
    pub fn apply_deposition(&mut self, deposited_flux: f64, species: &str, dt: f64) {
        // Deposition rate
        self.deposition_rate = deposited_flux / self.material.density;
        self.total_deposition += self.deposition_rate * dt;

        // Add to surface layer
        self.surface_layer.deposit(species, deposited_flux * dt, self.material.density);
    }

    /// Get net erosion (erosion - deposition)
    pub fn net_erosion(&self) -> f64 {
        self.total_erosion - self.total_deposition
    }

    /// Get lifetime estimate (time until wall thickness is eroded through)
    pub fn lifetime_estimate(&self, wall_thickness: f64) -> f64 {
        let net_rate = self.erosion_rate - self.deposition_rate;
        if net_rate > 0.0 {
            wall_thickness / net_rate
        } else {
            f64::INFINITY
        }
    }
}

// ============================================================================
// PWI HANDLER
// ============================================================================

/// Main PWI handler for wall surface
pub struct PWIHandler {
    /// Wall cells
    pub cells: Vec<PWICell>,
    /// Time (s)
    pub time: f64,
    /// Statistics
    pub stats: PWIStats,
}

#[derive(Debug, Clone, Default)]
pub struct PWIStats {
    /// Total eroded mass (kg)
    pub total_eroded_mass: f64,
    /// Total deposited mass (kg)
    pub total_deposited_mass: f64,
    /// Total hydrogen retention (atoms)
    pub total_hydrogen_retention: f64,
    /// Maximum erosion depth (m)
    pub max_erosion: f64,
    /// Minimum lifetime (s)
    pub min_lifetime: f64,
}

impl PWIHandler {
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            time: 0.0,
            stats: PWIStats::default(),
        }
    }

    /// Add wall cell
    pub fn add_cell(&mut self, cell: PWICell) {
        self.cells.push(cell);
    }

    /// Create divertor surface (simplified geometry)
    pub fn create_divertor(&mut self, n_cells: usize, material: WallMaterial) {
        let mat_props = match material {
            WallMaterial::Tungsten => MaterialProperties::tungsten(),
            WallMaterial::Carbon => MaterialProperties::carbon(),
            WallMaterial::Beryllium => MaterialProperties::beryllium(),
            WallMaterial::Lithium => MaterialProperties::lithium(),
            _ => MaterialProperties::tungsten(),
        };

        // Create cells along divertor surface
        for i in 0..n_cells {
            let x = (i as f64 / n_cells as f64) * 2.0 - 1.0;  // -1 to 1
            let pos = Vec3::new(x, -0.5, 0.0);  // Bottom of vessel
            let normal = Vec3::new(0.0, 1.0, 0.0);  // Pointing up
            let area = 0.01;  // 1 cm² per cell

            let mut cell = PWICell::new(pos, normal, area, mat_props.clone());

            // Set typical divertor conditions
            // Higher flux near strike point
            let strike_point = 0.0;
            let distance = (x - strike_point).abs();
            let flux_profile = 1e23 * (-distance.powi(2) / 0.5).exp();

            cell.set_flux(IncidentSpecies::Deuterium, flux_profile);
            cell.temperature = 800.0 + 400.0 * (-distance.powi(2) / 0.3).exp();

            // Energy distribution (Maxwell + sheath)
            let t_i = 50.0;  // Ion temperature (eV)
            let v_sheath = 3.0 * t_i;  // Sheath potential
            cell.set_energy_distribution(vec![
                (t_i + v_sheath, 0.5),
                (2.0 * t_i + v_sheath, 0.3),
                (3.0 * t_i + v_sheath, 0.15),
                (5.0 * t_i + v_sheath, 0.05),
            ]);

            self.add_cell(cell);
        }
    }

    /// Step all cells
    pub fn step(&mut self, dt: f64) {
        // Calculate sputtering for each cell
        for cell in &mut self.cells {
            cell.calculate_sputtering(dt);
        }

        // Simple redeposition model: fraction of sputtered material redeposits
        let redeposition_fraction = 0.5;  // Typical for divertor
        let total_sputtered: f64 = self.cells.iter()
            .map(|c| c.sputtered_flux * c.area)
            .sum();

        // Distribute redeposition (simplified: uniform)
        if total_sputtered > 0.0 && !self.cells.is_empty() {
            let deposited_per_cell = redeposition_fraction * total_sputtered / self.cells.len() as f64;
            for cell in &mut self.cells {
                cell.apply_deposition(deposited_per_cell / cell.area, "W", dt);
            }
        }

        self.time += dt;
        self.update_stats();
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.max_erosion = self.cells.iter()
            .map(|c| c.total_erosion)
            .fold(0.0, f64::max);

        self.stats.total_hydrogen_retention = self.cells.iter()
            .map(|c| c.surface_layer.hydrogen_content * c.area * c.surface_layer.thickness)
            .sum();

        let wall_thickness = 0.01;  // 1 cm wall
        self.stats.min_lifetime = self.cells.iter()
            .map(|c| c.lifetime_estimate(wall_thickness))
            .fold(f64::INFINITY, f64::min);
    }

    /// Get cell with maximum erosion
    pub fn max_erosion_cell(&self) -> Option<&PWICell> {
        self.cells.iter()
            .max_by(|a, b| a.total_erosion.partial_cmp(&b.total_erosion).unwrap())
    }

    /// Summary output
    pub fn summary(&self) -> String {
        format!(
            "PWI Summary (t = {:.3} s)\n\
             ========================\n\
             Cells: {}\n\
             Max erosion: {:.3} μm\n\
             Min lifetime: {:.2e} s ({:.1} years)\n\
             H retention: {:.2e} atoms\n",
            self.time,
            self.cells.len(),
            self.stats.max_erosion * 1e6,
            self.stats.min_lifetime,
            self.stats.min_lifetime / (365.25 * 24.0 * 3600.0),
            self.stats.total_hydrogen_retention,
        )
    }
}

impl Default for PWIHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_properties() {
        let w = MaterialProperties::tungsten();
        assert_eq!(w.z, 74.0);
        assert!(w.surface_binding_energy > 8.0);

        let c = MaterialProperties::carbon();
        assert!(c.displacement_energy < w.displacement_energy);
    }

    #[test]
    fn test_sputtering_yield() {
        let sputtering = SputteringYield::new(MaterialProperties::tungsten());

        // D on W threshold is ~200 eV, so test at 300 eV
        let y_300 = sputtering.yamamura_yield(300.0, IncidentSpecies::Deuterium, 0.0);
        assert!(y_300 > 0.0 && y_300 < 0.1);  // Should be very low

        // D on W at 1 keV (well above threshold)
        let y_1000 = sputtering.yamamura_yield(1000.0, IncidentSpecies::Deuterium, 0.0);
        assert!(y_1000 > y_300);  // Higher energy = higher yield

        // Self-sputtering (W on W) - lower threshold
        let y_self = sputtering.yamamura_yield(1000.0, IncidentSpecies::Tungsten, 0.0);
        assert!(y_self > y_1000);  // Self-sputtering is more efficient
    }

    #[test]
    fn test_chemical_erosion() {
        let chem = ChemicalErosion::carbon_deuterium();

        // At optimal temperature
        let y_850 = chem.chemical_yield(100.0, 850.0);

        // Below optimal
        let y_500 = chem.chemical_yield(100.0, 500.0);

        // Above optimal
        let y_1200 = chem.chemical_yield(100.0, 1200.0);

        assert!(y_850 > y_500);
        assert!(y_850 > y_1200);
    }

    #[test]
    fn test_surface_layer() {
        let mut layer = SurfaceLayer::new();

        layer.deposit("W", 1e20, 6.3e28);
        assert!(layer.thickness > 0.0);

        layer.implant_hydrogen(1e15);
        assert!(layer.hydrogen_content > 0.0);

        let released = layer.thermal_release(1000.0, 1.0);
        assert!(released > 0.0);
    }

    #[test]
    fn test_pwi_cell() {
        let mut cell = PWICell::new(
            Vec3::zero(),
            Vec3::new(0.0, 1.0, 0.0),
            0.01,
            MaterialProperties::tungsten(),
        );

        cell.set_flux(IncidentSpecies::Deuterium, 1e23);
        // Use energy above D on W threshold (~200 eV)
        cell.set_energy_distribution(vec![(500.0, 1.0)]);
        cell.temperature = 800.0;

        cell.calculate_sputtering(1e-3);

        assert!(cell.sputtered_flux > 0.0);
        assert!(cell.total_erosion > 0.0);
    }

    #[test]
    fn test_pwi_handler() {
        let mut handler = PWIHandler::new();
        handler.create_divertor(10, WallMaterial::Tungsten);

        assert_eq!(handler.cells.len(), 10);

        handler.step(1e-3);

        assert!(handler.stats.max_erosion > 0.0);
        assert!(handler.stats.min_lifetime > 0.0);
    }
}
