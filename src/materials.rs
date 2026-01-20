//! # TOKASIM-RS Materials Database
//!
//! Comprehensive database of materials used in tokamak fusion reactors.
//! Includes superconductors, structural materials, plasma-facing components,
//! and blanket materials with full physical properties.
//!
//! ## Author
//!
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use std::fmt;

/// Physical properties of a material
#[derive(Clone, Debug)]
pub struct Material {
    /// Unique identifier
    pub id: &'static str,
    /// Display name
    pub name: &'static str,
    /// Category of material
    pub category: MaterialCategory,

    // Thermal properties
    /// Maximum operating temperature [K]
    pub max_temperature: f32,
    /// Melting point [K]
    pub melting_point: f32,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f32,
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f32,

    // Magnetic properties (for superconductors)
    /// Maximum magnetic field [T] (0 for non-superconductors)
    pub max_magnetic_field: f32,
    /// Critical temperature [K] (0 for non-superconductors)
    pub critical_temperature: f32,
    /// Critical current density [A/mm²] at 4.2K, 12T
    pub critical_current_density: f32,

    // Mechanical properties
    /// Density [kg/m³]
    pub density: f32,
    /// Yield strength [MPa]
    pub yield_strength: f32,
    /// Young's modulus [GPa]
    pub youngs_modulus: f32,

    // Radiation resistance
    /// Neutron tolerance [dpa] (displacements per atom)
    pub neutron_tolerance: f32,

    // Economic (for future cost module)
    /// Cost per kilogram [USD/kg]
    pub cost_per_kg: f32,

    /// Description for UI tooltip
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialCategory {
    /// Low-temperature superconductors (LTS)
    SuperconductorLTS,
    /// High-temperature superconductors (HTS)
    SuperconductorHTS,
    /// Plasma-facing materials
    PlasmaFacing,
    /// Structural materials
    Structural,
    /// Blanket/breeding materials
    Blanket,
    /// Heat sink materials
    HeatSink,
}

impl fmt::Display for MaterialCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaterialCategory::SuperconductorLTS => write!(f, "LTS Superconductor"),
            MaterialCategory::SuperconductorHTS => write!(f, "HTS Superconductor"),
            MaterialCategory::PlasmaFacing => write!(f, "Plasma-Facing"),
            MaterialCategory::Structural => write!(f, "Structural"),
            MaterialCategory::Blanket => write!(f, "Blanket/Breeding"),
            MaterialCategory::HeatSink => write!(f, "Heat Sink"),
        }
    }
}

// ============================================================================
// SUPERCONDUCTORS - LOW TEMPERATURE (LTS)
// ============================================================================

pub const NB3SN: Material = Material {
    id: "nb3sn",
    name: "Nb₃Sn",
    category: MaterialCategory::SuperconductorLTS,
    max_temperature: 18.0,
    melting_point: 2400.0,
    thermal_conductivity: 0.2,  // Very low at cryogenic
    specific_heat: 200.0,
    max_magnetic_field: 13.0,
    critical_temperature: 18.3,
    critical_current_density: 1000.0,  // A/mm² at 4.2K, 12T
    density: 8900.0,
    yield_strength: 150.0,
    youngs_modulus: 165.0,
    neutron_tolerance: 10.0,
    cost_per_kg: 150.0,
    description: "Niobium-tin. Standard ITER superconductor. Max 13T at 4.2K. \
                  Brittle, requires wind-and-react or react-and-wind fabrication.",
};

pub const NBTI: Material = Material {
    id: "nbti",
    name: "NbTi",
    category: MaterialCategory::SuperconductorLTS,
    max_temperature: 10.0,
    melting_point: 2100.0,
    thermal_conductivity: 0.15,
    specific_heat: 180.0,
    max_magnetic_field: 9.0,
    critical_temperature: 9.8,
    critical_current_density: 3000.0,  // Higher Jc but lower field
    density: 6500.0,
    yield_strength: 200.0,
    youngs_modulus: 80.0,
    neutron_tolerance: 15.0,
    cost_per_kg: 80.0,
    description: "Niobium-titanium. Workhorse superconductor, easy to fabricate. \
                  Limited to 9T. Used in MRI, accelerators, and some fusion magnets.",
};

// ============================================================================
// SUPERCONDUCTORS - HIGH TEMPERATURE (HTS)
// ============================================================================

pub const REBCO: Material = Material {
    id: "rebco",
    name: "REBCO (HTS)",
    category: MaterialCategory::SuperconductorHTS,
    max_temperature: 92.0,
    melting_point: 1300.0,  // Decomposes before melting
    thermal_conductivity: 3.0,
    specific_heat: 250.0,
    max_magnetic_field: 20.0,
    critical_temperature: 92.0,
    critical_current_density: 500.0,  // At 20K, 20T
    density: 6300.0,
    yield_strength: 100.0,
    youngs_modulus: 150.0,
    neutron_tolerance: 5.0,  // Lower radiation tolerance
    cost_per_kg: 2000.0,
    description: "Rare Earth Barium Copper Oxide (REBa₂Cu₃O₇). HTS tape used in SPARC. \
                  Operates at 20K with liquid hydrogen or 77K with liquid nitrogen. \
                  Max 20T demonstrated. Enables compact high-field tokamaks.",
};

pub const YBCO: Material = Material {
    id: "ybco",
    name: "YBCO (Experimental)",
    category: MaterialCategory::SuperconductorHTS,
    max_temperature: 93.0,
    melting_point: 1300.0,
    thermal_conductivity: 4.0,
    specific_heat: 250.0,
    max_magnetic_field: 30.0,  // Experimental limit
    critical_temperature: 93.0,
    critical_current_density: 400.0,
    density: 6380.0,
    yield_strength: 90.0,
    youngs_modulus: 140.0,
    neutron_tolerance: 4.0,
    cost_per_kg: 5000.0,  // Very expensive
    description: "Yttrium Barium Copper Oxide. Next-generation HTS. \
                  Experimental fields up to 30T+ demonstrated in laboratory. \
                  Enables TS-1 design with 25T toroidal field.",
};

pub const BI2212: Material = Material {
    id: "bi2212",
    name: "Bi-2212",
    category: MaterialCategory::SuperconductorHTS,
    max_temperature: 85.0,
    melting_point: 1100.0,
    thermal_conductivity: 2.0,
    specific_heat: 230.0,
    max_magnetic_field: 25.0,
    critical_temperature: 85.0,
    critical_current_density: 300.0,
    density: 6500.0,
    yield_strength: 80.0,
    youngs_modulus: 100.0,
    neutron_tolerance: 3.0,
    cost_per_kg: 3000.0,
    description: "Bismuth Strontium Calcium Copper Oxide. Round wire HTS. \
                  Can be made into cables unlike REBCO tapes. Max 25T.",
};

// ============================================================================
// PLASMA-FACING MATERIALS
// ============================================================================

pub const TUNGSTEN: Material = Material {
    id: "tungsten",
    name: "Tungsten (W)",
    category: MaterialCategory::PlasmaFacing,
    max_temperature: 3000.0,  // Practical limit
    melting_point: 3695.0,
    thermal_conductivity: 173.0,
    specific_heat: 134.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 19300.0,
    yield_strength: 750.0,
    youngs_modulus: 411.0,
    neutron_tolerance: 5.0,
    cost_per_kg: 50.0,
    description: "Highest melting point of any element. Standard divertor material. \
                  High Z causes radiation losses if eroded into plasma. \
                  Excellent heat resistance but brittle below DBTT (~400°C).",
};

pub const BERYLLIUM: Material = Material {
    id: "beryllium",
    name: "Beryllium (Be)",
    category: MaterialCategory::PlasmaFacing,
    max_temperature: 1200.0,
    melting_point: 1560.0,
    thermal_conductivity: 200.0,
    specific_heat: 1825.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 1850.0,
    yield_strength: 240.0,
    youngs_modulus: 287.0,
    neutron_tolerance: 3.0,
    cost_per_kg: 800.0,
    description: "ITER first wall material. Low atomic number minimizes radiation losses. \
                  Excellent oxygen getter. Toxic - requires careful handling. \
                  Lower melting point limits heat flux capability.",
};

pub const CFC: Material = Material {
    id: "cfc",
    name: "Carbon Fiber Composite",
    category: MaterialCategory::PlasmaFacing,
    max_temperature: 2500.0,
    melting_point: 3800.0,  // Sublimes
    thermal_conductivity: 300.0,  // Very high
    specific_heat: 710.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 1800.0,
    yield_strength: 100.0,
    youngs_modulus: 30.0,
    neutron_tolerance: 1.0,  // Poor - swells under neutrons
    cost_per_kg: 200.0,
    description: "Legacy plasma-facing material (JET, TFTR). Excellent thermal shock \
                  resistance. Problem: retains tritium, limiting use in D-T reactors. \
                  Being phased out in favor of tungsten.",
};

pub const TUNGSTEN_COPPER: Material = Material {
    id: "wcu",
    name: "Tungsten-Copper (WCu)",
    category: MaterialCategory::PlasmaFacing,
    max_temperature: 2000.0,
    melting_point: 2500.0,  // Composite
    thermal_conductivity: 250.0,
    specific_heat: 200.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 15000.0,
    yield_strength: 400.0,
    youngs_modulus: 280.0,
    neutron_tolerance: 4.0,
    cost_per_kg: 100.0,
    description: "Tungsten-copper composite. Combines W heat resistance with Cu conductivity. \
                  Good for high heat flux components. Graded compositions available.",
};

// ============================================================================
// STRUCTURAL MATERIALS
// ============================================================================

pub const SS316L: Material = Material {
    id: "ss316l",
    name: "Stainless Steel 316L",
    category: MaterialCategory::Structural,
    max_temperature: 800.0,
    melting_point: 1673.0,
    thermal_conductivity: 16.0,
    specific_heat: 500.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 8000.0,
    yield_strength: 290.0,
    youngs_modulus: 193.0,
    neutron_tolerance: 10.0,
    cost_per_kg: 5.0,
    description: "Standard austenitic stainless steel. ITER vacuum vessel material. \
                  Good weldability and corrosion resistance. Activates under neutrons \
                  but acceptable for current designs.",
};

pub const INCONEL718: Material = Material {
    id: "inconel718",
    name: "Inconel 718",
    category: MaterialCategory::Structural,
    max_temperature: 980.0,
    melting_point: 1609.0,
    thermal_conductivity: 11.0,
    specific_heat: 435.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 8190.0,
    yield_strength: 1100.0,
    youngs_modulus: 200.0,
    neutron_tolerance: 8.0,
    cost_per_kg: 30.0,
    description: "Nickel-chromium superalloy. Excellent high-temperature strength. \
                  Used in SPARC and aerospace. More expensive than SS316L but \
                  maintains strength at higher temperatures.",
};

pub const EUROFER97: Material = Material {
    id: "eurofer97",
    name: "EUROFER-97",
    category: MaterialCategory::Structural,
    max_temperature: 823.0,
    melting_point: 1773.0,
    thermal_conductivity: 28.0,
    specific_heat: 500.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 7750.0,
    yield_strength: 530.0,
    youngs_modulus: 217.0,
    neutron_tolerance: 80.0,  // Designed for fusion
    cost_per_kg: 15.0,
    description: "Reduced Activation Ferritic-Martensitic (RAFM) steel. \
                  Designed specifically for fusion reactors. Low activation under \
                  neutron irradiation. Baseline structural material for DEMO.",
};

// ============================================================================
// HEAT SINK MATERIALS
// ============================================================================

pub const CUCRZR: Material = Material {
    id: "cucrzr",
    name: "CuCrZr",
    category: MaterialCategory::HeatSink,
    max_temperature: 623.0,
    melting_point: 1353.0,
    thermal_conductivity: 318.0,
    specific_heat: 390.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 8900.0,
    yield_strength: 340.0,
    youngs_modulus: 128.0,
    neutron_tolerance: 5.0,
    cost_per_kg: 20.0,
    description: "Copper-Chromium-Zirconium alloy. Precipitation hardened copper \
                  with high thermal conductivity. Used as heat sink beneath plasma-facing \
                  components. ITER divertor uses CuCrZr actively cooled.",
};

pub const OFHC_COPPER: Material = Material {
    id: "ofhc_cu",
    name: "OFHC Copper",
    category: MaterialCategory::HeatSink,
    max_temperature: 573.0,
    melting_point: 1358.0,
    thermal_conductivity: 391.0,
    specific_heat: 385.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 8940.0,
    yield_strength: 70.0,
    youngs_modulus: 117.0,
    neutron_tolerance: 3.0,
    cost_per_kg: 10.0,
    description: "Oxygen-Free High Conductivity copper. Highest thermal conductivity \
                  but low strength. Used where maximum heat removal is needed.",
};

// ============================================================================
// BLANKET / BREEDING MATERIALS
// ============================================================================

pub const LI4SIO4: Material = Material {
    id: "li4sio4",
    name: "Li₄SiO₄ (Lithium Orthosilicate)",
    category: MaterialCategory::Blanket,
    max_temperature: 1173.0,
    melting_point: 1528.0,
    thermal_conductivity: 2.5,
    specific_heat: 1200.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 2400.0,
    yield_strength: 50.0,
    youngs_modulus: 90.0,
    neutron_tolerance: 50.0,
    cost_per_kg: 100.0,
    description: "Solid tritium breeder ceramic. EU reference material. \
                  Li + n → T + He reaction breeds tritium fuel. \
                  Pebble bed design allows helium purge gas to extract tritium.",
};

pub const LI2TIO3: Material = Material {
    id: "li2tio3",
    name: "Li₂TiO₃ (Lithium Metatitanate)",
    category: MaterialCategory::Blanket,
    max_temperature: 1373.0,
    melting_point: 1806.0,
    thermal_conductivity: 3.0,
    specific_heat: 1000.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 3430.0,
    yield_strength: 60.0,
    youngs_modulus: 100.0,
    neutron_tolerance: 60.0,
    cost_per_kg: 120.0,
    description: "Solid tritium breeder ceramic. Japanese reference material. \
                  Higher density than Li₄SiO₄, better tritium release kinetics. \
                  Used in JA-DEMO blanket design.",
};

pub const PBLI: Material = Material {
    id: "pbli",
    name: "PbLi (Lead-Lithium)",
    category: MaterialCategory::Blanket,
    max_temperature: 823.0,
    melting_point: 508.0,
    thermal_conductivity: 17.0,
    specific_heat: 190.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 9400.0,
    yield_strength: 0.0,  // Liquid
    youngs_modulus: 0.0,
    neutron_tolerance: 100.0,  // Liquid - self-healing
    cost_per_kg: 50.0,
    description: "Liquid metal tritium breeder and coolant. Pb₁₇Li eutectic. \
                  Dual function: breeds tritium AND removes heat. \
                  MHD effects require insulating coatings on ducts. TS-1 baseline.",
};

pub const FLIBE: Material = Material {
    id: "flibe",
    name: "FLiBe (Molten Salt)",
    category: MaterialCategory::Blanket,
    max_temperature: 973.0,
    melting_point: 732.0,
    thermal_conductivity: 1.0,
    specific_heat: 2380.0,
    max_magnetic_field: 0.0,
    critical_temperature: 0.0,
    critical_current_density: 0.0,
    density: 1940.0,
    yield_strength: 0.0,  // Liquid
    youngs_modulus: 0.0,
    neutron_tolerance: 100.0,
    cost_per_kg: 200.0,
    description: "Lithium-Beryllium Fluoride molten salt (2LiF-BeF₂). \
                  Excellent neutron moderator, high heat capacity. \
                  Used in ARC/SPARC concepts. Requires corrosion-resistant materials.",
};

// ============================================================================
// MATERIAL COLLECTIONS
// ============================================================================

/// All available superconductor materials
pub const SUPERCONDUCTORS: &[&Material] = &[
    &NBTI, &NB3SN, &REBCO, &YBCO, &BI2212,
];

/// All plasma-facing materials
pub const PLASMA_FACING: &[&Material] = &[
    &TUNGSTEN, &BERYLLIUM, &CFC, &TUNGSTEN_COPPER,
];

/// All structural materials
pub const STRUCTURAL: &[&Material] = &[
    &SS316L, &INCONEL718, &EUROFER97,
];

/// All heat sink materials
pub const HEAT_SINKS: &[&Material] = &[
    &CUCRZR, &OFHC_COPPER,
];

/// All blanket materials
pub const BLANKET_MATERIALS: &[&Material] = &[
    &LI4SIO4, &LI2TIO3, &PBLI, &FLIBE,
];

/// All materials in the database
pub const ALL_MATERIALS: &[&Material] = &[
    // Superconductors
    &NBTI, &NB3SN, &REBCO, &YBCO, &BI2212,
    // Plasma-facing
    &TUNGSTEN, &BERYLLIUM, &CFC, &TUNGSTEN_COPPER,
    // Structural
    &SS316L, &INCONEL718, &EUROFER97,
    // Heat sinks
    &CUCRZR, &OFHC_COPPER,
    // Blanket
    &LI4SIO4, &LI2TIO3, &PBLI, &FLIBE,
];

/// Get material by ID
pub fn get_material(id: &str) -> Option<&'static Material> {
    ALL_MATERIALS.iter().find(|m| m.id == id).copied()
}

/// Get materials by category
pub fn get_materials_by_category(category: MaterialCategory) -> Vec<&'static Material> {
    ALL_MATERIALS.iter()
        .filter(|m| m.category == category)
        .copied()
        .collect()
}

// ============================================================================
// PRESET CONFIGURATIONS
// ============================================================================

/// Material configuration for a tokamak design
#[derive(Clone, Debug)]
pub struct TokamakMaterials {
    pub tf_coil: &'static Material,
    pub pf_coil: &'static Material,
    pub cs_coil: &'static Material,
    pub first_wall: &'static Material,
    pub divertor: &'static Material,
    pub blanket: &'static Material,
    pub vacuum_vessel: &'static Material,
    pub structure: &'static Material,
    pub heat_sink: &'static Material,
}

impl Default for TokamakMaterials {
    fn default() -> Self {
        Self::iter_preset()
    }
}

impl TokamakMaterials {
    /// ITER-like material configuration
    pub fn iter_preset() -> Self {
        Self {
            tf_coil: &NB3SN,
            pf_coil: &NB3SN,
            cs_coil: &NB3SN,
            first_wall: &BERYLLIUM,
            divertor: &TUNGSTEN,
            blanket: &LI4SIO4,
            vacuum_vessel: &SS316L,
            structure: &SS316L,
            heat_sink: &CUCRZR,
        }
    }

    /// SPARC-like material configuration
    pub fn sparc_preset() -> Self {
        Self {
            tf_coil: &REBCO,
            pf_coil: &REBCO,
            cs_coil: &REBCO,
            first_wall: &TUNGSTEN,
            divertor: &TUNGSTEN,
            blanket: &FLIBE,
            vacuum_vessel: &INCONEL718,
            structure: &INCONEL718,
            heat_sink: &CUCRZR,
        }
    }

    /// TS-1 optimized configuration (high-field compact)
    pub fn ts1_preset() -> Self {
        Self {
            tf_coil: &YBCO,  // Enables 25T+
            pf_coil: &REBCO,
            cs_coil: &REBCO,
            first_wall: &TUNGSTEN,
            divertor: &TUNGSTEN,
            blanket: &PBLI,
            vacuum_vessel: &EUROFER97,
            structure: &EUROFER97,
            heat_sink: &CUCRZR,
        }
    }

    /// Get maximum allowed toroidal field based on TF coil material
    pub fn max_toroidal_field(&self) -> f32 {
        self.tf_coil.max_magnetic_field
    }

    /// Get maximum operating temperature for first wall
    pub fn max_wall_temperature(&self) -> f32 {
        self.first_wall.max_temperature
    }
}

// ============================================================================
// THICKNESS CALCULATIONS
// ============================================================================

/// Calculate required TF coil thickness based on magnetic pressure
///
/// # Arguments
/// * `b_field` - Toroidal magnetic field [T]
/// * `major_radius` - Major radius [m]
/// * `material` - Coil structural material
/// * `safety_factor` - Design safety factor (typically 2-3)
///
/// # Returns
/// Required coil thickness [m]
pub fn calculate_tf_coil_thickness(
    b_field: f32,
    major_radius: f32,
    material: &Material,
    safety_factor: f32,
) -> f32 {
    const MU0: f32 = 1.2566e-6;  // H/m

    // Magnetic pressure: P = B² / (2*μ₀)
    let magnetic_pressure = b_field * b_field / (2.0 * MU0);

    // Hoop stress in coil: σ = P * R / t
    // Solving for t: t = P * R * SF / σ_yield
    let sigma_allow = material.yield_strength * 1e6;  // Convert MPa to Pa

    let thickness = magnetic_pressure * major_radius * safety_factor / sigma_allow;

    // Minimum practical thickness
    thickness.max(0.05)
}

/// Calculate required first wall thickness based on heat flux
///
/// # Arguments
/// * `heat_flux` - Surface heat flux [MW/m²]
/// * `pulse_duration` - Pulse duration [s]
/// * `material` - Wall material
/// * `delta_t_allow` - Allowed temperature rise [K]
///
/// # Returns
/// Required wall thickness [m]
pub fn calculate_wall_thickness(
    heat_flux: f32,
    pulse_duration: f32,
    material: &Material,
    delta_t_allow: f32,
) -> f32 {
    // Simplified thermal calculation
    // Energy deposited: E = q * t * A
    // Temperature rise: ΔT = E / (m * Cp) = q * t / (ρ * Cp * thickness)
    // Solving for thickness: t_wall = q * t_pulse / (ρ * Cp * ΔT)

    let heat_flux_w = heat_flux * 1e6;  // MW/m² to W/m²

    let thickness = heat_flux_w * pulse_duration /
                    (material.density * material.specific_heat * delta_t_allow);

    // Practical limits
    thickness.clamp(0.002, 0.05)  // 2mm to 50mm
}

/// Calculate required blanket thickness for tritium breeding
///
/// # Arguments
/// * `tbr_required` - Required Tritium Breeding Ratio (typically >1.1)
///
/// # Returns
/// Required blanket thickness [m]
pub fn calculate_blanket_thickness(tbr_required: f32) -> f32 {
    // Simplified: thickness scales roughly linearly with TBR requirement
    // Typical: 40-60 cm for TBR = 1.1-1.2
    let base_thickness = 0.4;  // 40 cm baseline
    let tbr_factor = (tbr_required - 1.0) / 0.1;  // Normalized

    (base_thickness * (1.0 + 0.2 * tbr_factor)).clamp(0.3, 0.8)
}

/// Calculate vacuum vessel thickness based on pressure
///
/// # Arguments
/// * `radius` - Vessel radius [m]
/// * `material` - Vessel material
/// * `pressure_diff` - Pressure differential [Pa] (typically 1 atm = 101325 Pa)
/// * `weld_efficiency` - Weld joint efficiency (typically 0.85)
///
/// # Returns
/// Required vessel thickness [m]
pub fn calculate_vessel_thickness(
    radius: f32,
    material: &Material,
    pressure_diff: f32,
    weld_efficiency: f32,
) -> f32 {
    // Thin-wall pressure vessel formula: t = P * R / (σ * η)
    let sigma_allow = material.yield_strength * 1e6 * 0.67;  // 2/3 of yield for pressure vessels

    let thickness = pressure_diff * radius / (sigma_allow * weld_efficiency);

    // Practical minimum
    thickness.max(0.02)  // 20mm minimum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_lookup() {
        assert!(get_material("nb3sn").is_some());
        assert!(get_material("rebco").is_some());
        assert!(get_material("nonexistent").is_none());
    }

    #[test]
    fn test_category_filter() {
        let hts = get_materials_by_category(MaterialCategory::SuperconductorHTS);
        assert!(hts.len() >= 2);  // REBCO, YBCO, Bi-2212
    }

    #[test]
    fn test_thickness_calculations() {
        let thickness = calculate_tf_coil_thickness(12.0, 6.0, &SS316L, 2.5);
        assert!(thickness > 0.1);  // Should be substantial

        let wall = calculate_wall_thickness(10.0, 400.0, &TUNGSTEN, 500.0);
        assert!(wall > 0.0 && wall < 0.1);
    }
}
