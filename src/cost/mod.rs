//! # Cost Model Module
//!
//! Advanced cost estimation with manufacturing complexity factors.
//!
//! ## Methodology
//!
//! Cost models include:
//! 1. **Direct costs**: Materials, labor, equipment
//! 2. **Complexity factors**: Geometric, tolerance, assembly difficulty
//! 3. **Learning curves**: Production rate effects
//! 4. **Risk adjustments**: Technical maturity factors
//!
//! ## Cost Estimation Relationships (CERs)
//!
//! Based on historical data from:
//! - ITER cost database
//! - Fusion power plant studies (ARIES, DEMO)
//! - High-field magnet development
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] Sheffield, J. "The physics of magnetic fusion reactors", Rev. Mod. Phys. 66, 1015 (1994)
//! [2] Maisonnier, D. et al. "DEMO and fusion power plant conceptual studies in Europe",
//!     Fusion Eng. Des. 81 (2006) 1123-1130
//! [3] Whyte, D.G. et al. "Smaller & Sooner: Exploiting High Magnetic Fields from New
//!     Superconductors for a More Attractive Fusion Energy Development Path",
//!     J. Fusion Energy 35, 41-53 (2016)

/// Currency unit (2026 USD millions)
pub type CostM = f64;

/// Manufacturing complexity factors
#[derive(Debug, Clone)]
pub struct ComplexityFactors {
    /// Geometric complexity (1.0 = simple, 3.0 = very complex)
    pub geometric: f64,
    /// Tolerance/precision factor (1.0 = standard, 2.0 = high precision)
    pub tolerance: f64,
    /// Material difficulty (1.0 = standard steel, 3.0 = exotic materials)
    pub material: f64,
    /// Assembly difficulty (1.0 = simple, 3.0 = complex integration)
    pub assembly: f64,
    /// Quality assurance level (1.0 = commercial, 2.0 = nuclear grade)
    pub qa_level: f64,
    /// Technical maturity (1.0 = proven, 2.0 = R&D phase)
    pub maturity: f64,
}

impl Default for ComplexityFactors {
    fn default() -> Self {
        Self {
            geometric: 1.0,
            tolerance: 1.0,
            material: 1.0,
            assembly: 1.0,
            qa_level: 1.0,
            maturity: 1.0,
        }
    }
}

impl ComplexityFactors {
    /// Create for HTS superconducting magnets
    pub fn hts_magnet() -> Self {
        Self {
            geometric: 1.8,    // Complex winding geometry
            tolerance: 2.0,    // Tight tolerances for field quality
            material: 2.5,     // REBCO is expensive
            assembly: 2.0,     // Delicate assembly
            qa_level: 2.0,     // Nuclear grade
            maturity: 1.5,     // Still developing
        }
    }

    /// Create for tungsten plasma-facing components
    pub fn tungsten_pfc() -> Self {
        Self {
            geometric: 2.0,    // Complex shapes for divertor
            tolerance: 1.8,    // Thermal management critical
            material: 2.0,     // Tungsten is difficult
            assembly: 1.5,     // Modular design
            qa_level: 2.0,     // Nuclear grade
            maturity: 1.2,     // Fairly mature
        }
    }

    /// Create for vacuum vessel
    pub fn vacuum_vessel() -> Self {
        Self {
            geometric: 2.5,    // Double-wall, complex ports
            tolerance: 1.5,    // Sealing critical
            material: 1.5,     // Special steel
            assembly: 2.5,     // On-site welding
            qa_level: 2.0,     // Nuclear grade
            maturity: 1.0,     // Well understood
        }
    }

    /// Create for cryogenic system
    pub fn cryogenic() -> Self {
        Self {
            geometric: 1.5,
            tolerance: 1.5,
            material: 1.2,
            assembly: 1.5,
            qa_level: 1.5,
            maturity: 1.0,     // Mature technology
        }
    }

    /// Combined complexity multiplier
    pub fn total_factor(&self) -> f64 {
        // Geometric mean gives balanced weighting
        (self.geometric * self.tolerance * self.material *
         self.assembly * self.qa_level * self.maturity).powf(1.0 / 6.0)
    }
}

/// Learning curve model
///
/// Cost for unit N: C_N = C_1 * N^b
/// where b = ln(learning_rate) / ln(2)
#[derive(Debug, Clone)]
pub struct LearningCurve {
    /// First unit cost
    pub first_unit_cost: CostM,
    /// Learning rate (0.8 = 80%, typical for manufacturing)
    pub learning_rate: f64,
}

impl LearningCurve {
    /// Create new learning curve
    pub fn new(first_unit_cost: CostM, learning_rate: f64) -> Self {
        Self { first_unit_cost, learning_rate }
    }

    /// Cost of unit N
    pub fn unit_cost(&self, n: usize) -> CostM {
        let b = self.learning_rate.ln() / 2.0_f64.ln();
        self.first_unit_cost * (n as f64).powf(b)
    }

    /// Total cost for N units (cumulative)
    pub fn cumulative_cost(&self, n: usize) -> CostM {
        (1..=n).map(|i| self.unit_cost(i)).sum()
    }

    /// Average cost per unit for N units
    pub fn average_cost(&self, n: usize) -> CostM {
        self.cumulative_cost(n) / n as f64
    }
}

/// Cost Estimation Relationship (CER) for major components
#[derive(Debug, Clone)]
pub enum CostEstimate {
    /// Parametric scaling: C = a * X^b where X is size parameter
    Parametric { coefficient: f64, exponent: f64, parameter_name: String },
    /// Engineering estimate (fixed)
    Engineering(CostM),
    /// Vendor quote with uncertainty
    Quote { base: CostM, uncertainty_fraction: f64 },
    /// Analogy with adjustment factor
    Analogy { reference_cost: CostM, adjustment: f64, reference_system: String },
}

impl CostEstimate {
    /// Evaluate cost for given parameter value
    pub fn evaluate(&self, parameter: f64) -> CostM {
        match self {
            Self::Parametric { coefficient, exponent, .. } => {
                coefficient * parameter.powf(*exponent)
            }
            Self::Engineering(cost) => *cost,
            Self::Quote { base, .. } => *base,
            Self::Analogy { reference_cost, adjustment, .. } => {
                reference_cost * adjustment
            }
        }
    }

    /// Evaluate with uncertainty bounds (low, nominal, high)
    pub fn with_uncertainty(&self, parameter: f64) -> (CostM, CostM, CostM) {
        let nominal = self.evaluate(parameter);

        match self {
            Self::Parametric { exponent, .. } => {
                // ±20% for well-characterized, ±50% for novel
                let factor = if exponent.abs() < 1.0 { 0.2 } else { 0.35 };
                (nominal * (1.0 - factor), nominal, nominal * (1.0 + factor))
            }
            Self::Engineering(_) => (nominal * 0.9, nominal, nominal * 1.3),
            Self::Quote { uncertainty_fraction, .. } => {
                (nominal * (1.0 - uncertainty_fraction),
                 nominal,
                 nominal * (1.0 + uncertainty_fraction))
            }
            Self::Analogy { .. } => (nominal * 0.7, nominal, nominal * 1.5),
        }
    }
}

/// Complete cost model for tokamak subsystem
#[derive(Debug, Clone)]
pub struct SubsystemCost {
    /// Subsystem name
    pub name: String,
    /// Cost estimation relationship
    pub cer: CostEstimate,
    /// Complexity factors
    pub complexity: ComplexityFactors,
    /// Learning curve (if multiple units)
    pub learning: Option<LearningCurve>,
    /// Installation factor (fraction of equipment cost)
    pub installation_factor: f64,
    /// Engineering factor (fraction of equipment cost)
    pub engineering_factor: f64,
}

impl SubsystemCost {
    /// Calculate total direct cost
    pub fn direct_cost(&self, parameter: f64, n_units: usize) -> CostM {
        let base = self.cer.evaluate(parameter);
        let complexity_adjusted = base * self.complexity.total_factor();

        let equipment = if let Some(ref lc) = self.learning {
            if n_units > 1 {
                lc.first_unit_cost * self.complexity.total_factor()
                    * (lc.cumulative_cost(n_units) / lc.cumulative_cost(1))
            } else {
                complexity_adjusted
            }
        } else {
            complexity_adjusted * n_units as f64
        };

        let installation = equipment * self.installation_factor;
        let engineering = equipment * self.engineering_factor;

        equipment + installation + engineering
    }
}

/// Complete cost model for fusion power plant
#[derive(Debug, Clone)]
pub struct FusionPlantCostModel {
    /// Subsystem costs
    pub subsystems: Vec<SubsystemCost>,
    /// Contingency factor
    pub contingency: f64,
    /// Indirect cost factor (management, overhead)
    pub indirect_factor: f64,
    /// Owner's cost factor
    pub owners_cost_factor: f64,
    /// Interest during construction (IDC) rate
    pub idc_rate: f64,
    /// Construction period (years)
    pub construction_years: f64,
}

impl FusionPlantCostModel {
    /// Create model for HTS compact tokamak (SPARC/ARC-like)
    pub fn hts_compact_tokamak() -> Self {
        let subsystems = vec![
            // Magnets: dominant cost for HTS machine
            SubsystemCost {
                name: "TF Magnets".to_string(),
                // Cost scales with stored energy: C ∝ E_mag^0.8
                cer: CostEstimate::Parametric {
                    coefficient: 50.0,  // $50M per GJ^0.8
                    exponent: 0.8,
                    parameter_name: "Stored energy (GJ)".to_string(),
                },
                complexity: ComplexityFactors::hts_magnet(),
                learning: Some(LearningCurve::new(100.0, 0.85)),
                installation_factor: 0.2,
                engineering_factor: 0.15,
            },
            SubsystemCost {
                name: "PF/CS Magnets".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 30.0,
                    exponent: 0.75,
                    parameter_name: "Stored energy (GJ)".to_string(),
                },
                complexity: ComplexityFactors::hts_magnet(),
                learning: Some(LearningCurve::new(80.0, 0.85)),
                installation_factor: 0.2,
                engineering_factor: 0.15,
            },

            // Vacuum Vessel
            SubsystemCost {
                name: "Vacuum Vessel".to_string(),
                // Cost scales with surface area: C ∝ A^0.7
                cer: CostEstimate::Parametric {
                    coefficient: 0.8,  // $0.8M per m² ^0.7
                    exponent: 0.7,
                    parameter_name: "Surface area (m²)".to_string(),
                },
                complexity: ComplexityFactors::vacuum_vessel(),
                learning: None,
                installation_factor: 0.35,
                engineering_factor: 0.1,
            },

            // First Wall / Blanket
            SubsystemCost {
                name: "First Wall & Blanket".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 1.5,
                    exponent: 0.65,
                    parameter_name: "Surface area (m²)".to_string(),
                },
                complexity: ComplexityFactors::tungsten_pfc(),
                learning: Some(LearningCurve::new(150.0, 0.90)),
                installation_factor: 0.25,
                engineering_factor: 0.15,
            },

            // Divertor
            SubsystemCost {
                name: "Divertor".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 5.0,
                    exponent: 0.6,
                    parameter_name: "Heat handling (MW)".to_string(),
                },
                complexity: ComplexityFactors::tungsten_pfc(),
                learning: Some(LearningCurve::new(50.0, 0.85)),
                installation_factor: 0.3,
                engineering_factor: 0.2,
            },

            // Cryogenic System
            SubsystemCost {
                name: "Cryogenic System".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 15.0,
                    exponent: 0.7,
                    parameter_name: "Cooling power (kW)".to_string(),
                },
                complexity: ComplexityFactors::cryogenic(),
                learning: None,
                installation_factor: 0.25,
                engineering_factor: 0.1,
            },

            // Heating Systems
            SubsystemCost {
                name: "RF Heating (ICRF/ECRH)".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 3.0,  // $3M per MW
                    exponent: 1.0,
                    parameter_name: "Power (MW)".to_string(),
                },
                complexity: ComplexityFactors::default(),
                learning: Some(LearningCurve::new(30.0, 0.90)),
                installation_factor: 0.15,
                engineering_factor: 0.1,
            },

            // Power Supply
            SubsystemCost {
                name: "Magnet Power Supply".to_string(),
                cer: CostEstimate::Parametric {
                    coefficient: 0.1,  // $0.1M per MVA
                    exponent: 0.85,
                    parameter_name: "Power (MVA)".to_string(),
                },
                complexity: ComplexityFactors::default(),
                learning: None,
                installation_factor: 0.2,
                engineering_factor: 0.1,
            },

            // Control System
            SubsystemCost {
                name: "Control & Instrumentation".to_string(),
                cer: CostEstimate::Engineering(100.0),  // Fixed ~$100M
                complexity: ComplexityFactors {
                    qa_level: 2.0,
                    maturity: 1.0,
                    ..Default::default()
                },
                learning: None,
                installation_factor: 0.3,
                engineering_factor: 0.25,
            },
        ];

        Self {
            subsystems,
            contingency: 0.25,           // 25% contingency
            indirect_factor: 0.20,       // 20% indirect costs
            owners_cost_factor: 0.10,    // 10% owner's costs
            idc_rate: 0.05,              // 5% interest rate
            construction_years: 8.0,      // 8 year construction
        }
    }

    /// Calculate total overnight capital cost
    pub fn overnight_cost(&self, parameters: &PlantParameters) -> CostBreakdown {
        let mut direct = 0.0;
        let mut subsystem_costs = Vec::new();

        // Calculate each subsystem
        for subsystem in &self.subsystems {
            let param = parameters.get_parameter(&subsystem.name);
            let n_units = parameters.get_units(&subsystem.name);
            let cost = subsystem.direct_cost(param, n_units);

            direct += cost;
            subsystem_costs.push((subsystem.name.clone(), cost));
        }

        let contingency = direct * self.contingency;
        let indirect = direct * self.indirect_factor;
        let overnight = direct + contingency + indirect;

        CostBreakdown {
            direct_cost: direct,
            contingency,
            indirect_cost: indirect,
            overnight_cost: overnight,
            subsystem_costs,
        }
    }

    /// Calculate total capital cost including IDC
    pub fn total_capital_cost(&self, parameters: &PlantParameters) -> f64 {
        let overnight = self.overnight_cost(parameters);

        // IDC using simple interest formula
        // IDC = Overnight * r * T / 2  (assuming linear spending profile)
        let idc = overnight.overnight_cost * self.idc_rate * self.construction_years / 2.0;
        let owners = overnight.overnight_cost * self.owners_cost_factor;

        overnight.overnight_cost + idc + owners
    }

    /// Calculate levelized cost of electricity (LCOE)
    pub fn lcoe(&self, parameters: &PlantParameters, capacity_factor: f64, lifetime_years: f64) -> f64 {
        let capital = self.total_capital_cost(parameters);

        // Assume fixed O&M is 2% of capital per year
        // Assume fuel cost is negligible for fusion

        let power_mw = parameters.net_electric_power;
        let annual_generation_mwh = power_mw * 8760.0 * capacity_factor;

        // Capital recovery factor
        let r = self.idc_rate;
        let n = lifetime_years;
        let crf = r * (1.0 + r).powf(n) / ((1.0 + r).powf(n) - 1.0);

        let annual_capital = capital * crf;
        let annual_om = capital * 0.02;

        // LCOE in $/MWh (convert from $M)
        (annual_capital + annual_om) * 1e6 / annual_generation_mwh
    }
}

/// Plant design parameters
#[derive(Debug, Clone)]
pub struct PlantParameters {
    /// Net electric power (MW)
    pub net_electric_power: f64,
    /// Major radius (m)
    pub major_radius: f64,
    /// Minor radius (m)
    pub minor_radius: f64,
    /// Toroidal field (T)
    pub toroidal_field: f64,
    /// Magnet stored energy (GJ)
    pub magnet_stored_energy: f64,
    /// First wall/vacuum vessel surface area (m²)
    pub first_wall_area: f64,
    /// Divertor heat load (MW)
    pub divertor_heat_load: f64,
    /// Cryogenic power (kW)
    pub cryo_power: f64,
    /// Heating power (MW)
    pub heating_power: f64,
    /// Magnet power supply (MVA)
    pub magnet_power: f64,
    /// Number of TF coils
    pub n_tf_coils: usize,
    /// Number of PF coils
    pub n_pf_coils: usize,
    /// Number of blanket modules
    pub n_blanket_modules: usize,
    /// Number of divertor cassettes
    pub n_divertor_cassettes: usize,
}

impl PlantParameters {
    /// Create parameters for SPARC-class device
    pub fn sparc_class() -> Self {
        Self {
            net_electric_power: 0.0,  // SPARC is Q>2, not a power plant
            major_radius: 1.85,
            minor_radius: 0.57,
            toroidal_field: 12.2,
            magnet_stored_energy: 1.5,  // ~1.5 GJ
            first_wall_area: 80.0,
            divertor_heat_load: 30.0,
            cryo_power: 50.0,
            heating_power: 25.0,
            magnet_power: 200.0,
            n_tf_coils: 18,
            n_pf_coils: 8,
            n_blanket_modules: 0,
            n_divertor_cassettes: 32,
        }
    }

    /// Create parameters for ARC-class pilot plant
    pub fn arc_class() -> Self {
        Self {
            net_electric_power: 200.0,  // ~200 MWe net
            major_radius: 3.3,
            minor_radius: 1.1,
            toroidal_field: 9.2,
            magnet_stored_energy: 8.0,  // ~8 GJ
            first_wall_area: 300.0,
            divertor_heat_load: 150.0,
            cryo_power: 100.0,
            heating_power: 40.0,
            magnet_power: 500.0,
            n_tf_coils: 18,
            n_pf_coils: 10,
            n_blanket_modules: 48,
            n_divertor_cassettes: 64,
        }
    }

    /// Get parameter value for a subsystem
    fn get_parameter(&self, subsystem: &str) -> f64 {
        match subsystem {
            "TF Magnets" | "PF/CS Magnets" => self.magnet_stored_energy,
            "Vacuum Vessel" | "First Wall & Blanket" => self.first_wall_area,
            "Divertor" => self.divertor_heat_load,
            "Cryogenic System" => self.cryo_power,
            "RF Heating (ICRF/ECRH)" => self.heating_power,
            "Magnet Power Supply" => self.magnet_power,
            _ => 1.0,
        }
    }

    /// Get number of units for a subsystem
    fn get_units(&self, subsystem: &str) -> usize {
        match subsystem {
            "TF Magnets" => self.n_tf_coils,
            "PF/CS Magnets" => self.n_pf_coils,
            "First Wall & Blanket" => self.n_blanket_modules.max(1),
            "Divertor" => self.n_divertor_cassettes.max(1),
            _ => 1,
        }
    }
}

/// Cost breakdown result
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    /// Direct cost (equipment + installation + engineering)
    pub direct_cost: CostM,
    /// Contingency
    pub contingency: CostM,
    /// Indirect costs
    pub indirect_cost: CostM,
    /// Total overnight cost
    pub overnight_cost: CostM,
    /// Cost by subsystem
    pub subsystem_costs: Vec<(String, CostM)>,
}

impl CostBreakdown {
    /// Generate cost report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== COST BREAKDOWN (2026 USD Millions) ===\n\n");

        report.push_str("SUBSYSTEM COSTS:\n");
        report.push_str("--------------\n");
        for (name, cost) in &self.subsystem_costs {
            report.push_str(&format!("  {:30} ${:>10.1}M\n", name, cost));
        }

        report.push_str("\nSUMMARY:\n");
        report.push_str("-------\n");
        report.push_str(&format!("  Direct Cost:      ${:>10.1}M\n", self.direct_cost));
        report.push_str(&format!("  Contingency:      ${:>10.1}M\n", self.contingency));
        report.push_str(&format!("  Indirect Cost:    ${:>10.1}M\n", self.indirect_cost));
        report.push_str(&format!("  --------------------------------\n"));
        report.push_str(&format!("  OVERNIGHT COST:   ${:>10.1}M\n", self.overnight_cost));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_factors() {
        let default = ComplexityFactors::default();
        assert!((default.total_factor() - 1.0).abs() < 1e-10);

        let hts = ComplexityFactors::hts_magnet();
        assert!(hts.total_factor() > 1.5);  // Should be notably higher
    }

    #[test]
    fn test_learning_curve() {
        let lc = LearningCurve::new(100.0, 0.8);

        // First unit
        assert!((lc.unit_cost(1) - 100.0).abs() < 1e-10);

        // Second unit should be 80% of first
        assert!((lc.unit_cost(2) - 80.0).abs() < 1.0);

        // Cumulative should be sum
        let cum_2 = lc.cumulative_cost(2);
        assert!((cum_2 - (100.0 + 80.0)).abs() < 1.0);
    }

    #[test]
    fn test_parametric_cer() {
        let cer = CostEstimate::Parametric {
            coefficient: 10.0,
            exponent: 0.7,
            parameter_name: "test".to_string(),
        };

        // C = 10 * 100^0.7 ≈ 251
        let cost = cer.evaluate(100.0);
        assert!((cost - 251.2).abs() < 1.0);
    }

    #[test]
    fn test_sparc_cost_estimate() {
        let model = FusionPlantCostModel::hts_compact_tokamak();
        let params = PlantParameters::sparc_class();

        let breakdown = model.overnight_cost(&params);

        // SPARC estimated at ~$2-5B with complexity factors
        // With manufacturing complexity, can exceed initial estimates
        assert!(breakdown.overnight_cost > 1000.0);  // > $1B
        assert!(breakdown.overnight_cost < 50000.0); // < $50B (sanity check)

        println!("{}", breakdown.report());
    }

    #[test]
    fn test_arc_lcoe() {
        let model = FusionPlantCostModel::hts_compact_tokamak();
        let params = PlantParameters::arc_class();

        // Calculate LCOE with 70% capacity factor, 40-year lifetime
        let lcoe = model.lcoe(&params, 0.70, 40.0);

        // First-of-a-kind (FOAK) fusion plants will have very high LCOE
        // This is expected for novel technology with high capital costs
        // LCOE decreases with:
        // - Learning curves (Nth-of-a-kind costs)
        // - Higher capacity factors (with experience)
        // - Design simplification
        // Current model includes full complexity factors for FOAK
        assert!(lcoe > 30.0);     // At least $30/MWh (floor)
        assert!(lcoe < 10000.0);  // Sanity upper bound

        // Print actual LCOE for reference
        println!("ARC-class LCOE: ${:.0}/MWh (FOAK estimate)", lcoe);
    }
}
