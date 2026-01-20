//! # Materials Module
//!
//! Temperature-dependent material properties database for tokamak components.
//!
//! ## Data Sources
//!
//! - ITER Material Properties Handbook (MPH)
//! - Nuclear Fusion Materials Database
//! - REBCO superconductor datasheet
//! - Tungsten monoblock literature
//!
//! ## Mathematical Models
//!
//! Properties are fitted to polynomial or exponential functions of temperature:
//! - Thermal conductivity: k(T) = a₀ + a₁T + a₂T² + a₃T³
//! - Specific heat: Cp(T) = b₀ + b₁T + b₂T² (Debye model corrections)
//! - Electrical resistivity: ρ(T) = ρ₀(1 + α(T - T₀)) (metals)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] ITER Material Properties Handbook, ITER Document G 74 MA 8 01-05-28 W 0.2
//! [2] Karditsas, P.J. & Baptiste, M.J. "Thermal and Structural Properties of
//!     Fusion related Materials", UKAEA FUS 294 (1995)
//! [3] Senatore, C. et al. "Progresses and challenges in the development of
//!     high-field solenoidal magnets based on RE123 coated conductors"
//!     Supercond. Sci. Technol. 27 (2014) 103001

/// Material identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Material {
    /// Pure tungsten (W) - plasma-facing component
    Tungsten,
    /// Tungsten-copper composite (W-Cu)
    TungstenCopper,
    /// EUROFER-97 reduced activation steel
    Eurofer97,
    /// 316LN stainless steel
    SS316LN,
    /// REBCO high-temperature superconductor tape
    REBCO,
    /// Nb3Sn low-temperature superconductor
    Nb3Sn,
    /// NbTi low-temperature superconductor
    NbTi,
    /// Copper (Cu) - stabilizer
    Copper,
    /// CuCrZr copper alloy - heat sink
    CuCrZr,
    /// Beryllium (Be) - first wall coating
    Beryllium,
    /// Carbon fiber composite (CFC)
    CarbonFiberComposite,
    /// Inconel 718 - structural
    Inconel718,
    /// Liquid helium (He)
    LiquidHelium,
    /// Water (H2O) - coolant
    Water,
    /// Lithium (Li) - breeding
    Lithium,
    /// Lead-lithium eutectic (PbLi)
    PbLi,
    /// Lithium orthosilicate ceramic breeder (Li4SiO4)
    Li4SiO4,
    /// Lithium titanate ceramic breeder (Li2TiO3)
    Li2TiO3,
}

/// Temperature-dependent material properties
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    /// Material name
    pub material: Material,
    /// Valid temperature range [K]
    pub temp_range: (f64, f64),
    /// Density [kg/m³] (assumed constant)
    pub density: f64,
    /// Melting point [K]
    pub melting_point: f64,
    /// Thermal conductivity coefficients [W/(m·K)]
    /// k(T) = sum(k_coeffs[i] * T^i)
    pub k_coeffs: Vec<f64>,
    /// Specific heat coefficients [J/(kg·K)]
    /// Cp(T) = sum(cp_coeffs[i] * T^i)
    pub cp_coeffs: Vec<f64>,
    /// Electrical resistivity at reference temperature [Ω·m]
    pub rho_0: f64,
    /// Temperature coefficient of resistivity [1/K]
    pub alpha_rho: f64,
    /// Reference temperature for resistivity [K]
    pub t_ref: f64,
    /// Young's modulus coefficients [Pa]
    /// E(T) = sum(e_coeffs[i] * T^i)
    pub e_coeffs: Vec<f64>,
    /// Poisson's ratio (assumed constant)
    pub poisson_ratio: f64,
    /// Thermal expansion coefficients [1/K]
    /// α(T) = sum(alpha_coeffs[i] * T^i)
    pub alpha_coeffs: Vec<f64>,
    /// Yield strength coefficients [Pa]
    /// σ_y(T) = sum(yield_coeffs[i] * T^i)
    pub yield_coeffs: Vec<f64>,
    /// For superconductors: critical temperature [K]
    pub tc: Option<f64>,
    /// For superconductors: critical current density at 4.2K, 12T [A/m²]
    pub jc_reference: Option<f64>,
}

impl MaterialProperties {
    /// Calculate thermal conductivity at temperature T [W/(m·K)]
    ///
    /// Uses polynomial fit: k(T) = Σᵢ aᵢTⁱ
    pub fn thermal_conductivity(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(self.temp_range.0, self.temp_range.1);
        self.k_coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * t_clamped.powi(i as i32))
            .sum()
    }

    /// Calculate specific heat at temperature T [J/(kg·K)]
    ///
    /// Uses polynomial fit: Cp(T) = Σᵢ bᵢTⁱ
    pub fn specific_heat(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(self.temp_range.0, self.temp_range.1);
        self.cp_coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * t_clamped.powi(i as i32))
            .sum()
    }

    /// Calculate electrical resistivity at temperature T [Ω·m]
    ///
    /// Uses linear model: ρ(T) = ρ₀[1 + α(T - T_ref)]
    pub fn electrical_resistivity(&self, t: f64) -> f64 {
        self.rho_0 * (1.0 + self.alpha_rho * (t - self.t_ref))
    }

    /// Calculate Young's modulus at temperature T [Pa]
    pub fn youngs_modulus(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(self.temp_range.0, self.temp_range.1);
        self.e_coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * t_clamped.powi(i as i32))
            .sum()
    }

    /// Calculate thermal expansion coefficient at temperature T [1/K]
    pub fn thermal_expansion(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(self.temp_range.0, self.temp_range.1);
        self.alpha_coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * t_clamped.powi(i as i32))
            .sum()
    }

    /// Calculate yield strength at temperature T [Pa]
    pub fn yield_strength(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(self.temp_range.0, self.temp_range.1);
        self.yield_coeffs.iter()
            .enumerate()
            .map(|(i, &c)| c * t_clamped.powi(i as i32))
            .sum()
    }

    /// Calculate thermal diffusivity [m²/s]
    ///
    /// α = k / (ρ · Cp)
    pub fn thermal_diffusivity(&self, t: f64) -> f64 {
        let k = self.thermal_conductivity(t);
        let cp = self.specific_heat(t);
        k / (self.density * cp)
    }

    /// For superconductors: calculate critical current density at given T and B
    ///
    /// Uses scaling law: Jc(T,B) = Jc₀ · (1 - T/Tc)^n · f(B)
    ///
    /// ## Arguments
    /// * `t` - Temperature [K]
    /// * `b` - Magnetic field [T]
    ///
    /// ## Returns
    /// Critical current density [A/m²], or None if not a superconductor
    pub fn critical_current_density(&self, t: f64, b: f64) -> Option<f64> {
        let tc = self.tc?;
        let jc0 = self.jc_reference?;

        if t >= tc {
            return Some(0.0);
        }

        // Temperature scaling: (1 - T/Tc)^n with n ≈ 1.5 for REBCO
        let t_factor = (1.0 - t / tc).powf(1.5);

        // Field scaling depends on material
        let b_factor = match self.material {
            Material::REBCO => {
                // REBCO: weak field dependence up to ~30T
                // Jc(B) ≈ Jc₀ / (1 + B/B₀) with B₀ ≈ 30T
                1.0 / (1.0 + b / 30.0)
            }
            Material::Nb3Sn => {
                // Nb3Sn: Kramer-type scaling
                // Jc(B) ∝ B^(-0.5) * (1 - B/Bc2)^2
                let bc2 = 25.0; // Upper critical field at 4.2K
                if b >= bc2 {
                    0.0
                } else {
                    b.powf(-0.5) * (1.0 - b / bc2).powi(2) * bc2.sqrt()
                }
            }
            Material::NbTi => {
                // NbTi: empirical fit
                let bc2 = 10.0; // Upper critical field at 4.2K
                if b >= bc2 {
                    0.0
                } else {
                    (1.0 - b / bc2).powi(2)
                }
            }
            _ => 1.0,
        };

        Some(jc0 * t_factor * b_factor)
    }

    /// Check if material is in superconducting state
    pub fn is_superconducting(&self, t: f64, b: f64) -> bool {
        self.critical_current_density(t, b)
            .map(|jc| jc > 0.0)
            .unwrap_or(false)
    }
}

/// Material database with pre-computed properties
pub struct MaterialDatabase {
    materials: std::collections::HashMap<Material, MaterialProperties>,
}

impl MaterialDatabase {
    /// Create database with ITER MPH data
    pub fn new() -> Self {
        let mut materials = std::collections::HashMap::new();

        // ========================================================================
        // TUNGSTEN (W) - ITER MPH G 74 MA 8
        // ========================================================================
        // Plasma-facing material for divertor and first wall armor
        materials.insert(Material::Tungsten, MaterialProperties {
            material: Material::Tungsten,
            temp_range: (300.0, 3500.0),
            density: 19300.0,  // kg/m³
            melting_point: 3695.0,  // K
            // Thermal conductivity fit: k = 174.9 - 0.1067T + 5.0e-5T² - 7.8e-9T³ [W/(m·K)]
            // From ITER MPH, valid 300-3000K
            k_coeffs: vec![174.9, -0.1067, 5.0e-5, -7.8e-9],
            // Specific heat fit: Cp = 128.3 + 0.0321T - 3.3e-6T² [J/(kg·K)]
            cp_coeffs: vec![128.3, 0.0321, -3.3e-6],
            // Electrical resistivity at 293K: 5.28e-8 Ω·m
            rho_0: 5.28e-8,
            alpha_rho: 4.82e-3,  // Temperature coefficient
            t_ref: 293.0,
            // Young's modulus fit: E = 411e9 - 1.0e7T [Pa]
            e_coeffs: vec![411e9, -1.0e7],
            poisson_ratio: 0.28,
            // Thermal expansion: α = 4.5e-6 + 3.0e-9T [1/K]
            alpha_coeffs: vec![4.5e-6, 3.0e-9],
            // Yield strength decreases with T
            yield_coeffs: vec![1.0e9, -5.0e5],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // EUROFER-97 - ITER MPH, reduced activation ferritic-martensitic steel
        // ========================================================================
        materials.insert(Material::Eurofer97, MaterialProperties {
            material: Material::Eurofer97,
            temp_range: (300.0, 900.0),
            density: 7870.0,
            melting_point: 1800.0,
            // k = 33.6 - 1.8e-2T + 2.3e-5T² [W/(m·K)]
            k_coeffs: vec![33.6, -1.8e-2, 2.3e-5],
            // Cp = 439 + 0.27T [J/(kg·K)]
            cp_coeffs: vec![439.0, 0.27],
            rho_0: 4.5e-7,
            alpha_rho: 1.0e-3,
            t_ref: 293.0,
            // E = 217e9 - 5.2e7T [Pa]
            e_coeffs: vec![217e9, -5.2e7],
            poisson_ratio: 0.3,
            alpha_coeffs: vec![1.05e-5, 4.0e-9],
            // σ_y at 20°C: 530 MPa, decreases with T
            yield_coeffs: vec![530e6, -3.0e5],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // 316LN STAINLESS STEEL - ITER MPH
        // ========================================================================
        materials.insert(Material::SS316LN, MaterialProperties {
            material: Material::SS316LN,
            temp_range: (4.0, 1000.0),
            density: 8000.0,
            melting_point: 1700.0,
            // Thermal conductivity varies significantly with T
            // Low T: k ≈ 0.2T, High T: k ≈ 15 + 0.013T
            k_coeffs: vec![12.0, 0.015],
            cp_coeffs: vec![450.0, 0.25],
            rho_0: 7.5e-7,
            alpha_rho: 4.0e-4,
            t_ref: 293.0,
            e_coeffs: vec![200e9, -6.0e7],
            poisson_ratio: 0.3,
            alpha_coeffs: vec![1.6e-5, 3.0e-9],
            yield_coeffs: vec![290e6, -2.0e5],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // COPPER (Cu) - ITER MPH
        // ========================================================================
        materials.insert(Material::Copper, MaterialProperties {
            material: Material::Copper,
            temp_range: (4.0, 1200.0),
            density: 8960.0,
            melting_point: 1358.0,
            // High purity Cu: k ≈ 400 at RT, increases at low T
            k_coeffs: vec![401.0, -0.0175],
            cp_coeffs: vec![385.0, 0.0],
            rho_0: 1.68e-8,
            alpha_rho: 4.3e-3,
            t_ref: 293.0,
            e_coeffs: vec![130e9, -3.5e7],
            poisson_ratio: 0.34,
            alpha_coeffs: vec![1.65e-5, 5.0e-9],
            yield_coeffs: vec![70e6, -5.0e4],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // CuCrZr - ITER MPH, precipitation-hardened copper alloy
        // ========================================================================
        materials.insert(Material::CuCrZr, MaterialProperties {
            material: Material::CuCrZr,
            temp_range: (300.0, 700.0),
            density: 8900.0,
            melting_point: 1350.0,
            // k = 318 + 0.055T [W/(m·K)]
            k_coeffs: vec![318.0, 0.055],
            cp_coeffs: vec![390.0, 0.05],
            rho_0: 2.2e-8,
            alpha_rho: 4.0e-3,
            t_ref: 293.0,
            e_coeffs: vec![128e9, -4.0e7],
            poisson_ratio: 0.33,
            alpha_coeffs: vec![1.7e-5, 3.0e-9],
            // Higher yield than pure Cu
            yield_coeffs: vec![350e6, -3.0e5],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // REBCO (ReBa2Cu3O7-x) - High-Temperature Superconductor
        // ========================================================================
        // Data from SuperPower and literature
        materials.insert(Material::REBCO, MaterialProperties {
            material: Material::REBCO,
            temp_range: (4.0, 92.0),
            density: 6300.0,  // Effective tape density
            melting_point: 1300.0,  // Decomposition
            // Thermal conductivity along tape
            k_coeffs: vec![10.0, 0.5],  // Simplified, dominated by substrate
            cp_coeffs: vec![200.0, 1.0],
            rho_0: 1e-6,  // Normal state
            alpha_rho: 0.0,
            t_ref: 100.0,
            e_coeffs: vec![150e9],  // Hastelloy substrate dominated
            poisson_ratio: 0.3,
            alpha_coeffs: vec![1.3e-5],
            yield_coeffs: vec![1e9],  // Limited by substrate
            // Superconducting properties
            tc: Some(92.0),  // Critical temperature [K]
            // Engineering Jc at 4.2K, 12T, parallel field: ~3000 A/mm² = 3e9 A/m²
            jc_reference: Some(3.0e9),
        });

        // ========================================================================
        // Nb3Sn - Low-Temperature Superconductor
        // ========================================================================
        materials.insert(Material::Nb3Sn, MaterialProperties {
            material: Material::Nb3Sn,
            temp_range: (1.8, 18.3),
            density: 8900.0,
            melting_point: 2400.0,
            k_coeffs: vec![0.1, 0.01],
            cp_coeffs: vec![0.5, 0.1],
            rho_0: 1e-6,
            alpha_rho: 0.0,
            t_ref: 20.0,
            e_coeffs: vec![160e9],
            poisson_ratio: 0.3,
            alpha_coeffs: vec![8.0e-6],
            yield_coeffs: vec![100e6],  // Brittle
            tc: Some(18.3),
            // Jc at 4.2K, 12T: ~3000 A/mm² = 3e9 A/m²
            jc_reference: Some(3.0e9),
        });

        // ========================================================================
        // BERYLLIUM (Be) - First wall coating
        // ========================================================================
        materials.insert(Material::Beryllium, MaterialProperties {
            material: Material::Beryllium,
            temp_range: (300.0, 1500.0),
            density: 1850.0,
            melting_point: 1560.0,
            // k = 200 - 0.05T [W/(m·K)]
            k_coeffs: vec![200.0, -0.05],
            cp_coeffs: vec![1825.0, 0.5],
            rho_0: 4.0e-8,
            alpha_rho: 6.0e-3,
            t_ref: 293.0,
            e_coeffs: vec![303e9, -5.0e7],
            poisson_ratio: 0.07,
            alpha_coeffs: vec![1.15e-5, 8.0e-9],
            yield_coeffs: vec![240e6, -1.5e5],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // WATER (H2O) - Coolant (saturated liquid at various T)
        // ========================================================================
        materials.insert(Material::Water, MaterialProperties {
            material: Material::Water,
            temp_range: (273.0, 473.0),
            density: 1000.0,  // Varies with T, simplified
            melting_point: 273.15,
            // k ≈ 0.6 W/(m·K) at RT
            k_coeffs: vec![0.569, 1.88e-3, -7.4e-6],
            // Cp ≈ 4186 J/(kg·K) at RT
            cp_coeffs: vec![4186.0, -0.1],
            rho_0: 1e10,  // Insulator
            alpha_rho: 0.0,
            t_ref: 293.0,
            e_coeffs: vec![2.2e9],  // Bulk modulus
            poisson_ratio: 0.5,
            alpha_coeffs: vec![2.1e-4],  // Volumetric
            yield_coeffs: vec![0.0],
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // Li4SiO4 - Lithium Orthosilicate Ceramic Breeder
        // ========================================================================
        // ITER Test Blanket Module reference material
        // Excellent tritium release at T > 300°C
        materials.insert(Material::Li4SiO4, MaterialProperties {
            material: Material::Li4SiO4,
            temp_range: (300.0, 1200.0),
            density: 2390.0,  // kg/m³ (pebble bed ~60% packing)
            melting_point: 1528.0,  // K (decomposition)
            // k = 2.5 - 0.001T (ceramic, decreases with T)
            k_coeffs: vec![2.5, -0.001],
            // Cp = 1100 + 0.3T [J/(kg·K)]
            cp_coeffs: vec![1100.0, 0.3],
            rho_0: 1e12,  // Insulator
            alpha_rho: 0.0,
            t_ref: 300.0,
            e_coeffs: vec![90e9],  // ~90 GPa
            poisson_ratio: 0.25,
            alpha_coeffs: vec![1.3e-5],
            yield_coeffs: vec![100e6],  // Brittle ceramic
            tc: None,
            jc_reference: None,
        });

        // ========================================================================
        // Li2TiO3 - Lithium Metatitanate Ceramic Breeder
        // ========================================================================
        // Alternative to Li4SiO4 with better irradiation stability
        materials.insert(Material::Li2TiO3, MaterialProperties {
            material: Material::Li2TiO3,
            temp_range: (300.0, 1400.0),
            density: 3430.0,  // kg/m³
            melting_point: 1806.0,  // K
            // k = 3.0 - 0.0015T
            k_coeffs: vec![3.0, -0.0015],
            cp_coeffs: vec![900.0, 0.25],
            rho_0: 1e11,  // Insulator
            alpha_rho: 0.0,
            t_ref: 300.0,
            e_coeffs: vec![120e9],
            poisson_ratio: 0.24,
            alpha_coeffs: vec![1.1e-5],
            yield_coeffs: vec![120e6],
            tc: None,
            jc_reference: None,
        });

        Self { materials }
    }

    /// Get material properties
    pub fn get(&self, material: Material) -> Option<&MaterialProperties> {
        self.materials.get(&material)
    }

    /// Get thermal conductivity at temperature
    pub fn thermal_conductivity(&self, material: Material, t: f64) -> Option<f64> {
        self.get(material).map(|m| m.thermal_conductivity(t))
    }

    /// Get specific heat at temperature
    pub fn specific_heat(&self, material: Material, t: f64) -> Option<f64> {
        self.get(material).map(|m| m.specific_heat(t))
    }

    /// Get all superconducting materials
    pub fn superconductors(&self) -> Vec<Material> {
        self.materials
            .values()
            .filter(|m| m.tc.is_some())
            .map(|m| m.material)
            .collect()
    }

    /// Interpolate between two materials (for composites)
    ///
    /// ## Arguments
    /// * `m1` - First material
    /// * `m2` - Second material
    /// * `f1` - Volume fraction of first material (0-1)
    ///
    /// ## Returns
    /// Effective thermal conductivity using Maxwell-Garnett mixing rule
    pub fn effective_thermal_conductivity(
        &self,
        m1: Material,
        m2: Material,
        f1: f64,
        t: f64,
    ) -> Option<f64> {
        let k1 = self.thermal_conductivity(m1, t)?;
        let k2 = self.thermal_conductivity(m2, t)?;
        let f2 = 1.0 - f1;

        // Maxwell-Garnett mixing rule for spherical inclusions
        // k_eff = k1 * (2k1 + k2 + 2f2(k2 - k1)) / (2k1 + k2 - f2(k2 - k1))
        let numerator = 2.0 * k1 + k2 + 2.0 * f2 * (k2 - k1);
        let denominator = 2.0 * k1 + k2 - f2 * (k2 - k1);

        Some(k1 * numerator / denominator)
    }
}

impl Default for MaterialDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate heat flux limit for material
///
/// Based on thermal stress criterion:
/// q_max = k · σ_y / (E · α · L)
///
/// ## Arguments
/// * `mat` - Material properties
/// * `t` - Operating temperature [K]
/// * `thickness` - Armor/coating thickness [m]
///
/// ## Returns
/// Maximum allowable heat flux [W/m²]
pub fn max_heat_flux(mat: &MaterialProperties, t: f64, thickness: f64) -> f64 {
    let k = mat.thermal_conductivity(t);
    let sigma_y = mat.yield_strength(t);
    let e = mat.youngs_modulus(t);
    let alpha = mat.thermal_expansion(t);

    // Safety factor
    let sf = 2.0;

    k * sigma_y / (sf * e * alpha * thickness)
}

/// Calculate critical heat flux for water cooling (Tong-75 correlation)
///
/// q''_CHF = 0.23 * h_fg * ρ_v * [σg(ρ_l - ρ_v)/ρ_v²]^0.25
///
/// ## Arguments
/// * `pressure` - System pressure [Pa]
/// * `mass_flux` - Mass flux [kg/(m²·s)]
/// * `quality` - Steam quality (0-1)
///
/// ## Returns
/// Critical heat flux [W/m²]
pub fn critical_heat_flux_water(pressure: f64, _mass_flux: f64, quality: f64) -> f64 {
    // Simplified Zuber-type correlation
    let p_mpa = pressure / 1e6;

    // Saturation properties (simplified polynomial fits)
    let h_fg = 2.257e6 * (1.0 - 0.13 * p_mpa);  // Latent heat [J/kg]
    let rho_l = 958.0 - 1.0 * p_mpa;  // Liquid density [kg/m³]
    let rho_v = 0.597 * p_mpa.powf(0.8);  // Vapor density [kg/m³]
    let sigma = 0.0589 * (1.0 - 0.1 * p_mpa);  // Surface tension [N/m]
    let g = 9.81;

    // Zuber correlation
    let q_chf_pool = 0.131 * h_fg * rho_v
        * (sigma * g * (rho_l - rho_v) / (rho_v * rho_v)).powf(0.25);

    // Quality correction (decreases CHF)
    let quality_factor = (1.0 - quality).max(0.1);

    q_chf_pool * quality_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tungsten_properties() {
        let db = MaterialDatabase::new();
        let w = db.get(Material::Tungsten).unwrap();

        // Check thermal conductivity at 500K (should be ~140-160 W/(m·K))
        let k_500 = w.thermal_conductivity(500.0);
        assert!(k_500 > 100.0 && k_500 < 200.0,
            "Tungsten k at 500K should be ~140 W/(m·K), got {}", k_500);

        // Check at 1000K (should be ~110-130 W/(m·K))
        let k_1000 = w.thermal_conductivity(1000.0);
        assert!(k_1000 > 80.0 && k_1000 < 150.0,
            "Tungsten k at 1000K should be ~120 W/(m·K), got {}", k_1000);

        // Thermal conductivity should decrease with temperature for W
        assert!(k_1000 < k_500);
    }

    #[test]
    fn test_rebco_superconductor() {
        let db = MaterialDatabase::new();
        let rebco = db.get(Material::REBCO).unwrap();

        // Check Tc
        assert_eq!(rebco.tc, Some(92.0));

        // Jc at 4.2K, 12T should be positive
        let jc = rebco.critical_current_density(4.2, 12.0).unwrap();
        assert!(jc > 1e9, "REBCO Jc should be > 1 GA/m², got {}", jc);

        // Jc should be 0 above Tc
        let jc_hot = rebco.critical_current_density(100.0, 0.0).unwrap();
        assert_eq!(jc_hot, 0.0);

        // Jc should decrease with temperature
        let jc_20k = rebco.critical_current_density(20.0, 12.0).unwrap();
        let jc_40k = rebco.critical_current_density(40.0, 12.0).unwrap();
        assert!(jc_20k > jc_40k);
    }

    #[test]
    fn test_composite_thermal_conductivity() {
        let db = MaterialDatabase::new();

        // W-Cu composite at 500K
        let k_eff = db.effective_thermal_conductivity(
            Material::Tungsten,
            Material::Copper,
            0.8,  // 80% W, 20% Cu
            500.0
        ).unwrap();

        let k_w = db.thermal_conductivity(Material::Tungsten, 500.0).unwrap();
        let k_cu = db.thermal_conductivity(Material::Copper, 500.0).unwrap();

        // Effective k should be between component values
        let k_min = k_w.min(k_cu);
        let k_max = k_w.max(k_cu);
        assert!(k_eff >= k_min && k_eff <= k_max);
    }

    #[test]
    fn test_max_heat_flux() {
        let db = MaterialDatabase::new();
        let w = db.get(Material::Tungsten).unwrap();

        // For 5mm tungsten armor at 1000K
        let q_max = max_heat_flux(w, 1000.0, 0.005);

        // Should be on the order of 10 MW/m² for tungsten
        assert!(q_max > 1e6 && q_max < 1e8,
            "Tungsten max heat flux should be ~10 MW/m², got {} MW/m²", q_max / 1e6);
    }

    #[test]
    fn test_chf() {
        // CHF at 1 MPa
        let q_chf = critical_heat_flux_water(1e6, 1000.0, 0.0);

        // Should be ~1-3 MW/m² for pool boiling
        assert!(q_chf > 0.5e6 && q_chf < 5e6,
            "CHF should be ~1-3 MW/m², got {} MW/m²", q_chf / 1e6);
    }
}
