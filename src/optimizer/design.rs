//! # Diseño de Reactor
//!
//! Estructura que representa un diseño completo de reactor tokamak.

use std::collections::HashMap;
use crate::optimizer::parameters::{MagnetType, WallMaterial, BlanketType};

/// Diseño completo de un reactor tokamak
#[derive(Debug, Clone)]
pub struct ReactorDesign {
    // ========== IDENTIFICACIÓN ==========
    /// ID único del diseño
    pub id: String,
    /// Generación (para algoritmos evolutivos)
    pub generation: usize,

    // ========== CAPA 1: PLASMA ==========
    /// Densidad electrónica (m⁻³)
    pub density: f64,
    /// Temperatura iónica (keV)
    pub ion_temperature_kev: f64,
    /// Temperatura electrónica (keV)
    pub electron_temperature_kev: f64,
    /// Z efectivo
    pub z_effective: f64,
    /// Fracción de deuterio
    pub deuterium_fraction: f64,

    // ========== CAPA 2: GEOMETRÍA ==========
    /// Radio mayor (m)
    pub major_radius: f64,
    /// Radio menor (m)
    pub minor_radius: f64,
    /// Elongación
    pub elongation: f64,
    /// Triangularidad
    pub triangularity: f64,

    // ========== CAPA 3: MAGNÉTICO ==========
    /// Campo toroidal en eje (T)
    pub toroidal_field: f64,
    /// Corriente de plasma (MA)
    pub plasma_current_ma: f64,
    /// Tecnología de imanes
    pub magnet_technology: MagnetType,
    /// Número de bobinas TF
    pub n_tf_coils: usize,
    /// Build radial de bobina TF (m)
    pub tf_coil_radial_build: f64,

    // ========== CAPA 4: CALENTAMIENTO ==========
    /// Potencia ICRF (MW)
    pub icrf_power_mw: f64,
    /// Potencia ECRH (MW)
    pub ecrh_power_mw: f64,
    /// Potencia NBI (MW)
    pub nbi_power_mw: f64,

    // ========== CAPA 5: BLINDAJE ==========
    /// Espesor de blindaje (m)
    pub shield_thickness: f64,
    /// Material de primera pared
    pub first_wall_material: WallMaterial,
    /// Tipo de blanket
    pub blanket_type: BlanketType,

    // ========== CAPA 6: INFRAESTRUCTURA ==========
    /// Margen de criostato (m)
    pub cryostat_margin: f64,
    /// Altura de grúa (m)
    pub crane_height: f64,

    // ========== RESULTADOS CALCULADOS ==========
    /// Beta normalizado
    pub beta_n: f64,
    /// Factor Q
    pub q_factor: f64,
    /// Potencia de fusión (MW)
    pub fusion_power_mw: f64,
    /// Tiempo de confinamiento (s)
    pub confinement_time: f64,
    /// Triple producto
    pub triple_product: f64,

    // ========== OBJETIVOS Y RESTRICCIONES ==========
    /// Valores de funciones objetivo
    pub objectives: HashMap<String, f64>,
    /// ¿Es factible?
    pub feasible: bool,
    /// Violaciones de restricciones
    pub constraint_violations: Vec<String>,
    /// Rank de Pareto (para NSGA-II)
    pub pareto_rank: usize,
    /// Distancia de crowding
    pub crowding_distance: f64,
}

impl ReactorDesign {
    /// Crea un nuevo diseño con valores por defecto
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            generation: 0,
            density: 1e20,
            ion_temperature_kev: 15.0,
            electron_temperature_kev: 15.0,
            z_effective: 1.7,
            deuterium_fraction: 0.5,
            major_radius: 3.0,
            minor_radius: 1.0,
            elongation: 1.8,
            triangularity: 0.4,
            toroidal_field: 8.0,
            plasma_current_ma: 8.0,
            magnet_technology: MagnetType::HtsRebco,
            n_tf_coils: 18,
            tf_coil_radial_build: 0.5,
            icrf_power_mw: 20.0,
            ecrh_power_mw: 10.0,
            nbi_power_mw: 20.0,
            shield_thickness: 0.6,
            first_wall_material: WallMaterial::Tungsten,
            blanket_type: BlanketType::Wcll,
            cryostat_margin: 1.2,
            crane_height: 18.0,
            beta_n: 0.0,
            q_factor: 0.0,
            fusion_power_mw: 0.0,
            confinement_time: 0.0,
            triple_product: 0.0,
            objectives: HashMap::new(),
            feasible: true,
            constraint_violations: Vec::new(),
            pareto_rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Aspect ratio R/a
    pub fn aspect_ratio(&self) -> f64 {
        self.major_radius / self.minor_radius
    }

    /// Inverso del aspect ratio (epsilon)
    pub fn inverse_aspect_ratio(&self) -> f64 {
        self.minor_radius / self.major_radius
    }

    /// Volumen del plasma (m³)
    pub fn plasma_volume(&self) -> f64 {
        2.0 * std::f64::consts::PI * std::f64::consts::PI
            * self.major_radius
            * self.minor_radius.powi(2)
            * self.elongation
    }

    /// Área de superficie del plasma (m²)
    pub fn plasma_surface(&self) -> f64 {
        // Aproximación para plasma elongado
        4.0 * std::f64::consts::PI * std::f64::consts::PI
            * self.major_radius
            * self.minor_radius
            * self.elongation.sqrt()
    }

    /// Potencia total de calentamiento (MW)
    pub fn total_heating_power(&self) -> f64 {
        self.icrf_power_mw + self.ecrh_power_mw + self.nbi_power_mw
    }

    /// Potencia total de calentamiento en Watts
    pub fn total_heating_power_w(&self) -> f64 {
        self.total_heating_power() * 1e6
    }

    /// Espesor del blanket según tipo
    pub fn blanket_thickness(&self) -> f64 {
        self.blanket_type.typical_thickness()
    }

    /// Factor de seguridad q95 aproximado
    pub fn q95(&self) -> f64 {
        5.0 * self.minor_radius.powi(2) * self.toroidal_field
            * (1.0 + self.elongation.powi(2))
            / (2.0 * self.major_radius * self.plasma_current_ma)
    }

    /// Límite de Greenwald (10²⁰ m⁻³)
    pub fn greenwald_density(&self) -> f64 {
        self.plasma_current_ma / (std::f64::consts::PI * self.minor_radius.powi(2))
    }

    /// Fracción de Greenwald
    pub fn greenwald_fraction(&self) -> f64 {
        self.density / (self.greenwald_density() * 1e20)
    }

    /// Beta toroidal aproximado
    pub fn beta_toroidal(&self) -> f64 {
        // β_t ≈ β_N * I_p / (a * B_t)
        self.beta_n * self.plasma_current_ma
            / (self.minor_radius * self.toroidal_field * 100.0)
    }

    /// Radio interno de bobina TF
    pub fn tf_inner_radius(&self) -> f64 {
        self.major_radius - self.minor_radius
            - self.blanket_thickness()
            - self.shield_thickness
            - self.tf_coil_radial_build / 2.0
    }

    /// Campo máximo en conductor
    pub fn max_field_at_conductor(&self) -> f64 {
        self.toroidal_field * self.major_radius / self.tf_inner_radius()
    }

    /// ¿El diseño es factible?
    pub fn is_feasible(&self) -> bool {
        self.feasible && self.constraint_violations.is_empty()
    }

    /// Marca como infactible con razón
    pub fn mark_infeasible(&mut self, reason: &str) {
        self.feasible = false;
        self.constraint_violations.push(reason.to_string());
    }

    /// Domina a otro diseño (para NSGA-II)
    pub fn dominates(&self, other: &Self) -> bool {
        if !self.feasible || !other.feasible {
            return self.feasible && !other.feasible;
        }

        let dominated_in_all = self.objectives.iter().all(|(key, &val)| {
            other.objectives.get(key).map_or(true, |&other_val| {
                // Asumimos minimización (para maximización, invertir)
                val <= other_val
            })
        });

        let strictly_better_in_one = self.objectives.iter().any(|(key, &val)| {
            other.objectives.get(key).map_or(false, |&other_val| {
                val < other_val
            })
        });

        dominated_in_all && strictly_better_in_one
    }

    /// Genera ID único
    pub fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("design_{:x}", timestamp)
    }
}

impl Default for ReactorDesign {
    fn default() -> Self {
        Self::new(&Self::generate_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasma_volume() {
        let design = ReactorDesign::new("test");
        let volume = design.plasma_volume();
        // Para R=3, a=1, κ=1.8: V ≈ 2π² * 3 * 1 * 1.8 ≈ 106.5 m³
        assert!(volume > 100.0 && volume < 120.0);
    }

    #[test]
    fn test_aspect_ratio() {
        let design = ReactorDesign::new("test");
        assert!((design.aspect_ratio() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_greenwald_fraction() {
        let mut design = ReactorDesign::new("test");
        design.density = 1e20;
        design.plasma_current_ma = 8.0;
        design.minor_radius = 1.0;
        // n_GW = 8 / (π * 1) ≈ 2.55 * 10²⁰
        // f_GW = 1e20 / (2.55e20) ≈ 0.39
        let f_gw = design.greenwald_fraction();
        assert!(f_gw > 0.3 && f_gw < 0.5);
    }
}
