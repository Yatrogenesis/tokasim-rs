//! # Funciones Objetivo para Optimización
//!
//! Define las funciones de mérito para optimización multi-objetivo.

use crate::optimizer::design::ReactorDesign;
use crate::optimizer::scaling_laws::ScalingLaws;
use crate::optimizer::infrastructure::InfrastructureCalculator;
use crate::optimizer::cost_model::CostModel;
use crate::optimizer::constraints::ConstraintEvaluator;

/// Tipo de objetivo
#[derive(Debug, Clone)]
pub enum ObjectiveType {
    /// Maximizar (valor negativo internamente)
    Maximize(String),
    /// Minimizar
    Minimize(String),
}

/// Funciones objetivo para optimización
pub struct ObjectiveFunctions;

impl ObjectiveFunctions {
    /// Potencia de fusión (MW) - MAXIMIZAR
    pub fn fusion_power(design: &ReactorDesign) -> f64 {
        ScalingLaws::fusion_power_mw(design)
    }

    /// Factor Q - MAXIMIZAR
    pub fn q_factor(design: &ReactorDesign) -> f64 {
        ScalingLaws::q_factor(design)
    }

    /// Triple producto (m⁻³ keV s) - MAXIMIZAR
    pub fn triple_product(design: &ReactorDesign) -> f64 {
        ScalingLaws::triple_product(design)
    }

    /// Tiempo de confinamiento (s) - MAXIMIZAR
    pub fn confinement_time(design: &ReactorDesign) -> f64 {
        ScalingLaws::confinement_time_ipb98(design)
    }

    /// Costo de capital (USD) - MINIMIZAR
    pub fn capital_cost(design: &ReactorDesign) -> f64 {
        CostModel::default().estimate_capex(design)
    }

    /// LCOE ($/MWh) - MINIMIZAR
    pub fn lcoe(design: &ReactorDesign) -> f64 {
        CostModel::default().calculate_lcoe(design)
    }

    /// Área de sitio (m²) - MINIMIZAR
    pub fn site_area(design: &ReactorDesign) -> f64 {
        InfrastructureCalculator::new().total_site_area(design)
    }

    /// Volumen total (m³) - MINIMIZAR
    pub fn total_volume(design: &ReactorDesign) -> f64 {
        InfrastructureCalculator::new().total_building_volume(design)
    }

    /// Masa total del sistema (toneladas) - MINIMIZAR
    pub fn total_mass(design: &ReactorDesign) -> f64 {
        // Aproximación basada en escalamiento
        let r = design.major_radius;
        let b = design.toroidal_field;

        // Masa de bobinas TF (escala con R * B²)
        let m_tf = 500.0 * r * b.powi(2) / 25.0; // Normalizado a ITER

        // Masa de vacuum vessel (escala con R * a)
        let m_vv = 100.0 * r * design.minor_radius / 12.4;

        // Masa de blindaje
        let m_shield = 200.0 * design.plasma_surface() * design.shield_thickness;

        // Masa de criostato
        let m_cryo = 300.0 * r.powi(2) / 40.0;

        m_tf + m_vv + m_shield + m_cryo
    }

    /// Índice de seguridad compuesto - MAXIMIZAR
    pub fn safety_index(design: &ReactorDesign) -> f64 {
        let evaluator = ConstraintEvaluator::new();
        let result = evaluator.evaluate(design);

        if !result.feasible {
            return 0.0;
        }

        // Promedio geométrico de márgenes
        let margins: Vec<f64> = result.margins.values()
            .map(|&m| (m + 0.1).max(0.01)) // Evitar ceros
            .collect();

        if margins.is_empty() {
            1.0
        } else {
            margins.iter().product::<f64>().powf(1.0 / margins.len() as f64)
        }
    }

    /// Evalúa todos los objetivos para un diseño
    pub fn evaluate_all(design: &mut ReactorDesign) {
        design.objectives.insert("P_fusion".to_string(), Self::fusion_power(design));
        design.objectives.insert("Q".to_string(), Self::q_factor(design));
        design.objectives.insert("triple_product".to_string(), Self::triple_product(design));
        design.objectives.insert("tau_E".to_string(), Self::confinement_time(design));
        design.objectives.insert("CAPEX".to_string(), Self::capital_cost(design));
        design.objectives.insert("site_area".to_string(), Self::site_area(design));
        design.objectives.insert("total_mass".to_string(), Self::total_mass(design));
        design.objectives.insert("safety_index".to_string(), Self::safety_index(design));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_power() {
        let mut design = ReactorDesign::default();
        design.density = 1e20;
        design.ion_temperature_kev = 15.0;

        let p_f = ObjectiveFunctions::fusion_power(&design);
        println!("Fusion power: {:.2} MW", p_f);
        assert!(p_f >= 0.0);
    }

    #[test]
    fn test_q_factor() {
        let mut design = ReactorDesign::default();
        design.density = 1e20;
        design.ion_temperature_kev = 15.0;
        design.icrf_power_mw = 50.0;

        let q = ObjectiveFunctions::q_factor(&design);
        println!("Q factor: {:.2}", q);
    }
}
