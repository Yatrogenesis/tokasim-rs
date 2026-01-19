//! # Sistema de Optimización Paramétrica Total (SOPT)
//!
//! Implementa optimización multi-objetivo con restricciones no lineales
//! para diseño de reactores de fusión tokamak.
//!
//! ## Arquitectura de 7 Capas
//!
//! ```text
//! CAPA 7: Costos y Cronograma (CAPEX, OPEX, LCOE)
//! CAPA 6: Estructura y Edificio (criostato, edificio, terreno)
//! CAPA 5: Blindaje y Primera Pared (neutrónica, TBR)
//! CAPA 4: Sistemas Auxiliares (calentamiento, diagnósticos, criogenia)
//! CAPA 3: Sistema Magnético (bobinas TF/PF/CS, fuerzas, quench)
//! CAPA 2: Geometría del Plasma (R, a, κ, δ, volumen)
//! CAPA 1: Física del Plasma (n, T, β, τ_E, Q)
//! ```
//!
//! ## Algoritmos Disponibles
//!
//! - NSGA-II: Optimización multi-objetivo con frentes de Pareto
//! - Differential Evolution: Optimización global robusta
//! - Bayesian Optimization: Búsqueda eficiente con surrogate
//!
//! ## Autor
//!
//! Francisco Molina-Burgos, Avermex Research Division
//! ORCID: 0009-0008-6093-8267

pub mod parameters;
pub mod constraints;
pub mod objectives;
pub mod algorithms;
pub mod scaling_laws;
pub mod cost_model;
pub mod infrastructure;
pub mod reports;
pub mod design;
pub mod utils;

// Re-exports principales
pub use parameters::{ReactorParameterSpace, ParameterDef, MagnetType, WallMaterial, BlanketType};
pub use constraints::{ConstraintEvaluator, ConstraintResult, Violation, Warning};
pub use objectives::ObjectiveFunctions;
pub use algorithms::{NSGA2Optimizer, DifferentialEvolution, OptimizationConfig};
pub use scaling_laws::ScalingLaws;
pub use cost_model::CostModel;
pub use infrastructure::{InfrastructureCalculator, InfrastructureSpec};
pub use reports::ReportGenerator;
pub use design::ReactorDesign;

/// Versión del módulo de optimización
pub const OPTIMIZER_VERSION: &str = "1.0.0";

/// Información del optimizador
pub fn info() -> String {
    format!(
        "TOKASIM-RS SOPT v{}\n\
         Sistema de Optimización Paramétrica Total\n\
         Autor: Francisco Molina-Burgos\n\
         ORCID: 0009-0008-6093-8267\n\
         Avermex Research Division",
        OPTIMIZER_VERSION
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info() {
        let info_str = info();
        assert!(info_str.contains("TOKASIM-RS SOPT"));
        assert!(info_str.contains("1.0.0"));
    }
}
