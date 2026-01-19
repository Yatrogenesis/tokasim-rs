//! # Evaluador de Restricciones Físicas y de Ingeniería
//!
//! Verifica que un diseño cumpla con todos los límites operacionales.

use crate::optimizer::design::ReactorDesign;
use crate::optimizer::scaling_laws::ScalingLaws;

/// Resultado de evaluación de restricciones
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// ¿Es factible el diseño?
    pub feasible: bool,
    /// Violaciones (restricciones no cumplidas)
    pub violations: Vec<Violation>,
    /// Advertencias (cerca de límites)
    pub warnings: Vec<Warning>,
    /// Márgenes respecto a límites
    pub margins: std::collections::HashMap<String, f64>,
}

/// Violación de restricción
#[derive(Debug, Clone)]
pub enum Violation {
    /// Densidad excede límite de Greenwald
    Greenwald { actual: f64, limit: f64 },
    /// Beta excede límite de Troyon
    Troyon { actual: f64, limit: f64 },
    /// q95 muy bajo (inestabilidad kink)
    KinkInstability { actual: f64, limit: f64 },
    /// Campo máximo excede capacidad del conductor
    MagneticFieldLimit { actual: f64, limit: f64, technology: String },
    /// Carga de pared excede límite del material
    WallLoadExceeded { actual: f64, limit: f64 },
    /// TBR insuficiente para autosuficiencia de tritio
    InsufficientTBR { actual: f64, limit: f64 },
    /// Aspect ratio fuera de rango
    AspectRatio { actual: f64, min: f64, max: f64 },
    /// Espacio insuficiente para CS
    InsufficientCSSpace { available: f64, required: f64 },
    /// Otro
    Other { name: String, actual: f64, limit: f64 },
}

/// Advertencia (cerca de límite)
#[derive(Debug, Clone)]
pub enum Warning {
    HighGreenwald(f64),
    HighBeta(f64),
    LowQ95(f64),
    HighWallLoad(f64),
    LowTBR(f64),
    HighAspectRatio(f64),
}

/// Evaluador de restricciones
#[derive(Debug, Clone)]
pub struct ConstraintEvaluator {
    /// Factor de Troyon (conservador: 2.8, optimista: 3.5)
    pub g_troyon: f64,
    /// q95 mínimo
    pub q95_min: f64,
    /// TBR mínimo requerido
    pub tbr_min: f64,
    /// Aspect ratio mínimo
    pub aspect_ratio_min: f64,
    /// Aspect ratio máximo
    pub aspect_ratio_max: f64,
}

impl Default for ConstraintEvaluator {
    fn default() -> Self {
        Self {
            g_troyon: 2.8,
            q95_min: 2.0,
            tbr_min: 1.05,
            aspect_ratio_min: 2.0,
            aspect_ratio_max: 5.0,
        }
    }
}

impl ConstraintEvaluator {
    /// Crea evaluador con valores por defecto
    pub fn new() -> Self {
        Self::default()
    }

    /// Evalúa todas las restricciones para un diseño
    pub fn evaluate(&self, design: &ReactorDesign) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut margins = std::collections::HashMap::new();

        // 1. Límite de Greenwald
        let n_gw = ScalingLaws::greenwald_limit(design) * 1e20;
        let f_gw = design.density / n_gw;
        margins.insert("greenwald".to_string(), 1.0 - f_gw);

        if f_gw > 1.0 {
            violations.push(Violation::Greenwald { actual: f_gw, limit: 1.0 });
        } else if f_gw > 0.85 {
            warnings.push(Warning::HighGreenwald(f_gw));
        }

        // 2. Límite de Troyon
        let beta_n_max = ScalingLaws::troyon_limit(design, self.g_troyon);
        let beta_margin = 1.0 - design.beta_n / beta_n_max;
        margins.insert("troyon".to_string(), beta_margin);

        if design.beta_n > beta_n_max {
            violations.push(Violation::Troyon { actual: design.beta_n, limit: beta_n_max });
        } else if design.beta_n > 0.85 * beta_n_max {
            warnings.push(Warning::HighBeta(design.beta_n / beta_n_max));
        }

        // 3. Estabilidad kink (q95)
        let q95 = ScalingLaws::q95(design);
        let q95_margin = (q95 - self.q95_min) / self.q95_min;
        margins.insert("q95".to_string(), q95_margin);

        if q95 < self.q95_min {
            violations.push(Violation::KinkInstability { actual: q95, limit: self.q95_min });
        } else if q95 < 2.5 {
            warnings.push(Warning::LowQ95(q95));
        }

        // 4. Campo máximo en conductor
        let b_max = design.max_field_at_conductor();
        let b_limit = design.magnet_technology.max_field();
        let b_margin = 1.0 - b_max / b_limit;
        margins.insert("b_max".to_string(), b_margin);

        if b_max > b_limit {
            violations.push(Violation::MagneticFieldLimit {
                actual: b_max,
                limit: b_limit,
                technology: format!("{:?}", design.magnet_technology),
            });
        }

        // 5. Carga de pared
        let wall_load = ScalingLaws::neutron_wall_load(design);
        let wall_limit = design.first_wall_material.max_wall_load();
        let wall_margin = 1.0 - wall_load / wall_limit;
        margins.insert("wall_load".to_string(), wall_margin);

        if wall_load > wall_limit {
            violations.push(Violation::WallLoadExceeded { actual: wall_load, limit: wall_limit });
        } else if wall_load > 0.8 * wall_limit {
            warnings.push(Warning::HighWallLoad(wall_load / wall_limit));
        }

        // 6. TBR (Tritium Breeding Ratio)
        let tbr = design.blanket_type.expected_tbr();
        let tbr_margin = (tbr - self.tbr_min) / self.tbr_min;
        margins.insert("tbr".to_string(), tbr_margin);

        if tbr < self.tbr_min {
            violations.push(Violation::InsufficientTBR { actual: tbr, limit: self.tbr_min });
        } else if tbr < 1.08 {
            warnings.push(Warning::LowTBR(tbr));
        }

        // 7. Aspect ratio
        let ar = design.aspect_ratio();
        let ar_margin_low = (ar - self.aspect_ratio_min) / self.aspect_ratio_min;
        let ar_margin_high = (self.aspect_ratio_max - ar) / self.aspect_ratio_max;
        margins.insert("aspect_ratio".to_string(), ar_margin_low.min(ar_margin_high));

        if ar < self.aspect_ratio_min || ar > self.aspect_ratio_max {
            violations.push(Violation::AspectRatio {
                actual: ar,
                min: self.aspect_ratio_min,
                max: self.aspect_ratio_max,
            });
        } else if ar > 4.5 {
            warnings.push(Warning::HighAspectRatio(ar));
        }

        ConstraintResult {
            feasible: violations.is_empty(),
            violations,
            warnings,
            margins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_evaluation() {
        let design = ReactorDesign::default();
        let evaluator = ConstraintEvaluator::new();
        let result = evaluator.evaluate(&design);

        // Un diseño por defecto debería ser factible
        println!("Violations: {:?}", result.violations);
        println!("Warnings: {:?}", result.warnings);
    }
}
