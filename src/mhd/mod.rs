//! # MHD Module
//!
//! Magnetohydrodynamics for plasma equilibrium and stability.
//!
//! ## Key Equations
//!
//! Grad-Shafranov equation for equilibrium:
//! Δ*ψ = -μ₀R²(dp/dψ) - F(dF/dψ)

use crate::types::Vec3;

/// MHD equilibrium solver (Grad-Shafranov)
pub struct EquilibriumSolver {
    /// Major radius (m)
    pub r0: f64,
    /// Minor radius (m)
    pub a: f64,
    /// Toroidal field (T)
    pub b0: f64,
    /// Plasma current (A)
    pub ip: f64,
    /// Pressure profile exponent
    pub alpha_p: f64,
    /// Current profile exponent
    pub alpha_j: f64,
}

impl EquilibriumSolver {
    /// Create new equilibrium solver
    pub fn new(r0: f64, a: f64, b0: f64, ip: f64) -> Self {
        Self {
            r0, a, b0, ip,
            alpha_p: 2.0,
            alpha_j: 1.0,
        }
    }

    /// Calculate safety factor at normalized radius
    /// q(ρ) ≈ q0 + (q_edge - q0) * ρ²
    pub fn safety_factor(&self, rho_norm: f64, q0: f64, q_edge: f64) -> f64 {
        q0 + (q_edge - q0) * rho_norm * rho_norm
    }

    /// Calculate pressure profile (normalized)
    /// p(ρ) = p0 * (1 - ρ²)^α_p
    pub fn pressure_profile(&self, rho_norm: f64, p0: f64) -> f64 {
        p0 * (1.0 - rho_norm * rho_norm).max(0.0).powf(self.alpha_p)
    }

    /// Calculate current density profile (normalized)
    /// j(ρ) = j0 * (1 - ρ²)^α_j
    pub fn current_profile(&self, rho_norm: f64, j0: f64) -> f64 {
        j0 * (1.0 - rho_norm * rho_norm).max(0.0).powf(self.alpha_j)
    }

    /// Calculate poloidal flux function (simplified)
    pub fn poloidal_flux(&self, r: f64, z: f64) -> f64 {
        // Simplified: circular flux surfaces
        let dr = r - self.r0;
        let rho_sq = (dr * dr + z * z) / (self.a * self.a);

        if rho_sq <= 1.0 {
            // Inside plasma
            0.5 * crate::constants::MU_0 * self.ip * (1.0 - rho_sq)
        } else {
            // Outside plasma (vacuum)
            0.0
        }
    }

    /// Calculate magnetic field from flux
    pub fn magnetic_field(&self, r: f64, z: f64) -> Vec3 {
        // Toroidal component
        let b_phi = self.b0 * self.r0 / r.max(0.1 * self.r0);

        // Poloidal components (from derivatives of flux)
        let dr = r - self.r0;
        let rho_sq = (dr * dr + z * z) / (self.a * self.a);

        let (b_r, b_z) = if rho_sq <= 1.0 {
            let factor = crate::constants::MU_0 * self.ip / (self.a * self.a * r.max(0.1));
            (-factor * z, factor * dr)
        } else {
            (0.0, 0.0)
        };

        Vec3::new(b_r, b_phi, b_z)
    }
}

/// Stability analyzer for MHD modes
pub struct StabilityAnalyzer {
    /// Elongation
    pub kappa: f64,
    /// Triangularity
    pub delta: f64,
    /// Internal inductance
    pub li: f64,
}

impl StabilityAnalyzer {
    /// Create new stability analyzer
    pub fn new(kappa: f64, delta: f64) -> Self {
        Self {
            kappa,
            delta,
            li: 1.0,  // Default internal inductance
        }
    }

    /// Calculate Troyon beta limit
    /// β_N,max ≈ 2.8 * I_N (Troyon scaling)
    /// I_N = I_p / (a * B_t) in MA, m, T
    pub fn troyon_beta_limit(&self, ip_ma: f64, a: f64, bt: f64) -> f64 {
        let i_n = ip_ma / (a * bt);
        2.8 * i_n
    }

    /// Calculate kink stability (simplified Kruskal-Shafranov)
    /// Stable if q_edge > 2 for m=2, n=1 kink
    pub fn kink_stability(&self, q_edge: f64) -> bool {
        q_edge > 2.0
    }

    /// Calculate vertical stability parameter
    /// n_ext = (1/2) * (1 + κ²) / (κ² - 1 + δ/κ)
    pub fn vertical_stability_index(&self) -> f64 {
        if self.kappa > 1.01 {
            0.5 * (1.0 + self.kappa * self.kappa) /
                (self.kappa * self.kappa - 1.0 + self.delta / self.kappa)
        } else {
            f64::INFINITY  // Circular - stable
        }
    }

    /// Check overall stability
    pub fn is_stable(&self, beta_n: f64, q_edge: f64, ip_ma: f64, a: f64, bt: f64) -> StabilityResult {
        let mut result = StabilityResult::default();

        // Beta limit
        let beta_limit = self.troyon_beta_limit(ip_ma, a, bt);
        result.beta_margin = (beta_limit - beta_n) / beta_limit;
        result.beta_stable = beta_n < beta_limit;

        // Kink stability
        result.kink_stable = self.kink_stability(q_edge);
        result.q_margin = (q_edge - 2.0) / q_edge;

        // Vertical stability
        let n_ext = self.vertical_stability_index();
        result.vertical_stable = n_ext > 0.0;

        // Overall
        result.overall_stable = result.beta_stable && result.kink_stable && result.vertical_stable;

        result
    }
}

/// Result of stability analysis
#[derive(Debug, Clone, Default)]
pub struct StabilityResult {
    /// Overall stable?
    pub overall_stable: bool,
    /// Beta limit stable?
    pub beta_stable: bool,
    /// Margin to beta limit (fraction)
    pub beta_margin: f64,
    /// Kink stable?
    pub kink_stable: bool,
    /// Margin to q=2 (fraction)
    pub q_margin: f64,
    /// Vertically stable?
    pub vertical_stable: bool,
}

/// Disruption predictor
pub struct DisruptionPredictor {
    /// Historical data for pattern recognition
    history: Vec<DisruptionIndicators>,
    /// Window size for analysis
    window_size: usize,
}

/// Indicators used for disruption prediction
#[derive(Debug, Clone, Default)]
pub struct DisruptionIndicators {
    /// Time (s)
    pub time: f64,
    /// Beta_N
    pub beta_n: f64,
    /// q95
    pub q95: f64,
    /// Vertical position (m)
    pub z_position: f64,
    /// dIp/dt (A/s)
    pub current_rate: f64,
    /// Locked mode amplitude (T)
    pub locked_mode: f64,
    /// Radiated power fraction
    pub rad_fraction: f64,
}

impl DisruptionPredictor {
    /// Create new predictor
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(1000),
            window_size: 100,
        }
    }

    /// Add new data point
    pub fn add_data(&mut self, indicators: DisruptionIndicators) {
        self.history.push(indicators);
        if self.history.len() > self.window_size * 10 {
            self.history.remove(0);
        }
    }

    /// Predict disruption risk (0-1)
    pub fn predict_risk(&self, current: &DisruptionIndicators) -> f64 {
        let mut risk = 0.0;

        // High beta_N risk
        if current.beta_n > 3.0 {
            risk += 0.2 * (current.beta_n - 3.0) / 0.5;
        }

        // Low q95 risk
        if current.q95 < 2.5 {
            risk += 0.3 * (2.5 - current.q95) / 0.5;
        }

        // Vertical displacement risk
        if current.z_position.abs() > 0.1 {
            risk += 0.2 * (current.z_position.abs() - 0.1) / 0.1;
        }

        // Locked mode risk
        if current.locked_mode > 1e-4 {
            risk += 0.3;
        }

        // High radiation risk (density limit approach)
        if current.rad_fraction > 0.5 {
            risk += 0.2 * (current.rad_fraction - 0.5) / 0.3;
        }

        // Trend analysis if enough history
        if self.history.len() >= 10 {
            let recent: Vec<&DisruptionIndicators> = self.history.iter()
                .rev().take(10).collect();

            // Check for rapid beta increase
            if recent.len() >= 2 {
                let d_beta = recent[0].beta_n - recent[recent.len()-1].beta_n;
                if d_beta > 0.5 {
                    risk += 0.1;
                }
            }
        }

        risk.clamp(0.0, 1.0)
    }

    /// Get time to disruption estimate (s)
    /// Returns None if no disruption predicted
    pub fn time_to_disruption(&self, current: &DisruptionIndicators) -> Option<f64> {
        let risk = self.predict_risk(current);

        if risk > 0.5 {
            // Empirical scaling: higher risk = shorter time
            Some(1.0 * (1.0 - risk) + 0.01)
        } else {
            None
        }
    }
}

impl Default for DisruptionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_equilibrium() {
        let eq = EquilibriumSolver::new(TS1_MAJOR_RADIUS, TS1_MINOR_RADIUS, TS1_TOROIDAL_FIELD, TS1_PLASMA_CURRENT);

        // Flux should be maximum at center
        let psi_center = eq.poloidal_flux(TS1_MAJOR_RADIUS, 0.0);
        let psi_edge = eq.poloidal_flux(TS1_MAJOR_RADIUS + TS1_MINOR_RADIUS, 0.0);
        assert!(psi_center > psi_edge);
    }

    #[test]
    fn test_stability() {
        let analyzer = StabilityAnalyzer::new(TS1_ELONGATION, TS1_TRIANGULARITY);

        let result = analyzer.is_stable(
            2.5,  // beta_N
            3.5,  // q_edge
            TS1_PLASMA_CURRENT_MA,
            TS1_MINOR_RADIUS,
            TS1_TOROIDAL_FIELD
        );

        assert!(result.overall_stable);
    }

    #[test]
    fn test_disruption_predictor() {
        let mut predictor = DisruptionPredictor::new();

        // Normal conditions
        let normal = DisruptionIndicators {
            beta_n: 2.5,
            q95: 3.5,
            z_position: 0.0,
            ..Default::default()
        };
        assert!(predictor.predict_risk(&normal) < 0.5);

        // Dangerous conditions
        let danger = DisruptionIndicators {
            beta_n: 3.8,
            q95: 1.8,
            z_position: 0.15,
            locked_mode: 1e-3,
            ..Default::default()
        };
        assert!(predictor.predict_risk(&danger) > 0.5);
    }
}
