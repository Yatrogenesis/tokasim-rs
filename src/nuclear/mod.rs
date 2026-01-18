//! # Nuclear Module
//!
//! Fusion reactions and nuclear physics.
//!
//! ## Key Reactions
//!
//! D + T → He⁴ (3.5 MeV) + n (14.1 MeV)
//! D + D → He³ (0.82 MeV) + n (2.45 MeV) [50%]
//! D + D → T (1.01 MeV) + p (3.02 MeV) [50%]

use crate::types::{Vec3, Species};
use crate::constants::*;
use crate::particle::Particle;

/// Fusion reaction rates using Bosch-Hale parametrization
pub struct FusionRates;

impl FusionRates {
    /// D-T fusion cross-section (m²) at given center-of-mass energy (keV)
    ///
    /// Bosch-Hale parametrization
    pub fn dt_cross_section(e_cm_kev: f64) -> f64 {
        // Bosch-Hale parameters for D-T
        const BG: f64 = 34.3827;  // Gamow constant (keV^1/2)
        const A1: f64 = 6.927e4;
        const A2: f64 = 7.454e8;
        const A3: f64 = 2.050e6;
        const A4: f64 = 5.2002e4;
        const A5: f64 = 0.0;
        const B1: f64 = 6.38e1;
        const B2: f64 = -9.95e-1;
        const B3: f64 = 6.981e-5;
        const B4: f64 = 1.728e-4;

        let e = e_cm_kev;
        let theta = e / (1.0 - (B1 * e + B2 * e * e + B3 * e.powi(3) + B4 * e.powi(4))
            / (1.0 + A1 * e + A2 * e * e + A3 * e.powi(3) + A4 * e.powi(4) + A5 * e.powi(5)));

        let s = (A1 + e * (A2 + e * (A3 + e * (A4 + e * A5))))
            / (1.0 + e * (B1 + e * (B2 + e * (B3 + e * B4))));

        // Cross section in barns, convert to m²
        let sigma_barns = s / e * (-BG / e.sqrt()).exp();
        sigma_barns * 1e-28
    }

    /// D-T reactivity <σv> (m³/s) at given temperature (keV)
    ///
    /// Maxwellian-averaged
    pub fn dt_reactivity(t_kev: f64) -> f64 {
        // Bosch-Hale fit for reactivity
        const C1: f64 = 1.17302e-9;
        const C2: f64 = 1.51361e-2;
        const C3: f64 = 7.51886e-2;
        const C4: f64 = 4.60643e-3;
        const C5: f64 = 1.35000e-2;
        const C6: f64 = -1.06750e-4;
        const C7: f64 = 1.36600e-5;
        const BG_SQ: f64 = 34.3827_f64 * 34.3827;

        let t = t_kev;
        let theta = t / (1.0 - t * (C2 + t * (C4 + t * C6)) / (1.0 + t * (C3 + t * (C5 + t * C7))));

        C1 * theta.sqrt() * (-3.0 * (BG_SQ / (4.0 * theta)).powf(1.0/3.0)).exp()
    }

    /// D-D reactivity <σv> (m³/s) at given temperature (keV)
    /// Sum of both branches
    pub fn dd_reactivity(t_kev: f64) -> f64 {
        // Simplified fit
        let t = t_kev.max(0.1);
        1e-27 * (t / 10.0).powf(2.0) * (-20.0 / t.sqrt()).exp()
    }

    /// Calculate fusion power density (W/m³)
    ///
    /// P = n_D * n_T * <σv> * E_fusion
    pub fn fusion_power_density(n_d: f64, n_t: f64, t_kev: f64) -> f64 {
        let reactivity = Self::dt_reactivity(t_kev);
        n_d * n_t * reactivity * DT_FUSION_ENERGY_J
    }

    /// Calculate total fusion power (W) for tokamak
    pub fn total_fusion_power(density: f64, temp_kev: f64, volume: f64, d_fraction: f64, t_fraction: f64) -> f64 {
        let n_d = density * d_fraction;
        let n_t = density * t_fraction;
        Self::fusion_power_density(n_d, n_t, temp_kev) * volume
    }

    /// Calculate Q factor (fusion power / input power)
    pub fn q_factor(fusion_power: f64, input_power: f64) -> f64 {
        if input_power > 0.0 {
            fusion_power / input_power
        } else {
            0.0
        }
    }

    /// Calculate triple product n*T*τ (keV·s/m³)
    pub fn triple_product(density: f64, temp_kev: f64, confinement_time: f64) -> f64 {
        density * temp_kev * confinement_time
    }

    /// Lawson criterion value for D-T ignition
    /// n*T*τ > 3×10²¹ keV·s/m³
    pub const LAWSON_DT: f64 = 3e21;
}

/// Fusion event during simulation
#[derive(Debug, Clone)]
pub struct FusionEvent {
    /// Time of event (s)
    pub time: f64,
    /// Position of event
    pub position: Vec3,
    /// Reaction type
    pub reaction: FusionReaction,
    /// Energy released (J)
    pub energy: f64,
    /// Products created
    pub products: Vec<(Species, Vec3)>,  // (species, velocity)
}

/// Type of fusion reaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionReaction {
    /// D + T → He⁴ + n
    DT,
    /// D + D → He³ + n
    DDHe3,
    /// D + D → T + p
    DDT,
}

impl FusionReaction {
    /// Energy released (J)
    pub fn energy(&self) -> f64 {
        match self {
            FusionReaction::DT => DT_FUSION_ENERGY_J,
            FusionReaction::DDHe3 => DD_HE3_ENERGY_MEV * 1.602e-13,
            FusionReaction::DDT => DD_T_ENERGY_MEV * 1.602e-13,
        }
    }

    /// Product particles
    pub fn products(&self) -> (Species, Species) {
        match self {
            FusionReaction::DT => (Species::Alpha, Species::Neutron),
            FusionReaction::DDHe3 => (Species::Helium3, Species::Neutron),
            FusionReaction::DDT => (Species::Tritium, Species::Proton),
        }
    }
}

/// Monte Carlo fusion handler
pub struct FusionHandler {
    /// Events this timestep
    pub events: Vec<FusionEvent>,
    /// Total fusion count
    pub total_fusions: u64,
    /// Total energy released (J)
    pub total_energy: f64,
    /// RNG state
    rng_state: u64,
}

impl FusionHandler {
    /// Create new fusion handler
    pub fn new(seed: u64) -> Self {
        Self {
            events: Vec::new(),
            total_fusions: 0,
            total_energy: 0.0,
            rng_state: seed,
        }
    }

    /// Random number (0-1)
    fn random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Check for fusion between two particles
    ///
    /// Returns fusion event if reaction occurs
    pub fn check_fusion(
        &mut self,
        p1: &Particle,
        p2: &Particle,
        dt: f64,
        time: f64,
    ) -> Option<FusionEvent> {
        // Determine reaction type
        let reaction = match (p1.species, p2.species) {
            (Species::Deuterium, Species::Tritium) |
            (Species::Tritium, Species::Deuterium) => FusionReaction::DT,

            (Species::Deuterium, Species::Deuterium) => {
                // 50% each branch
                if self.random() < 0.5 {
                    FusionReaction::DDHe3
                } else {
                    FusionReaction::DDT
                }
            }

            _ => return None,  // No fusion for other combinations
        };

        // Calculate relative velocity and center-of-mass energy
        let v_rel = p1.velocity.sub(&p2.velocity);
        let v_rel_mag = v_rel.mag();

        let m_reduced = (p1.species.mass() * p2.species.mass())
            / (p1.species.mass() + p2.species.mass());
        let e_cm_j = 0.5 * m_reduced * v_rel_mag * v_rel_mag;
        let e_cm_kev = e_cm_j / (1000.0 * EV_TO_J);

        // Get cross section
        let sigma = match reaction {
            FusionReaction::DT => FusionRates::dt_cross_section(e_cm_kev.max(0.1)),
            FusionReaction::DDHe3 | FusionReaction::DDT => {
                FusionRates::dt_cross_section(e_cm_kev.max(0.1)) * 0.01  // DD much lower
            }
        };

        // Probability of fusion in this timestep
        // P = n₂ * σ * v_rel * dt * weight
        let prob = p2.weight * sigma * v_rel_mag * dt;

        if self.random() < prob {
            // Fusion occurs!
            let (prod1, prod2) = reaction.products();

            // Position at midpoint
            let pos = p1.position.add(&p2.position).scale(0.5);

            // Calculate product velocities (momentum conservation + energy release)
            let m1 = prod1.mass();
            let m2 = prod2.mass();
            let e_release = reaction.energy();

            // Simplified: isotropic emission in CM frame
            let theta = self.random() * std::f64::consts::PI;
            let phi = self.random() * 2.0 * std::f64::consts::PI;

            // Speed from energy
            let v1 = (2.0 * e_release * m2 / (m1 * (m1 + m2))).sqrt();
            let v2 = (2.0 * e_release * m1 / (m2 * (m1 + m2))).sqrt();

            let dir = Vec3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );

            let event = FusionEvent {
                time,
                position: pos,
                reaction,
                energy: e_release,
                products: vec![
                    (prod1, dir.scale(v1)),
                    (prod2, dir.scale(-v2)),
                ],
            };

            self.events.push(event.clone());
            self.total_fusions += 1;
            self.total_energy += e_release;

            Some(event)
        } else {
            None
        }
    }

    /// Clear events for new timestep
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Get current fusion power (W)
    pub fn fusion_power(&self, dt: f64) -> f64 {
        if dt > 0.0 {
            self.events.iter().map(|e| e.energy).sum::<f64>() / dt
        } else {
            0.0
        }
    }
}

/// Alpha particle heating
pub struct AlphaHeating {
    /// Alpha slowing-down time (s)
    pub tau_sd: f64,
}

impl AlphaHeating {
    /// Create alpha heating model
    pub fn new(density: f64, temp_kev: f64) -> Self {
        // Slowing-down time: τ_sd ≈ 0.05 * T_e^(3/2) / n_e [s, keV, 10²⁰ m⁻³]
        let tau_sd = 0.05 * temp_kev.powf(1.5) / (density / 1e20);
        Self { tau_sd }
    }

    /// Calculate alpha heating power density (W/m³)
    ///
    /// Alpha carries 3.5 MeV = 20% of fusion energy
    pub fn alpha_heating_density(fusion_power_density: f64) -> f64 {
        fusion_power_density * (DT_ALPHA_ENERGY_MEV / DT_FUSION_ENERGY_MEV)
    }

    /// Fraction of alpha energy deposited to ions vs electrons
    pub fn ion_fraction(temp_kev: f64) -> f64 {
        // Higher temperature → more to ions
        // Empirical: f_i ≈ T_e / (T_e + 33 keV)
        temp_kev / (temp_kev + 33.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_reactivity() {
        // At 10 keV, reactivity should be around 1e-22 m³/s
        let sv = FusionRates::dt_reactivity(10.0);
        assert!(sv > 1e-23 && sv < 1e-21);

        // Peak at ~60-70 keV
        let sv_peak = FusionRates::dt_reactivity(65.0);
        assert!(sv_peak > FusionRates::dt_reactivity(10.0));
        assert!(sv_peak > FusionRates::dt_reactivity(200.0));
    }

    #[test]
    fn test_fusion_power() {
        // TS-1 parameters
        let power = FusionRates::total_fusion_power(
            TS1_PLASMA_DENSITY,
            TS1_TEMPERATURE_KEV,
            TS1_PLASMA_VOLUME,
            0.5,
            0.5,
        );

        // Should be hundreds of MW
        assert!(power > 100e6);
        assert!(power < 1000e6);
    }

    #[test]
    fn test_triple_product() {
        let ntt = FusionRates::triple_product(3e20, 15.0, 2.0);

        // Should exceed Lawson criterion for good performance
        assert!(ntt > FusionRates::LAWSON_DT);
    }

    #[test]
    fn test_alpha_heating() {
        let fusion_power = 500e6 / TS1_PLASMA_VOLUME;  // W/m³
        let alpha_power = AlphaHeating::alpha_heating_density(fusion_power);

        // Alpha should be ~20% of fusion
        assert!((alpha_power / fusion_power - 0.2).abs() < 0.01);
    }
}
