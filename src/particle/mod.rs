//! # Particle Module
//!
//! Particle-In-Cell (PIC) implementation for plasma simulation.
//!
//! ## Physics
//!
//! Lorentz force: F = q(E + v × B)
//! Boris pusher for accurate magnetic field integration.

use crate::types::{Vec3, Species};
use crate::constants::*;

/// A macro-particle representing many real particles
#[derive(Debug, Clone)]
pub struct Particle {
    /// Position (m)
    pub position: Vec3,
    /// Velocity (m/s)
    pub velocity: Vec3,
    /// Species (determines mass and charge)
    pub species: Species,
    /// Weight (number of real particles represented)
    pub weight: f64,
    /// Is particle still active (not absorbed by wall)
    pub active: bool,
}

impl Particle {
    /// Create new particle
    pub fn new(species: Species, position: Vec3, velocity: Vec3, weight: f64) -> Self {
        Self {
            position,
            velocity,
            species,
            weight,
            active: true,
        }
    }

    /// Get kinetic energy (J)
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.species.mass() * self.velocity.mag_squared()
    }

    /// Get thermal speed from temperature (m/s)
    pub fn thermal_speed(species: Species, temperature_kev: f64) -> f64 {
        let temp_j = temperature_kev * 1000.0 * EV_TO_J;
        (2.0 * temp_j / species.mass()).sqrt()
    }

    /// Get Larmor radius for given magnetic field (m)
    pub fn larmor_radius(&self, b_field: f64) -> f64 {
        let v_perp = (self.velocity.x * self.velocity.x + self.velocity.y * self.velocity.y).sqrt();
        let q = self.species.charge().abs();
        if q > 0.0 && b_field > 0.0 {
            self.species.mass() * v_perp / (q * b_field)
        } else {
            f64::INFINITY
        }
    }

    /// Get cyclotron frequency for given magnetic field (rad/s)
    pub fn cyclotron_frequency(&self, b_field: f64) -> f64 {
        let q = self.species.charge().abs();
        q * b_field / self.species.mass()
    }
}

/// Boris pusher for particle motion in electromagnetic fields
pub struct BorisPusher {
    /// Timestep (s)
    pub dt: f64,
}

impl BorisPusher {
    /// Create new Boris pusher
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    /// Push particle through one timestep
    ///
    /// Uses Boris algorithm for accuracy in magnetic fields:
    /// 1. Half electric acceleration
    /// 2. Magnetic rotation
    /// 3. Half electric acceleration
    pub fn push(&self, particle: &mut Particle, e_field: Vec3, b_field: Vec3) {
        if !particle.active || particle.species.charge() == 0.0 {
            // Neutrons: just drift
            particle.position += particle.velocity * self.dt;
            return;
        }

        let q = particle.species.charge();
        let m = particle.species.mass();
        let qm_dt_2 = q / m * self.dt * 0.5;

        // Half acceleration from E field
        let v_minus = particle.velocity + e_field * qm_dt_2;

        // Magnetic rotation
        let t = b_field * (q * self.dt / (2.0 * m));
        let t_mag_sq = t.mag_squared();
        let s = t * (2.0 / (1.0 + t_mag_sq));

        let v_prime = v_minus + v_minus.cross(&t);
        let v_plus = v_minus + v_prime.cross(&s);

        // Half acceleration from E field
        particle.velocity = v_plus + e_field * qm_dt_2;

        // Update position
        particle.position += particle.velocity * self.dt;
    }

    /// Push multiple particles (batch processing for efficiency)
    pub fn push_batch(&self, particles: &mut [Particle], e_field: &[Vec3], b_field: &[Vec3]) {
        assert_eq!(particles.len(), e_field.len());
        assert_eq!(particles.len(), b_field.len());

        for i in 0..particles.len() {
            self.push(&mut particles[i], e_field[i], b_field[i]);
        }
    }
}

/// Particle species population
#[derive(Debug)]
pub struct ParticlePopulation {
    /// All particles of this species
    pub particles: Vec<Particle>,
    /// Species type
    pub species: Species,
    /// Statistics
    stats: PopulationStats,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]  // Fields reserved for analytics/visualization
struct PopulationStats {
    total_particles: usize,
    active_particles: usize,
    total_kinetic_energy: f64,
    mean_velocity: Vec3,
}

impl ParticlePopulation {
    /// Create new empty population
    pub fn new(species: Species) -> Self {
        Self {
            particles: Vec::new(),
            species,
            stats: PopulationStats::default(),
        }
    }

    /// Create population with initial particles
    pub fn with_capacity(species: Species, capacity: usize) -> Self {
        Self {
            particles: Vec::with_capacity(capacity),
            species,
            stats: PopulationStats::default(),
        }
    }

    /// Add a particle
    pub fn add(&mut self, particle: Particle) {
        debug_assert_eq!(particle.species, self.species);
        self.particles.push(particle);
    }

    /// Create Maxwellian distribution
    pub fn create_maxwellian(
        &mut self,
        n_particles: usize,
        temperature_kev: f64,
        density: f64,
        volume: f64,
        rng_seed: u64,
    ) {
        let weight = density * volume / n_particles as f64;
        let v_th = Particle::thermal_speed(self.species, temperature_kev);

        // Simple LCG random number generator (no dependencies)
        let mut rng_state = rng_seed;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state as f64) / (u64::MAX as f64)
        };

        // Box-Muller transform for Gaussian distribution
        let gaussian = |state: &mut u64| -> f64 {
            let u1 = lcg_next(state).max(1e-10);
            let u2 = lcg_next(state);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        for _ in 0..n_particles {
            let vx = v_th * gaussian(&mut rng_state);
            let vy = v_th * gaussian(&mut rng_state);
            let vz = v_th * gaussian(&mut rng_state);

            // Random position within volume (simplified: uniform in box)
            let x = lcg_next(&mut rng_state) * volume.powf(1.0/3.0);
            let y = lcg_next(&mut rng_state) * volume.powf(1.0/3.0);
            let z = lcg_next(&mut rng_state) * volume.powf(1.0/3.0);

            self.particles.push(Particle::new(
                self.species,
                Vec3::new(x, y, z),
                Vec3::new(vx, vy, vz),
                weight,
            ));
        }
    }

    /// Update statistics
    pub fn update_stats(&mut self) {
        let mut ke = 0.0;
        let mut vsum = Vec3::zero();
        let mut active = 0;

        for p in &self.particles {
            if p.active {
                ke += p.kinetic_energy() * p.weight;
                vsum += p.velocity * p.weight;
                active += 1;
            }
        }

        self.stats = PopulationStats {
            total_particles: self.particles.len(),
            active_particles: active,
            total_kinetic_energy: ke,
            mean_velocity: if active > 0 {
                vsum * (1.0 / active as f64)
            } else {
                Vec3::zero()
            },
        };
    }

    /// Get total kinetic energy
    pub fn total_kinetic_energy(&self) -> f64 {
        self.stats.total_kinetic_energy
    }

    /// Get number of active particles
    pub fn active_count(&self) -> usize {
        self.stats.active_particles
    }

    /// Remove inactive particles
    pub fn compact(&mut self) {
        self.particles.retain(|p| p.active);
    }
}

/// Collision operator (Fokker-Planck approximation)
pub struct CollisionOperator {
    /// Collision frequency (s⁻¹)
    pub nu: f64,
    /// Timestep
    pub dt: f64,
}

impl CollisionOperator {
    /// Create collision operator with given frequency
    pub fn new(nu: f64, dt: f64) -> Self {
        Self { nu, dt }
    }

    /// Calculate electron-ion collision frequency (s⁻¹)
    ///
    /// ν_ei ≈ 2.9 × 10⁻¹² n_e Z² ln(Λ) / T_e^(3/2)
    ///
    /// where T_e is in keV, n_e in m⁻³
    pub fn electron_ion_frequency(n_e: f64, t_e_kev: f64, z: i32, coulomb_log: f64) -> f64 {
        2.9e-12 * n_e * (z * z) as f64 * coulomb_log / t_e_kev.powf(1.5)
    }

    /// Calculate Coulomb logarithm
    ///
    /// ln(Λ) ≈ 17 - 0.5 ln(n_e/10²⁰) + 1.5 ln(T_e/keV)
    pub fn coulomb_logarithm(n_e: f64, t_e_kev: f64) -> f64 {
        17.0 - 0.5 * (n_e / 1e20).ln() + 1.5 * t_e_kev.ln()
    }

    /// Apply collision operator (pitch-angle scattering)
    pub fn apply(&self, particle: &mut Particle, rng_state: &mut u64) {
        // Simple model: random deflection with rate ν
        let prob = 1.0 - (-self.nu * self.dt).exp();

        // LCG random number
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (*rng_state as f64) / (u64::MAX as f64);

        if r < prob {
            // Apply small random deflection
            let v_mag = particle.velocity.mag();
            if v_mag > 0.0 {
                // Random angle deflection (simplified)
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let theta = ((*rng_state as f64) / (u64::MAX as f64) - 0.5) * 0.1;
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let phi = ((*rng_state as f64) / (u64::MAX as f64)) * 2.0 * std::f64::consts::PI;

                // Rotate velocity (simplified rotation)
                let vn = particle.velocity.normalize();
                let perp = if vn.z.abs() < 0.9 {
                    vn.cross(&Vec3::unit_z()).normalize()
                } else {
                    vn.cross(&Vec3::unit_x()).normalize()
                };

                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let new_dir = vn * cos_t + perp * (sin_t * phi.cos())
                    + vn.cross(&perp) * (sin_t * phi.sin());

                particle.velocity = new_dir * v_mag;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(
            Species::Deuterium,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1e6, 0.0, 0.0),
            1e6,
        );
        assert!(p.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_boris_pusher_no_field() {
        let mut p = Particle::new(
            Species::Proton,
            Vec3::zero(),
            Vec3::new(1e6, 0.0, 0.0),
            1.0,
        );
        let pusher = BorisPusher::new(1e-9);

        pusher.push(&mut p, Vec3::zero(), Vec3::zero());

        // Should drift in x direction
        assert!(p.position.x > 0.0);
        assert!((p.position.y).abs() < 1e-15);
    }

    #[test]
    fn test_boris_pusher_magnetic() {
        let mut p = Particle::new(
            Species::Proton,
            Vec3::zero(),
            Vec3::new(1e6, 0.0, 0.0),
            1.0,
        );
        let pusher = BorisPusher::new(1e-9);
        let b = Vec3::new(0.0, 0.0, 1.0); // B in z

        // Push many times - should gyrate
        for _ in 0..1000 {
            pusher.push(&mut p, Vec3::zero(), b);
        }

        // Should stay roughly same distance from z-axis
        let r = (p.position.x * p.position.x + p.position.y * p.position.y).sqrt();
        assert!(r < 0.1); // Gyration radius should be small
    }

    #[test]
    fn test_maxwellian() {
        let mut pop = ParticlePopulation::new(Species::Electron);
        pop.create_maxwellian(1000, 1.0, 1e20, 1.0, 12345);

        assert_eq!(pop.particles.len(), 1000);

        pop.update_stats();
        assert!(pop.total_kinetic_energy() > 0.0);
    }

    #[test]
    fn test_coulomb_log() {
        let ln_lambda = CollisionOperator::coulomb_logarithm(1e20, 10.0);
        assert!(ln_lambda > 10.0 && ln_lambda < 25.0);
    }
}
