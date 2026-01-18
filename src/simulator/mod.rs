//! # Simulator Module
//!
//! Main simulation engine combining all components.

use crate::types::*;
use crate::particle::*;
use crate::field::*;
use crate::mhd::*;
use crate::nuclear::*;
use crate::control::*;
use crate::geometry::*;
use crate::constants::*;

/// Main tokamak simulator
pub struct TokamakSimulator {
    /// Configuration
    pub config: TokamakConfig,
    /// Geometry
    pub geometry: TokamakGeometry,
    /// Simulation state
    pub state: SimulationState,
    /// Particle populations
    pub electrons: ParticlePopulation,
    pub deuterium: ParticlePopulation,
    pub tritium: ParticlePopulation,
    pub alphas: ParticlePopulation,
    /// Field grid
    pub fields: FieldGrid,
    /// Boris pusher
    pusher: BorisPusher,
    /// FDTD solver
    fdtd: FDTDSolver,
    /// Equilibrium solver
    pub equilibrium: EquilibriumSolver,
    /// Stability analyzer
    pub stability: StabilityAnalyzer,
    /// Disruption predictor
    pub disruption: DisruptionPredictor,
    /// Fusion handler
    pub fusion: FusionHandler,
    /// Plasma controller
    pub controller: PlasmaController,
    /// Timestep (s)
    pub dt: f64,
    /// Output interval (steps)
    pub output_interval: u64,
}

/// Simulation parameters
#[derive(Debug, Clone)]
pub struct SimulationParams {
    /// Number of macro-particles per species
    pub n_particles: usize,
    /// Grid resolution (cells per minor radius)
    pub grid_resolution: usize,
    /// Timestep (s)
    pub dt: f64,
    /// Total simulation time (s)
    pub total_time: f64,
    /// Output interval (s)
    pub output_interval: f64,
    /// RNG seed
    pub seed: u64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            n_particles: 100_000,
            grid_resolution: 50,
            dt: 1e-10,
            total_time: 1e-6,
            output_interval: 1e-8,
            seed: 12345,
        }
    }
}

impl TokamakSimulator {
    /// Create new simulator
    pub fn new(config: TokamakConfig, params: SimulationParams) -> Self {
        // Setup geometry
        let geometry = TokamakGeometry::from_config(&config);

        // Setup field grid
        let grid_size = params.grid_resolution;
        let cell_size = 2.0 * config.minor_radius / grid_size as f64;
        let mut fields = FieldGrid::cubic(grid_size, cell_size);

        // Initialize toroidal field
        fields.set_toroidal_field(config.toroidal_field, config.major_radius);

        // Setup particles
        let mut electrons = ParticlePopulation::with_capacity(Species::Electron, params.n_particles);
        let mut deuterium = ParticlePopulation::with_capacity(Species::Deuterium, params.n_particles);
        let mut tritium = ParticlePopulation::with_capacity(Species::Tritium, params.n_particles);
        let alphas = ParticlePopulation::new(Species::Alpha);

        // Initialize Maxwellian distributions
        let volume = geometry.plasma_volume();
        let n_per_species = params.n_particles / 2;

        electrons.create_maxwellian(
            params.n_particles,
            config.electron_temperature_kev,
            config.density,
            volume,
            params.seed,
        );

        deuterium.create_maxwellian(
            n_per_species,
            config.ion_temperature_kev,
            config.density * config.deuterium_fraction,
            volume,
            params.seed + 1,
        );

        tritium.create_maxwellian(
            n_per_species,
            config.ion_temperature_kev,
            config.density * config.tritium_fraction,
            volume,
            params.seed + 2,
        );

        // Setup solvers
        let pusher = BorisPusher::new(params.dt);
        let fdtd = FDTDSolver::new(&fields);

        let equilibrium = EquilibriumSolver::new(
            config.major_radius,
            config.minor_radius,
            config.toroidal_field,
            config.plasma_current,
        );

        let stability = StabilityAnalyzer::new(config.elongation, config.triangularity);
        let disruption = DisruptionPredictor::new();
        let fusion = FusionHandler::new(params.seed + 100);
        let controller = PlasmaController::new();

        Self {
            config,
            geometry,
            state: SimulationState::default(),
            electrons,
            deuterium,
            tritium,
            alphas,
            fields,
            pusher,
            fdtd,
            equilibrium,
            stability,
            disruption,
            fusion,
            controller,
            dt: params.dt,
            output_interval: (params.output_interval / params.dt) as u64,
        }
    }

    /// Create with default TS-1 configuration
    pub fn ts1(params: SimulationParams) -> Self {
        Self::new(TokamakConfig::ts1(), params)
    }

    /// Advance simulation by one timestep
    pub fn step(&mut self) {
        // 1. Get field at particle positions and push particles
        self.push_particles();

        // 2. Deposit charge and current from particles to grid
        self.deposit_to_grid();

        // 3. Update fields (FDTD)
        self.fdtd.step(&mut self.fields);

        // 4. Check for fusion reactions
        self.check_fusion();

        // 5. Update MHD stability
        self.check_stability();

        // 6. Apply control
        self.apply_control();

        // 7. Update state
        self.state.step += 1;
        self.state.time += self.dt;
        self.state.status = SimulationStatus::Running;

        // Update statistics
        self.electrons.update_stats();
        self.deuterium.update_stats();
        self.tritium.update_stats();

        self.state.kinetic_energy = self.electrons.total_kinetic_energy()
            + self.deuterium.total_kinetic_energy()
            + self.tritium.total_kinetic_energy()
            + self.alphas.total_kinetic_energy();

        self.state.field_energy = self.fields.total_energy();
        self.state.particle_count = self.electrons.active_count()
            + self.deuterium.active_count()
            + self.tritium.active_count()
            + self.alphas.active_count();
    }

    /// Push all particles through one timestep
    fn push_particles(&mut self) {
        // Get fields at particle positions (simplified: uniform background)
        let b_field = Vec3::new(0.0, self.config.toroidal_field, 0.0);
        let e_field = Vec3::zero();

        for p in &mut self.electrons.particles {
            if p.active {
                self.pusher.push(p, e_field, b_field);
            }
        }

        for p in &mut self.deuterium.particles {
            if p.active {
                self.pusher.push(p, e_field, b_field);
            }
        }

        for p in &mut self.tritium.particles {
            if p.active {
                self.pusher.push(p, e_field, b_field);
            }
        }

        for p in &mut self.alphas.particles {
            if p.active {
                self.pusher.push(p, e_field, b_field);
            }
        }
    }

    /// Deposit charge and current to grid
    fn deposit_to_grid(&mut self) {
        self.fields.clear_current();

        // TODO: Implement proper charge/current deposition
        // For now, just track that particles exist
    }

    /// Check for fusion reactions
    fn check_fusion(&mut self) {
        self.fusion.clear_events();

        // Sample D-T pairs for fusion
        let n_d = self.deuterium.particles.len();
        let n_t = self.tritium.particles.len();

        if n_d == 0 || n_t == 0 {
            return;
        }

        // Monte Carlo sampling of pairs (not all pairs - that's O(n²))
        let n_samples = (n_d * n_t).min(10_000);

        for _ in 0..n_samples {
            // TODO: proper random sampling
            let i_d = (self.state.step as usize) % n_d;
            let i_t = ((self.state.step as usize) * 7) % n_t;

            let d = &self.deuterium.particles[i_d];
            let t = &self.tritium.particles[i_t];

            if d.active && t.active {
                if let Some(event) = self.fusion.check_fusion(d, t, self.dt, self.state.time) {
                    // Create alpha particle
                    let alpha_v = event.products[0].1;
                    let alpha = Particle::new(
                        Species::Alpha,
                        event.position,
                        alpha_v,
                        d.weight,
                    );
                    self.alphas.particles.push(alpha);
                }
            }
        }

        self.state.fusion_power = self.fusion.fusion_power(self.dt);
        self.state.fusion_count = self.fusion.total_fusions;
    }

    /// Check MHD stability
    fn check_stability(&mut self) {
        // Calculate beta_N
        let beta = self.calculate_beta();
        let beta_n = beta * self.config.minor_radius * self.config.toroidal_field
            / (self.config.plasma_current / 1e6);

        // Estimate q95
        let q95 = 5.0 * self.config.minor_radius.powi(2) * self.config.toroidal_field
            / (self.config.major_radius * self.config.plasma_current / 1e6)
            * ((1.0 + self.config.elongation.powi(2)) / 2.0);

        let result = self.stability.is_stable(
            beta_n,
            q95,
            self.config.plasma_current / 1e6,
            self.config.minor_radius,
            self.config.toroidal_field,
        );

        if !result.overall_stable {
            self.state.status = SimulationStatus::Disruption;
        }

        // Update disruption predictor
        let indicators = DisruptionIndicators {
            time: self.state.time,
            beta_n,
            q95,
            z_position: 0.0,  // TODO: calculate actual position
            current_rate: 0.0,
            locked_mode: 0.0,
            rad_fraction: 0.0,
        };
        self.disruption.add_data(indicators);
    }

    /// Apply plasma control
    fn apply_control(&mut self) {
        let measurements = PlasmaMeasurements {
            plasma_current: self.config.plasma_current,
            vertical_position: 0.0,
            radial_position: self.config.major_radius,
            density: self.config.density,
            ion_temperature_kev: self.config.ion_temperature_kev,
            electron_temperature_kev: self.config.electron_temperature_kev,
            beta_n: self.calculate_beta_n(),
            q95: self.calculate_q95(),
            stored_energy: self.state.kinetic_energy + self.state.field_energy,
            fusion_power: self.state.fusion_power,
        };

        let output = self.controller.compute(&measurements, self.dt);

        if output.emergency_shutdown {
            self.state.status = SimulationStatus::Error;
        }
    }

    /// Calculate plasma beta (ratio of plasma pressure to magnetic pressure)
    pub fn calculate_beta(&self) -> f64 {
        let n = self.config.density;
        let t = self.config.ion_temperature_kev * 1000.0 * EV_TO_J;
        let p = 2.0 * n * t;  // Pressure = 2nkT (ions + electrons)

        let b = self.config.toroidal_field;
        let p_mag = b * b / (2.0 * MU_0);

        p / p_mag
    }

    /// Calculate normalized beta
    pub fn calculate_beta_n(&self) -> f64 {
        let beta = self.calculate_beta();
        beta * self.config.minor_radius * self.config.toroidal_field
            / (self.config.plasma_current / 1e6) * 100.0
    }

    /// Calculate safety factor at 95% flux
    pub fn calculate_q95(&self) -> f64 {
        5.0 * self.config.minor_radius.powi(2) * self.config.toroidal_field
            / (self.config.major_radius * self.config.plasma_current / 1e6)
            * ((1.0 + self.config.elongation.powi(2)) / 2.0)
    }

    /// Run simulation for specified time
    pub fn run(&mut self, total_time: f64) {
        let n_steps = (total_time / self.dt) as u64;

        for _ in 0..n_steps {
            self.step();

            if self.state.status == SimulationStatus::Error ||
               self.state.status == SimulationStatus::Disruption {
                break;
            }
        }

        self.state.status = SimulationStatus::Completed;
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "TOKASIM-RS Simulation Summary\n\
             ==============================\n\
             Device: {}\n\
             Time: {:.3e} s ({} steps)\n\
             Status: {:?}\n\n\
             Plasma Parameters:\n\
             - Kinetic energy: {:.3e} J\n\
             - Field energy: {:.3e} J\n\
             - Fusion power: {:.3e} W\n\
             - Total fusions: {}\n\
             - Active particles: {}\n\n\
             Performance Metrics:\n\
             - β_N: {:.2}\n\
             - q95: {:.2}\n\
             - Q factor: {:.2}",
            self.config.name,
            self.state.time,
            self.state.step,
            self.state.status,
            self.state.kinetic_energy,
            self.state.field_energy,
            self.state.fusion_power,
            self.state.fusion_count,
            self.state.particle_count,
            self.calculate_beta_n(),
            self.calculate_q95(),
            FusionRates::q_factor(self.state.fusion_power, self.config.total_heating_mw() * 1e6),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let params = SimulationParams {
            n_particles: 1000,
            grid_resolution: 10,
            ..Default::default()
        };

        let sim = TokamakSimulator::ts1(params);
        assert_eq!(sim.config.name, "TS-1");
        assert!(sim.electrons.particles.len() > 0);
    }

    #[test]
    fn test_simulator_step() {
        let params = SimulationParams {
            n_particles: 100,
            grid_resolution: 10,
            dt: 1e-12,
            ..Default::default()
        };

        let mut sim = TokamakSimulator::ts1(params);
        sim.step();

        assert_eq!(sim.state.step, 1);
        assert!(sim.state.time > 0.0);
    }

    #[test]
    fn test_beta_calculation() {
        let params = SimulationParams::default();
        let sim = TokamakSimulator::ts1(params);

        let beta = sim.calculate_beta();
        assert!(beta > 0.0);
        assert!(beta < 0.1);  // Beta should be a few percent
    }
}
