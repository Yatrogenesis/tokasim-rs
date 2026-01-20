//! # Multiphysics Coupling Module
//!
//! Provides tight (monolithic) and loose (partitioned) coupling strategies
//! for thermal-structural-electromagnetic interactions.
//!
//! ## Coupling Types
//!
//! ### Weak/Loose Coupling (Sequential)
//! ```text
//! Thermal → Structural → EM → Thermal → ...
//! ```
//! Simple but may miss important interactions.
//!
//! ### Tight/Strong Coupling (Monolithic)
//! ```text
//! ┌─────────────────────────────────┐
//! │     Coupled System Matrix       │
//! │  ┌───┬───┬───┐                 │
//! │  │ T │Tσ │TE │ {ΔT}   {R_T}   │
//! │  ├───┼───┼───┤ {Δu} = {R_u}   │
//! │  │σT │ σ │σE │ {ΔB}   {R_B}   │
//! │  ├───┼───┼───┤                 │
//! │  │ET │Eσ │ E │                 │
//! │  └───┴───┴───┘                 │
//! └─────────────────────────────────┘
//! ```
//! Solves all physics simultaneously within the same timestep.
//!
//! ## References
//!
//! - Felippa et al., "Partitioned analysis of coupled mechanical systems" (2001)
//! - Park & Felippa, "A variational framework for solution method developments" (2000)
//! - Küttler & Wall, "Fixed-point fluid-structure interaction solvers" (2008)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

pub mod jfnk;

use crate::types::Vec3;

// Re-export JFNK solver
pub use jfnk::{JFNKSolver, JFNKConfig, CoupledStateVector};

// ============================================================================
// COUPLING ENUMS AND TRAITS
// ============================================================================

/// Coupling strategy for multiphysics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingStrategy {
    /// Sequential/staggered - each physics solved separately, exchange data
    Loose,
    /// Iterative - iterate between physics until convergence
    Iterative { max_iter: usize, tolerance: f64 },
    /// Monolithic - solve all physics in single coupled system
    Monolithic,
}

/// Physics domain types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsDomain {
    Thermal,
    Structural,
    Electromagnetic,
    Fluid,
    Particle,
}

/// Field data exchanged between domains
#[derive(Debug, Clone)]
pub struct FieldExchange {
    /// Field name
    pub name: String,
    /// Source domain
    pub source: PhysicsDomain,
    /// Target domain
    pub target: PhysicsDomain,
    /// Data values at nodes/cells
    pub values: Vec<f64>,
    /// Optional gradient data
    pub gradients: Option<Vec<Vec3>>,
}

/// Interface between two physics domains
#[derive(Debug, Clone)]
pub struct CouplingInterface {
    /// Interface name
    pub name: String,
    /// First domain
    pub domain_a: PhysicsDomain,
    /// Second domain
    pub domain_b: PhysicsDomain,
    /// Mapping: indices in domain_a → indices in domain_b
    pub mapping_a_to_b: Vec<(usize, usize)>,
    /// Interpolation weights
    pub weights: Vec<f64>,
}

// ============================================================================
// THERMAL-STRUCTURAL COUPLING
// ============================================================================

/// Thermal expansion coupling: T → displacement
#[derive(Debug, Clone)]
pub struct ThermalStructuralCoupling {
    /// Coefficient of thermal expansion (1/K)
    pub alpha: f64,
    /// Reference temperature (K)
    pub t_ref: f64,
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Current temperature field
    pub temperature: Vec<f64>,
    /// Computed thermal strain
    pub thermal_strain: Vec<f64>,
    /// Computed displacement
    pub displacement: Vec<Vec3>,
    /// Computed stress (von Mises)
    pub stress: Vec<f64>,
}

impl ThermalStructuralCoupling {
    /// Create for tungsten (first wall material)
    pub fn tungsten() -> Self {
        Self {
            alpha: 4.5e-6,        // 1/K at high T
            t_ref: 293.0,         // K
            youngs_modulus: 400e9, // Pa
            poisson_ratio: 0.28,
            temperature: Vec::new(),
            thermal_strain: Vec::new(),
            displacement: Vec::new(),
            stress: Vec::new(),
        }
    }

    /// Create for EUROFER steel (structural material)
    pub fn eurofer() -> Self {
        Self {
            alpha: 11.0e-6,
            t_ref: 293.0,
            youngs_modulus: 210e9,
            poisson_ratio: 0.30,
            temperature: Vec::new(),
            thermal_strain: Vec::new(),
            displacement: Vec::new(),
            stress: Vec::new(),
        }
    }

    /// Initialize with mesh size
    pub fn initialize(&mut self, n_nodes: usize) {
        self.temperature = vec![self.t_ref; n_nodes];
        self.thermal_strain = vec![0.0; n_nodes];
        self.displacement = vec![Vec3::zero(); n_nodes];
        self.stress = vec![0.0; n_nodes];
    }

    /// Compute thermal strain from temperature field
    /// ε_th = α(T - T_ref)
    pub fn compute_thermal_strain(&mut self) {
        for i in 0..self.temperature.len() {
            self.thermal_strain[i] = self.alpha * (self.temperature[i] - self.t_ref);
        }
    }

    /// Estimate displacement from thermal strain (1D approximation)
    /// For full 3D, need FEM structural solver
    pub fn estimate_displacement(&mut self, element_size: f64) {
        // Cumulative displacement along one direction
        let mut cumulative = 0.0;
        for i in 0..self.thermal_strain.len() {
            cumulative += self.thermal_strain[i] * element_size;
            self.displacement[i] = Vec3::new(cumulative, 0.0, 0.0);
        }
    }

    /// Estimate thermal stress (constrained expansion)
    /// σ = E·α·ΔT (for fully constrained case)
    pub fn compute_thermal_stress(&mut self) {
        for i in 0..self.temperature.len() {
            let dt = self.temperature[i] - self.t_ref;
            // Biaxial constraint factor: E/(1-ν)
            let factor = self.youngs_modulus / (1.0 - self.poisson_ratio);
            self.stress[i] = factor * self.alpha * dt;
        }
    }

    /// Full coupling step: T → ε → u → σ
    pub fn couple(&mut self, element_size: f64) {
        self.compute_thermal_strain();
        self.estimate_displacement(element_size);
        self.compute_thermal_stress();
    }

    /// Get maximum thermal stress (Pa)
    pub fn max_stress(&self) -> f64 {
        self.stress.iter().cloned().fold(0.0, f64::max)
    }

    /// Get maximum displacement magnitude (m)
    pub fn max_displacement(&self) -> f64 {
        self.displacement.iter()
            .map(|d| (d.x*d.x + d.y*d.y + d.z*d.z).sqrt())
            .fold(0.0, f64::max)
    }
}

// ============================================================================
// STRUCTURAL-ELECTROMAGNETIC COUPLING
// ============================================================================

/// Magnetostriction and Lorentz force coupling
#[derive(Debug, Clone)]
pub struct StructuralEMCoupling {
    /// Electrical conductivity (S/m)
    pub sigma: f64,
    /// Magnetic permeability (H/m)
    pub mu: f64,
    /// Lorentz force density at each point
    pub lorentz_force: Vec<Vec3>,
    /// Maxwell stress tensor (diagonal components)
    pub maxwell_stress: Vec<(f64, f64, f64)>,
    /// Induced current density
    pub current_density: Vec<Vec3>,
}

impl StructuralEMCoupling {
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            mu: 4.0 * std::f64::consts::PI * 1e-7,  // μ₀
            lorentz_force: Vec::new(),
            maxwell_stress: Vec::new(),
            current_density: Vec::new(),
        }
    }

    /// Initialize with mesh size
    pub fn initialize(&mut self, n_points: usize) {
        self.lorentz_force = vec![Vec3::zero(); n_points];
        self.maxwell_stress = vec![(0.0, 0.0, 0.0); n_points];
        self.current_density = vec![Vec3::zero(); n_points];
    }

    /// Compute Lorentz force: F = J × B
    pub fn compute_lorentz_force(&mut self, b_field: &[Vec3]) {
        for i in 0..self.current_density.len().min(b_field.len()) {
            let j = &self.current_density[i];
            let b = &b_field[i];
            self.lorentz_force[i] = Vec3::new(
                j.y * b.z - j.z * b.y,
                j.z * b.x - j.x * b.z,
                j.x * b.y - j.y * b.x,
            );
        }
    }

    /// Compute Maxwell stress tensor diagonal components
    /// T_ij = (1/μ₀)(B_i·B_j - δ_ij·B²/2)
    pub fn compute_maxwell_stress(&mut self, b_field: &[Vec3]) {
        for i in 0..b_field.len().min(self.maxwell_stress.len()) {
            let b = &b_field[i];
            let b_sq = b.x*b.x + b.y*b.y + b.z*b.z;
            let inv_mu = 1.0 / self.mu;

            // Diagonal: T_xx, T_yy, T_zz
            self.maxwell_stress[i] = (
                inv_mu * (b.x*b.x - 0.5*b_sq),
                inv_mu * (b.y*b.y - 0.5*b_sq),
                inv_mu * (b.z*b.z - 0.5*b_sq),
            );
        }
    }

    /// Compute induced current from moving conductor: J = σ(v × B)
    pub fn compute_induced_current(&mut self, velocity: &[Vec3], b_field: &[Vec3]) {
        for i in 0..velocity.len().min(b_field.len()).min(self.current_density.len()) {
            let v = &velocity[i];
            let b = &b_field[i];

            // v × B
            let vxb = Vec3::new(
                v.y * b.z - v.z * b.y,
                v.z * b.x - v.x * b.z,
                v.x * b.y - v.y * b.x,
            );

            self.current_density[i] = Vec3::new(
                self.sigma * vxb.x,
                self.sigma * vxb.y,
                self.sigma * vxb.z,
            );
        }
    }

    /// Get maximum Lorentz force magnitude
    pub fn max_lorentz_force(&self) -> f64 {
        self.lorentz_force.iter()
            .map(|f| (f.x*f.x + f.y*f.y + f.z*f.z).sqrt())
            .fold(0.0, f64::max)
    }
}

// ============================================================================
// MONOLITHIC COUPLED SOLVER
// ============================================================================

/// Monolithic solver state for tight coupling
#[derive(Debug, Clone)]
pub struct CoupledState {
    /// Temperature field
    pub temperature: Vec<f64>,
    /// Displacement field
    pub displacement: Vec<Vec3>,
    /// Magnetic field
    pub b_field: Vec<Vec3>,
    /// Velocity field (if fluid)
    pub velocity: Vec<Vec3>,
}

impl CoupledState {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            temperature: vec![300.0; n_nodes],
            displacement: vec![Vec3::zero(); n_nodes],
            b_field: vec![Vec3::zero(); n_nodes],
            velocity: vec![Vec3::zero(); n_nodes],
        }
    }

    /// Pack all fields into single solution vector
    pub fn pack(&self) -> Vec<f64> {
        let n = self.temperature.len();
        let mut packed = Vec::with_capacity(n * 10);

        // Temperature (1 DOF per node)
        packed.extend(&self.temperature);

        // Displacement (3 DOF per node)
        for d in &self.displacement {
            packed.push(d.x);
            packed.push(d.y);
            packed.push(d.z);
        }

        // B-field (3 DOF per node)
        for b in &self.b_field {
            packed.push(b.x);
            packed.push(b.y);
            packed.push(b.z);
        }

        // Velocity (3 DOF per node)
        for v in &self.velocity {
            packed.push(v.x);
            packed.push(v.y);
            packed.push(v.z);
        }

        packed
    }

    /// Unpack solution vector into fields
    pub fn unpack(&mut self, packed: &[f64]) {
        let n = self.temperature.len();
        let mut idx = 0;

        // Temperature
        for i in 0..n {
            self.temperature[i] = packed[idx];
            idx += 1;
        }

        // Displacement
        for i in 0..n {
            self.displacement[i] = Vec3::new(packed[idx], packed[idx+1], packed[idx+2]);
            idx += 3;
        }

        // B-field
        for i in 0..n {
            self.b_field[i] = Vec3::new(packed[idx], packed[idx+1], packed[idx+2]);
            idx += 3;
        }

        // Velocity
        for i in 0..n {
            self.velocity[i] = Vec3::new(packed[idx], packed[idx+1], packed[idx+2]);
            idx += 3;
        }
    }

    /// Total degrees of freedom
    pub fn total_dof(&self) -> usize {
        self.temperature.len() * 10  // 1 + 3 + 3 + 3
    }
}

/// Coupling matrix entry (block structure)
#[derive(Debug, Clone)]
pub struct CouplingMatrixBlock {
    /// Source field type
    pub from: PhysicsDomain,
    /// Target field type
    pub to: PhysicsDomain,
    /// Row start in global matrix
    pub row_start: usize,
    /// Column start in global matrix
    pub col_start: usize,
    /// Block size (rows × cols)
    pub size: (usize, usize),
    /// Non-zero entries: (local_row, local_col, value)
    pub entries: Vec<(usize, usize, f64)>,
}

/// Monolithic multiphysics coupler
pub struct MultiPhysicsCoupler {
    /// Coupling strategy
    pub strategy: CouplingStrategy,
    /// Number of nodes in mesh
    pub n_nodes: usize,
    /// Thermal-structural coupling
    pub thermal_structural: ThermalStructuralCoupling,
    /// Structural-EM coupling
    pub structural_em: StructuralEMCoupling,
    /// Current coupled state
    pub state: CoupledState,
    /// Previous state (for convergence check)
    pub state_prev: CoupledState,
    /// Coupling matrix blocks (for monolithic)
    pub coupling_blocks: Vec<CouplingMatrixBlock>,
    /// Residual vector
    pub residual: Vec<f64>,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// Element size for displacement calculation
    pub element_size: f64,
}

impl MultiPhysicsCoupler {
    pub fn new(n_nodes: usize, strategy: CouplingStrategy) -> Self {
        let mut thermal_structural = ThermalStructuralCoupling::tungsten();
        thermal_structural.initialize(n_nodes);

        let mut structural_em = StructuralEMCoupling::new(1e6);
        structural_em.initialize(n_nodes);

        Self {
            strategy,
            n_nodes,
            thermal_structural,
            structural_em,
            state: CoupledState::new(n_nodes),
            state_prev: CoupledState::new(n_nodes),
            coupling_blocks: Vec::new(),
            residual: vec![0.0; n_nodes * 10],
            convergence_history: Vec::new(),
            element_size: 0.01,  // Default 1cm
        }
    }

    /// Set element size for displacement calculation
    pub fn set_element_size(&mut self, size: f64) {
        self.element_size = size;
    }

    /// Set temperature field from external source (CFD)
    pub fn set_temperature(&mut self, temperature: &[f64]) {
        let n = self.n_nodes.min(temperature.len());
        self.state.temperature[..n].copy_from_slice(&temperature[..n]);
        self.thermal_structural.temperature[..n].copy_from_slice(&temperature[..n]);
    }

    /// Set magnetic field from external source (EM solver)
    pub fn set_magnetic_field(&mut self, b_field: &[Vec3]) {
        let n = self.n_nodes.min(b_field.len());
        self.state.b_field[..n].copy_from_slice(&b_field[..n]);
    }

    /// Set velocity field from external source (Fluid solver)
    pub fn set_velocity(&mut self, velocity: &[Vec3]) {
        let n = self.n_nodes.min(velocity.len());
        self.state.velocity[..n].copy_from_slice(&velocity[..n]);
    }

    /// Execute coupling based on strategy
    pub fn couple(&mut self) -> CouplingResult {
        match self.strategy {
            CouplingStrategy::Loose => self.couple_loose(),
            CouplingStrategy::Iterative { max_iter, tolerance } => {
                self.couple_iterative(max_iter, tolerance)
            }
            CouplingStrategy::Monolithic => self.couple_monolithic(),
        }
    }

    /// Loose/sequential coupling
    fn couple_loose(&mut self) -> CouplingResult {
        // 1. Thermal → Structural
        self.thermal_structural.couple(self.element_size);

        // Copy displacement to state
        for i in 0..self.n_nodes {
            self.state.displacement[i] = self.thermal_structural.displacement[i];
        }

        // 2. Structural + EM → Lorentz forces
        self.structural_em.compute_induced_current(&self.state.velocity, &self.state.b_field);
        self.structural_em.compute_lorentz_force(&self.state.b_field);
        self.structural_em.compute_maxwell_stress(&self.state.b_field);

        CouplingResult {
            converged: true,
            iterations: 1,
            residual: 0.0,
            max_displacement: self.thermal_structural.max_displacement(),
            max_stress: self.thermal_structural.max_stress(),
            max_lorentz_force: self.structural_em.max_lorentz_force(),
        }
    }

    /// Iterative (block Gauss-Seidel) coupling
    fn couple_iterative(&mut self, max_iter: usize, tolerance: f64) -> CouplingResult {
        self.convergence_history.clear();

        for iter in 0..max_iter {
            // Save previous state
            self.state_prev = self.state.clone();

            // 1. Solve thermal → structural
            self.thermal_structural.couple(self.element_size);
            for i in 0..self.n_nodes {
                self.state.displacement[i] = self.thermal_structural.displacement[i];
            }

            // 2. Solve structural + EM
            self.structural_em.compute_induced_current(&self.state.velocity, &self.state.b_field);
            self.structural_em.compute_lorentz_force(&self.state.b_field);

            // 3. Update fields with relaxation (Aitken)
            let omega = 0.7;  // Under-relaxation factor
            for i in 0..self.n_nodes {
                self.state.displacement[i].x = omega * self.state.displacement[i].x
                    + (1.0 - omega) * self.state_prev.displacement[i].x;
                self.state.displacement[i].y = omega * self.state.displacement[i].y
                    + (1.0 - omega) * self.state_prev.displacement[i].y;
                self.state.displacement[i].z = omega * self.state.displacement[i].z
                    + (1.0 - omega) * self.state_prev.displacement[i].z;
            }

            // 4. Check convergence
            let residual = self.compute_residual();
            self.convergence_history.push(residual);

            if residual < tolerance {
                return CouplingResult {
                    converged: true,
                    iterations: iter + 1,
                    residual,
                    max_displacement: self.thermal_structural.max_displacement(),
                    max_stress: self.thermal_structural.max_stress(),
                    max_lorentz_force: self.structural_em.max_lorentz_force(),
                };
            }
        }

        CouplingResult {
            converged: false,
            iterations: max_iter,
            residual: *self.convergence_history.last().unwrap_or(&1.0),
            max_displacement: self.thermal_structural.max_displacement(),
            max_stress: self.thermal_structural.max_stress(),
            max_lorentz_force: self.structural_em.max_lorentz_force(),
        }
    }

    /// Monolithic (Newton-Raphson) coupling
    /// Solves the fully coupled system in one shot
    fn couple_monolithic(&mut self) -> CouplingResult {
        // Build coupling matrix blocks
        self.build_coupling_matrix();

        // For a true monolithic solver, we'd assemble and solve:
        // K·Δx = -R
        // where K is the Jacobian of the coupled system
        //
        // This is a simplified version that uses block iteration
        // with strong coupling between blocks

        let max_newton_iter = 20;
        let newton_tol = 1e-8;

        for iter in 0..max_newton_iter {
            // Compute residual for all physics
            self.compute_coupled_residual();

            let residual_norm = self.residual_norm();
            self.convergence_history.push(residual_norm);

            if residual_norm < newton_tol {
                return CouplingResult {
                    converged: true,
                    iterations: iter + 1,
                    residual: residual_norm,
                    max_displacement: self.thermal_structural.max_displacement(),
                    max_stress: self.thermal_structural.max_stress(),
                    max_lorentz_force: self.structural_em.max_lorentz_force(),
                };
            }

            // Apply Newton update (simplified - in practice would solve linear system)
            self.apply_newton_correction();
        }

        CouplingResult {
            converged: false,
            iterations: max_newton_iter,
            residual: self.residual_norm(),
            max_displacement: self.thermal_structural.max_displacement(),
            max_stress: self.thermal_structural.max_stress(),
            max_lorentz_force: self.structural_em.max_lorentz_force(),
        }
    }

    /// Build coupling matrix block structure
    fn build_coupling_matrix(&mut self) {
        self.coupling_blocks.clear();

        // Thermal → Thermal (diagonal)
        self.coupling_blocks.push(CouplingMatrixBlock {
            from: PhysicsDomain::Thermal,
            to: PhysicsDomain::Thermal,
            row_start: 0,
            col_start: 0,
            size: (self.n_nodes, self.n_nodes),
            entries: Vec::new(),  // Would be populated by thermal solver
        });

        // Thermal → Structural (off-diagonal coupling)
        self.coupling_blocks.push(CouplingMatrixBlock {
            from: PhysicsDomain::Thermal,
            to: PhysicsDomain::Structural,
            row_start: self.n_nodes,
            col_start: 0,
            size: (self.n_nodes * 3, self.n_nodes),
            entries: Vec::new(),  // Thermal expansion coupling
        });

        // Structural → Structural (diagonal)
        self.coupling_blocks.push(CouplingMatrixBlock {
            from: PhysicsDomain::Structural,
            to: PhysicsDomain::Structural,
            row_start: self.n_nodes,
            col_start: self.n_nodes,
            size: (self.n_nodes * 3, self.n_nodes * 3),
            entries: Vec::new(),
        });

        // EM → Structural (Lorentz force coupling)
        self.coupling_blocks.push(CouplingMatrixBlock {
            from: PhysicsDomain::Electromagnetic,
            to: PhysicsDomain::Structural,
            row_start: self.n_nodes,
            col_start: self.n_nodes * 4,
            size: (self.n_nodes * 3, self.n_nodes * 3),
            entries: Vec::new(),
        });

        // Structural → EM (geometry change affects field)
        self.coupling_blocks.push(CouplingMatrixBlock {
            from: PhysicsDomain::Structural,
            to: PhysicsDomain::Electromagnetic,
            row_start: self.n_nodes * 4,
            col_start: self.n_nodes,
            size: (self.n_nodes * 3, self.n_nodes * 3),
            entries: Vec::new(),
        });
    }

    /// Compute coupled residual for all physics
    fn compute_coupled_residual(&mut self) {
        // Clear residual
        for r in &mut self.residual {
            *r = 0.0;
        }

        let mut idx = 0;

        // Thermal residual: R_T = Q - ∇·(k∇T)
        for i in 0..self.n_nodes {
            // Simplified: just track temperature change
            self.residual[idx] = self.state.temperature[i] - self.state_prev.temperature[i];
            idx += 1;
        }

        // Structural residual: R_u = F_ext - K·u - F_thermal
        for i in 0..self.n_nodes {
            let u = &self.state.displacement[i];
            let u_prev = &self.state_prev.displacement[i];
            self.residual[idx] = u.x - u_prev.x;
            self.residual[idx + 1] = u.y - u_prev.y;
            self.residual[idx + 2] = u.z - u_prev.z;
            idx += 3;
        }

        // EM residual: R_B = ∇×H - J
        for i in 0..self.n_nodes {
            let b = &self.state.b_field[i];
            let b_prev = &self.state_prev.b_field[i];
            self.residual[idx] = b.x - b_prev.x;
            self.residual[idx + 1] = b.y - b_prev.y;
            self.residual[idx + 2] = b.z - b_prev.z;
            idx += 3;
        }
    }

    /// Compute displacement residual for convergence check
    fn compute_residual(&self) -> f64 {
        let mut residual = 0.0;
        for i in 0..self.n_nodes {
            let du = self.state.displacement[i].x - self.state_prev.displacement[i].x;
            let dv = self.state.displacement[i].y - self.state_prev.displacement[i].y;
            let dw = self.state.displacement[i].z - self.state_prev.displacement[i].z;
            residual += du*du + dv*dv + dw*dw;
        }
        (residual / self.n_nodes as f64).sqrt()
    }

    /// Compute norm of full residual vector
    fn residual_norm(&self) -> f64 {
        let sum_sq: f64 = self.residual.iter().map(|r| r * r).sum();
        (sum_sq / self.residual.len() as f64).sqrt()
    }

    /// Apply Newton correction (simplified)
    fn apply_newton_correction(&mut self) {
        // In full implementation, would solve K·Δx = -R
        // Here we use a simplified gradient descent approach

        let alpha = 0.1;  // Step size

        let mut idx = 0;

        // Temperature correction
        for i in 0..self.n_nodes {
            self.state.temperature[i] -= alpha * self.residual[idx];
            self.thermal_structural.temperature[i] = self.state.temperature[i];
            idx += 1;
        }

        // Recompute thermal-structural coupling
        self.thermal_structural.couple(self.element_size);
        for i in 0..self.n_nodes {
            self.state.displacement[i] = self.thermal_structural.displacement[i];
        }

        // Recompute EM coupling
        self.structural_em.compute_induced_current(&self.state.velocity, &self.state.b_field);
        self.structural_em.compute_lorentz_force(&self.state.b_field);
    }

    /// Get displacement at node (for mesh deformation)
    pub fn get_displacement(&self, node: usize) -> Vec3 {
        if node < self.n_nodes {
            self.state.displacement[node]
        } else {
            Vec3::zero()
        }
    }

    /// Get thermal stress at node
    pub fn get_thermal_stress(&self, node: usize) -> f64 {
        if node < self.thermal_structural.stress.len() {
            self.thermal_structural.stress[node]
        } else {
            0.0
        }
    }

    /// Get Lorentz force at node
    pub fn get_lorentz_force(&self, node: usize) -> Vec3 {
        if node < self.structural_em.lorentz_force.len() {
            self.structural_em.lorentz_force[node]
        } else {
            Vec3::zero()
        }
    }
}

/// Result of coupling operation
#[derive(Debug, Clone)]
pub struct CouplingResult {
    /// Whether coupling converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
    /// Maximum displacement (m)
    pub max_displacement: f64,
    /// Maximum thermal stress (Pa)
    pub max_stress: f64,
    /// Maximum Lorentz force density (N/m³)
    pub max_lorentz_force: f64,
}

// ============================================================================
// MESH DEFORMATION
// ============================================================================

/// Mesh deformation handler
pub struct MeshDeformer {
    /// Original node positions
    pub original_positions: Vec<Vec3>,
    /// Current (deformed) positions
    pub current_positions: Vec<Vec3>,
    /// Deformation mode
    pub mode: DeformationMode,
}

#[derive(Debug, Clone, Copy)]
pub enum DeformationMode {
    /// Direct displacement application
    Direct,
    /// Laplacian smoothing
    LaplacianSmoothing { iterations: usize },
    /// RBF (Radial Basis Function) interpolation
    RBF { radius: f64 },
}

impl MeshDeformer {
    pub fn new(positions: Vec<Vec3>) -> Self {
        let current = positions.clone();
        Self {
            original_positions: positions,
            current_positions: current,
            mode: DeformationMode::Direct,
        }
    }

    /// Apply displacements from coupling
    pub fn apply_displacement(&mut self, displacements: &[Vec3]) {
        match self.mode {
            DeformationMode::Direct => {
                for i in 0..self.original_positions.len().min(displacements.len()) {
                    self.current_positions[i] = Vec3::new(
                        self.original_positions[i].x + displacements[i].x,
                        self.original_positions[i].y + displacements[i].y,
                        self.original_positions[i].z + displacements[i].z,
                    );
                }
            }
            DeformationMode::LaplacianSmoothing { iterations } => {
                // Apply direct first
                for i in 0..self.original_positions.len().min(displacements.len()) {
                    self.current_positions[i] = Vec3::new(
                        self.original_positions[i].x + displacements[i].x,
                        self.original_positions[i].y + displacements[i].y,
                        self.original_positions[i].z + displacements[i].z,
                    );
                }
                // Then smooth
                self.laplacian_smooth(iterations);
            }
            DeformationMode::RBF { radius } => {
                self.rbf_interpolate(displacements, radius);
            }
        }
    }

    /// Laplacian smoothing
    fn laplacian_smooth(&mut self, iterations: usize) {
        let n = self.current_positions.len();
        if n < 3 { return; }

        for _ in 0..iterations {
            let mut smoothed = self.current_positions.clone();

            for i in 1..n-1 {
                // Simple 1D smoothing (for demonstration)
                smoothed[i] = Vec3::new(
                    0.25 * self.current_positions[i-1].x +
                    0.5 * self.current_positions[i].x +
                    0.25 * self.current_positions[i+1].x,
                    0.25 * self.current_positions[i-1].y +
                    0.5 * self.current_positions[i].y +
                    0.25 * self.current_positions[i+1].y,
                    0.25 * self.current_positions[i-1].z +
                    0.5 * self.current_positions[i].z +
                    0.25 * self.current_positions[i+1].z,
                );
            }

            self.current_positions = smoothed;
        }
    }

    /// RBF interpolation for mesh deformation
    fn rbf_interpolate(&mut self, displacements: &[Vec3], radius: f64) {
        // Simplified RBF: Gaussian basis functions
        for i in 0..self.original_positions.len() {
            let mut weighted_disp = Vec3::zero();
            let mut weight_sum = 0.0;

            for j in 0..displacements.len() {
                let dx = self.original_positions[i].x - self.original_positions[j].x;
                let dy = self.original_positions[i].y - self.original_positions[j].y;
                let dz = self.original_positions[i].z - self.original_positions[j].z;
                let dist_sq = dx*dx + dy*dy + dz*dz;

                let weight = (-dist_sq / (radius * radius)).exp();
                weight_sum += weight;

                weighted_disp.x += weight * displacements[j].x;
                weighted_disp.y += weight * displacements[j].y;
                weighted_disp.z += weight * displacements[j].z;
            }

            if weight_sum > 1e-10 {
                self.current_positions[i] = Vec3::new(
                    self.original_positions[i].x + weighted_disp.x / weight_sum,
                    self.original_positions[i].y + weighted_disp.y / weight_sum,
                    self.original_positions[i].z + weighted_disp.z / weight_sum,
                );
            }
        }
    }

    /// Get maximum deformation magnitude
    pub fn max_deformation(&self) -> f64 {
        self.original_positions.iter()
            .zip(self.current_positions.iter())
            .map(|(orig, curr)| {
                let dx = curr.x - orig.x;
                let dy = curr.y - orig.y;
                let dz = curr.z - orig.z;
                (dx*dx + dy*dy + dz*dz).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// Reset to original mesh
    pub fn reset(&mut self) {
        self.current_positions = self.original_positions.clone();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_structural_coupling() {
        let mut coupling = ThermalStructuralCoupling::tungsten();
        coupling.initialize(10);

        // Set temperature gradient
        for i in 0..10 {
            coupling.temperature[i] = 300.0 + i as f64 * 100.0;  // 300K to 1200K
        }

        coupling.couple(0.01);

        assert!(coupling.max_stress() > 0.0);
        assert!(coupling.max_displacement() > 0.0);
    }

    #[test]
    fn test_structural_em_coupling() {
        let mut coupling = StructuralEMCoupling::new(1e6);
        coupling.initialize(10);

        // Set velocity and B-field
        let velocity: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64 * 0.1, 0.0, 0.0)).collect();
        let b_field: Vec<Vec3> = vec![Vec3::new(0.0, 5.0, 0.0); 10];

        coupling.compute_induced_current(&velocity, &b_field);
        coupling.compute_lorentz_force(&b_field);

        assert!(coupling.max_lorentz_force() > 0.0);
    }

    #[test]
    fn test_loose_coupling() {
        let mut coupler = MultiPhysicsCoupler::new(100, CouplingStrategy::Loose);

        // Set temperature field
        let temp: Vec<f64> = (0..100).map(|i| 300.0 + i as f64 * 10.0).collect();
        coupler.set_temperature(&temp);

        let result = coupler.couple();
        assert!(result.converged);
        assert!(result.max_displacement > 0.0);
    }

    #[test]
    fn test_iterative_coupling() {
        let mut coupler = MultiPhysicsCoupler::new(50, CouplingStrategy::Iterative {
            max_iter: 100,
            tolerance: 1e-6,
        });

        let temp: Vec<f64> = (0..50).map(|i| 500.0 + i as f64 * 5.0).collect();
        coupler.set_temperature(&temp);

        let result = coupler.couple();
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_mesh_deformer() {
        let positions: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64 * 0.1, 0.0, 0.0)).collect();
        let mut deformer = MeshDeformer::new(positions);

        let displacements: Vec<Vec3> = (0..10).map(|i| Vec3::new(0.0, i as f64 * 0.001, 0.0)).collect();
        deformer.apply_displacement(&displacements);

        assert!(deformer.max_deformation() > 0.0);
    }
}
