//! # Computational Fluid Dynamics Module
//!
//! Navier-Stokes solver for tokamak coolant systems (liquid metal, helium, water).
//!
//! ## Governing Equations
//!
//! ### Incompressible Navier-Stokes:
//!
//! ```text
//! ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
//! ∇·u = 0  (continuity)
//! ```
//!
//! ### Energy equation:
//!
//! ```text
//! ∂T/∂t + (u·∇)T = α∇²T + Q/(ρCp)
//! ```
//!
//! ### MHD effects (liquid metals):
//!
//! ```text
//! f_Lorentz = J × B = σ(E + u × B) × B
//! ```
//!
//! ## Numerical Methods
//!
//! - Finite Volume Method (FVM) on structured grids
//! - SIMPLE algorithm for pressure-velocity coupling
//! - Upwind/Central differencing for convection
//! - Implicit time stepping
//! - k-ε turbulence model
//!
//! ## References
//!
//! - Patankar, "Numerical Heat Transfer and Fluid Flow" (1980)
//! - Versteeg & Malalasekera, "Computational Fluid Dynamics" (2007)
//! - Müller & Bühler, "Magnetofluiddynamics in Channels and Containers" (2001)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use crate::types::Vec3;

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// Universal gas constant (J/mol·K)
pub const R_GAS: f64 = 8.314462618;

/// Stefan-Boltzmann constant (W/m²·K⁴)
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

// ============================================================================
// FLUID PROPERTIES
// ============================================================================

/// Coolant fluid types for fusion reactors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolantType {
    /// Water (subcritical)
    Water,
    /// Helium gas (high pressure)
    Helium,
    /// Lead-Lithium eutectic (Pb-17Li)
    PbLi,
    /// FLiBe molten salt (2LiF-BeF2)
    FLiBe,
    /// Liquid Lithium
    LiquidLithium,
}

/// Temperature-dependent fluid properties
#[derive(Debug, Clone)]
pub struct FluidProperties {
    pub coolant: CoolantType,
    /// Reference temperature (K)
    pub t_ref: f64,
    /// Density at T_ref (kg/m³)
    pub rho_ref: f64,
    /// Dynamic viscosity at T_ref (Pa·s)
    pub mu_ref: f64,
    /// Thermal conductivity at T_ref (W/m·K)
    pub k_ref: f64,
    /// Specific heat at T_ref (J/kg·K)
    pub cp_ref: f64,
    /// Thermal expansion coefficient (1/K)
    pub beta: f64,
    /// Electrical conductivity (S/m) - for MHD
    pub sigma_e: f64,
    /// Prandtl number
    pub pr: f64,
}

impl FluidProperties {
    /// Water at 300°C (subcritical)
    pub fn water_300c() -> Self {
        Self {
            coolant: CoolantType::Water,
            t_ref: 573.15,
            rho_ref: 712.0,
            mu_ref: 8.58e-5,
            k_ref: 0.545,
            cp_ref: 5650.0,
            beta: 3.0e-3,
            sigma_e: 0.05,  // Low conductivity
            pr: 0.89,
        }
    }

    /// Helium at 500°C, 8 MPa
    pub fn helium_8mpa() -> Self {
        Self {
            coolant: CoolantType::Helium,
            t_ref: 773.15,
            rho_ref: 3.67,
            mu_ref: 4.0e-5,
            k_ref: 0.32,
            cp_ref: 5193.0,
            beta: 1.29e-3,
            sigma_e: 0.0,  // Insulator
            pr: 0.65,
        }
    }

    /// Pb-17Li at 450°C
    pub fn pbli_450c() -> Self {
        Self {
            coolant: CoolantType::PbLi,
            t_ref: 723.15,
            rho_ref: 9490.0,
            mu_ref: 1.8e-3,
            k_ref: 20.6,
            cp_ref: 188.0,
            beta: 1.18e-4,
            sigma_e: 7.7e5,  // High conductivity - MHD effects!
            pr: 0.016,       // Very low Prandtl
        }
    }

    /// FLiBe at 600°C
    pub fn flibe_600c() -> Self {
        Self {
            coolant: CoolantType::FLiBe,
            t_ref: 873.15,
            rho_ref: 1940.0,
            mu_ref: 5.6e-3,
            k_ref: 1.1,
            cp_ref: 2380.0,
            beta: 2.0e-4,
            sigma_e: 155.0,
            pr: 12.1,
        }
    }

    /// Liquid Lithium at 500°C
    pub fn liquid_lithium() -> Self {
        Self {
            coolant: CoolantType::LiquidLithium,
            t_ref: 773.15,
            rho_ref: 480.0,
            mu_ref: 3.5e-4,
            k_ref: 50.0,
            cp_ref: 4170.0,
            beta: 2.0e-4,
            sigma_e: 3.0e6,  // Very high conductivity
            pr: 0.029,
        }
    }

    /// Density at temperature T (Boussinesq approximation)
    pub fn density(&self, t: f64) -> f64 {
        self.rho_ref * (1.0 - self.beta * (t - self.t_ref))
    }

    /// Dynamic viscosity at temperature (Arrhenius-type)
    pub fn viscosity(&self, t: f64) -> f64 {
        // Simplified temperature dependence
        let t_ratio = self.t_ref / t;
        self.mu_ref * t_ratio.powf(0.7)
    }

    /// Kinematic viscosity ν = μ/ρ
    pub fn kinematic_viscosity(&self, t: f64) -> f64 {
        self.viscosity(t) / self.density(t)
    }

    /// Thermal diffusivity α = k/(ρ·Cp)
    pub fn thermal_diffusivity(&self, t: f64) -> f64 {
        self.k_ref / (self.density(t) * self.cp_ref)
    }

    /// Hartmann number Ha = B·L·sqrt(σ/(ρ·ν))
    /// Measures MHD effects
    pub fn hartmann_number(&self, b: f64, l: f64, t: f64) -> f64 {
        let rho = self.density(t);
        let nu = self.kinematic_viscosity(t);
        b * l * (self.sigma_e / (rho * nu)).sqrt()
    }

    /// Reynolds number Re = ρ·u·L/μ
    pub fn reynolds_number(&self, u: f64, l: f64, t: f64) -> f64 {
        self.density(t) * u * l / self.viscosity(t)
    }

    /// Nusselt number for fully developed turbulent pipe flow (Dittus-Boelter)
    pub fn nusselt_turbulent(&self, re: f64, heating: bool) -> f64 {
        let n = if heating { 0.4 } else { 0.3 };
        0.023 * re.powf(0.8) * self.pr.powf(n)
    }
}

// ============================================================================
// MESH DEFINITION
// ============================================================================

/// Structured 3D mesh for CFD
#[derive(Debug, Clone)]
pub struct StructuredMesh {
    /// Number of cells in each direction
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Domain bounds
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    /// Cell sizes
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Cell volumes
    pub volumes: Vec<f64>,
    /// Face areas
    pub area_x: f64,
    pub area_y: f64,
    pub area_z: f64,
}

impl StructuredMesh {
    /// Create uniform Cartesian mesh
    pub fn cartesian(nx: usize, ny: usize, nz: usize,
                     bounds: ((f64, f64), (f64, f64), (f64, f64))) -> Self {
        let ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounds;

        let dx = (x_max - x_min) / nx as f64;
        let dy = (y_max - y_min) / ny as f64;
        let dz = (z_max - z_min) / nz as f64;

        let volume = dx * dy * dz;
        let volumes = vec![volume; nx * ny * nz];

        Self {
            nx, ny, nz,
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            dx, dy, dz,
            volumes,
            area_x: dy * dz,
            area_y: dx * dz,
            area_z: dx * dy,
        }
    }

    /// Total number of cells
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Cell index from (i, j, k)
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }

    /// (i, j, k) from cell index
    pub fn cell_ijk(&self, idx: usize) -> (usize, usize, usize) {
        let k = idx / (self.nx * self.ny);
        let rem = idx % (self.nx * self.ny);
        let j = rem / self.nx;
        let i = rem % self.nx;
        (i, j, k)
    }

    /// Cell center position
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.x_min + (i as f64 + 0.5) * self.dx,
            self.y_min + (j as f64 + 0.5) * self.dy,
            self.z_min + (k as f64 + 0.5) * self.dz,
        )
    }
}

// ============================================================================
// FIELD VARIABLES
// ============================================================================

/// Scalar field on mesh
#[derive(Debug, Clone)]
pub struct ScalarField {
    pub values: Vec<f64>,
    pub name: String,
}

impl ScalarField {
    pub fn new(name: &str, n_cells: usize, init_value: f64) -> Self {
        Self {
            values: vec![init_value; n_cells],
            name: name.into(),
        }
    }

    pub fn zeros(name: &str, n_cells: usize) -> Self {
        Self::new(name, n_cells, 0.0)
    }
}

/// Vector field on mesh (velocity, etc.)
#[derive(Debug, Clone)]
pub struct VectorField {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub name: String,
}

impl VectorField {
    pub fn new(name: &str, n_cells: usize) -> Self {
        Self {
            x: vec![0.0; n_cells],
            y: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            name: name.into(),
        }
    }

    pub fn magnitude(&self, idx: usize) -> f64 {
        (self.x[idx].powi(2) + self.y[idx].powi(2) + self.z[idx].powi(2)).sqrt()
    }

    pub fn set(&mut self, idx: usize, v: Vec3) {
        self.x[idx] = v.x;
        self.y[idx] = v.y;
        self.z[idx] = v.z;
    }

    pub fn get(&self, idx: usize) -> Vec3 {
        Vec3::new(self.x[idx], self.y[idx], self.z[idx])
    }
}

// ============================================================================
// BOUNDARY CONDITIONS
// ============================================================================

/// Boundary condition types
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    /// Dirichlet (fixed value)
    Dirichlet(f64),
    /// Neumann (fixed gradient)
    Neumann(f64),
    /// Wall (no-slip for velocity)
    Wall,
    /// Inlet with specified velocity
    Inlet { velocity: Vec3, temperature: f64 },
    /// Outlet (zero gradient)
    Outlet,
    /// Symmetry
    Symmetry,
    /// Periodic
    Periodic,
    /// Heat flux (W/m²)
    HeatFlux(f64),
    /// Convection h(T - T_inf)
    Convection { h: f64, t_inf: f64 },
}

/// Boundary face
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryFace {
    XMin, XMax,
    YMin, YMax,
    ZMin, ZMax,
}

/// Boundary condition set
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    pub velocity: std::collections::HashMap<BoundaryFace, BoundaryCondition>,
    pub pressure: std::collections::HashMap<BoundaryFace, BoundaryCondition>,
    pub temperature: std::collections::HashMap<BoundaryFace, BoundaryCondition>,
}

impl BoundaryConditions {
    pub fn new() -> Self {
        Self {
            velocity: std::collections::HashMap::new(),
            pressure: std::collections::HashMap::new(),
            temperature: std::collections::HashMap::new(),
        }
    }

    /// Pipe flow setup (inlet-outlet)
    pub fn pipe_flow(u_inlet: f64, t_inlet: f64, t_wall: f64) -> Self {
        let mut bc = Self::new();

        // X direction is flow direction
        bc.velocity.insert(BoundaryFace::XMin, BoundaryCondition::Inlet {
            velocity: Vec3::new(u_inlet, 0.0, 0.0),
            temperature: t_inlet,
        });
        bc.velocity.insert(BoundaryFace::XMax, BoundaryCondition::Outlet);

        // Walls in Y and Z
        bc.velocity.insert(BoundaryFace::YMin, BoundaryCondition::Wall);
        bc.velocity.insert(BoundaryFace::YMax, BoundaryCondition::Wall);
        bc.velocity.insert(BoundaryFace::ZMin, BoundaryCondition::Wall);
        bc.velocity.insert(BoundaryFace::ZMax, BoundaryCondition::Wall);

        // Pressure
        bc.pressure.insert(BoundaryFace::XMin, BoundaryCondition::Neumann(0.0));
        bc.pressure.insert(BoundaryFace::XMax, BoundaryCondition::Dirichlet(0.0));

        // Temperature
        bc.temperature.insert(BoundaryFace::XMin, BoundaryCondition::Dirichlet(t_inlet));
        bc.temperature.insert(BoundaryFace::XMax, BoundaryCondition::Neumann(0.0));
        bc.temperature.insert(BoundaryFace::YMin, BoundaryCondition::Dirichlet(t_wall));
        bc.temperature.insert(BoundaryFace::YMax, BoundaryCondition::Dirichlet(t_wall));
        bc.temperature.insert(BoundaryFace::ZMin, BoundaryCondition::Dirichlet(t_wall));
        bc.temperature.insert(BoundaryFace::ZMax, BoundaryCondition::Dirichlet(t_wall));

        bc
    }
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TURBULENCE MODELS
// ============================================================================

/// k-ε turbulence model coefficients
#[derive(Debug, Clone)]
pub struct KepsilonModel {
    /// C_μ coefficient
    pub c_mu: f64,
    /// C_ε1 coefficient
    pub c_eps1: f64,
    /// C_ε2 coefficient
    pub c_eps2: f64,
    /// σ_k (turbulent Prandtl number for k)
    pub sigma_k: f64,
    /// σ_ε (turbulent Prandtl number for ε)
    pub sigma_eps: f64,
}

impl Default for KepsilonModel {
    fn default() -> Self {
        // Standard k-ε coefficients (Launder & Spalding)
        Self {
            c_mu: 0.09,
            c_eps1: 1.44,
            c_eps2: 1.92,
            sigma_k: 1.0,
            sigma_eps: 1.3,
        }
    }
}

impl KepsilonModel {
    /// Turbulent viscosity ν_t = C_μ · k² / ε
    pub fn turbulent_viscosity(&self, k: f64, epsilon: f64) -> f64 {
        if epsilon > 1e-30 {
            self.c_mu * k * k / epsilon
        } else {
            0.0
        }
    }

    /// Production term P_k = ν_t · |∇u|²
    pub fn production(&self, nu_t: f64, strain_rate_sq: f64) -> f64 {
        nu_t * strain_rate_sq
    }
}

/// Turbulence field variables
#[derive(Debug, Clone)]
pub struct TurbulenceFields {
    /// Turbulent kinetic energy k (m²/s²)
    pub k: ScalarField,
    /// Dissipation rate ε (m²/s³)
    pub epsilon: ScalarField,
    /// Turbulent viscosity ν_t (m²/s)
    pub nu_t: ScalarField,
}

impl TurbulenceFields {
    pub fn new(n_cells: usize) -> Self {
        Self {
            k: ScalarField::new("k", n_cells, 1e-4),
            epsilon: ScalarField::new("epsilon", n_cells, 1e-5),
            nu_t: ScalarField::zeros("nu_t", n_cells),
        }
    }
}

// ============================================================================
// MHD EFFECTS
// ============================================================================

/// MHD (magnetohydrodynamic) effects for liquid metal flows
#[derive(Debug, Clone)]
pub struct MHDModel {
    /// Applied magnetic field (T)
    pub b_field: Vec3,
    /// Electrical conductivity (S/m)
    pub sigma: f64,
}

impl MHDModel {
    pub fn new(b_field: Vec3, sigma: f64) -> Self {
        Self { b_field, sigma }
    }

    /// Lorentz force: F = J × B = σ(u × B) × B
    /// For incompressible flow with uniform B:
    /// F = -σ·B²·u_perp (damping of perpendicular velocity)
    pub fn lorentz_force(&self, velocity: &Vec3) -> Vec3 {
        // J = σ(E + u × B), assuming E = 0 (no applied electric field)
        // F = J × B = σ(u × B) × B

        // u × B
        let uxb = Vec3::new(
            velocity.y * self.b_field.z - velocity.z * self.b_field.y,
            velocity.z * self.b_field.x - velocity.x * self.b_field.z,
            velocity.x * self.b_field.y - velocity.y * self.b_field.x,
        );

        // (u × B) × B
        let force = Vec3::new(
            uxb.y * self.b_field.z - uxb.z * self.b_field.y,
            uxb.z * self.b_field.x - uxb.x * self.b_field.z,
            uxb.x * self.b_field.y - uxb.y * self.b_field.x,
        );

        Vec3::new(
            self.sigma * force.x,
            self.sigma * force.y,
            self.sigma * force.z,
        )
    }

    /// Hartmann layer thickness δ_H = L/Ha
    pub fn hartmann_layer_thickness(&self, l: f64, rho: f64, nu: f64) -> f64 {
        let b_mag = (self.b_field.x.powi(2) + self.b_field.y.powi(2) + self.b_field.z.powi(2)).sqrt();
        if b_mag < 1e-10 { return l; }

        let ha = b_mag * l * (self.sigma / (rho * nu)).sqrt();
        l / ha
    }

    /// MHD pressure drop in channel flow
    /// Δp = σ·u·B²·L (for insulating walls)
    pub fn pressure_drop(&self, u: f64, length: f64) -> f64 {
        let b_perp_sq = self.b_field.y.powi(2) + self.b_field.z.powi(2);
        self.sigma * u * b_perp_sq * length
    }
}

// ============================================================================
// SIMPLE ALGORITHM SOLVER
// ============================================================================

/// SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) solver
pub struct SimpleSolver {
    pub mesh: StructuredMesh,
    pub fluid: FluidProperties,
    pub bc: BoundaryConditions,
    /// Velocity field
    pub velocity: VectorField,
    /// Pressure field
    pub pressure: ScalarField,
    /// Temperature field
    pub temperature: ScalarField,
    /// Turbulence model
    pub turbulence: Option<(KepsilonModel, TurbulenceFields)>,
    /// MHD model
    pub mhd: Option<MHDModel>,
    /// Heat source (W/m³)
    pub heat_source: ScalarField,
    /// Under-relaxation factor for velocity
    pub alpha_u: f64,
    /// Under-relaxation factor for pressure
    pub alpha_p: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Time step (0 for steady-state)
    pub dt: f64,
    /// Current simulation time
    pub time: f64,
}

impl SimpleSolver {
    pub fn new(mesh: StructuredMesh, fluid: FluidProperties, bc: BoundaryConditions) -> Self {
        let n_cells = mesh.n_cells();

        Self {
            mesh,
            fluid,
            bc,
            velocity: VectorField::new("U", n_cells),
            pressure: ScalarField::zeros("p", n_cells),
            temperature: ScalarField::new("T", n_cells, 300.0),
            turbulence: None,
            mhd: None,
            heat_source: ScalarField::zeros("Q", n_cells),
            alpha_u: 0.7,
            alpha_p: 0.3,
            tolerance: 1e-6,
            max_iterations: 1000,
            dt: 0.0,
            time: 0.0,
        }
    }

    /// Enable k-ε turbulence model
    pub fn enable_turbulence(&mut self) {
        let n_cells = self.mesh.n_cells();
        self.turbulence = Some((KepsilonModel::default(), TurbulenceFields::new(n_cells)));
    }

    /// Enable MHD effects
    pub fn enable_mhd(&mut self, b_field: Vec3) {
        self.mhd = Some(MHDModel::new(b_field, self.fluid.sigma_e));
    }

    /// Set volumetric heat source
    pub fn set_heat_source(&mut self, source_fn: impl Fn(Vec3) -> f64) {
        for k in 0..self.mesh.nz {
            for j in 0..self.mesh.ny {
                for i in 0..self.mesh.nx {
                    let idx = self.mesh.cell_index(i, j, k);
                    let pos = self.mesh.cell_center(i, j, k);
                    self.heat_source.values[idx] = source_fn(pos);
                }
            }
        }
    }

    /// Initialize with uniform flow
    pub fn initialize_uniform(&mut self, u: Vec3, t: f64) {
        for idx in 0..self.mesh.n_cells() {
            self.velocity.set(idx, u);
            self.temperature.values[idx] = t;
        }
    }

    /// Solve steady-state (SIMPLE algorithm)
    pub fn solve_steady(&mut self) -> SolverResult {
        let mut residuals = Vec::new();

        for iter in 0..self.max_iterations {
            // 1. Solve momentum equations (predict velocity)
            let u_residual = self.solve_momentum();

            // 2. Solve pressure correction equation
            let p_residual = self.solve_pressure_correction();

            // 3. Correct velocity and pressure
            self.correct_velocity_pressure();

            // 4. Update turbulence (if enabled)
            if self.turbulence.is_some() {
                self.solve_turbulence();
            }

            // 5. Solve energy equation
            let t_residual = self.solve_energy();

            let total_residual = (u_residual.powi(2) + p_residual.powi(2) + t_residual.powi(2)).sqrt();
            residuals.push(total_residual);

            if total_residual < self.tolerance {
                return SolverResult {
                    converged: true,
                    iterations: iter + 1,
                    final_residual: total_residual,
                    residual_history: residuals,
                };
            }
        }

        SolverResult {
            converged: false,
            iterations: self.max_iterations,
            final_residual: *residuals.last().unwrap_or(&1.0),
            residual_history: residuals,
        }
    }

    /// Solve transient (time-stepping)
    pub fn solve_transient(&mut self, end_time: f64) -> TransientResult {
        if self.dt <= 0.0 {
            self.dt = self.estimate_time_step();
        }

        let mut time_history = Vec::new();
        let mut max_velocity_history = Vec::new();
        let mut max_temperature_history = Vec::new();

        while self.time < end_time {
            // Inner iterations for each time step
            for _ in 0..10 {
                self.solve_momentum();
                self.solve_pressure_correction();
                self.correct_velocity_pressure();
                self.solve_energy();
            }

            self.time += self.dt;

            // Record history
            time_history.push(self.time);
            max_velocity_history.push(self.max_velocity());
            max_temperature_history.push(self.max_temperature());
        }

        TransientResult {
            final_time: self.time,
            time_steps: time_history.len(),
            time_history,
            max_velocity_history,
            max_temperature_history,
        }
    }

    fn estimate_time_step(&self) -> f64 {
        // CFL-based time step
        let u_max = self.max_velocity().max(1e-6);
        let dx_min = self.mesh.dx.min(self.mesh.dy).min(self.mesh.dz);
        0.5 * dx_min / u_max
    }

    fn solve_momentum(&mut self) -> f64 {
        let n_cells = self.mesh.n_cells();
        let mut u_new = vec![0.0; n_cells];
        let mut v_new = vec![0.0; n_cells];
        let mut w_new = vec![0.0; n_cells];

        let t_avg = self.temperature.values.iter().sum::<f64>() / n_cells as f64;
        let rho = self.fluid.density(t_avg);
        let mu = self.fluid.viscosity(t_avg);

        // Get turbulent viscosity if enabled
        let nu_t: Vec<f64> = if let Some((_, ref turb_fields)) = self.turbulence {
            turb_fields.nu_t.values.clone()
        } else {
            vec![0.0; n_cells]
        };

        for k in 1..self.mesh.nz - 1 {
            for j in 1..self.mesh.ny - 1 {
                for i in 1..self.mesh.nx - 1 {
                    let idx = self.mesh.cell_index(i, j, k);

                    // Effective viscosity (laminar + turbulent)
                    let mu_eff = mu + rho * nu_t[idx];

                    // Neighbors
                    let idx_e = self.mesh.cell_index(i + 1, j, k);
                    let idx_w = self.mesh.cell_index(i - 1, j, k);
                    let idx_n = self.mesh.cell_index(i, j + 1, k);
                    let idx_s = self.mesh.cell_index(i, j - 1, k);
                    let idx_t = self.mesh.cell_index(i, j, k + 1);
                    let idx_b = self.mesh.cell_index(i, j, k - 1);

                    // Diffusion coefficients
                    let d_e = mu_eff * self.mesh.area_x / self.mesh.dx;
                    let d_w = mu_eff * self.mesh.area_x / self.mesh.dx;
                    let d_n = mu_eff * self.mesh.area_y / self.mesh.dy;
                    let d_s = mu_eff * self.mesh.area_y / self.mesh.dy;
                    let d_t = mu_eff * self.mesh.area_z / self.mesh.dz;
                    let d_b = mu_eff * self.mesh.area_z / self.mesh.dz;

                    // Convection (upwind)
                    let f_e = rho * self.velocity.x[idx] * self.mesh.area_x;
                    let f_w = rho * self.velocity.x[idx_w] * self.mesh.area_x;
                    let f_n = rho * self.velocity.y[idx] * self.mesh.area_y;
                    let f_s = rho * self.velocity.y[idx_s] * self.mesh.area_y;
                    let f_t = rho * self.velocity.z[idx] * self.mesh.area_z;
                    let f_b = rho * self.velocity.z[idx_b] * self.mesh.area_z;

                    // Coefficients (hybrid scheme)
                    let a_e = (d_e + (-f_e).max(0.0)).max(0.0);
                    let a_w = (d_w + f_w.max(0.0)).max(0.0);
                    let a_n = (d_n + (-f_n).max(0.0)).max(0.0);
                    let a_s = (d_s + f_s.max(0.0)).max(0.0);
                    let a_t = (d_t + (-f_t).max(0.0)).max(0.0);
                    let a_b = (d_b + f_b.max(0.0)).max(0.0);

                    let a_p = a_e + a_w + a_n + a_s + a_t + a_b + (f_e - f_w + f_n - f_s + f_t - f_b);

                    // Pressure gradient source
                    let sp_x = -(self.pressure.values[idx_e] - self.pressure.values[idx_w]) /
                               (2.0 * self.mesh.dx) * self.mesh.volumes[idx];
                    let sp_y = -(self.pressure.values[idx_n] - self.pressure.values[idx_s]) /
                               (2.0 * self.mesh.dy) * self.mesh.volumes[idx];
                    let sp_z = -(self.pressure.values[idx_t] - self.pressure.values[idx_b]) /
                               (2.0 * self.mesh.dz) * self.mesh.volumes[idx];

                    // MHD Lorentz force
                    let (mhd_x, mhd_y, mhd_z) = if let Some(ref mhd) = self.mhd {
                        let vel = self.velocity.get(idx);
                        let f_lorentz = mhd.lorentz_force(&vel);
                        (f_lorentz.x * self.mesh.volumes[idx],
                         f_lorentz.y * self.mesh.volumes[idx],
                         f_lorentz.z * self.mesh.volumes[idx])
                    } else {
                        (0.0, 0.0, 0.0)
                    };

                    // Solve for new velocities
                    if a_p.abs() > 1e-30 {
                        u_new[idx] = (a_e * self.velocity.x[idx_e] +
                                     a_w * self.velocity.x[idx_w] +
                                     a_n * self.velocity.x[idx_n] +
                                     a_s * self.velocity.x[idx_s] +
                                     a_t * self.velocity.x[idx_t] +
                                     a_b * self.velocity.x[idx_b] +
                                     sp_x + mhd_x) / a_p;

                        v_new[idx] = (a_e * self.velocity.y[idx_e] +
                                     a_w * self.velocity.y[idx_w] +
                                     a_n * self.velocity.y[idx_n] +
                                     a_s * self.velocity.y[idx_s] +
                                     a_t * self.velocity.y[idx_t] +
                                     a_b * self.velocity.y[idx_b] +
                                     sp_y + mhd_y) / a_p;

                        w_new[idx] = (a_e * self.velocity.z[idx_e] +
                                     a_w * self.velocity.z[idx_w] +
                                     a_n * self.velocity.z[idx_n] +
                                     a_s * self.velocity.z[idx_s] +
                                     a_t * self.velocity.z[idx_t] +
                                     a_b * self.velocity.z[idx_b] +
                                     sp_z + mhd_z) / a_p;
                    }
                }
            }
        }

        // Under-relaxation and residual calculation
        let mut residual = 0.0;
        for idx in 0..n_cells {
            let du = u_new[idx] - self.velocity.x[idx];
            let dv = v_new[idx] - self.velocity.y[idx];
            let dw = w_new[idx] - self.velocity.z[idx];

            residual += du * du + dv * dv + dw * dw;

            self.velocity.x[idx] += self.alpha_u * du;
            self.velocity.y[idx] += self.alpha_u * dv;
            self.velocity.z[idx] += self.alpha_u * dw;
        }

        // Apply boundary conditions
        self.apply_velocity_bc();

        (residual / n_cells as f64).sqrt()
    }

    fn solve_pressure_correction(&mut self) -> f64 {
        // Simplified pressure correction
        // Real implementation would solve Poisson equation for p'

        let n_cells = self.mesh.n_cells();
        let mut p_correction = vec![0.0; n_cells];

        let t_avg = self.temperature.values.iter().sum::<f64>() / n_cells as f64;
        let rho = self.fluid.density(t_avg);

        // Gauss-Seidel iterations for pressure Poisson equation
        for _ in 0..50 {
            for k in 1..self.mesh.nz - 1 {
                for j in 1..self.mesh.ny - 1 {
                    for i in 1..self.mesh.nx - 1 {
                        let idx = self.mesh.cell_index(i, j, k);

                        let idx_e = self.mesh.cell_index(i + 1, j, k);
                        let idx_w = self.mesh.cell_index(i - 1, j, k);
                        let idx_n = self.mesh.cell_index(i, j + 1, k);
                        let idx_s = self.mesh.cell_index(i, j - 1, k);
                        let idx_t = self.mesh.cell_index(i, j, k + 1);
                        let idx_b = self.mesh.cell_index(i, j, k - 1);

                        // Mass imbalance (source for pressure correction)
                        let mass_imbalance = rho * (
                            (self.velocity.x[idx_e] - self.velocity.x[idx_w]) / (2.0 * self.mesh.dx) +
                            (self.velocity.y[idx_n] - self.velocity.y[idx_s]) / (2.0 * self.mesh.dy) +
                            (self.velocity.z[idx_t] - self.velocity.z[idx_b]) / (2.0 * self.mesh.dz)
                        );

                        // Laplacian coefficients
                        let a = 2.0 * (1.0 / self.mesh.dx.powi(2) +
                                      1.0 / self.mesh.dy.powi(2) +
                                      1.0 / self.mesh.dz.powi(2));

                        if a.abs() > 1e-30 {
                            p_correction[idx] = (
                                (p_correction[idx_e] + p_correction[idx_w]) / self.mesh.dx.powi(2) +
                                (p_correction[idx_n] + p_correction[idx_s]) / self.mesh.dy.powi(2) +
                                (p_correction[idx_t] + p_correction[idx_b]) / self.mesh.dz.powi(2) -
                                mass_imbalance
                            ) / a;
                        }
                    }
                }
            }
        }

        // Apply correction
        let mut residual = 0.0;
        for idx in 0..n_cells {
            let dp = self.alpha_p * p_correction[idx];
            residual += dp * dp;
            self.pressure.values[idx] += dp;
        }

        (residual / n_cells as f64).sqrt()
    }

    fn correct_velocity_pressure(&mut self) {
        // Velocity correction from pressure gradient
        // u' = -dt/rho * ∂p'/∂x (simplified)

        for k in 1..self.mesh.nz - 1 {
            for j in 1..self.mesh.ny - 1 {
                for i in 1..self.mesh.nx - 1 {
                    let idx = self.mesh.cell_index(i, j, k);
                    let idx_e = self.mesh.cell_index(i + 1, j, k);
                    let idx_w = self.mesh.cell_index(i - 1, j, k);
                    let idx_n = self.mesh.cell_index(i, j + 1, k);
                    let idx_s = self.mesh.cell_index(i, j - 1, k);
                    let idx_t = self.mesh.cell_index(i, j, k + 1);
                    let idx_b = self.mesh.cell_index(i, j, k - 1);

                    let dp_dx = (self.pressure.values[idx_e] - self.pressure.values[idx_w]) /
                               (2.0 * self.mesh.dx);
                    let dp_dy = (self.pressure.values[idx_n] - self.pressure.values[idx_s]) /
                               (2.0 * self.mesh.dy);
                    let dp_dz = (self.pressure.values[idx_t] - self.pressure.values[idx_b]) /
                               (2.0 * self.mesh.dz);

                    // Correction factor (simplified)
                    let factor = 0.1 * self.mesh.dx;

                    self.velocity.x[idx] -= factor * dp_dx;
                    self.velocity.y[idx] -= factor * dp_dy;
                    self.velocity.z[idx] -= factor * dp_dz;
                }
            }
        }
    }

    fn solve_turbulence(&mut self) {
        if let Some((ref model, ref mut fields)) = self.turbulence {
            let n_cells = self.mesh.n_cells();

            // Update turbulent viscosity
            for idx in 0..n_cells {
                fields.nu_t.values[idx] = model.turbulent_viscosity(
                    fields.k.values[idx],
                    fields.epsilon.values[idx]
                );
            }

            // Simplified k-ε update (production-dissipation balance)
            for k in 1..self.mesh.nz - 1 {
                for j in 1..self.mesh.ny - 1 {
                    for i in 1..self.mesh.nx - 1 {
                        let idx = self.mesh.cell_index(i, j, k);

                        // Strain rate squared (simplified)
                        let idx_e = self.mesh.cell_index(i + 1, j, k);
                        let idx_w = self.mesh.cell_index(i - 1, j, k);
                        let du_dx = (self.velocity.x[idx_e] - self.velocity.x[idx_w]) /
                                   (2.0 * self.mesh.dx);

                        let s2 = 2.0 * du_dx * du_dx;  // Simplified

                        // Production
                        let p_k = model.production(fields.nu_t.values[idx], s2);

                        // Update k (simplified explicit)
                        let dk = 0.1 * (p_k - fields.epsilon.values[idx]);
                        fields.k.values[idx] = (fields.k.values[idx] + dk).max(1e-10);

                        // Update ε
                        let k_val = fields.k.values[idx];
                        let eps_val = fields.epsilon.values[idx];
                        if k_val > 1e-30 {
                            let d_eps = 0.1 * eps_val / k_val *
                                       (model.c_eps1 * p_k - model.c_eps2 * eps_val);
                            fields.epsilon.values[idx] = (eps_val + d_eps).max(1e-10);
                        }
                    }
                }
            }
        }
    }

    fn solve_energy(&mut self) -> f64 {
        let n_cells = self.mesh.n_cells();
        let mut t_new = vec![0.0; n_cells];

        let t_avg = self.temperature.values.iter().sum::<f64>() / n_cells as f64;
        let rho = self.fluid.density(t_avg);
        let alpha = self.fluid.thermal_diffusivity(t_avg);
        let cp = self.fluid.cp_ref;

        // Turbulent thermal diffusivity
        let alpha_t: Vec<f64> = if let Some((_, ref turb_fields)) = self.turbulence {
            turb_fields.nu_t.values.iter().map(|&nu_t| nu_t / 0.9).collect()  // Pr_t ≈ 0.9
        } else {
            vec![0.0; n_cells]
        };

        for k in 1..self.mesh.nz - 1 {
            for j in 1..self.mesh.ny - 1 {
                for i in 1..self.mesh.nx - 1 {
                    let idx = self.mesh.cell_index(i, j, k);

                    let alpha_eff = alpha + alpha_t[idx];

                    let idx_e = self.mesh.cell_index(i + 1, j, k);
                    let idx_w = self.mesh.cell_index(i - 1, j, k);
                    let idx_n = self.mesh.cell_index(i, j + 1, k);
                    let idx_s = self.mesh.cell_index(i, j - 1, k);
                    let idx_t = self.mesh.cell_index(i, j, k + 1);
                    let idx_b = self.mesh.cell_index(i, j, k - 1);

                    // Diffusion
                    let diff_x = alpha_eff * (self.temperature.values[idx_e] -
                                             2.0 * self.temperature.values[idx] +
                                             self.temperature.values[idx_w]) / self.mesh.dx.powi(2);
                    let diff_y = alpha_eff * (self.temperature.values[idx_n] -
                                             2.0 * self.temperature.values[idx] +
                                             self.temperature.values[idx_s]) / self.mesh.dy.powi(2);
                    let diff_z = alpha_eff * (self.temperature.values[idx_t] -
                                             2.0 * self.temperature.values[idx] +
                                             self.temperature.values[idx_b]) / self.mesh.dz.powi(2);

                    // Convection (upwind)
                    let conv_x = -self.velocity.x[idx] *
                                (self.temperature.values[idx] - self.temperature.values[idx_w]) / self.mesh.dx;
                    let conv_y = -self.velocity.y[idx] *
                                (self.temperature.values[idx] - self.temperature.values[idx_s]) / self.mesh.dy;
                    let conv_z = -self.velocity.z[idx] *
                                (self.temperature.values[idx] - self.temperature.values[idx_b]) / self.mesh.dz;

                    // Source term
                    let source = self.heat_source.values[idx] / (rho * cp);

                    t_new[idx] = self.temperature.values[idx] +
                                0.5 * (diff_x + diff_y + diff_z + conv_x + conv_y + conv_z + source);
                }
            }
        }

        // Residual and update
        let mut residual = 0.0;
        for idx in 0..n_cells {
            let dt_val = t_new[idx] - self.temperature.values[idx];
            residual += dt_val * dt_val;
            self.temperature.values[idx] = t_new[idx];
        }

        // Apply boundary conditions
        self.apply_temperature_bc();

        (residual / n_cells as f64).sqrt()
    }

    fn apply_velocity_bc(&mut self) {
        // Apply boundary conditions for velocity

        // X boundaries
        for k in 0..self.mesh.nz {
            for j in 0..self.mesh.ny {
                let idx_min = self.mesh.cell_index(0, j, k);
                let idx_max = self.mesh.cell_index(self.mesh.nx - 1, j, k);

                if let Some(bc) = self.bc.velocity.get(&BoundaryFace::XMin) {
                    match bc {
                        BoundaryCondition::Wall => {
                            self.velocity.x[idx_min] = 0.0;
                            self.velocity.y[idx_min] = 0.0;
                            self.velocity.z[idx_min] = 0.0;
                        }
                        BoundaryCondition::Inlet { velocity, .. } => {
                            self.velocity.set(idx_min, *velocity);
                        }
                        _ => {}
                    }
                }

                if let Some(bc) = self.bc.velocity.get(&BoundaryFace::XMax) {
                    match bc {
                        BoundaryCondition::Wall => {
                            self.velocity.x[idx_max] = 0.0;
                            self.velocity.y[idx_max] = 0.0;
                            self.velocity.z[idx_max] = 0.0;
                        }
                        BoundaryCondition::Outlet => {
                            let idx_prev = self.mesh.cell_index(self.mesh.nx - 2, j, k);
                            self.velocity.x[idx_max] = self.velocity.x[idx_prev];
                            self.velocity.y[idx_max] = self.velocity.y[idx_prev];
                            self.velocity.z[idx_max] = self.velocity.z[idx_prev];
                        }
                        _ => {}
                    }
                }
            }
        }

        // Y boundaries (walls)
        for k in 0..self.mesh.nz {
            for i in 0..self.mesh.nx {
                let idx_min = self.mesh.cell_index(i, 0, k);
                let idx_max = self.mesh.cell_index(i, self.mesh.ny - 1, k);

                if matches!(self.bc.velocity.get(&BoundaryFace::YMin), Some(BoundaryCondition::Wall)) {
                    self.velocity.x[idx_min] = 0.0;
                    self.velocity.y[idx_min] = 0.0;
                    self.velocity.z[idx_min] = 0.0;
                }
                if matches!(self.bc.velocity.get(&BoundaryFace::YMax), Some(BoundaryCondition::Wall)) {
                    self.velocity.x[idx_max] = 0.0;
                    self.velocity.y[idx_max] = 0.0;
                    self.velocity.z[idx_max] = 0.0;
                }
            }
        }

        // Z boundaries (walls)
        for j in 0..self.mesh.ny {
            for i in 0..self.mesh.nx {
                let idx_min = self.mesh.cell_index(i, j, 0);
                let idx_max = self.mesh.cell_index(i, j, self.mesh.nz - 1);

                if matches!(self.bc.velocity.get(&BoundaryFace::ZMin), Some(BoundaryCondition::Wall)) {
                    self.velocity.x[idx_min] = 0.0;
                    self.velocity.y[idx_min] = 0.0;
                    self.velocity.z[idx_min] = 0.0;
                }
                if matches!(self.bc.velocity.get(&BoundaryFace::ZMax), Some(BoundaryCondition::Wall)) {
                    self.velocity.x[idx_max] = 0.0;
                    self.velocity.y[idx_max] = 0.0;
                    self.velocity.z[idx_max] = 0.0;
                }
            }
        }
    }

    fn apply_temperature_bc(&mut self) {
        // X boundaries
        for k in 0..self.mesh.nz {
            for j in 0..self.mesh.ny {
                let idx_min = self.mesh.cell_index(0, j, k);
                let idx_max = self.mesh.cell_index(self.mesh.nx - 1, j, k);

                if let Some(BoundaryCondition::Dirichlet(t)) = self.bc.temperature.get(&BoundaryFace::XMin) {
                    self.temperature.values[idx_min] = *t;
                }
                if let Some(BoundaryCondition::Neumann(_)) = self.bc.temperature.get(&BoundaryFace::XMax) {
                    let idx_prev = self.mesh.cell_index(self.mesh.nx - 2, j, k);
                    self.temperature.values[idx_max] = self.temperature.values[idx_prev];
                }
            }
        }

        // Y boundaries
        for k in 0..self.mesh.nz {
            for i in 0..self.mesh.nx {
                let idx_min = self.mesh.cell_index(i, 0, k);
                let idx_max = self.mesh.cell_index(i, self.mesh.ny - 1, k);

                if let Some(BoundaryCondition::Dirichlet(t)) = self.bc.temperature.get(&BoundaryFace::YMin) {
                    self.temperature.values[idx_min] = *t;
                }
                if let Some(BoundaryCondition::Dirichlet(t)) = self.bc.temperature.get(&BoundaryFace::YMax) {
                    self.temperature.values[idx_max] = *t;
                }
            }
        }

        // Z boundaries
        for j in 0..self.mesh.ny {
            for i in 0..self.mesh.nx {
                let idx_min = self.mesh.cell_index(i, j, 0);
                let idx_max = self.mesh.cell_index(i, j, self.mesh.nz - 1);

                if let Some(BoundaryCondition::Dirichlet(t)) = self.bc.temperature.get(&BoundaryFace::ZMin) {
                    self.temperature.values[idx_min] = *t;
                }
                if let Some(BoundaryCondition::Dirichlet(t)) = self.bc.temperature.get(&BoundaryFace::ZMax) {
                    self.temperature.values[idx_max] = *t;
                }
            }
        }
    }

    pub fn max_velocity(&self) -> f64 {
        (0..self.mesh.n_cells())
            .map(|idx| self.velocity.magnitude(idx))
            .fold(0.0, f64::max)
    }

    pub fn max_temperature(&self) -> f64 {
        self.temperature.values.iter().cloned().fold(f64::MIN, f64::max)
    }

    pub fn min_temperature(&self) -> f64 {
        self.temperature.values.iter().cloned().fold(f64::MAX, f64::min)
    }

    pub fn average_temperature(&self) -> f64 {
        self.temperature.values.iter().sum::<f64>() / self.mesh.n_cells() as f64
    }

    /// Calculate heat transfer coefficient
    pub fn heat_transfer_coefficient(&self, wall_face: BoundaryFace) -> f64 {
        let t_wall = match self.bc.temperature.get(&wall_face) {
            Some(BoundaryCondition::Dirichlet(t)) => *t,
            _ => return 0.0,
        };

        let t_bulk = self.average_temperature();
        if (t_wall - t_bulk).abs() < 1e-10 {
            return 0.0;
        }

        // Estimate wall heat flux from temperature gradient
        let q_wall = match wall_face {
            BoundaryFace::YMin => {
                let mut q_sum = 0.0;
                let mut count = 0;
                for k in 1..self.mesh.nz - 1 {
                    for i in 1..self.mesh.nx - 1 {
                        let idx_wall = self.mesh.cell_index(i, 0, k);
                        let idx_fluid = self.mesh.cell_index(i, 1, k);
                        let dt_dy = (self.temperature.values[idx_fluid] -
                                    self.temperature.values[idx_wall]) / self.mesh.dy;
                        q_sum += self.fluid.k_ref * dt_dy;
                        count += 1;
                    }
                }
                if count > 0 { q_sum / count as f64 } else { 0.0 }
            }
            _ => 0.0,  // Implement for other faces as needed
        };

        q_wall / (t_wall - t_bulk)
    }

    /// Summary output
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== CFD Solution Summary ===\n");
        s.push_str(&format!("Mesh: {}×{}×{} = {} cells\n",
                           self.mesh.nx, self.mesh.ny, self.mesh.nz, self.mesh.n_cells()));
        s.push_str(&format!("Coolant: {:?}\n", self.fluid.coolant));
        s.push_str(&format!("Max velocity: {:.3} m/s\n", self.max_velocity()));
        s.push_str(&format!("Temperature range: {:.1} - {:.1} K\n",
                           self.min_temperature(), self.max_temperature()));
        s.push_str(&format!("Average temperature: {:.1} K\n", self.average_temperature()));

        if self.mhd.is_some() {
            let re = self.fluid.reynolds_number(self.max_velocity(), self.mesh.dy, self.average_temperature());
            let ha = self.fluid.hartmann_number(5.0, self.mesh.dy, self.average_temperature());  // Assume 5T
            s.push_str(&format!("Reynolds number: {:.0}\n", re));
            s.push_str(&format!("Hartmann number: {:.0}\n", ha));
        }

        s
    }
}

/// Steady-state solver result
#[derive(Debug, Clone)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub residual_history: Vec<f64>,
}

/// Transient solver result
#[derive(Debug, Clone)]
pub struct TransientResult {
    pub final_time: f64,
    pub time_steps: usize,
    pub time_history: Vec<f64>,
    pub max_velocity_history: Vec<f64>,
    pub max_temperature_history: Vec<f64>,
}

// ============================================================================
// CONJUGATE HEAT TRANSFER
// ============================================================================

/// Solid region for conjugate heat transfer
#[derive(Debug, Clone)]
pub struct SolidRegion {
    pub name: String,
    pub mesh: StructuredMesh,
    /// Thermal conductivity (W/m·K)
    pub k: f64,
    /// Density (kg/m³)
    pub rho: f64,
    /// Specific heat (J/kg·K)
    pub cp: f64,
    /// Temperature field
    pub temperature: ScalarField,
    /// Heat source (W/m³)
    pub heat_source: ScalarField,
}

impl SolidRegion {
    pub fn new(name: &str, mesh: StructuredMesh, k: f64, rho: f64, cp: f64) -> Self {
        let n_cells = mesh.n_cells();
        Self {
            name: name.into(),
            mesh,
            k, rho, cp,
            temperature: ScalarField::new("T_solid", n_cells, 300.0),
            heat_source: ScalarField::zeros("Q_solid", n_cells),
        }
    }

    /// First wall with neutron heating profile
    pub fn first_wall_tungsten(mesh: StructuredMesh) -> Self {
        let mut region = Self::new("First Wall (W)", mesh, 173.0, 19300.0, 134.0);

        // Exponential heating decay from plasma side
        for k in 0..region.mesh.nz {
            for j in 0..region.mesh.ny {
                for i in 0..region.mesh.nx {
                    let idx = region.mesh.cell_index(i, j, k);
                    let y = region.mesh.cell_center(i, j, k).y;
                    let y_norm = (y - region.mesh.y_min) / (region.mesh.y_max - region.mesh.y_min);

                    // Surface heat flux ~10 MW/m² decaying into material
                    let q_surface = 10.0e6;  // W/m²
                    let decay_length = 0.005;  // 5 mm
                    region.heat_source.values[idx] = q_surface / decay_length *
                        (-y_norm * (region.mesh.y_max - region.mesh.y_min) / decay_length).exp();
                }
            }
        }

        region
    }

    /// Thermal diffusivity
    pub fn thermal_diffusivity(&self) -> f64 {
        self.k / (self.rho * self.cp)
    }

    /// Solve transient heat conduction
    pub fn solve_conduction(&mut self, dt: f64, n_steps: usize) {
        let alpha = self.thermal_diffusivity();
        let _n_cells = self.mesh.n_cells();

        for _ in 0..n_steps {
            let mut t_new = self.temperature.values.clone();

            for k in 1..self.mesh.nz - 1 {
                for j in 1..self.mesh.ny - 1 {
                    for i in 1..self.mesh.nx - 1 {
                        let idx = self.mesh.cell_index(i, j, k);

                        let idx_e = self.mesh.cell_index(i + 1, j, k);
                        let idx_w = self.mesh.cell_index(i - 1, j, k);
                        let idx_n = self.mesh.cell_index(i, j + 1, k);
                        let idx_s = self.mesh.cell_index(i, j - 1, k);
                        let idx_t = self.mesh.cell_index(i, j, k + 1);
                        let idx_b = self.mesh.cell_index(i, j, k - 1);

                        let laplacian =
                            (self.temperature.values[idx_e] - 2.0 * self.temperature.values[idx] +
                             self.temperature.values[idx_w]) / self.mesh.dx.powi(2) +
                            (self.temperature.values[idx_n] - 2.0 * self.temperature.values[idx] +
                             self.temperature.values[idx_s]) / self.mesh.dy.powi(2) +
                            (self.temperature.values[idx_t] - 2.0 * self.temperature.values[idx] +
                             self.temperature.values[idx_b]) / self.mesh.dz.powi(2);

                        let source = self.heat_source.values[idx] / (self.rho * self.cp);

                        t_new[idx] = self.temperature.values[idx] + dt * (alpha * laplacian + source);
                    }
                }
            }

            self.temperature.values = t_new;
        }
    }

    pub fn max_temperature(&self) -> f64 {
        self.temperature.values.iter().cloned().fold(f64::MIN, f64::max)
    }
}

// ============================================================================
// PIPE FLOW CORRELATIONS
// ============================================================================

/// Engineering correlations for pipe/channel flow
pub struct PipeFlowCorrelations;

impl PipeFlowCorrelations {
    /// Darcy friction factor (Colebrook-White, turbulent)
    pub fn friction_factor_turbulent(re: f64, roughness: f64, diameter: f64) -> f64 {
        let eps_d = roughness / diameter;

        // Haaland approximation (explicit)
        let f = 1.0 / (-1.8 * ((eps_d / 3.7).powf(1.11) + 6.9 / re).log10()).powi(2);
        f
    }

    /// Darcy friction factor (laminar)
    pub fn friction_factor_laminar(re: f64) -> f64 {
        64.0 / re
    }

    /// Pressure drop in pipe
    pub fn pressure_drop(f: f64, length: f64, diameter: f64, rho: f64, velocity: f64) -> f64 {
        f * (length / diameter) * 0.5 * rho * velocity.powi(2)
    }

    /// Nusselt number - Gnielinski correlation (turbulent, 0.5 < Pr < 2000, 3000 < Re < 5×10⁶)
    pub fn nusselt_gnielinski(re: f64, pr: f64, f: f64) -> f64 {
        let f8 = f / 8.0;
        f8 * (re - 1000.0) * pr /
            (1.0 + 12.7 * f8.sqrt() * (pr.powf(2.0 / 3.0) - 1.0))
    }

    /// Nusselt number for liquid metals (Lyon-Martinelli)
    pub fn nusselt_liquid_metal(pe: f64) -> f64 {
        // Pe = Re × Pr (Peclet number)
        7.0 + 0.025 * pe.powf(0.8)
    }

    /// MHD friction factor increase
    pub fn mhd_friction_factor_ratio(ha: f64) -> f64 {
        // For high Hartmann number
        if ha < 10.0 {
            1.0
        } else {
            // Approximately Ha for Ha >> 1
            ha / 3.0
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_properties() {
        let pbli = FluidProperties::pbli_450c();

        assert!(pbli.rho_ref > 9000.0);
        assert!(pbli.sigma_e > 1e5);  // High electrical conductivity
        assert!(pbli.pr < 0.1);  // Very low Prandtl

        let ha = pbli.hartmann_number(5.0, 0.1, 723.15);
        assert!(ha > 1000.0);  // Strong MHD effects expected
    }

    #[test]
    fn test_mesh_creation() {
        let mesh = StructuredMesh::cartesian(10, 10, 10,
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)));

        assert_eq!(mesh.n_cells(), 1000);
        assert!((mesh.dx - 0.1).abs() < 1e-10);

        let center = mesh.cell_center(5, 5, 5);
        assert!((center.x - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_simple_solver_creation() {
        let mesh = StructuredMesh::cartesian(5, 5, 5,
            ((0.0, 0.1), (0.0, 0.01), (0.0, 0.01)));

        let fluid = FluidProperties::water_300c();
        let bc = BoundaryConditions::pipe_flow(1.0, 573.15, 623.15);

        let solver = SimpleSolver::new(mesh, fluid, bc);

        assert_eq!(solver.mesh.n_cells(), 125);
    }

    #[test]
    fn test_mhd_lorentz_force() {
        let mhd = MHDModel::new(Vec3::new(0.0, 5.0, 0.0), 7.7e5);

        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let force = mhd.lorentz_force(&velocity);

        // Force should oppose velocity component perpendicular to B
        assert!(force.x < 0.0);  // Damping in x direction
        assert!(force.y.abs() < 1e-10);  // No force along B
    }

    #[test]
    fn test_pipe_flow_correlations() {
        // Turbulent flow in smooth pipe
        let re = 50000.0;
        let f = PipeFlowCorrelations::friction_factor_turbulent(re, 0.0, 0.1);

        assert!(f > 0.01 && f < 0.05);  // Reasonable range

        // Liquid metal Nusselt
        let pe = 1000.0;
        let nu = PipeFlowCorrelations::nusselt_liquid_metal(pe);
        assert!(nu > 7.0);  // Must be > 7 for liquid metals
    }

    #[test]
    fn test_solid_region() {
        let mesh = StructuredMesh::cartesian(10, 5, 10,
            ((0.0, 0.1), (0.0, 0.01), (0.0, 0.1)));

        let mut solid = SolidRegion::first_wall_tungsten(mesh);

        // Set initial temperature
        for t in &mut solid.temperature.values {
            *t = 800.0;
        }

        // Solve a few time steps
        solid.solve_conduction(1e-6, 10);

        // Temperature should increase due to heating
        assert!(solid.max_temperature() > 800.0);
    }

    #[test]
    fn test_boundary_conditions() {
        let bc = BoundaryConditions::pipe_flow(2.0, 300.0, 400.0);

        assert!(bc.velocity.contains_key(&BoundaryFace::XMin));
        assert!(bc.temperature.contains_key(&BoundaryFace::YMin));
    }
}
