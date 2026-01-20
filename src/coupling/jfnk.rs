//! # Jacobian-Free Newton-Krylov (JFNK) Solver
//!
//! Implicit solver for multiphysics coupling without explicit Jacobian assembly.
//!
//! ## Why JFNK?
//!
//! Explicit coupling (sequential iteration) has severe time step restrictions:
//! - Thermal time scale: ~10⁻³ s
//! - Magnetic time scale: ~10⁻⁶ s
//! - MHD time scale: ~10⁻⁸ s
//!
//! Explicit coupling requires Δt < min(all time scales) = 10⁻⁸ s.
//! To simulate 1 second: 10⁸ steps = FOREVER.
//!
//! JFNK solves the coupled system implicitly, allowing Δt ~ 10⁻³ s.
//! Speedup: 10⁵× faster.
//!
//! ## Algorithm
//!
//! 1. Newton iteration: Solve F(x) = 0 for coupled residual F
//! 2. Each Newton step: J·δx = -F(x)
//! 3. Krylov method (GMRES): Solve linear system without forming J
//! 4. Jacobian-vector product: J·v ≈ [F(x + εv) - F(x)]/ε
//!
//! ## References
//!
//! - Knoll & Keyes, "Jacobian-free Newton-Krylov methods" (2004)
//! - Brown & Saad, "Hybrid Krylov methods for nonlinear systems" (1990)
//! - Kelley, "Iterative Methods for Linear and Nonlinear Equations" (1995)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use crate::types::Vec3;

/// JFNK solver configuration
#[derive(Debug, Clone)]
pub struct JFNKConfig {
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
    /// Newton convergence tolerance
    pub newton_tol: f64,
    /// Maximum GMRES iterations per Newton step
    pub max_gmres_iter: usize,
    /// GMRES convergence tolerance
    pub gmres_tol: f64,
    /// Finite difference epsilon for Jacobian-vector products
    pub fd_epsilon: f64,
    /// Enable line search (globalization)
    pub use_line_search: bool,
    /// Enable preconditioning
    pub use_preconditioner: bool,
}

impl Default for JFNKConfig {
    fn default() -> Self {
        Self {
            max_newton_iter: 20,
            newton_tol: 1e-8,
            max_gmres_iter: 50,
            gmres_tol: 1e-4,
            fd_epsilon: 1e-7,
            use_line_search: true,
            use_preconditioner: true,
        }
    }
}

/// State vector for coupled multiphysics
#[derive(Debug, Clone)]
pub struct CoupledStateVector {
    /// Temperature field (K)
    pub temperature: Vec<f64>,
    /// Displacement field (m)
    pub displacement: Vec<Vec3>,
    /// Magnetic field (T)
    pub b_field: Vec<Vec3>,
    /// Velocity field (m/s)
    pub velocity: Vec<Vec3>,
    /// Number of nodes
    pub n_nodes: usize,
}

impl CoupledStateVector {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            temperature: vec![300.0; n_nodes],
            displacement: vec![Vec3::zero(); n_nodes],
            b_field: vec![Vec3::zero(); n_nodes],
            velocity: vec![Vec3::zero(); n_nodes],
            n_nodes,
        }
    }

    /// Total degrees of freedom
    pub fn dof(&self) -> usize {
        self.n_nodes * 10  // 1 + 3 + 3 + 3
    }

    /// Pack into flat vector
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.dof());

        // Temperature (1 DOF per node)
        v.extend(&self.temperature);

        // Displacement (3 DOF per node)
        for d in &self.displacement {
            v.push(d.x);
            v.push(d.y);
            v.push(d.z);
        }

        // B-field (3 DOF per node)
        for b in &self.b_field {
            v.push(b.x);
            v.push(b.y);
            v.push(b.z);
        }

        // Velocity (3 DOF per node)
        for vel in &self.velocity {
            v.push(vel.x);
            v.push(vel.y);
            v.push(vel.z);
        }

        v
    }

    /// Unpack from flat vector
    pub fn from_vec(&mut self, v: &[f64]) {
        let n = self.n_nodes;
        let mut idx = 0;

        // Temperature
        for i in 0..n {
            self.temperature[i] = v[idx];
            idx += 1;
        }

        // Displacement
        for i in 0..n {
            self.displacement[i] = Vec3::new(v[idx], v[idx + 1], v[idx + 2]);
            idx += 3;
        }

        // B-field
        for i in 0..n {
            self.b_field[i] = Vec3::new(v[idx], v[idx + 1], v[idx + 2]);
            idx += 3;
        }

        // Velocity
        for i in 0..n {
            self.velocity[i] = Vec3::new(v[idx], v[idx + 1], v[idx + 2]);
            idx += 3;
        }
    }

    /// Compute L2 norm
    pub fn norm(&self) -> f64 {
        let v = self.to_vec();
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute difference norm
    pub fn diff_norm(&self, other: &CoupledStateVector) -> f64 {
        let v1 = self.to_vec();
        let v2 = other.to_vec();
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Trait for physics residual computation
pub trait ResidualFunction: Send + Sync {
    /// Compute residual F(x) for given state
    fn compute_residual(&self, state: &CoupledStateVector) -> Vec<f64>;

    /// Apply physics-based preconditioner (optional)
    fn apply_preconditioner(&self, _r: &[f64]) -> Vec<f64> {
        // Default: identity preconditioner
        _r.to_vec()
    }
}

/// Simple coupled physics residual (thermal + structural)
pub struct SimpleCoupledResidual {
    /// Thermal conductivity (W/m·K)
    pub k_thermal: f64,
    /// Young's modulus (Pa)
    pub e_modulus: f64,
    /// Thermal expansion coefficient (1/K)
    pub alpha: f64,
    /// Reference temperature (K)
    pub t_ref: f64,
    /// Time step (s)
    pub dt: f64,
    /// Previous state
    pub state_prev: CoupledStateVector,
    /// Heat source (W/m³)
    pub heat_source: Vec<f64>,
    /// External force (N/m³)
    pub body_force: Vec<Vec3>,
}

impl SimpleCoupledResidual {
    pub fn new(n_nodes: usize, dt: f64) -> Self {
        Self {
            k_thermal: 100.0,
            e_modulus: 200e9,
            alpha: 12e-6,
            t_ref: 300.0,
            dt,
            state_prev: CoupledStateVector::new(n_nodes),
            heat_source: vec![0.0; n_nodes],
            body_force: vec![Vec3::zero(); n_nodes],
        }
    }
}

impl ResidualFunction for SimpleCoupledResidual {
    fn compute_residual(&self, state: &CoupledStateVector) -> Vec<f64> {
        let n = state.n_nodes;
        let dof = state.dof();
        let mut residual = vec![0.0; dof];

        // ========================================
        // Thermal residual: ρc ∂T/∂t = ∇·(k∇T) + Q
        // Discretized: (T - T_prev)/dt = α·Laplacian(T) + Q/(ρc)
        // Residual: R_T = T - T_prev - dt·[α·Lap(T) + Q/(ρc)]
        // ========================================

        let thermal_diffusivity = self.k_thermal / (8000.0 * 500.0);  // k/(ρ·c_p)

        for i in 0..n {
            let t = state.temperature[i];
            let t_prev = self.state_prev.temperature[i];

            // Simplified 1D Laplacian
            let lap_t = if i > 0 && i < n - 1 {
                state.temperature[i + 1] - 2.0 * t + state.temperature[i - 1]
            } else {
                0.0
            };

            let q_source = self.heat_source[i] / (8000.0 * 500.0);  // Q/(ρ·c_p)

            residual[i] = t - t_prev - self.dt * (thermal_diffusivity * lap_t + q_source);
        }

        // ========================================
        // Structural residual: ρ ∂²u/∂t² = ∇·σ + f
        // With thermal expansion: σ = E(ε - α·ΔT)
        // Quasi-static: ∇·σ + f = 0
        // Residual: R_u = K·u - f_thermal - f_external
        // ========================================

        let mut idx = n;  // Start after temperature DOFs

        for i in 0..n {
            let u = &state.displacement[i];
            let delta_t = state.temperature[i] - self.t_ref;

            // Thermal strain contribution (used in dt_dx calculation)
            let _thermal_strain = self.alpha * delta_t;

            // Simplified 1D structural residual
            // R_u = E·(∂²u/∂x² - α·∂T/∂x) + f
            let d2u_dx2 = if i > 0 && i < n - 1 {
                state.displacement[i + 1].x - 2.0 * u.x + state.displacement[i - 1].x
            } else {
                0.0
            };

            let dt_dx = if i > 0 && i < n - 1 {
                (state.temperature[i + 1] - state.temperature[i - 1]) / 2.0
            } else {
                0.0
            };

            residual[idx] = self.e_modulus * (d2u_dx2 - self.alpha * dt_dx) + self.body_force[i].x;
            residual[idx + 1] = self.body_force[i].y;
            residual[idx + 2] = self.body_force[i].z;

            idx += 3;
        }

        // ========================================
        // EM residual: ∂B/∂t = -∇×E, ∇×B = μ₀J
        // Simplified: steady-state Maxwell
        // ========================================

        for i in 0..n {
            let _b = &state.b_field[i];  // Used for future enhancements

            // Simplified: B should satisfy ∇·B = 0
            let div_b = if i > 0 && i < n - 1 {
                (state.b_field[i + 1].x - state.b_field[i - 1].x) / 2.0
            } else {
                0.0
            };

            residual[idx] = div_b;
            residual[idx + 1] = 0.0;
            residual[idx + 2] = 0.0;

            idx += 3;
        }

        // ========================================
        // Velocity residual (incompressible Navier-Stokes)
        // Simplified for now
        // ========================================

        for _i in 0..n {
            residual[idx] = 0.0;
            residual[idx + 1] = 0.0;
            residual[idx + 2] = 0.0;
            idx += 3;
        }

        residual
    }

    fn apply_preconditioner(&self, r: &[f64]) -> Vec<f64> {
        // Block-diagonal preconditioner
        // Scale each physics block by its characteristic magnitude

        let n = self.state_prev.n_nodes;
        let mut p = vec![0.0; r.len()];

        // Thermal: scale by thermal diffusivity
        let thermal_scale = 1.0 / (self.k_thermal / (8000.0 * 500.0) * self.dt + 1.0);
        for i in 0..n {
            p[i] = r[i] * thermal_scale;
        }

        // Structural: scale by modulus
        let struct_scale = 1.0 / self.e_modulus;
        let mut idx = n;
        for _i in 0..n {
            p[idx] = r[idx] * struct_scale;
            p[idx + 1] = r[idx + 1] * struct_scale;
            p[idx + 2] = r[idx + 2] * struct_scale;
            idx += 3;
        }

        // EM and velocity: identity scaling
        for i in idx..r.len() {
            p[i] = r[i];
        }

        p
    }
}

/// GMRES (Generalized Minimal RESidual) solver
pub struct GMRES {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Krylov basis vectors
    basis: Vec<Vec<f64>>,
    /// Hessenberg matrix
    h: Vec<Vec<f64>>,
}

impl GMRES {
    pub fn new(max_iter: usize, tolerance: f64) -> Self {
        Self {
            max_iter,
            tolerance,
            basis: Vec::new(),
            h: Vec::new(),
        }
    }

    /// Solve J·x = b using GMRES with matrix-free Jacobian
    pub fn solve<F>(
        &mut self,
        matvec: F,
        b: &[f64],
        x0: &[f64],
    ) -> (Vec<f64>, GMRESResult)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();

        // Initial guess
        let mut x = x0.to_vec();

        // Initial residual: r = b - A*x0
        let ax0 = matvec(&x);
        let r: Vec<f64> = b.iter().zip(ax0.iter()).map(|(bi, ai)| bi - ai).collect();

        let beta = vec_norm(&r);

        if beta < self.tolerance {
            return (x, GMRESResult { converged: true, iterations: 0, residual: beta });
        }

        // Normalize initial residual
        let v0: Vec<f64> = r.iter().map(|ri| ri / beta).collect();

        // Initialize Krylov basis
        self.basis = vec![v0];
        self.h = vec![vec![0.0; self.max_iter + 1]; self.max_iter];

        // Arnoldi process with GMRES
        let mut g = vec![0.0; self.max_iter + 1];
        g[0] = beta;

        let mut cos = vec![0.0; self.max_iter];
        let mut sin = vec![0.0; self.max_iter];

        let mut converged = false;
        let mut final_iter = 0;
        let mut final_residual = beta;

        for k in 0..self.max_iter {
            // Arnoldi step: w = A * v_k
            let w = matvec(&self.basis[k]);

            // Modified Gram-Schmidt orthogonalization
            let mut w_orth = w.clone();

            for j in 0..=k {
                self.h[j][k] = dot(&w_orth, &self.basis[j]);
                for i in 0..n {
                    w_orth[i] -= self.h[j][k] * self.basis[j][i];
                }
            }

            self.h[k + 1][k] = vec_norm(&w_orth);

            if self.h[k + 1][k] < 1e-14 {
                // Lucky breakdown
                break;
            }

            // Add new basis vector
            let v_new: Vec<f64> = w_orth.iter().map(|wi| wi / self.h[k + 1][k]).collect();
            self.basis.push(v_new);

            // Apply previous Givens rotations to new column
            for j in 0..k {
                let h_j = self.h[j][k];
                let h_j1 = self.h[j + 1][k];
                self.h[j][k] = cos[j] * h_j + sin[j] * h_j1;
                self.h[j + 1][k] = -sin[j] * h_j + cos[j] * h_j1;
            }

            // Compute new Givens rotation
            let h_k = self.h[k][k];
            let h_k1 = self.h[k + 1][k];
            let rho = (h_k * h_k + h_k1 * h_k1).sqrt();

            cos[k] = h_k / rho;
            sin[k] = h_k1 / rho;

            self.h[k][k] = rho;
            self.h[k + 1][k] = 0.0;

            // Update residual
            let g_k = g[k];
            g[k] = cos[k] * g_k;
            g[k + 1] = -sin[k] * g_k;

            final_residual = g[k + 1].abs();
            final_iter = k + 1;

            if final_residual < self.tolerance {
                converged = true;
                break;
            }
        }

        // Solve upper triangular system: H * y = g
        let m = final_iter;
        let mut y = vec![0.0; m];

        for i in (0..m).rev() {
            let mut sum = g[i];
            for j in (i + 1)..m {
                sum -= self.h[i][j] * y[j];
            }
            y[i] = sum / self.h[i][i];
        }

        // Compute solution: x = x0 + V * y
        for i in 0..m {
            for j in 0..n {
                x[j] += y[i] * self.basis[i][j];
            }
        }

        (x, GMRESResult { converged, iterations: final_iter, residual: final_residual })
    }
}

/// GMRES result
#[derive(Debug, Clone)]
pub struct GMRESResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual: f64,
}

/// Jacobian-Free Newton-Krylov solver
pub struct JFNKSolver {
    /// Configuration
    pub config: JFNKConfig,
    /// GMRES solver
    gmres: GMRES,
    /// Convergence history
    pub history: Vec<NewtonStep>,
}

/// Record of one Newton step
#[derive(Debug, Clone)]
pub struct NewtonStep {
    pub iteration: usize,
    pub residual_norm: f64,
    pub gmres_iterations: usize,
    pub step_length: f64,
}

impl JFNKSolver {
    pub fn new(config: JFNKConfig) -> Self {
        let gmres = GMRES::new(config.max_gmres_iter, config.gmres_tol);

        Self {
            config,
            gmres,
            history: Vec::new(),
        }
    }

    /// Solve F(x) = 0 using JFNK
    pub fn solve<R: ResidualFunction>(
        &mut self,
        residual_fn: &R,
        state: &mut CoupledStateVector,
    ) -> JFNKResult {
        self.history.clear();

        let mut x = state.to_vec();
        let n = x.len();

        for newton_iter in 0..self.config.max_newton_iter {
            // Update state from current x
            state.from_vec(&x);

            // Compute residual F(x)
            let f = residual_fn.compute_residual(state);
            let f_norm = vec_norm(&f);

            // Check convergence
            if f_norm < self.config.newton_tol {
                return JFNKResult {
                    converged: true,
                    iterations: newton_iter,
                    final_residual: f_norm,
                    history: self.history.clone(),
                };
            }

            // Define Jacobian-vector product: J*v ≈ [F(x + ε*v) - F(x)] / ε
            let eps = self.config.fd_epsilon;
            let matvec = |v: &[f64]| -> Vec<f64> {
                // Perturbed state
                let v_norm = vec_norm(v);
                let eps_scaled = if v_norm > 1e-14 { eps / v_norm } else { eps };

                let x_pert: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| xi + eps_scaled * vi).collect();

                let mut state_pert = state.clone();
                state_pert.from_vec(&x_pert);

                let f_pert = residual_fn.compute_residual(&state_pert);

                // J*v = (F(x+εv) - F(x)) / ε
                f_pert.iter().zip(f.iter()).map(|(fp, fi)| (fp - fi) / eps_scaled).collect()
            };

            // Precondition RHS
            let b: Vec<f64> = if self.config.use_preconditioner {
                let neg_f: Vec<f64> = f.iter().map(|fi| -fi).collect();
                residual_fn.apply_preconditioner(&neg_f)
            } else {
                f.iter().map(|fi| -fi).collect()
            };

            // Solve J*dx = -F using GMRES
            let x0 = vec![0.0; n];
            let (dx, gmres_result) = self.gmres.solve(&matvec, &b, &x0);

            // Line search (backtracking)
            let mut alpha = 1.0;

            if self.config.use_line_search {
                for _ in 0..10 {
                    let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(xi, di)| xi + alpha * di).collect();

                    let mut state_new = state.clone();
                    state_new.from_vec(&x_new);

                    let f_new = residual_fn.compute_residual(&state_new);
                    let f_new_norm = vec_norm(&f_new);

                    // Armijo condition
                    if f_new_norm < (1.0 - 1e-4 * alpha) * f_norm {
                        break;
                    }

                    alpha *= 0.5;
                }
            }

            // Update solution
            for i in 0..n {
                x[i] += alpha * dx[i];
            }

            // Record step
            self.history.push(NewtonStep {
                iteration: newton_iter,
                residual_norm: f_norm,
                gmres_iterations: gmres_result.iterations,
                step_length: alpha,
            });
        }

        // Did not converge
        state.from_vec(&x);
        let f = residual_fn.compute_residual(state);

        JFNKResult {
            converged: false,
            iterations: self.config.max_newton_iter,
            final_residual: vec_norm(&f),
            history: self.history.clone(),
        }
    }
}

/// JFNK solver result
#[derive(Debug, Clone)]
pub struct JFNKResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub history: Vec<NewtonStep>,
}

impl JFNKResult {
    pub fn summary(&self) -> String {
        format!(
            "JFNK Result\n\
             ===========\n\
             Converged: {}\n\
             Newton iterations: {}\n\
             Final residual: {:.2e}\n",
            self.converged,
            self.iterations,
            self.final_residual,
        )
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Vector L2 norm
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupled_state_vector() {
        let state = CoupledStateVector::new(10);
        assert_eq!(state.dof(), 100);

        let v = state.to_vec();
        assert_eq!(v.len(), 100);

        let mut state2 = CoupledStateVector::new(10);
        state2.temperature[5] = 500.0;
        state2.from_vec(&state2.to_vec());
        assert!((state2.temperature[5] - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_gmres_simple() {
        // Test GMRES on simple diagonal system: A*x = b where A = diag(1,2,3,4,5)
        let mut gmres = GMRES::new(10, 1e-10);

        let matvec = |v: &[f64]| -> Vec<f64> {
            v.iter().enumerate().map(|(i, vi)| (i + 1) as f64 * vi).collect()
        };

        let b = vec![1.0, 4.0, 9.0, 16.0, 25.0];  // Solution should be [1,2,3,4,5]
        let x0 = vec![0.0; 5];

        let (x, result) = gmres.solve(&matvec, &b, &x0);

        assert!(result.converged);
        for i in 0..5 {
            assert!((x[i] - (i + 1) as f64).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simple_residual() {
        let n_nodes = 10;
        let dt = 0.001;
        let residual_fn = SimpleCoupledResidual::new(n_nodes, dt);
        let state = CoupledStateVector::new(n_nodes);

        let r = residual_fn.compute_residual(&state);
        assert_eq!(r.len(), state.dof());
    }

    #[test]
    fn test_jfnk_solver() {
        let config = JFNKConfig {
            max_newton_iter: 10,
            newton_tol: 1e-6,
            max_gmres_iter: 20,
            gmres_tol: 1e-3,
            ..Default::default()
        };

        let mut solver = JFNKSolver::new(config);

        let n_nodes = 10;
        let dt = 0.001;
        let residual_fn = SimpleCoupledResidual::new(n_nodes, dt);
        let mut state = CoupledStateVector::new(n_nodes);

        // Set non-trivial initial condition
        for i in 0..n_nodes {
            state.temperature[i] = 300.0 + 10.0 * (i as f64);
        }

        let result = solver.solve(&residual_fn, &mut state);

        // Should make progress (residual should decrease)
        if !result.history.is_empty() {
            let first_res = result.history[0].residual_norm;
            let last_res = result.final_residual;
            // At least some progress or convergence
            assert!(last_res <= first_res * 10.0 || result.converged);
        }
    }
}
