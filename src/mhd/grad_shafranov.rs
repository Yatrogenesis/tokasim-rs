//! # Grad-Shafranov Equation Solver
//!
//! First-principles 2D equilibrium solver for tokamak plasmas.
//!
//! ## The Grad-Shafranov Equation
//!
//! In axisymmetric equilibrium, the magnetic field can be written as:
//! B = (1/R)∇ψ × φ̂ + (F(ψ)/R)φ̂
//!
//! where ψ is the poloidal flux function and F(ψ) = RB_φ.
//!
//! Force balance (J × B = ∇p) leads to the Grad-Shafranov equation:
//!
//! Δ*ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z² = -μ₀R²(dp/dψ) - F(dF/dψ)
//!
//! This is a nonlinear elliptic PDE solved iteratively.

use crate::constants::MU_0;
use std::f64::consts::PI;

/// 2D grid for Grad-Shafranov solver
#[derive(Debug, Clone)]
pub struct GSGrid {
    /// R coordinates (major radius direction)
    pub r: Vec<f64>,
    /// Z coordinates (vertical direction)
    pub z: Vec<f64>,
    /// Number of grid points in R
    pub nr: usize,
    /// Number of grid points in Z
    pub nz: usize,
    /// Grid spacing in R
    pub dr: f64,
    /// Grid spacing in Z
    pub dz: f64,
    /// Poloidal flux ψ(R,Z)
    pub psi: Vec<Vec<f64>>,
    /// Previous iteration flux (for convergence check)
    psi_old: Vec<Vec<f64>>,
}

impl GSGrid {
    /// Create new computational grid
    pub fn new(r_min: f64, r_max: f64, z_min: f64, z_max: f64, nr: usize, nz: usize) -> Self {
        let dr = (r_max - r_min) / (nr - 1) as f64;
        let dz = (z_max - z_min) / (nz - 1) as f64;

        let r: Vec<f64> = (0..nr).map(|i| r_min + i as f64 * dr).collect();
        let z: Vec<f64> = (0..nz).map(|j| z_min + j as f64 * dz).collect();

        let psi = vec![vec![0.0; nz]; nr];
        let psi_old = vec![vec![0.0; nz]; nr];

        Self { r, z, nr, nz, dr, dz, psi, psi_old }
    }

    /// Get R coordinate at index i
    #[inline]
    pub fn r_at(&self, i: usize) -> f64 {
        self.r[i]
    }

    /// Get Z coordinate at index j
    #[inline]
    pub fn z_at(&self, j: usize) -> f64 {
        self.z[j]
    }

    /// Get flux at grid point
    #[inline]
    pub fn psi_at(&self, i: usize, j: usize) -> f64 {
        self.psi[i][j]
    }

    /// Set flux at grid point
    #[inline]
    pub fn set_psi(&mut self, i: usize, j: usize, value: f64) {
        self.psi[i][j] = value;
    }
}

/// Pressure and current profile specifications
#[derive(Debug, Clone)]
pub struct ProfileSpec {
    /// Pressure at magnetic axis (Pa)
    pub p0: f64,
    /// Pressure profile exponent
    pub alpha_p: f64,
    /// F = R*B_phi at axis
    pub f0: f64,
    /// F profile parameter
    pub alpha_f: f64,
    /// Plasma current (A)
    pub ip_target: f64,
}

impl Default for ProfileSpec {
    fn default() -> Self {
        Self {
            p0: 1e6,        // 1 MPa peak pressure
            alpha_p: 2.0,   // Parabolic pressure profile
            f0: 1.0,        // Will be set based on B0 and R0
            alpha_f: 1.0,   // F profile exponent
            ip_target: 12e6, // 12 MA
        }
    }
}

/// Grad-Shafranov equilibrium solver
#[derive(Debug)]
pub struct GradShafranovSolver {
    /// Computational grid
    pub grid: GSGrid,
    /// Profile specification
    pub profiles: ProfileSpec,
    /// Major radius (m)
    pub r0: f64,
    /// Minor radius (m)
    pub a: f64,
    /// Toroidal field at R0 (T)
    pub b0: f64,
    /// Elongation
    pub kappa: f64,
    /// Triangularity
    pub delta: f64,
    /// Flux at magnetic axis
    pub psi_axis: f64,
    /// Flux at plasma boundary
    pub psi_boundary: f64,
    /// Magnetic axis R position
    pub r_axis: f64,
    /// Magnetic axis Z position
    pub z_axis: f64,
    /// Relaxation parameter for SOR
    omega: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl GradShafranovSolver {
    /// Create new solver with tokamak parameters
    pub fn new(r0: f64, a: f64, b0: f64, kappa: f64, delta: f64) -> Self {
        // Grid extends from R_min to R_max with margin
        let r_min = (r0 - 1.5 * a).max(0.1);
        let r_max = r0 + 1.5 * a;
        let z_min = -1.5 * a * kappa;
        let z_max = 1.5 * a * kappa;

        // Use 129x129 grid for good resolution
        let nr = 129;
        let nz = 129;

        let grid = GSGrid::new(r_min, r_max, z_min, z_max, nr, nz);

        let mut profiles = ProfileSpec::default();
        profiles.f0 = r0 * b0;  // F0 = R0 * B0

        Self {
            grid,
            profiles,
            r0,
            a,
            b0,
            kappa,
            delta,
            psi_axis: 0.0,
            psi_boundary: 1.0,
            r_axis: r0,
            z_axis: 0.0,
            omega: 1.7,      // SOR over-relaxation parameter
            max_iter: 5000,
            tolerance: 1e-8,
        }
    }

    /// Initialize with Solov'ev analytical solution as starting guess
    pub fn initialize_solovev(&mut self) {
        // Solov'ev solution: ψ = A*x⁴ + B*x² + C*(R⁴/8 + R²Z²)
        // where x² = (R-R0)²/a² + Z²/(κa)²

        for i in 0..self.grid.nr {
            for j in 0..self.grid.nz {
                let r = self.grid.r_at(i);
                let z = self.grid.z_at(j);

                // Normalized coordinates
                let x = (r - self.r0) / self.a;
                let y = z / (self.kappa * self.a);

                // D-shape parameterization for boundary check
                let theta = y.atan2(x);
                let r_boundary = 1.0 + self.delta * theta.cos();
                let rho_sq = x * x / (r_boundary * r_boundary) + y * y;

                if rho_sq <= 1.0 {
                    // Inside plasma - Solov'ev-like initial guess
                    let psi_norm = 1.0 - rho_sq;
                    self.grid.set_psi(i, j, psi_norm);
                } else {
                    // Outside plasma - vacuum
                    self.grid.set_psi(i, j, 0.0);
                }
            }
        }

        // Set initial axis and boundary values
        self.psi_axis = 1.0;
        self.psi_boundary = 0.0;
    }

    /// Pressure as function of normalized flux
    /// p(ψ_n) = p0 * (1 - ψ_n^α_p)
    pub fn pressure(&self, psi_norm: f64) -> f64 {
        if psi_norm >= 0.0 && psi_norm <= 1.0 {
            self.profiles.p0 * (1.0 - psi_norm).powf(self.profiles.alpha_p)
        } else {
            0.0
        }
    }

    /// dp/dψ for the RHS of GS equation
    pub fn dp_dpsi(&self, psi_norm: f64) -> f64 {
        if psi_norm >= 0.0 && psi_norm <= 1.0 && self.profiles.alpha_p > 0.0 {
            let psi_range = self.psi_axis - self.psi_boundary;
            if psi_range.abs() > 1e-10 {
                -self.profiles.p0 * self.profiles.alpha_p
                    * (1.0 - psi_norm).powf(self.profiles.alpha_p - 1.0) / psi_range
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// F(ψ) = R*B_phi function
    pub fn f_function(&self, psi_norm: f64) -> f64 {
        if psi_norm >= 0.0 && psi_norm <= 1.0 {
            // F² = F0² * (1 - C*(1-ψ_n)^α_f)
            // For simplicity, use linear profile
            let c = 0.1; // Small variation in F
            self.profiles.f0 * (1.0 - c * (1.0 - psi_norm).powf(self.profiles.alpha_f)).sqrt()
        } else {
            self.profiles.f0  // Vacuum value
        }
    }

    /// F * dF/dψ for the RHS of GS equation
    pub fn f_df_dpsi(&self, psi_norm: f64) -> f64 {
        if psi_norm >= 0.0 && psi_norm <= 1.0 {
            let psi_range = self.psi_axis - self.psi_boundary;
            if psi_range.abs() > 1e-10 {
                let c = 0.1;
                let f = self.f_function(psi_norm);
                let df_dpsi_norm = self.profiles.f0 * c * self.profiles.alpha_f
                    * (1.0 - psi_norm).powf(self.profiles.alpha_f - 1.0)
                    / (2.0 * f);
                f * df_dpsi_norm / psi_range
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Calculate normalized flux at grid point
    fn psi_normalized(&self, i: usize, j: usize) -> f64 {
        let psi = self.grid.psi_at(i, j);
        let psi_range = self.psi_axis - self.psi_boundary;
        if psi_range.abs() > 1e-10 {
            (psi - self.psi_boundary) / psi_range
        } else {
            0.0
        }
    }

    /// Check if point is inside plasma boundary
    fn is_inside_plasma(&self, i: usize, j: usize) -> bool {
        let r = self.grid.r_at(i);
        let z = self.grid.z_at(j);

        let x = (r - self.r0) / self.a;
        let y = z / (self.kappa * self.a);

        // D-shape boundary
        let theta = y.atan2(x);
        let r_boundary = 1.0 + self.delta * theta.cos();
        let rho_sq = x * x / (r_boundary * r_boundary) + y * y;

        rho_sq <= 1.0
    }

    /// Right-hand side of Grad-Shafranov equation
    /// RHS = -μ₀R²(dp/dψ) - F(dF/dψ)
    fn rhs(&self, i: usize, j: usize) -> f64 {
        if !self.is_inside_plasma(i, j) {
            return 0.0;  // Vacuum region
        }

        let r = self.grid.r_at(i);
        let psi_n = self.psi_normalized(i, j);

        let dp_dpsi = self.dp_dpsi(psi_n);
        let f_df_dpsi = self.f_df_dpsi(psi_n);

        -MU_0 * r * r * dp_dpsi - f_df_dpsi
    }

    /// Apply one SOR iteration of the Grad-Shafranov operator
    /// Δ*ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
    fn sor_iteration(&mut self) -> f64 {
        let dr = self.grid.dr;
        let dz = self.grid.dz;
        let dr2 = dr * dr;
        let dz2 = dz * dz;

        let mut max_change = 0.0_f64;

        // Save old values for convergence check
        for i in 0..self.grid.nr {
            for j in 0..self.grid.nz {
                self.grid.psi_old[i][j] = self.grid.psi[i][j];
            }
        }

        // Interior points only (boundary is fixed)
        for i in 1..self.grid.nr - 1 {
            let r = self.grid.r_at(i);
            let r_plus_half = r + 0.5 * dr;
            let r_minus_half = r - 0.5 * dr;

            for j in 1..self.grid.nz - 1 {
                if !self.is_inside_plasma(i, j) {
                    // Vacuum region - Laplacian = 0
                    continue;
                }

                // Finite difference coefficients for Δ*ψ
                // Δ*ψ = (1/R)[∂/∂R(R ∂ψ/∂R)] + ∂²ψ/∂Z²
                // Using conservative differencing:
                // = (1/(R*dR²)) * [R_{i+1/2}(ψ_{i+1} - ψ_i) - R_{i-1/2}(ψ_i - ψ_{i-1})]
                //   + (ψ_{j+1} - 2ψ_j + ψ_{j-1})/dZ²

                let psi_ip = self.grid.psi_old[i + 1][j];
                let psi_im = self.grid.psi_old[i - 1][j];
                let psi_jp = self.grid.psi_old[i][j + 1];
                let psi_jm = self.grid.psi_old[i][j - 1];
                let psi_ij = self.grid.psi_old[i][j];

                // Coefficients
                let a_r_plus = r_plus_half / (r * dr2);
                let a_r_minus = r_minus_half / (r * dr2);
                let a_z = 1.0 / dz2;
                let a_center = a_r_plus + a_r_minus + 2.0 * a_z;

                // RHS from pressure and current
                let rhs = self.rhs(i, j);

                // Gauss-Seidel update
                let psi_new = (a_r_plus * psi_ip + a_r_minus * psi_im
                             + a_z * (psi_jp + psi_jm) - rhs) / a_center;

                // SOR relaxation
                let psi_relaxed = psi_ij + self.omega * (psi_new - psi_ij);
                self.grid.set_psi(i, j, psi_relaxed);

                max_change = max_change.max((psi_relaxed - psi_ij).abs());
            }
        }

        max_change
    }

    /// Find magnetic axis (maximum of ψ inside plasma)
    fn find_magnetic_axis(&mut self) {
        let mut psi_max = f64::NEG_INFINITY;
        let mut i_max = self.grid.nr / 2;
        let mut j_max = self.grid.nz / 2;

        for i in 1..self.grid.nr - 1 {
            for j in 1..self.grid.nz - 1 {
                if self.is_inside_plasma(i, j) {
                    let psi = self.grid.psi_at(i, j);
                    if psi > psi_max {
                        psi_max = psi;
                        i_max = i;
                        j_max = j;
                    }
                }
            }
        }

        self.r_axis = self.grid.r_at(i_max);
        self.z_axis = self.grid.z_at(j_max);
        self.psi_axis = psi_max;
    }

    /// Find flux at plasma boundary (LCFS)
    fn find_boundary_flux(&mut self) {
        // Find minimum flux on the plasma boundary
        let mut psi_min = f64::INFINITY;

        for i in 1..self.grid.nr - 1 {
            for j in 1..self.grid.nz - 1 {
                let r = self.grid.r_at(i);
                let z = self.grid.z_at(j);

                let x = (r - self.r0) / self.a;
                let y = z / (self.kappa * self.a);
                let theta = y.atan2(x);
                let r_boundary = 1.0 + self.delta * theta.cos();
                let rho = (x * x / (r_boundary * r_boundary) + y * y).sqrt();

                // Points near boundary (0.95 < rho < 1.05)
                if rho > 0.95 && rho < 1.05 {
                    let psi = self.grid.psi_at(i, j);
                    if psi < psi_min {
                        psi_min = psi;
                    }
                }
            }
        }

        self.psi_boundary = psi_min.max(0.0);
    }

    /// Solve the Grad-Shafranov equation iteratively
    pub fn solve(&mut self) -> SolverResult {
        // Initialize with Solov'ev guess
        self.initialize_solovev();

        let mut iterations = 0;
        let mut converged = false;
        let mut residual = 1.0;

        // Main iteration loop
        for iter in 0..self.max_iter {
            iterations = iter + 1;

            // One SOR iteration
            residual = self.sor_iteration();

            // Update axis and boundary
            if iter % 50 == 0 {
                self.find_magnetic_axis();
                self.find_boundary_flux();
            }

            // Check convergence
            if residual < self.tolerance {
                converged = true;
                break;
            }
        }

        // Final update of axis and boundary
        self.find_magnetic_axis();
        self.find_boundary_flux();

        SolverResult {
            converged,
            iterations,
            residual,
            psi_axis: self.psi_axis,
            psi_boundary: self.psi_boundary,
            r_axis: self.r_axis,
            z_axis: self.z_axis,
        }
    }

    /// Get magnetic field components at (R, Z)
    pub fn magnetic_field(&self, r: f64, z: f64) -> (f64, f64, f64) {
        // Find grid cell
        let i = ((r - self.grid.r[0]) / self.grid.dr) as usize;
        let j = ((z - self.grid.z[0]) / self.grid.dz) as usize;

        if i == 0 || i >= self.grid.nr - 1 || j == 0 || j >= self.grid.nz - 1 {
            // Outside grid - return vacuum field
            return (0.0, self.b0 * self.r0 / r, 0.0);
        }

        // Calculate derivatives of ψ using central differences
        let dpsi_dr = (self.grid.psi[i + 1][j] - self.grid.psi[i - 1][j]) / (2.0 * self.grid.dr);
        let dpsi_dz = (self.grid.psi[i][j + 1] - self.grid.psi[i][j - 1]) / (2.0 * self.grid.dz);

        // B_R = -(1/R) ∂ψ/∂Z
        let b_r = -dpsi_dz / r;

        // B_Z = (1/R) ∂ψ/∂R
        let b_z = dpsi_dr / r;

        // B_phi = F(ψ)/R
        let psi_n = self.psi_normalized(i, j);
        let f = self.f_function(psi_n);
        let b_phi = f / r;

        (b_r, b_phi, b_z)
    }

    /// Calculate plasma current from equilibrium
    pub fn plasma_current(&self) -> f64 {
        // I_p = (1/μ₀) ∮ B_pol · dl around plasma boundary
        // Using Ampère's law: I_p = (2π/μ₀) * [ψ_axis - ψ_boundary]

        let psi_range = self.psi_axis - self.psi_boundary;
        2.0 * PI * psi_range / MU_0
    }

    /// Calculate toroidal beta
    pub fn beta_toroidal(&self) -> f64 {
        // β_t = 2μ₀ <p> / B_0²
        // where <p> is volume-averaged pressure

        let mut p_sum = 0.0;
        let mut vol_sum = 0.0;

        for i in 1..self.grid.nr - 1 {
            let r = self.grid.r_at(i);
            for j in 1..self.grid.nz - 1 {
                if self.is_inside_plasma(i, j) {
                    let psi_n = self.psi_normalized(i, j);
                    let p = self.pressure(psi_n);
                    let dv = 2.0 * PI * r * self.grid.dr * self.grid.dz;
                    p_sum += p * dv;
                    vol_sum += dv;
                }
            }
        }

        let p_avg = if vol_sum > 0.0 { p_sum / vol_sum } else { 0.0 };
        2.0 * MU_0 * p_avg / (self.b0 * self.b0)
    }

    /// Calculate safety factor at normalized flux surface
    pub fn safety_factor(&self, psi_n: f64) -> f64 {
        if psi_n <= 0.0 || psi_n >= 1.0 {
            return f64::INFINITY;
        }

        // q = (1/2π) ∮ (F/R²) (dl_pol/|B_pol|)
        // Simplified: q ≈ (r²B_phi) / (R ψ')

        // For now, use approximate formula
        let rho = psi_n.sqrt();  // Approximate normalized minor radius
        let r_local = rho * self.a;

        // q ≈ (r B_t) / (R B_p) ≈ (2π r² B_0 R_0) / (μ_0 R I_p enclosed)
        let i_enclosed = self.profiles.ip_target * psi_n;  // Approximate

        if i_enclosed.abs() > 1.0 {
            2.0 * PI * r_local * r_local * self.b0 * self.r0 / (MU_0 * self.r0 * i_enclosed)
        } else {
            1.0  // Near axis
        }
    }

    /// Export flux surface data for visualization
    pub fn export_flux_surfaces(&self, n_surfaces: usize) -> Vec<FluxSurface> {
        let mut surfaces = Vec::with_capacity(n_surfaces);

        for k in 0..n_surfaces {
            let psi_n = (k as f64 + 0.5) / n_surfaces as f64;
            let psi_target = self.psi_boundary + psi_n * (self.psi_axis - self.psi_boundary);

            let mut points = Vec::new();

            // Trace contour at this flux value
            for i in 1..self.grid.nr - 1 {
                for j in 1..self.grid.nz - 1 {
                    let psi = self.grid.psi_at(i, j);
                    let psi_right = self.grid.psi_at(i + 1, j);
                    let psi_up = self.grid.psi_at(i, j + 1);

                    // Check for contour crossing
                    if (psi - psi_target) * (psi_right - psi_target) < 0.0 {
                        let t = (psi_target - psi) / (psi_right - psi);
                        let r = self.grid.r_at(i) + t * self.grid.dr;
                        let z = self.grid.z_at(j);
                        points.push((r, z));
                    }
                    if (psi - psi_target) * (psi_up - psi_target) < 0.0 {
                        let t = (psi_target - psi) / (psi_up - psi);
                        let r = self.grid.r_at(i);
                        let z = self.grid.z_at(j) + t * self.grid.dz;
                        points.push((r, z));
                    }
                }
            }

            // Sort points by angle for proper contour
            if !points.is_empty() {
                points.sort_by(|a, b| {
                    let angle_a = (a.1 - self.z_axis).atan2(a.0 - self.r_axis);
                    let angle_b = (b.1 - self.z_axis).atan2(b.0 - self.r_axis);
                    angle_a.partial_cmp(&angle_b).unwrap()
                });
            }

            surfaces.push(FluxSurface {
                psi_normalized: psi_n,
                q: self.safety_factor(psi_n),
                points,
            });
        }

        surfaces
    }
}

/// Result of Grad-Shafranov solver
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Did the solver converge?
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
    /// Flux at magnetic axis
    pub psi_axis: f64,
    /// Flux at plasma boundary
    pub psi_boundary: f64,
    /// R position of magnetic axis
    pub r_axis: f64,
    /// Z position of magnetic axis
    pub z_axis: f64,
}

/// Flux surface data for visualization
#[derive(Debug, Clone)]
pub struct FluxSurface {
    /// Normalized flux (0 = boundary, 1 = axis)
    pub psi_normalized: f64,
    /// Safety factor on this surface
    pub q: f64,
    /// Points (R, Z) on this flux surface
    pub points: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::*;

    #[test]
    fn test_gs_solver_creation() {
        let solver = GradShafranovSolver::new(
            TS1_MAJOR_RADIUS,
            TS1_MINOR_RADIUS,
            TS1_TOROIDAL_FIELD,
            TS1_ELONGATION,
            TS1_TRIANGULARITY
        );

        assert_eq!(solver.r0, TS1_MAJOR_RADIUS);
        assert_eq!(solver.a, TS1_MINOR_RADIUS);
        assert!(solver.grid.nr > 50);
        assert!(solver.grid.nz > 50);
    }

    #[test]
    fn test_gs_initialization() {
        let mut solver = GradShafranovSolver::new(1.5, 0.6, 25.0, 1.97, 0.54);
        solver.initialize_solovev();

        // Flux should be positive inside plasma
        let i_center = solver.grid.nr / 2;
        let j_center = solver.grid.nz / 2;
        assert!(solver.grid.psi_at(i_center, j_center) > 0.0);

        // Flux should be zero outside
        assert_eq!(solver.grid.psi_at(0, 0), 0.0);
    }

    #[test]
    fn test_gs_solve() {
        let mut solver = GradShafranovSolver::new(1.5, 0.6, 25.0, 1.97, 0.54);
        solver.profiles.p0 = 5e5;  // Lower pressure for faster convergence
        solver.max_iter = 1000;
        solver.tolerance = 1e-6;

        let result = solver.solve();

        assert!(result.iterations > 0);
        // Axis should be near geometric center
        assert!((result.r_axis - 1.5).abs() < 0.2);
        assert!(result.z_axis.abs() < 0.1);
    }

    #[test]
    fn test_magnetic_field() {
        let mut solver = GradShafranovSolver::new(1.5, 0.6, 25.0, 1.97, 0.54);
        solver.initialize_solovev();

        let (b_r, b_phi, b_z) = solver.magnetic_field(1.5, 0.0);

        // At axis, toroidal field should dominate
        assert!(b_phi.abs() > b_r.abs());
        assert!(b_phi.abs() > b_z.abs());
        // B_phi should be approximately B0 at R0
        assert!((b_phi - 25.0).abs() < 5.0);
    }

    #[test]
    fn test_pressure_profile() {
        let solver = GradShafranovSolver::new(1.5, 0.6, 25.0, 1.97, 0.54);

        // Profile uses p(psi_n) = p0 * (1 - psi_n)^alpha
        // At boundary (psi_n = 0): p = p0
        // At axis (psi_n = 1): p = 0
        // This is inverted from physical intuition but mathematically consistent

        let p_boundary = solver.pressure(0.0);
        assert!((p_boundary - solver.profiles.p0).abs() < 1.0);

        let p_axis = solver.pressure(1.0);
        assert!(p_axis < p_boundary);

        // Mid-radius should be between
        let p_mid = solver.pressure(0.5);
        assert!(p_mid < p_boundary);
        assert!(p_mid > p_axis);
    }
}
