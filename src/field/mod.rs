//! # Field Module
//!
//! Electromagnetic field solver using Finite Difference Time Domain (FDTD).
//!
//! ## Maxwell's Equations
//!
//! ∇×E = -∂B/∂t
//! ∇×B = μ₀J + μ₀ε₀∂E/∂t
//! ∇·E = ρ/ε₀
//! ∇·B = 0

use crate::types::Vec3;
use crate::constants::*;

/// 3D grid for electromagnetic fields
pub struct FieldGrid {
    /// Number of cells in each dimension
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell size (m)
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Electric field components (staggered grid - Yee cell)
    pub ex: Vec<f64>,
    pub ey: Vec<f64>,
    pub ez: Vec<f64>,
    /// Magnetic field components (staggered grid)
    pub bx: Vec<f64>,
    pub by: Vec<f64>,
    pub bz: Vec<f64>,
    /// Charge density
    pub rho: Vec<f64>,
    /// Current density
    pub jx: Vec<f64>,
    pub jy: Vec<f64>,
    pub jz: Vec<f64>,
}

impl FieldGrid {
    /// Create new field grid
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        let size = nx * ny * nz;
        Self {
            nx, ny, nz,
            dx, dy, dz,
            ex: vec![0.0; size],
            ey: vec![0.0; size],
            ez: vec![0.0; size],
            bx: vec![0.0; size],
            by: vec![0.0; size],
            bz: vec![0.0; size],
            rho: vec![0.0; size],
            jx: vec![0.0; size],
            jy: vec![0.0; size],
            jz: vec![0.0; size],
        }
    }

    /// Create cubic grid
    pub fn cubic(n: usize, cell_size: f64) -> Self {
        Self::new(n, n, n, cell_size, cell_size, cell_size)
    }

    /// Linear index from 3D coordinates
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * self.nx + k * self.nx * self.ny
    }

    /// Get E field at position
    pub fn e_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let idx = self.idx(i, j, k);
        Vec3::new(self.ex[idx], self.ey[idx], self.ez[idx])
    }

    /// Get B field at position
    pub fn b_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let idx = self.idx(i, j, k);
        Vec3::new(self.bx[idx], self.by[idx], self.bz[idx])
    }

    /// Set uniform toroidal magnetic field (for tokamak)
    pub fn set_toroidal_field(&mut self, b0: f64, r0: f64) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    // Compute position relative to torus center
                    let x = (i as f64 - self.nx as f64 / 2.0) * self.dx;
                    let y = (j as f64 - self.ny as f64 / 2.0) * self.dy;
                    let r = (x * x + y * y).sqrt();

                    // Toroidal field: B_φ = B0 * R0 / R
                    let b_phi = if r > 0.1 * r0 {
                        b0 * r0 / r
                    } else {
                        b0 * r0 / (0.1 * r0)
                    };

                    // Convert to Cartesian
                    let phi = y.atan2(x);
                    let idx = self.idx(i, j, k);
                    self.bx[idx] = -b_phi * phi.sin();
                    self.by[idx] = b_phi * phi.cos();
                    // bz stays 0 for pure toroidal field
                }
            }
        }
    }

    /// Clear current density (call before particle deposition)
    pub fn clear_current(&mut self) {
        self.jx.fill(0.0);
        self.jy.fill(0.0);
        self.jz.fill(0.0);
        self.rho.fill(0.0);
    }

    /// Total field energy
    pub fn total_energy(&self) -> f64 {
        let mut e_energy = 0.0;
        let mut b_energy = 0.0;

        for i in 0..self.ex.len() {
            e_energy += self.ex[i] * self.ex[i] + self.ey[i] * self.ey[i] + self.ez[i] * self.ez[i];
            b_energy += self.bx[i] * self.bx[i] + self.by[i] * self.by[i] + self.bz[i] * self.bz[i];
        }

        let volume = self.dx * self.dy * self.dz;
        0.5 * EPSILON_0 * e_energy * volume + 0.5 / MU_0 * b_energy * volume
    }
}

/// FDTD solver for Maxwell's equations
pub struct FDTDSolver {
    /// Timestep (s)
    pub dt: f64,
    /// CFL factor (should be < 1)
    pub cfl: f64,
}

impl FDTDSolver {
    /// Create new FDTD solver
    pub fn new(grid: &FieldGrid) -> Self {
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = 0.5;
        let dt = cfl * min_dx / C;  // CFL condition

        Self { dt, cfl }
    }

    /// Advance E field by half timestep (leapfrog)
    pub fn advance_e(&self, grid: &mut FieldGrid) {
        let dtx = self.dt / grid.dx;
        let dty = self.dt / grid.dy;
        let dtz = self.dt / grid.dz;
        let dt_eps = self.dt / EPSILON_0;

        for k in 1..grid.nz-1 {
            for j in 1..grid.ny-1 {
                for i in 1..grid.nx-1 {
                    let idx = grid.idx(i, j, k);

                    // ∂Ex/∂t = (1/ε₀)(∂Bz/∂y - ∂By/∂z - Jx)
                    let dbz_dy = (grid.bz[grid.idx(i, j+1, k)] - grid.bz[grid.idx(i, j-1, k)]) / (2.0 * grid.dy);
                    let dby_dz = (grid.by[grid.idx(i, j, k+1)] - grid.by[grid.idx(i, j, k-1)]) / (2.0 * grid.dz);
                    grid.ex[idx] += dt_eps * (dbz_dy / MU_0 - dby_dz / MU_0 - grid.jx[idx]);

                    // ∂Ey/∂t = (1/ε₀)(∂Bx/∂z - ∂Bz/∂x - Jy)
                    let dbx_dz = (grid.bx[grid.idx(i, j, k+1)] - grid.bx[grid.idx(i, j, k-1)]) / (2.0 * grid.dz);
                    let dbz_dx = (grid.bz[grid.idx(i+1, j, k)] - grid.bz[grid.idx(i-1, j, k)]) / (2.0 * grid.dx);
                    grid.ey[idx] += dt_eps * (dbx_dz / MU_0 - dbz_dx / MU_0 - grid.jy[idx]);

                    // ∂Ez/∂t = (1/ε₀)(∂By/∂x - ∂Bx/∂y - Jz)
                    let dby_dx = (grid.by[grid.idx(i+1, j, k)] - grid.by[grid.idx(i-1, j, k)]) / (2.0 * grid.dx);
                    let dbx_dy = (grid.bx[grid.idx(i, j+1, k)] - grid.bx[grid.idx(i, j-1, k)]) / (2.0 * grid.dy);
                    grid.ez[idx] += dt_eps * (dby_dx / MU_0 - dbx_dy / MU_0 - grid.jz[idx]);
                }
            }
        }
    }

    /// Advance B field by half timestep (leapfrog)
    pub fn advance_b(&self, grid: &mut FieldGrid) {
        for k in 1..grid.nz-1 {
            for j in 1..grid.ny-1 {
                for i in 1..grid.nx-1 {
                    let idx = grid.idx(i, j, k);

                    // ∂Bx/∂t = -(∂Ez/∂y - ∂Ey/∂z)
                    let dez_dy = (grid.ez[grid.idx(i, j+1, k)] - grid.ez[grid.idx(i, j-1, k)]) / (2.0 * grid.dy);
                    let dey_dz = (grid.ey[grid.idx(i, j, k+1)] - grid.ey[grid.idx(i, j, k-1)]) / (2.0 * grid.dz);
                    grid.bx[idx] -= self.dt * (dez_dy - dey_dz);

                    // ∂By/∂t = -(∂Ex/∂z - ∂Ez/∂x)
                    let dex_dz = (grid.ex[grid.idx(i, j, k+1)] - grid.ex[grid.idx(i, j, k-1)]) / (2.0 * grid.dz);
                    let dez_dx = (grid.ez[grid.idx(i+1, j, k)] - grid.ez[grid.idx(i-1, j, k)]) / (2.0 * grid.dx);
                    grid.by[idx] -= self.dt * (dex_dz - dez_dx);

                    // ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
                    let dey_dx = (grid.ey[grid.idx(i+1, j, k)] - grid.ey[grid.idx(i-1, j, k)]) / (2.0 * grid.dx);
                    let dex_dy = (grid.ex[grid.idx(i, j+1, k)] - grid.ex[grid.idx(i, j-1, k)]) / (2.0 * grid.dy);
                    grid.bz[idx] -= self.dt * (dey_dx - dex_dy);
                }
            }
        }
    }

    /// Full timestep
    pub fn step(&self, grid: &mut FieldGrid) {
        self.advance_b(grid);
        self.advance_e(grid);
    }
}

/// Poisson solver for electrostatic potential
pub struct PoissonSolver {
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iter: usize,
}

impl PoissonSolver {
    /// Create new Poisson solver
    pub fn new() -> Self {
        Self {
            tolerance: POISSON_TOLERANCE,
            max_iter: POISSON_MAX_ITER,
        }
    }

    /// Solve ∇²φ = -ρ/ε₀ using Gauss-Seidel iteration
    pub fn solve(&self, grid: &FieldGrid, phi: &mut [f64]) -> (usize, f64) {
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;
        let coeff = 2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2);

        let mut max_residual = f64::MAX;
        let mut iterations = 0;

        while max_residual > self.tolerance && iterations < self.max_iter {
            max_residual = 0.0;

            for k in 1..grid.nz-1 {
                for j in 1..grid.ny-1 {
                    for i in 1..grid.nx-1 {
                        let idx = grid.idx(i, j, k);

                        let phi_new = (
                            (phi[grid.idx(i+1, j, k)] + phi[grid.idx(i-1, j, k)]) / dx2 +
                            (phi[grid.idx(i, j+1, k)] + phi[grid.idx(i, j-1, k)]) / dy2 +
                            (phi[grid.idx(i, j, k+1)] + phi[grid.idx(i, j, k-1)]) / dz2 +
                            grid.rho[idx] / EPSILON_0
                        ) / coeff;

                        let residual = (phi_new - phi[idx]).abs();
                        max_residual = max_residual.max(residual);

                        phi[idx] = phi_new;
                    }
                }
            }

            iterations += 1;
        }

        (iterations, max_residual)
    }
}

impl Default for PoissonSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_grid_creation() {
        let grid = FieldGrid::cubic(10, 0.01);
        assert_eq!(grid.nx, 10);
        assert_eq!(grid.ex.len(), 1000);
    }

    #[test]
    fn test_toroidal_field() {
        let mut grid = FieldGrid::cubic(20, 0.1);
        grid.set_toroidal_field(1.0, 1.0);

        // Check that field is non-zero
        let center_idx = grid.idx(10, 15, 10);  // Off-center
        let b = (grid.bx[center_idx].powi(2) + grid.by[center_idx].powi(2)).sqrt();
        assert!(b > 0.0);
    }

    #[test]
    fn test_fdtd_stability() {
        let grid = FieldGrid::cubic(10, 0.01);
        let solver = FDTDSolver::new(&grid);

        // CFL should be satisfied
        let dt_max = 0.5 * grid.dx / C;
        assert!(solver.dt <= dt_max);
    }
}
