//! # Geometry Module
//!
//! Tokamak geometry definitions and coordinate systems.

use crate::types::{Vec3, TokamakConfig};
use std::f64::consts::PI;

/// Tokamak geometry handler
pub struct TokamakGeometry {
    /// Major radius R₀ (m)
    pub r0: f64,
    /// Minor radius a (m)
    pub a: f64,
    /// Elongation κ
    pub kappa: f64,
    /// Triangularity δ
    pub delta: f64,
}

impl TokamakGeometry {
    /// Create from config
    pub fn from_config(config: &TokamakConfig) -> Self {
        Self {
            r0: config.major_radius,
            a: config.minor_radius,
            kappa: config.elongation,
            delta: config.triangularity,
        }
    }

    /// Convert toroidal coordinates (R, φ, Z) to Cartesian (x, y, z)
    pub fn toroidal_to_cartesian(&self, r: f64, phi: f64, z: f64) -> Vec3 {
        Vec3::new(r * phi.cos(), r * phi.sin(), z)
    }

    /// Convert Cartesian (x, y, z) to toroidal coordinates (R, φ, Z)
    pub fn cartesian_to_toroidal(&self, pos: &Vec3) -> (f64, f64, f64) {
        let r = (pos.x * pos.x + pos.y * pos.y).sqrt();
        let phi = pos.y.atan2(pos.x);
        (r, phi, pos.z)
    }

    /// Get flux surface at normalized radius ρ and poloidal angle θ
    ///
    /// Uses D-shaped parametrization with Shafranov shift
    pub fn flux_surface(&self, rho: f64, theta: f64) -> (f64, f64) {
        // Shafranov shift (simplified)
        let delta_sh = 0.1 * rho * rho * self.a;

        // D-shaped cross-section
        let r = self.r0 + delta_sh + rho * self.a * (theta.cos() + self.delta * (2.0 * theta).cos());
        let z = rho * self.a * self.kappa * theta.sin();

        (r, z)
    }

    /// Check if point is inside the plasma
    pub fn is_inside_plasma(&self, r: f64, z: f64) -> bool {
        let dr = r - self.r0;
        let rho_sq = (dr / self.a).powi(2) + (z / (self.kappa * self.a)).powi(2);
        rho_sq <= 1.0
    }

    /// Get normalized radius ρ at point (R, Z)
    pub fn normalized_radius(&self, r: f64, z: f64) -> f64 {
        let dr = r - self.r0;
        ((dr / self.a).powi(2) + (z / (self.kappa * self.a)).powi(2)).sqrt()
    }

    /// Calculate plasma volume
    pub fn plasma_volume(&self) -> f64 {
        2.0 * PI * PI * self.r0 * self.a * self.a * self.kappa
    }

    /// Calculate plasma surface area
    pub fn plasma_surface(&self) -> f64 {
        // Approximate for elongated plasma
        4.0 * PI * PI * self.r0 * self.a * ((1.0 + self.kappa * self.kappa) / 2.0).sqrt()
    }

    /// Get boundary points for plotting (R, Z pairs)
    pub fn boundary_points(&self, n_points: usize) -> Vec<(f64, f64)> {
        (0..n_points)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (n_points as f64);
                self.flux_surface(1.0, theta)
            })
            .collect()
    }
}

/// Grid in tokamak coordinates
pub struct TokamakGrid {
    /// Number of radial points
    pub nr: usize,
    /// Number of poloidal points
    pub ntheta: usize,
    /// Number of toroidal points
    pub nphi: usize,
    /// Radial coordinate (normalized)
    pub rho: Vec<f64>,
    /// Poloidal angle
    pub theta: Vec<f64>,
    /// Toroidal angle
    pub phi: Vec<f64>,
}

impl TokamakGrid {
    /// Create uniform grid
    pub fn new(nr: usize, ntheta: usize, nphi: usize) -> Self {
        let rho: Vec<f64> = (0..nr).map(|i| (i as f64 + 0.5) / nr as f64).collect();
        let theta: Vec<f64> = (0..ntheta).map(|i| 2.0 * PI * i as f64 / ntheta as f64).collect();
        let phi: Vec<f64> = (0..nphi).map(|i| 2.0 * PI * i as f64 / nphi as f64).collect();

        Self { nr, ntheta, nphi, rho, theta, phi }
    }

    /// Get 3D index
    pub fn idx(&self, ir: usize, itheta: usize, iphi: usize) -> usize {
        ir + itheta * self.nr + iphi * self.nr * self.ntheta
    }

    /// Total number of grid points
    pub fn total_points(&self) -> usize {
        self.nr * self.ntheta * self.nphi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokamakConfig;

    #[test]
    fn test_geometry() {
        let config = TokamakConfig::ts1();
        let geom = TokamakGeometry::from_config(&config);

        // Check volume is reasonable
        let vol = geom.plasma_volume();
        assert!(vol > 0.0);
        assert!(vol < 100.0);  // Should be ~10 m³ for TS-1

        // Check center is inside
        assert!(geom.is_inside_plasma(geom.r0, 0.0));

        // Check outside is outside
        assert!(!geom.is_inside_plasma(geom.r0 + 2.0 * geom.a, 0.0));
    }

    #[test]
    fn test_coordinate_conversion() {
        let config = TokamakConfig::ts1();
        let geom = TokamakGeometry::from_config(&config);

        let pos = geom.toroidal_to_cartesian(1.5, 0.0, 0.0);
        assert!((pos.x - 1.5).abs() < 1e-10);
        assert!(pos.y.abs() < 1e-10);
        assert!(pos.z.abs() < 1e-10);
    }
}
