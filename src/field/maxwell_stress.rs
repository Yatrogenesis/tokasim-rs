//! # Maxwell Stress Tensor
//!
//! Electromagnetic stress tensor for Fluid-Structure Interaction (FSI) coupling.
//!
//! ## Mathematical Foundation
//!
//! The Maxwell stress tensor describes the mechanical stress in the electromagnetic field:
//!
//! ```text
//! T_ij = ε₀(E_iE_j - ½δ_ij|E|²) + (1/μ₀)(B_iB_j - ½δ_ij|B|²)
//! ```
//!
//! In component form:
//! ```text
//! T_xx = ε₀(E_x² - E²/2) + (B_x² - B²/2)/μ₀
//! T_xy = ε₀E_xE_y + B_xB_y/μ₀
//! ...etc
//! ```
//!
//! ## Physical Interpretation
//!
//! - Diagonal components: Pressure + tension along field lines
//! - Off-diagonal components: Shear stress
//! - The force per unit area on a surface is: f_i = T_ij · n_j
//!
//! ## Applications in Tokamaks
//!
//! 1. **Halo currents**: During disruptions, large currents flow through the vessel
//! 2. **Vertical displacement events (VDE)**: Asymmetric forces during plasma motion
//! 3. **Magnetic pressure**: B²/(2μ₀) at first wall during normal operation
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] Jackson, J.D. "Classical Electrodynamics", 3rd ed., Chapter 6
//! [2] ITER Structural Design Criteria (SDC-IC), Section 4.3

use crate::types::Vec3;
use crate::constants::{EPSILON_0, MU_0};

/// 3x3 Maxwell stress tensor at a point
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxwellStressTensor {
    /// Tensor components T[i][j]
    pub t: [[f64; 3]; 3],
}

impl MaxwellStressTensor {
    /// Create zero tensor
    pub fn zero() -> Self {
        Self { t: [[0.0; 3]; 3] }
    }

    /// Calculate Maxwell stress tensor from E and B fields
    ///
    /// ## Arguments
    /// * `e` - Electric field vector [V/m]
    /// * `b` - Magnetic field vector [T]
    ///
    /// ## Returns
    /// Maxwell stress tensor [Pa]
    ///
    /// ## Formula
    /// T_ij = ε₀(E_iE_j - ½δ_ij|E|²) + (1/μ₀)(B_iB_j - ½δ_ij|B|²)
    pub fn from_fields(e: &Vec3, b: &Vec3) -> Self {
        let e_arr = [e.x, e.y, e.z];
        let b_arr = [b.x, b.y, b.z];

        let e_sq = e.mag_squared();
        let b_sq = b.mag_squared();

        let mut t = [[0.0; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                let delta_ij = if i == j { 1.0 } else { 0.0 };

                // Electric field contribution
                let t_e = EPSILON_0 * (e_arr[i] * e_arr[j] - 0.5 * delta_ij * e_sq);

                // Magnetic field contribution
                let t_b = (1.0 / MU_0) * (b_arr[i] * b_arr[j] - 0.5 * delta_ij * b_sq);

                t[i][j] = t_e + t_b;
            }
        }

        Self { t }
    }

    /// Calculate from magnetic field only (electrostatic limit)
    ///
    /// For tokamaks at equilibrium, |E| << c|B|, so magnetic term dominates
    ///
    /// T_ij = (1/μ₀)(B_iB_j - ½δ_ij|B|²)
    pub fn from_magnetic_field(b: &Vec3) -> Self {
        Self::from_fields(&Vec3::zero(), b)
    }

    /// Get diagonal component (pressure/tension along axis i)
    ///
    /// Returns T[i][i] in Pa
    pub fn diagonal(&self, i: usize) -> f64 {
        self.t[i][i]
    }

    /// Get off-diagonal component (shear stress)
    pub fn shear(&self, i: usize, j: usize) -> f64 {
        self.t[i][j]
    }

    /// Calculate magnetic pressure
    ///
    /// p_mag = B²/(2μ₀)
    ///
    /// This is the isotropic part of the magnetic stress
    pub fn magnetic_pressure(b: &Vec3) -> f64 {
        b.mag_squared() / (2.0 * MU_0)
    }

    /// Calculate magnetic tension
    ///
    /// The tension along field lines is B²/μ₀
    pub fn magnetic_tension(b: &Vec3) -> f64 {
        b.mag_squared() / MU_0
    }

    /// Calculate force per unit area on surface with normal n
    ///
    /// ## Arguments
    /// * `n` - Unit normal vector of surface (pointing outward)
    ///
    /// ## Returns
    /// Force per unit area vector [N/m²] = [Pa]
    ///
    /// ## Formula
    /// f_i = Σⱼ T_ij · n_j
    pub fn force_on_surface(&self, n: &Vec3) -> Vec3 {
        let n_arr = [n.x, n.y, n.z];

        let fx = self.t[0][0] * n_arr[0] + self.t[0][1] * n_arr[1] + self.t[0][2] * n_arr[2];
        let fy = self.t[1][0] * n_arr[0] + self.t[1][1] * n_arr[1] + self.t[1][2] * n_arr[2];
        let fz = self.t[2][0] * n_arr[0] + self.t[2][1] * n_arr[1] + self.t[2][2] * n_arr[2];

        Vec3::new(fx, fy, fz)
    }

    /// Calculate trace (invariant)
    ///
    /// For Maxwell tensor: Tr(T) = -ε₀|E|²/2 - |B|²/(2μ₀)
    pub fn trace(&self) -> f64 {
        self.t[0][0] + self.t[1][1] + self.t[2][2]
    }

    /// Calculate von Mises equivalent stress
    ///
    /// σ_vm = √(3/2 · s_ij · s_ij)
    ///
    /// where s_ij is the deviatoric stress tensor
    pub fn von_mises(&self) -> f64 {
        // Deviatoric stress: s_ij = T_ij - (1/3)δ_ij·Tr(T)
        let p = self.trace() / 3.0;

        let mut s_sq = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let delta_ij = if i == j { 1.0 } else { 0.0 };
                let s_ij = self.t[i][j] - p * delta_ij;
                s_sq += s_ij * s_ij;
            }
        }

        (1.5 * s_sq).sqrt()
    }

    /// Calculate principal stresses (eigenvalues)
    ///
    /// Returns (σ₁, σ₂, σ₃) in descending order
    pub fn principal_stresses(&self) -> (f64, f64, f64) {
        // For 3x3 symmetric matrix, use analytical formula
        // Based on Cardano's formula

        let i1 = self.trace();
        let i2 = self.t[0][0] * self.t[1][1] + self.t[1][1] * self.t[2][2] + self.t[2][2] * self.t[0][0]
            - self.t[0][1].powi(2) - self.t[1][2].powi(2) - self.t[2][0].powi(2);
        let i3 = self.t[0][0] * self.t[1][1] * self.t[2][2]
            + 2.0 * self.t[0][1] * self.t[1][2] * self.t[2][0]
            - self.t[0][0] * self.t[1][2].powi(2)
            - self.t[1][1] * self.t[2][0].powi(2)
            - self.t[2][2] * self.t[0][1].powi(2);

        // Solve characteristic equation: λ³ - I₁λ² + I₂λ - I₃ = 0
        let p = i2 - i1.powi(2) / 3.0;
        let q = i3 - i1 * i2 / 3.0 + 2.0 * i1.powi(3) / 27.0;

        // Discriminant
        let discriminant = (p / 3.0).powi(3) + (q / 2.0).powi(2);

        if discriminant < 0.0 {
            // Three real roots (trigonometric solution)
            let r = (-p / 3.0).sqrt();
            let phi = ((-q / 2.0) / r.powi(3)).acos();

            let sigma1 = 2.0 * r * (phi / 3.0).cos() + i1 / 3.0;
            let sigma2 = 2.0 * r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() + i1 / 3.0;
            let sigma3 = 2.0 * r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() + i1 / 3.0;

            // Sort descending
            let mut sigmas = [sigma1, sigma2, sigma3];
            sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap());

            (sigmas[0], sigmas[1], sigmas[2])
        } else {
            // Degenerate case (repeated eigenvalue)
            let sqrt_d = discriminant.sqrt();
            let u = (-q / 2.0 + sqrt_d).cbrt();
            let v = (-q / 2.0 - sqrt_d).cbrt();

            let sigma1 = u + v + i1 / 3.0;
            let sigma23 = -(u + v) / 2.0 + i1 / 3.0;

            if sigma1 > sigma23 {
                (sigma1, sigma23, sigma23)
            } else {
                (sigma23, sigma23, sigma1)
            }
        }
    }

    /// Add two stress tensors
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.t[i][j] = self.t[i][j] + other.t[i][j];
            }
        }
        result
    }

    /// Scale tensor by scalar
    pub fn scale(&self, s: f64) -> Self {
        let mut result = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                result.t[i][j] = self.t[i][j] * s;
            }
        }
        result
    }
}

/// Calculate total electromagnetic force on a volume bounded by surface S
///
/// F = ∮_S T · dA
///
/// ## Arguments
/// * `stress_tensor` - Maxwell stress tensor at each surface element
/// * `normals` - Outward normal vectors
/// * `areas` - Surface element areas
///
/// ## Returns
/// Total force vector [N]
pub fn total_em_force(
    stress_tensors: &[MaxwellStressTensor],
    normals: &[Vec3],
    areas: &[f64],
) -> Vec3 {
    let n = stress_tensors.len();
    assert_eq!(n, normals.len());
    assert_eq!(n, areas.len());

    let mut total_force = Vec3::zero();

    for i in 0..n {
        let f_density = stress_tensors[i].force_on_surface(&normals[i]);
        total_force = total_force + f_density * areas[i];
    }

    total_force
}

/// Calculate halo current force on vacuum vessel during VDE
///
/// ## Arguments
/// * `i_halo` - Halo current [A]
/// * `b_t` - Toroidal magnetic field [T]
/// * `r` - Radial position [m]
/// * `path_length` - Current path length [m]
///
/// ## Returns
/// Force per unit length [N/m]
///
/// ## Formula
/// F/L = I × B (Lorentz force)
pub fn halo_current_force(i_halo: f64, b_t: f64, _r: f64, _path_length: f64) -> f64 {
    i_halo * b_t
}

/// Calculate electromagnetic pressure on first wall during normal operation
///
/// ## Arguments
/// * `b_pol` - Poloidal field at wall [T]
/// * `b_tor` - Toroidal field at wall [T]
///
/// ## Returns
/// Magnetic pressure [Pa]
pub fn first_wall_em_pressure(b_pol: f64, b_tor: f64) -> f64 {
    let b_total_sq = b_pol * b_pol + b_tor * b_tor;
    b_total_sq / (2.0 * MU_0)
}

/// Calculate vertical force during VDE
///
/// During a vertical displacement event, the plasma column moves vertically
/// and induces large currents in the vessel, creating asymmetric forces.
///
/// ## Arguments
/// * `i_p` - Plasma current [A]
/// * `b_r` - Radial field at plasma edge [T]
/// * `circumference` - Plasma circumference [m]
///
/// ## Returns
/// Vertical force [N]
///
/// ## Formula
/// F_z = I_p × B_r × L (integrated Lorentz force)
pub fn vde_vertical_force(i_p: f64, b_r: f64, circumference: f64) -> f64 {
    i_p * b_r * circumference
}

/// Calculate disruption loads on vacuum vessel
///
/// ## Arguments
/// * `stored_energy` - Plasma stored energy [J]
/// * `tau_quench` - Current quench time [s]
/// * `l_vessel` - Vessel inductance [H]
///
/// ## Returns
/// * `i_eddy` - Peak eddy current [A]
/// * `p_dissipated` - Power dissipated [W]
pub fn disruption_loads(stored_energy: f64, tau_quench: f64, l_vessel: f64) -> (f64, f64) {
    // Peak eddy current from flux conservation
    // I_eddy ≈ W_mag / (L_vessel)^0.5 * (τ_quench)^-0.5
    let i_eddy = (2.0 * stored_energy / l_vessel).sqrt() / (tau_quench / l_vessel).sqrt();

    // Power dissipated in vessel
    let p_dissipated = stored_energy / tau_quench;

    (i_eddy, p_dissipated)
}

/// Stress tensor for tokamak plasma equilibrium
///
/// In toroidal coordinates (R, φ, Z), the equilibrium stress tensor has
/// special structure due to axisymmetry.
pub struct ToroidalStressTensor {
    /// Magnetic pressure B²/(2μ₀) [Pa]
    pub p_mag: f64,
    /// Plasma kinetic pressure [Pa]
    pub p_plasma: f64,
    /// Radial field component [T]
    pub b_r: f64,
    /// Toroidal field component [T]
    pub b_phi: f64,
    /// Vertical field component [T]
    pub b_z: f64,
}

impl ToroidalStressTensor {
    /// Create from field components and plasma pressure
    pub fn new(b_r: f64, b_phi: f64, b_z: f64, p_plasma: f64) -> Self {
        let b_sq = b_r * b_r + b_phi * b_phi + b_z * b_z;
        Self {
            p_mag: b_sq / (2.0 * MU_0),
            p_plasma,
            b_r,
            b_phi,
            b_z,
        }
    }

    /// Calculate total pressure (magnetic + kinetic)
    pub fn total_pressure(&self) -> f64 {
        self.p_mag + self.p_plasma
    }

    /// Calculate beta (plasma pressure / magnetic pressure)
    pub fn beta(&self) -> f64 {
        if self.p_mag > 0.0 {
            self.p_plasma / self.p_mag
        } else {
            0.0
        }
    }

    /// Convert to Cartesian Maxwell tensor at position (R, phi)
    pub fn to_cartesian(&self, phi: f64) -> MaxwellStressTensor {
        // Transform B from cylindrical to Cartesian
        let b_x = self.b_r * phi.cos() - self.b_phi * phi.sin();
        let b_y = self.b_r * phi.sin() + self.b_phi * phi.cos();
        let b_z = self.b_z;

        let b = Vec3::new(b_x, b_y, b_z);
        MaxwellStressTensor::from_magnetic_field(&b)
    }

    /// Calculate Grad-Shafranov force balance
    ///
    /// For equilibrium: J × B = ∇p
    ///
    /// Returns the residual (should be 0 at equilibrium)
    pub fn force_balance_residual(&self, dp_dr: f64, j_phi: f64) -> f64 {
        // J × B in radial direction
        let j_cross_b = j_phi * self.b_z;

        // Should equal dp/dR
        (j_cross_b - dp_dr).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxwell_tensor_symmetric() {
        let b = Vec3::new(1.0, 2.0, 3.0);
        let tensor = MaxwellStressTensor::from_magnetic_field(&b);

        // Maxwell tensor should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((tensor.t[i][j] - tensor.t[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_magnetic_pressure() {
        let b = Vec3::new(5.0, 0.0, 0.0);  // 5 T field

        let p_mag = MaxwellStressTensor::magnetic_pressure(&b);

        // B²/(2μ₀) = 25 / (2 * 4π × 10⁻⁷) ≈ 9.95 × 10⁶ Pa
        let expected = 25.0 / (2.0 * MU_0);
        assert!((p_mag - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_force_on_surface() {
        // Uniform B field in z direction
        let b = Vec3::new(0.0, 0.0, 1.0);
        let tensor = MaxwellStressTensor::from_magnetic_field(&b);

        // Force on surface with normal in z direction
        let n = Vec3::new(0.0, 0.0, 1.0);
        let f = tensor.force_on_surface(&n);

        // Should be magnetic tension: B²/μ₀ - B²/(2μ₀) = B²/(2μ₀) in z direction
        let expected_fz = 1.0 / (2.0 * MU_0);
        assert!((f.z - expected_fz).abs() / expected_fz < 1e-10);
    }

    #[test]
    fn test_principal_stresses() {
        // Simple diagonal tensor
        let mut tensor = MaxwellStressTensor::zero();
        tensor.t[0][0] = 3.0;
        tensor.t[1][1] = 2.0;
        tensor.t[2][2] = 1.0;

        let (s1, s2, s3) = tensor.principal_stresses();

        assert!((s1 - 3.0).abs() < 1e-10);
        assert!((s2 - 2.0).abs() < 1e-10);
        assert!((s3 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let b = Vec3::new(1.0, 1.0, 1.0);
        let tensor = MaxwellStressTensor::from_magnetic_field(&b);

        // For pure magnetic field: Tr(T) = -B²/(2μ₀) = -3/(2μ₀)
        let expected_trace = -3.0 / (2.0 * MU_0);
        assert!((tensor.trace() - expected_trace).abs() / expected_trace.abs() < 1e-10);
    }

    #[test]
    fn test_first_wall_pressure() {
        // Typical ITER values: B_pol ≈ 1 T, B_tor ≈ 5 T
        let p = first_wall_em_pressure(1.0, 5.0);

        // Should be ~10 MPa
        assert!(p > 1e6 && p < 1e8, "Expected ~10 MPa, got {} MPa", p / 1e6);
    }

    #[test]
    fn test_halo_current_force() {
        // Typical disruption: I_halo ~ 1 MA, B_t ~ 5 T
        let f_per_length = halo_current_force(1e6, 5.0, 1.5, 10.0);

        // F/L = I × B = 1e6 × 5 = 5 MN/m
        assert!((f_per_length - 5e6).abs() / 5e6 < 1e-10);
    }

    #[test]
    fn test_toroidal_stress() {
        let stress = ToroidalStressTensor::new(0.1, 5.0, 0.5, 1e6);

        // Check beta calculation
        let beta = stress.beta();
        assert!(beta > 0.0 && beta < 1.0);

        // Check pressure sum
        let p_total = stress.total_pressure();
        assert!(p_total > stress.p_mag);
    }
}
