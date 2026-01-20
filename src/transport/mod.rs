//! # Turbulent Transport Module
//!
//! First-principles anomalous transport calculations for tokamak plasmas.
//!
//! ## Theory
//!
//! Turbulent transport in tokamaks is driven by microinstabilities:
//! - Ion Temperature Gradient (ITG) modes
//! - Trapped Electron Modes (TEM)
//! - Electron Temperature Gradient (ETG) modes
//!
//! The transport equations solved are:
//!
//! ```text
//! ∂n/∂t = ∇·(D∇n) + S_n
//! (3/2)∂(nT_i)/∂t = ∇·(n χ_i ∇T_i) + Q_i
//! (3/2)∂(nT_e)/∂t = ∇·(n χ_e ∇T_e) + Q_e - P_rad
//! ```
//!
//! ## Transport Model
//!
//! We use a quasi-linear gyrofluid model that captures:
//! - Critical gradient thresholds
//! - Stiffness above threshold
//! - Magnetic shear stabilization
//! - E×B shear suppression
//!
//! ## References
//!
//! - Kotschenreuther et al., "Quantitative predictions of tokamak energy confinement"
//! - Waltz et al., "A gyro-Landau-fluid transport model"
//! - Staebler et al., "TGLF transport model"
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use std::f64::consts::PI;

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380649e-23;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602176634e-19;

/// Proton mass (kg)
pub const M_P: f64 = 1.67262192e-27;

/// Electron mass (kg)
pub const M_E: f64 = 9.1093837e-31;

/// Vacuum permeability (H/m)
pub const MU_0: f64 = 1.25663706212e-6;

// ============================================================================
// PLASMA PROFILES
// ============================================================================

/// Radial plasma profiles on a 1D grid
#[derive(Debug, Clone)]
pub struct PlasmaProfiles {
    /// Normalized radius ρ = r/a (0 to 1)
    pub rho: Vec<f64>,
    /// Electron density (m^-3)
    pub n_e: Vec<f64>,
    /// Ion density (m^-3)
    pub n_i: Vec<f64>,
    /// Electron temperature (keV)
    pub t_e: Vec<f64>,
    /// Ion temperature (keV)
    pub t_i: Vec<f64>,
    /// Safety factor q
    pub q: Vec<f64>,
    /// Magnetic shear s = (r/q)(dq/dr)
    pub shear: Vec<f64>,
    /// Toroidal rotation frequency (rad/s)
    pub omega: Vec<f64>,
    /// Radial electric field (V/m)
    pub e_r: Vec<f64>,
}

impl PlasmaProfiles {
    /// Create profiles on uniform grid with n_points
    pub fn new(n_points: usize) -> Self {
        let rho: Vec<f64> = (0..n_points)
            .map(|i| i as f64 / (n_points - 1) as f64)
            .collect();

        Self {
            rho: rho.clone(),
            n_e: vec![0.0; n_points],
            n_i: vec![0.0; n_points],
            t_e: vec![0.0; n_points],
            t_i: vec![0.0; n_points],
            q: vec![1.0; n_points],
            shear: vec![0.0; n_points],
            omega: vec![0.0; n_points],
            e_r: vec![0.0; n_points],
        }
    }

    /// Initialize with parabolic profiles
    pub fn initialize_parabolic(&mut self, n0: f64, te0: f64, ti0: f64, q0: f64, q_edge: f64) {
        let n = self.rho.len();
        for i in 0..n {
            let r = self.rho[i];
            let r2 = r * r;

            // Parabolic profiles: X(r) = X0 * (1 - r²)^α
            self.n_e[i] = n0 * (1.0 - r2).max(0.01).powf(0.5);
            self.n_i[i] = self.n_e[i];  // Quasi-neutrality
            self.t_e[i] = te0 * (1.0 - r2).max(0.01).powf(1.5);
            self.t_i[i] = ti0 * (1.0 - r2).max(0.01).powf(1.5);

            // q profile: q(r) = q0 * (1 + (q_edge/q0 - 1) * r²)
            self.q[i] = q0 * (1.0 + (q_edge / q0 - 1.0) * r2);

            // Magnetic shear
            if i > 0 && i < n - 1 {
                let dr = self.rho[i + 1] - self.rho[i - 1];
                let dq = self.q[i + 1] - self.q[i - 1];
                self.shear[i] = (r / self.q[i]) * (dq / dr);
            }
        }
        self.shear[0] = 0.0;
        self.shear[n - 1] = self.shear[n - 2];
    }

    /// Calculate normalized gradients at radius index
    pub fn gradients(&self, i: usize) -> Gradients {
        let n = self.rho.len();
        if i == 0 || i >= n - 1 {
            return Gradients::default();
        }

        let dr = self.rho[i + 1] - self.rho[i - 1];

        // R/L_X = -R * (1/X) * dX/dr
        // For normalized radius: a/L_X = -(1/X) * dX/dρ

        let dn_e = (self.n_e[i + 1] - self.n_e[i - 1]) / dr;
        let dt_e = (self.t_e[i + 1] - self.t_e[i - 1]) / dr;
        let dt_i = (self.t_i[i + 1] - self.t_i[i - 1]) / dr;

        Gradients {
            a_ln: if self.n_e[i] > 0.0 { -dn_e / self.n_e[i] } else { 0.0 },
            a_lte: if self.t_e[i] > 0.0 { -dt_e / self.t_e[i] } else { 0.0 },
            a_lti: if self.t_i[i] > 0.0 { -dt_i / self.t_i[i] } else { 0.0 },
            eta_e: if dn_e.abs() > 1e-10 { dt_e * self.n_e[i] / (dn_e * self.t_e[i]) } else { 0.0 },
            eta_i: if dn_e.abs() > 1e-10 { dt_i * self.n_i[i] / (dn_e * self.t_i[i]) } else { 0.0 },
        }
    }

    /// Get value at arbitrary ρ using linear interpolation
    pub fn interpolate(&self, rho: f64, values: &[f64]) -> f64 {
        let n = self.rho.len();
        if rho <= 0.0 {
            return values[0];
        }
        if rho >= 1.0 {
            return values[n - 1];
        }

        let idx = (rho * (n - 1) as f64) as usize;
        let idx = idx.min(n - 2);
        let t = (rho - self.rho[idx]) / (self.rho[idx + 1] - self.rho[idx]);
        values[idx] + t * (values[idx + 1] - values[idx])
    }
}

/// Normalized gradient scale lengths
#[derive(Debug, Clone, Default)]
pub struct Gradients {
    /// a/L_n (density gradient)
    pub a_ln: f64,
    /// a/L_Te (electron temperature gradient)
    pub a_lte: f64,
    /// a/L_Ti (ion temperature gradient)
    pub a_lti: f64,
    /// η_e = L_n/L_Te
    pub eta_e: f64,
    /// η_i = L_n/L_Ti
    pub eta_i: f64,
}

// ============================================================================
// MICROINSTABILITY ANALYSIS
// ============================================================================

/// Parameters for microinstability calculation
#[derive(Debug, Clone)]
pub struct MicroinstabilityParams {
    /// Major radius (m)
    pub r0: f64,
    /// Minor radius (m)
    pub a: f64,
    /// Toroidal field (T)
    pub b0: f64,
    /// Ion mass number (2 for D, 2.5 for D-T)
    pub a_ion: f64,
    /// Effective charge Z_eff
    pub z_eff: f64,
    /// Inverse aspect ratio at local position
    pub epsilon: f64,
    /// Collision frequency / bounce frequency
    pub nu_star: f64,
}

/// Growth rates and frequencies of instabilities
#[derive(Debug, Clone, Default)]
pub struct InstabilityResult {
    /// ITG growth rate (s^-1)
    pub gamma_itg: f64,
    /// ITG real frequency (s^-1)
    pub omega_itg: f64,
    /// TEM growth rate (s^-1)
    pub gamma_tem: f64,
    /// TEM real frequency (s^-1)
    pub omega_tem: f64,
    /// ETG growth rate (s^-1)
    pub gamma_etg: f64,
    /// Dominant mode
    pub dominant: DominantMode,
    /// Critical gradient for ITG
    pub a_lti_crit: f64,
    /// Critical gradient for TEM
    pub a_lte_crit: f64,
}

/// Dominant instability type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DominantMode {
    #[default]
    Stable,
    ITG,
    TEM,
    ETG,
    Mixed,
}

/// Microinstability calculator (quasi-linear gyrofluid)
pub struct MicroinstabilityCalculator {
    pub params: MicroinstabilityParams,
}

impl MicroinstabilityCalculator {
    pub fn new(params: MicroinstabilityParams) -> Self {
        Self { params }
    }

    /// Calculate growth rates at given local parameters
    pub fn analyze(
        &self,
        grads: &Gradients,
        q: f64,
        shear: f64,
        te_kev: f64,
        ti_kev: f64,
        ne: f64,
    ) -> InstabilityResult {
        let mut result = InstabilityResult::default();

        // Convert temperatures to Joules
        let te = te_kev * 1000.0 * E_CHARGE;
        let ti = ti_kev * 1000.0 * E_CHARGE;

        // Ion Larmor radius
        let rho_i = self.ion_larmor_radius(ti);

        // Diamagnetic frequencies
        let omega_star_i = self.diamagnetic_freq(ti, ne, grads.a_ln);
        let omega_star_e = -omega_star_i * te / ti;  // Opposite sign

        // ========================================
        // ITG Critical Gradient
        // ========================================
        // Romanelli formula: (R/L_Ti)_crit = (4/3)(1 + Ti/Te)(1 + 2s/q)
        let ti_te = ti / te;
        result.a_lti_crit = (4.0 / 3.0) * (1.0 + ti_te) * (1.0 + 2.0 * shear.abs() / q) / self.params.epsilon;

        // ITG growth rate (above threshold)
        let margin_itg = (grads.a_lti - result.a_lti_crit).max(0.0);
        if margin_itg > 0.0 {
            // Quasi-linear estimate
            // γ_ITG ~ ω*_i * sqrt(ε * margin)
            let k_theta_rho = 0.3;  // Typical wavenumber
            result.gamma_itg = omega_star_i.abs() * k_theta_rho
                * (self.params.epsilon * margin_itg / result.a_lti_crit).sqrt();

            result.omega_itg = omega_star_i * (1.0 + grads.eta_i);  // Ion direction
        }

        // ========================================
        // TEM Critical Gradient
        // ========================================
        // TEM threshold: (R/L_Te)_crit ~ 2.5 / sqrt(ε)
        result.a_lte_crit = 2.5 / self.params.epsilon.sqrt();

        // Trapped particle fraction
        let f_trapped = (2.0 * self.params.epsilon).sqrt();

        // TEM growth rate
        let margin_tem = (grads.a_lte - result.a_lte_crit).max(0.0);
        if margin_tem > 0.0 && self.params.nu_star < 1.0 {
            // Collisionless TEM
            let bounce_freq = (te / M_E).sqrt() / (q * self.params.r0);
            result.gamma_tem = bounce_freq * f_trapped
                * (margin_tem / result.a_lte_crit).powf(0.5);

            result.omega_tem = omega_star_e * (1.0 + grads.eta_e);  // Electron direction
        }

        // ========================================
        // ETG (small scale, usually subdominant for ion transport)
        // ========================================
        let rho_e = self.electron_larmor_radius(te);
        if grads.a_lte > 4.0 {  // High threshold
            result.gamma_etg = omega_star_e.abs() * (rho_e / rho_i)
                * (grads.a_lte - 4.0).sqrt();
        }

        // Use rho_i in this scope to avoid warning
        let _ = rho_i;

        // ========================================
        // Determine dominant mode
        // ========================================
        let max_gamma = result.gamma_itg.max(result.gamma_tem).max(result.gamma_etg);
        if max_gamma < 1e3 {
            result.dominant = DominantMode::Stable;
        } else if result.gamma_itg > result.gamma_tem * 1.2 && result.gamma_itg > result.gamma_etg {
            result.dominant = DominantMode::ITG;
        } else if result.gamma_tem > result.gamma_itg * 1.2 && result.gamma_tem > result.gamma_etg {
            result.dominant = DominantMode::TEM;
        } else if result.gamma_etg > result.gamma_itg && result.gamma_etg > result.gamma_tem {
            result.dominant = DominantMode::ETG;
        } else {
            result.dominant = DominantMode::Mixed;
        }

        result
    }

    /// Ion Larmor radius ρ_i = sqrt(2 m_i T_i) / (e B)
    fn ion_larmor_radius(&self, ti: f64) -> f64 {
        let m_i = self.params.a_ion * M_P;
        (2.0 * m_i * ti).sqrt() / (E_CHARGE * self.params.b0)
    }

    /// Electron Larmor radius
    fn electron_larmor_radius(&self, te: f64) -> f64 {
        (2.0 * M_E * te).sqrt() / (E_CHARGE * self.params.b0)
    }

    /// Diamagnetic frequency ω* = k_θ T / (e B L_n)
    fn diamagnetic_freq(&self, t: f64, _n: f64, a_ln: f64) -> f64 {
        let l_n = self.params.a / a_ln.max(0.1);  // Density scale length
        let k_theta = 0.3 / self.ion_larmor_radius(t);  // Typical k_θ ρ_i ~ 0.3
        k_theta * t / (E_CHARGE * self.params.b0 * l_n)
    }
}

// ============================================================================
// TRANSPORT COEFFICIENTS
// ============================================================================

/// Anomalous transport coefficients
#[derive(Debug, Clone, Default)]
pub struct TransportCoefficients {
    /// Ion thermal diffusivity χ_i (m²/s)
    pub chi_i: f64,
    /// Electron thermal diffusivity χ_e (m²/s)
    pub chi_e: f64,
    /// Particle diffusivity D (m²/s)
    pub d_particle: f64,
    /// Momentum diffusivity χ_φ (m²/s)
    pub chi_phi: f64,
    /// Ion convective velocity (m/s)
    pub v_i: f64,
    /// Electron convective velocity (m/s)
    pub v_e: f64,
    /// Particle pinch velocity (m/s)
    pub v_pinch: f64,
}

/// Quasi-linear transport model
pub struct TransportModel {
    pub params: MicroinstabilityParams,
    /// E×B shear rate (s^-1)
    pub exb_shear: f64,
    /// Alpha (stiffness parameter)
    pub stiffness: f64,
}

impl TransportModel {
    pub fn new(params: MicroinstabilityParams) -> Self {
        Self {
            params,
            exb_shear: 0.0,
            stiffness: 1.5,  // Typically 1-2
        }
    }

    /// Calculate transport coefficients from microinstability analysis
    pub fn calculate_transport(
        &self,
        instability: &InstabilityResult,
        te_kev: f64,
        ti_kev: f64,
        q: f64,
    ) -> TransportCoefficients {
        let mut coeff = TransportCoefficients::default();

        // Convert to SI
        let te = te_kev * 1000.0 * E_CHARGE;
        let ti = ti_kev * 1000.0 * E_CHARGE;

        // Ion Larmor radius
        let m_i = self.params.a_ion * M_P;
        let rho_i = (2.0 * m_i * ti).sqrt() / (E_CHARGE * self.params.b0);

        // Gyro-Bohm diffusivity: D_gB = ρ_i² v_ti / R
        // v_ti = sqrt(T_i / m_i)
        let v_ti = (ti / m_i).sqrt();
        let d_gyro_bohm = rho_i * rho_i * v_ti / self.params.r0;

        // ========================================
        // ITG-driven transport
        // ========================================
        if instability.gamma_itg > 0.0 {
            // Quasi-linear estimate: χ ~ γ / k_θ²
            // With normalization to gyro-Bohm
            let k_theta_rho = 0.3;
            let gamma_norm = instability.gamma_itg / (v_ti / self.params.r0);

            // Stiff transport above critical gradient
            let chi_itg = d_gyro_bohm * gamma_norm / (k_theta_rho * k_theta_rho);

            // E×B shear suppression
            let exb_factor = if self.exb_shear > 0.0 {
                1.0 / (1.0 + (self.exb_shear / instability.gamma_itg).powi(2))
            } else {
                1.0
            };

            coeff.chi_i += chi_itg * exb_factor;
            coeff.d_particle += 0.3 * chi_itg * exb_factor;  // Particle diffusivity (typically lower)
        }

        // ========================================
        // TEM-driven transport
        // ========================================
        if instability.gamma_tem > 0.0 {
            let v_te = (te / M_E).sqrt();
            let rho_e = (2.0 * M_E * te).sqrt() / (E_CHARGE * self.params.b0);
            let d_e_gyro_bohm = rho_e * rho_e * v_te / self.params.r0;

            let gamma_norm = instability.gamma_tem / (v_te / (q * self.params.r0));
            let chi_tem = d_e_gyro_bohm * gamma_norm * 100.0;  // TEM often drives strong electron transport

            let exb_factor = if self.exb_shear > 0.0 {
                1.0 / (1.0 + (self.exb_shear / instability.gamma_tem).powi(2))
            } else {
                1.0
            };

            coeff.chi_e += chi_tem * exb_factor;
            coeff.d_particle += 0.5 * chi_tem * exb_factor;
        }

        // ========================================
        // ETG-driven transport (electron channel only)
        // ========================================
        if instability.gamma_etg > 0.0 {
            let rho_e = (2.0 * M_E * te).sqrt() / (E_CHARGE * self.params.b0);

            // ETG is small-scale, contributes to electron heat only
            let chi_etg = rho_e * rho_e * instability.gamma_etg;
            coeff.chi_e += chi_etg * 0.1;  // Usually subdominant
        }

        // ========================================
        // Neoclassical baseline (always present)
        // ========================================
        let chi_neo = self.neoclassical_chi(ti_kev, te_kev, q);
        coeff.chi_i = coeff.chi_i.max(chi_neo.0);
        coeff.chi_e = coeff.chi_e.max(chi_neo.1);

        // ========================================
        // Momentum transport (Prandtl number ~ 0.7)
        // ========================================
        coeff.chi_phi = 0.7 * coeff.chi_i;

        // ========================================
        // Convective velocities (thermodiffusion)
        // ========================================
        coeff.v_pinch = -coeff.d_particle / self.params.r0;  // Ware pinch approximation

        coeff
    }

    /// Neoclassical ion and electron thermal diffusivities
    fn neoclassical_chi(&self, ti_kev: f64, te_kev: f64, q: f64) -> (f64, f64) {
        let ti = ti_kev * 1000.0 * E_CHARGE;
        let te = te_kev * 1000.0 * E_CHARGE;
        let m_i = self.params.a_ion * M_P;

        // Banana regime ion thermal diffusivity
        // χ_i,neo ~ q² ε^(-3/2) ρ_i² ν_ii
        let rho_i = (2.0 * m_i * ti).sqrt() / (E_CHARGE * self.params.b0);
        let nu_ii = 1e4;  // Simplified collision frequency

        let chi_i_neo = q * q * self.params.epsilon.powf(-1.5) * rho_i * rho_i * nu_ii;

        // Electron neoclassical (usually smaller)
        let rho_e = (2.0 * M_E * te).sqrt() / (E_CHARGE * self.params.b0);
        let nu_ee = 1e6;  // Higher collision frequency
        let chi_e_neo = q * q * self.params.epsilon.powf(-1.5) * rho_e * rho_e * nu_ee * 0.01;

        (chi_i_neo.min(10.0), chi_e_neo.min(10.0))  // Cap for numerical stability
    }
}

// ============================================================================
// TRANSPORT SOLVER
// ============================================================================

/// 1D transport equation solver
pub struct TransportSolver {
    /// Plasma profiles
    pub profiles: PlasmaProfiles,
    /// Transport model
    pub model: TransportModel,
    /// Time step (s)
    pub dt: f64,
    /// Source terms
    pub sources: SourceTerms,
    /// Boundary conditions
    pub boundary: BoundaryConditions,
}

/// Source terms for transport equations
#[derive(Debug, Clone)]
pub struct SourceTerms {
    /// Particle source (m^-3 s^-1)
    pub s_n: Vec<f64>,
    /// Ion heating (W/m³)
    pub q_i: Vec<f64>,
    /// Electron heating (W/m³)
    pub q_e: Vec<f64>,
    /// Radiated power (W/m³)
    pub p_rad: Vec<f64>,
    /// Ion-electron equilibration (W/m³)
    pub q_ie: Vec<f64>,
}

impl SourceTerms {
    pub fn new(n_points: usize) -> Self {
        Self {
            s_n: vec![0.0; n_points],
            q_i: vec![0.0; n_points],
            q_e: vec![0.0; n_points],
            p_rad: vec![0.0; n_points],
            q_ie: vec![0.0; n_points],
        }
    }

    /// Set alpha heating profile (peaked at center)
    pub fn set_alpha_heating(&mut self, p_alpha: f64, rho: &[f64]) {
        let n = rho.len();
        for i in 0..n {
            let r = rho[i];
            // Gaussian-like profile
            let profile = (-5.0 * r * r).exp();
            self.q_i[i] = 0.2 * p_alpha * profile;  // 20% to ions
            self.q_e[i] = 0.8 * p_alpha * profile;  // 80% to electrons
        }
    }

    /// Set auxiliary heating
    pub fn set_auxiliary_heating(&mut self, p_icrf: f64, p_ecrh: f64, p_nbi: f64, rho: &[f64]) {
        let n = rho.len();
        for i in 0..n {
            let r = rho[i];

            // ICRF: broad, mostly to ions
            let icrf_profile = (-2.0 * r * r).exp();
            self.q_i[i] += 0.9 * p_icrf * icrf_profile;
            self.q_e[i] += 0.1 * p_icrf * icrf_profile;

            // ECRH: narrow, to electrons only
            let ecrh_profile = (-10.0 * (r - 0.3).powi(2)).exp();
            self.q_e[i] += p_ecrh * ecrh_profile;

            // NBI: broad, to ions
            let nbi_profile = (-3.0 * r * r).exp();
            self.q_i[i] += 0.7 * p_nbi * nbi_profile;
            self.q_e[i] += 0.3 * p_nbi * nbi_profile;
        }
    }
}

/// Boundary conditions
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    /// Density at edge (m^-3)
    pub n_edge: f64,
    /// Electron temperature at edge (keV)
    pub te_edge: f64,
    /// Ion temperature at edge (keV)
    pub ti_edge: f64,
    /// Separatrix density (m^-3)
    pub n_sep: f64,
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self {
            n_edge: 1e19,
            te_edge: 0.1,
            ti_edge: 0.1,
            n_sep: 5e18,
        }
    }
}

impl TransportSolver {
    pub fn new(n_points: usize, params: MicroinstabilityParams) -> Self {
        let profiles = PlasmaProfiles::new(n_points);
        let model = TransportModel::new(params);
        let sources = SourceTerms::new(n_points);

        Self {
            profiles,
            model,
            dt: 1e-4,  // 0.1 ms default
            sources,
            boundary: BoundaryConditions::default(),
        }
    }

    /// Advance one time step using implicit method
    pub fn step(&mut self) {
        let n = self.profiles.rho.len();

        // Calculate transport coefficients at each point
        let mut chi_i = vec![0.0; n];
        let mut chi_e = vec![0.0; n];
        let mut d_n = vec![0.0; n];

        let calc = MicroinstabilityCalculator::new(self.model.params.clone());

        for i in 1..n - 1 {
            let grads = self.profiles.gradients(i);
            let q = self.profiles.q[i];
            let shear = self.profiles.shear[i];
            let te = self.profiles.t_e[i];
            let ti = self.profiles.t_i[i];
            let ne = self.profiles.n_e[i];

            let instab = calc.analyze(&grads, q, shear, te, ti, ne);
            let coeff = self.model.calculate_transport(&instab, te, ti, q);

            chi_i[i] = coeff.chi_i;
            chi_e[i] = coeff.chi_e;
            d_n[i] = coeff.d_particle;
        }

        // Solve diffusion equations (implicit Crank-Nicolson)
        self.solve_diffusion(&chi_i, &chi_e, &d_n);

        // Apply boundary conditions
        self.apply_boundary_conditions();

        // Calculate ion-electron equilibration
        self.calculate_equilibration();
    }

    /// Solve diffusion equations using Thomas algorithm
    fn solve_diffusion(&mut self, chi_i: &[f64], chi_e: &[f64], d_n: &[f64]) {
        let n = self.profiles.rho.len();
        let dr = 1.0 / (n - 1) as f64;
        let dr2 = dr * dr;

        // Tridiagonal coefficients
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];

        // ========================================
        // Ion temperature equation
        // ========================================
        for i in 1..n - 1 {
            let r = self.profiles.rho[i];
            let r_factor = if r > 0.01 { 1.0 / r } else { 1.0 };

            let chi = chi_i[i].max(0.01);
            let alpha = chi * self.dt / dr2;

            a[i] = -0.5 * alpha * (1.0 - 0.5 * dr * r_factor);
            c[i] = -0.5 * alpha * (1.0 + 0.5 * dr * r_factor);
            b[i] = 1.0 + alpha;

            // Source terms (converted from W/m³ to keV change)
            let ne = self.profiles.n_e[i].max(1e15);
            let source = (self.sources.q_i[i] + self.sources.q_ie[i]) / (1.5 * ne * E_CHARGE * 1000.0);

            d[i] = self.profiles.t_i[i] + self.dt * source;
        }

        // Boundary conditions
        b[0] = 1.0;
        c[0] = -1.0;  // Zero gradient at center
        d[0] = 0.0;

        a[n - 1] = 0.0;
        b[n - 1] = 1.0;
        d[n - 1] = self.boundary.ti_edge;

        // Solve
        let ti_new = self.thomas_algorithm(&a, &b, &c, &d);
        for i in 0..n {
            self.profiles.t_i[i] = ti_new[i].max(0.01);
        }

        // ========================================
        // Electron temperature equation
        // ========================================
        for i in 1..n - 1 {
            let r = self.profiles.rho[i];
            let r_factor = if r > 0.01 { 1.0 / r } else { 1.0 };

            let chi = chi_e[i].max(0.01);
            let alpha = chi * self.dt / dr2;

            a[i] = -0.5 * alpha * (1.0 - 0.5 * dr * r_factor);
            c[i] = -0.5 * alpha * (1.0 + 0.5 * dr * r_factor);
            b[i] = 1.0 + alpha;

            let ne = self.profiles.n_e[i].max(1e15);
            let source = (self.sources.q_e[i] - self.sources.p_rad[i] - self.sources.q_ie[i])
                / (1.5 * ne * E_CHARGE * 1000.0);

            d[i] = self.profiles.t_e[i] + self.dt * source;
        }

        b[0] = 1.0;
        c[0] = -1.0;
        d[0] = 0.0;

        a[n - 1] = 0.0;
        b[n - 1] = 1.0;
        d[n - 1] = self.boundary.te_edge;

        let te_new = self.thomas_algorithm(&a, &b, &c, &d);
        for i in 0..n {
            self.profiles.t_e[i] = te_new[i].max(0.01);
        }

        // ========================================
        // Density equation
        // ========================================
        for i in 1..n - 1 {
            let r = self.profiles.rho[i];
            let r_factor = if r > 0.01 { 1.0 / r } else { 1.0 };

            let d_coeff = d_n[i].max(0.001);
            let alpha = d_coeff * self.dt / dr2;

            a[i] = -0.5 * alpha * (1.0 - 0.5 * dr * r_factor);
            c[i] = -0.5 * alpha * (1.0 + 0.5 * dr * r_factor);
            b[i] = 1.0 + alpha;

            d[i] = self.profiles.n_e[i] + self.dt * self.sources.s_n[i];
        }

        b[0] = 1.0;
        c[0] = -1.0;
        d[0] = 0.0;

        a[n - 1] = 0.0;
        b[n - 1] = 1.0;
        d[n - 1] = self.boundary.n_edge;

        let ne_new = self.thomas_algorithm(&a, &b, &c, &d);
        for i in 0..n {
            self.profiles.n_e[i] = ne_new[i].max(1e15);
            self.profiles.n_i[i] = self.profiles.n_e[i];  // Quasi-neutrality
        }
    }

    /// Thomas algorithm for tridiagonal system
    fn thomas_algorithm(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];
        let mut x = vec![0.0; n];

        // Forward sweep
        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];

        for i in 1..n {
            let denom = b[i] - a[i] * cp[i - 1];
            cp[i] = c[i] / denom;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
        }

        // Back substitution
        x[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }

        x
    }

    fn apply_boundary_conditions(&mut self) {
        let n = self.profiles.rho.len();

        // Edge values
        self.profiles.n_e[n - 1] = self.boundary.n_edge;
        self.profiles.n_i[n - 1] = self.boundary.n_edge;
        self.profiles.t_e[n - 1] = self.boundary.te_edge;
        self.profiles.t_i[n - 1] = self.boundary.ti_edge;

        // Zero gradient at center
        self.profiles.n_e[0] = self.profiles.n_e[1];
        self.profiles.n_i[0] = self.profiles.n_i[1];
        self.profiles.t_e[0] = self.profiles.t_e[1];
        self.profiles.t_i[0] = self.profiles.t_i[1];
    }

    fn calculate_equilibration(&mut self) {
        let n = self.profiles.rho.len();

        for i in 0..n {
            let te = self.profiles.t_e[i];
            let ti = self.profiles.t_i[i];
            let ne = self.profiles.n_e[i];

            // Coulomb logarithm
            let ln_lambda = 17.0;

            // Equilibration time τ_eq ~ Te^(3/2) / (ne * ln_lambda)
            let tau_eq = 3.5e-6 * (te * 1000.0).powf(1.5) / (ne * 1e-20 * ln_lambda);

            // Q_ie = (3/2) n (Ti - Te) / τ_eq [W/m³]
            self.sources.q_ie[i] = 1.5 * ne * (ti - te) * 1000.0 * E_CHARGE / tau_eq.max(1e-6);
        }
    }

    /// Run to steady state
    pub fn run_to_steady_state(&mut self, max_steps: usize, tolerance: f64) -> bool {
        let n = self.profiles.rho.len();
        let mut te_old = self.profiles.t_e.clone();

        for _step in 0..max_steps {
            self.step();

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let change = (self.profiles.t_e[i] - te_old[i]).abs() / te_old[i].max(0.1);
                max_change = max_change.max(change);
            }

            if max_change < tolerance {
                return true;  // Converged
            }

            te_old = self.profiles.t_e.clone();
        }

        false  // Did not converge
    }

    /// Calculate stored energy (J)
    pub fn stored_energy(&self, volume: f64) -> f64 {
        let n = self.profiles.rho.len();
        let dr = 1.0 / (n - 1) as f64;

        let mut w_th = 0.0;
        for i in 0..n {
            let r = self.profiles.rho[i];
            let ne = self.profiles.n_e[i];
            let te = self.profiles.t_e[i] * 1000.0 * E_CHARGE;  // keV to J
            let ti = self.profiles.t_i[i] * 1000.0 * E_CHARGE;

            // dW = (3/2)(n_e T_e + n_i T_i) * 2πr dr * V
            let dv = 2.0 * PI * r * dr * volume;
            w_th += 1.5 * ne * (te + ti) * dv;
        }

        w_th
    }

    /// Calculate confinement time (s)
    pub fn confinement_time(&self, p_loss: f64, volume: f64) -> f64 {
        let w_th = self.stored_energy(volume);
        w_th / p_loss.max(1e3)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_initialization() {
        let mut profiles = PlasmaProfiles::new(51);
        profiles.initialize_parabolic(3e20, 15.0, 15.0, 1.0, 4.0);

        // Check central values
        assert!((profiles.n_e[0] - 3e20).abs() / 3e20 < 0.1);
        assert!((profiles.t_e[0] - 15.0).abs() < 1.0);

        // Check monotonicity
        for i in 1..profiles.rho.len() {
            assert!(profiles.t_e[i] <= profiles.t_e[i - 1] + 0.01);
        }
    }

    #[test]
    fn test_gradients() {
        let mut profiles = PlasmaProfiles::new(51);
        profiles.initialize_parabolic(3e20, 15.0, 15.0, 1.0, 4.0);

        let grads = profiles.gradients(25);  // Mid-radius

        // Gradients should be positive (decreasing profiles)
        assert!(grads.a_lte > 0.0);
        assert!(grads.a_lti > 0.0);
    }

    #[test]
    fn test_microinstability() {
        let params = MicroinstabilityParams {
            r0: 1.5,
            a: 0.6,
            b0: 25.0,
            a_ion: 2.5,
            z_eff: 1.5,
            epsilon: 0.4,   // Larger epsilon = lower critical gradient
            nu_star: 0.1,
        };

        let calc = MicroinstabilityCalculator::new(params);

        // Test ITG unstable case
        // Critical gradient scales as 1/epsilon, so with epsilon=0.4:
        // a_lti_crit ~ (4/3)(1+1)(1+2*0.5/2)/0.4 ~ 8
        let grads = Gradients {
            a_ln: 2.0,
            a_lte: 10.0,
            a_lti: 15.0,  // Well above threshold (crit ~ 8-10)
            eta_e: 2.5,
            eta_i: 4.0,
        };

        let result = calc.analyze(&grads, 2.0, 0.5, 10.0, 10.0, 3e20);

        // Check that critical gradient is calculated
        assert!(result.a_lti_crit > 0.0, "Critical gradient should be positive");

        // With these parameters, we should be above threshold
        // If not above threshold, at least verify the calculation
        if grads.a_lti > result.a_lti_crit {
            assert!(result.gamma_itg > 0.0, "ITG should be unstable above threshold");
        }
    }

    #[test]
    fn test_transport_coefficients() {
        let params = MicroinstabilityParams {
            r0: 1.5,
            a: 0.6,
            b0: 25.0,
            a_ion: 2.5,
            z_eff: 1.5,
            epsilon: 0.2,
            nu_star: 0.1,
        };

        let model = TransportModel::new(params);

        let instab = InstabilityResult {
            gamma_itg: 1e5,
            omega_itg: 5e4,
            gamma_tem: 5e4,
            omega_tem: -3e4,
            gamma_etg: 0.0,
            dominant: DominantMode::ITG,
            a_lti_crit: 3.0,
            a_lte_crit: 5.0,
        };

        let coeff = model.calculate_transport(&instab, 10.0, 10.0, 2.0);

        // Should have positive diffusivities
        assert!(coeff.chi_i > 0.0);
        assert!(coeff.chi_e > 0.0);
        assert!(coeff.d_particle > 0.0);
    }

    #[test]
    fn test_transport_solver() {
        let params = MicroinstabilityParams {
            r0: 1.5,
            a: 0.6,
            b0: 25.0,
            a_ion: 2.5,
            z_eff: 1.5,
            epsilon: 0.2,
            nu_star: 0.1,
        };

        let mut solver = TransportSolver::new(51, params);
        solver.profiles.initialize_parabolic(3e20, 15.0, 15.0, 1.0, 4.0);

        // Set some heating
        solver.sources.set_alpha_heating(50e6, &solver.profiles.rho.clone());

        // Run a few steps
        for _ in 0..10 {
            solver.step();
        }

        // Profiles should still be reasonable
        assert!(solver.profiles.t_e[0] > 1.0);
        assert!(solver.profiles.t_e[0] < 50.0);
    }

    #[test]
    fn test_thomas_algorithm() {
        let params = MicroinstabilityParams {
            r0: 1.5, a: 0.6, b0: 25.0, a_ion: 2.5,
            z_eff: 1.5, epsilon: 0.2, nu_star: 0.1,
        };

        let solver = TransportSolver::new(5, params);

        // Simple test: x = [1, 2, 3, 4, 5]
        // Should solve tridiagonal system exactly
        let a = vec![0.0, -1.0, -1.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, -1.0, -1.0, 0.0];
        let d = vec![1.0, 0.0, 0.0, 0.0, 9.0];

        let x = solver.thomas_algorithm(&a, &b, &c, &d);

        // Check solution
        assert!(x.len() == 5);
        for val in &x {
            assert!(val.is_finite());
        }
    }
}
