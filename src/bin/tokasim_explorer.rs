//! # TOKASIM-RS Parameter Explorer
//!
//! Interactive parameter exploration for tokamak design optimization.
//! Allows real-time adjustment of geometric and physics parameters
//! with instant feedback on structural and operational limits.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin tokasim-explorer --features bevy-viz --release
//! ```
//!
//! ## Features
//!
//! - Interactive sliders for all major parameters
//! - Real-time geometry regeneration
//! - Physics limits visualization (β, Greenwald, q95)
//! - Structural stress indicators
//! - Operating point analysis
//! - Control system comparison (PIRS vs PCS vs DeepMind)
//! - Heat map visualization with cross-sectional cuts
//! - Pizza slice selection for toroidal segments
//!
//! ## Author
//!
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::pbr::wireframe::{WireframePlugin, WireframeConfig, Wireframe, WireframeColor};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use std::f32::consts::{PI, TAU};

// ============================================================================
// COMPONENTS
// ============================================================================

#[derive(Component)]
struct Plasma;

#[derive(Component)]
struct PlasmaWireframe;

#[derive(Component)]
struct VacuumVessel;

#[derive(Component)]
struct TFCoil(usize);

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    radius: f32,
    azimuth: f32,
    elevation: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            radius: 8.0,
            azimuth: PI / 4.0,
            elevation: PI / 6.0,
        }
    }
}

// ============================================================================
// TOKAMAK PARAMETERS (ADJUSTABLE)
// ============================================================================

#[derive(Resource)]
struct TokamakParams {
    // Geometry
    major_radius: f32,      // R₀ [m] - 0.5 to 10.0
    minor_radius: f32,      // a [m] - 0.1 to 3.0
    elongation: f32,        // κ - 1.0 to 2.5
    triangularity: f32,     // δ - 0.0 to 0.8

    // Magnetic field
    toroidal_field: f32,    // Bt [T] - 1.0 to 30.0
    plasma_current: f32,    // Ip [MA] - 0.1 to 20.0

    // Plasma
    density: f32,           // n̄e [10²⁰ m⁻³] - 0.1 to 5.0
    temperature: f32,       // T [keV] - 1.0 to 50.0

    // Derived/calculated
    needs_geometry_update: bool,
}

impl Default for TokamakParams {
    fn default() -> Self {
        Self {
            // TS-1 defaults
            major_radius: 1.5,
            minor_radius: 0.6,
            elongation: 1.97,
            triangularity: 0.54,
            toroidal_field: 25.0,
            plasma_current: 12.0,
            density: 2.0,
            temperature: 20.0,
            needs_geometry_update: false,
        }
    }
}

// ============================================================================
// PHYSICS LIMITS
// ============================================================================

#[derive(Resource, Default)]
struct PhysicsLimits {
    // Calculated limits
    aspect_ratio: f32,
    plasma_volume: f32,         // m³

    // MHD Stability
    beta_actual: f32,           // %
    beta_troyon_limit: f32,     // % (Troyon limit)
    beta_margin: f32,           // % (how close to limit)

    // Density limit
    greenwald_density: f32,     // 10²⁰ m⁻³
    greenwald_fraction: f32,    // n/n_G

    // Safety factor
    q95: f32,                   // Edge safety factor
    q95_min_safe: f32,          // Minimum safe q95

    // Confinement
    tau_e_iter98: f32,          // Energy confinement time [s]
    triple_product: f32,        // n·T·τ [10²¹ keV·s/m³]

    // Fusion performance
    fusion_power: f32,          // MW
    q_factor: f32,              // Q = Pfus/Pheat

    // Structural
    magnetic_pressure: f32,     // MPa
    hoop_stress: f32,           // MPa (simplified)

    // Status flags
    beta_exceeded: bool,
    density_exceeded: bool,
    q95_too_low: bool,
    structural_warning: bool,
}

impl PhysicsLimits {
    fn calculate(params: &TokamakParams) -> Self {
        let r0 = params.major_radius;
        let a = params.minor_radius;
        let kappa = params.elongation;
        let delta = params.triangularity;
        let bt = params.toroidal_field;
        let ip = params.plasma_current;  // MA
        let ne = params.density;         // 10²⁰ m⁻³
        let te = params.temperature;     // keV

        let aspect_ratio = r0 / a;

        // Plasma volume (D-shaped approximation)
        let plasma_volume = 2.0 * PI * PI * r0 * a * a * kappa;

        // Beta calculation
        // β = 2μ₀ * n * T / B²
        // β_N = β * a * B / I  (normalized beta)
        let mu0 = 4.0 * PI * 1e-7;
        let n_si = ne * 1e20;  // Convert to m⁻³
        let t_si = te * 1.602e-16;  // Convert keV to J
        let pressure = n_si * t_si;  // Pa
        let beta_actual = 100.0 * 2.0 * mu0 as f32 * pressure as f32 / (bt * bt);

        // Troyon limit: β_N ≤ 2.8 (conservative) to 3.5 (optimistic)
        // β_max = 2.8 * I / (a * B) [%]
        let beta_troyon_limit = 2.8 * ip / (a * bt);
        let beta_margin = beta_troyon_limit - beta_actual;

        // Greenwald density limit
        // n_G = I / (π * a²) [10²⁰ m⁻³]
        let greenwald_density = ip / (PI * a * a);
        let greenwald_fraction = ne / greenwald_density;

        // Safety factor q95
        // q95 ≈ 5 * a² * B * κ / (R * I) * (1 + κ²)/2
        let kappa_factor = (1.0 + kappa * kappa) / 2.0;
        let q95 = 5.0 * a * a * bt * kappa_factor / (r0 * ip);
        let q95_min_safe = 2.0;  // Below 2 is dangerous (disruptions)

        // ITER98 H-mode confinement scaling
        // τ_E = 0.0562 * I^0.93 * B^0.15 * n^0.41 * P^-0.69 * R^1.97 * a^0.58 * κ^0.78 * M^0.19
        // Simplified version assuming P_heat ~ 50 MW, M = 2.5 (D-T)
        let p_heat = 50.0_f32;  // MW assumed
        let tau_e = 0.0562 * ip.powf(0.93) * bt.powf(0.15) * ne.powf(0.41)
                    * p_heat.powf(-0.69) * r0.powf(1.97) * a.powf(0.58)
                    * kappa.powf(0.78) * 2.5_f32.powf(0.19);

        // Triple product
        let triple_product = ne * te * tau_e;

        // Fusion power (simplified D-T)
        // P_fus ≈ n² * <σv> * E_fus * V
        // At 20 keV, <σv> ≈ 4e-22 m³/s
        let sigma_v = 4e-22_f32 * (te / 20.0).powf(2.0).min(2.0);
        let e_fus = 17.6e6 * 1.602e-19;  // MeV to J
        let fusion_power = (n_si * n_si) as f32 * sigma_v * e_fus as f32 * plasma_volume / 4.0 / 1e6;

        // Q factor
        let q_factor = fusion_power / p_heat;

        // Magnetic pressure
        // P_mag = B² / (2μ₀)
        let magnetic_pressure = (bt * bt) / (2.0 * mu0 as f32) / 1e6;  // MPa

        // Simplified hoop stress in TF coils
        // σ ≈ B² * R / (2μ₀ * t) where t ~ 0.3m coil thickness
        let coil_thickness = 0.3;
        let hoop_stress = (bt * bt * r0) / (2.0 * mu0 as f32 * coil_thickness) / 1e6;

        // Status flags
        let beta_exceeded = beta_actual > beta_troyon_limit * 0.9;
        let density_exceeded = greenwald_fraction > 0.85;
        let q95_too_low = q95 < q95_min_safe * 1.2;
        let structural_warning = hoop_stress > 500.0;  // 500 MPa is high

        Self {
            aspect_ratio,
            plasma_volume,
            beta_actual,
            beta_troyon_limit,
            beta_margin,
            greenwald_density,
            greenwald_fraction,
            q95,
            q95_min_safe,
            tau_e_iter98: tau_e,
            triple_product,
            fusion_power,
            q_factor,
            magnetic_pressure,
            hoop_stress,
            beta_exceeded,
            density_exceeded,
            q95_too_low,
            structural_warning,
        }
    }
}

// ============================================================================
// CONTROL SYSTEMS COMPARISON
// ============================================================================

/// Control system response characteristics
///
/// NOTE: These response times represent theoretical processing latency only.
/// Actual control loop performance depends on additional factors:
/// - Sensor acquisition latency (typically 1-10 μs)
/// - Communication bus delays (μs to ms depending on protocol)
/// - Actuator response time (ms for magnetic coils)
/// - Feedback loop delays
///
/// The comparison demonstrates algorithmic response capability, not complete
/// control loop performance.
#[derive(Resource)]
struct ControlSystems {
    // Response times (seconds)
    pirs_response_time: f32,      // ~0.1 μs = 1e-7 s (deterministic symbolic)
    pcs_response_time: f32,       // ~100 μs = 1e-4 s (traditional PID+MPC)
    deepmind_response_time: f32,  // ~1-10 ms = 1e-3 to 1e-2 s (ML inference)

    // Current instability timescales
    current_event_rate: f32,      // Events per second requiring response
    fastest_event_timescale: f32, // Fastest instability timescale [s]

    // System status
    pirs_can_respond: bool,
    pcs_can_respond: bool,
    deepmind_can_respond: bool,

    // Failure counters (accumulated)
    pirs_failures: u32,
    pcs_failures: u32,
    deepmind_failures: u32,

    // Active instability type
    instability_type: InstabilityType,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum InstabilityType {
    #[default]
    None,
    EdgeLocalizedMode,       // ELM: 0.1-1 ms timescale
    SawtoothCrash,           // Sawtooth: 10-100 ms
    VerticalDisplacement,    // VDE: 1-10 ms
    LockedMode,              // Locked mode: 10-100 ms
    Disruption,              // Full disruption: 1-10 ms thermal quench
    MinorDisruption,         // Minor: 10-50 ms
    NeoClassicalTearing,     // NTM: 100 ms - 1 s growth
    ResistiveWallMode,       // RWM: 1-100 ms
}

impl InstabilityType {
    fn timescale(&self) -> f32 {
        match self {
            InstabilityType::None => 1.0,                    // 1 s (no urgency)
            InstabilityType::EdgeLocalizedMode => 5e-4,      // 0.5 ms
            InstabilityType::SawtoothCrash => 5e-2,          // 50 ms
            InstabilityType::VerticalDisplacement => 5e-3,   // 5 ms
            InstabilityType::LockedMode => 5e-2,             // 50 ms
            InstabilityType::Disruption => 2e-3,             // 2 ms thermal quench
            InstabilityType::MinorDisruption => 3e-2,        // 30 ms
            InstabilityType::NeoClassicalTearing => 5e-1,    // 500 ms growth
            InstabilityType::ResistiveWallMode => 1e-2,      // 10 ms
        }
    }

    fn name(&self) -> &'static str {
        match self {
            InstabilityType::None => "Stable Operation",
            InstabilityType::EdgeLocalizedMode => "ELM (Edge Localized Mode)",
            InstabilityType::SawtoothCrash => "Sawtooth Crash",
            InstabilityType::VerticalDisplacement => "VDE (Vertical Displacement)",
            InstabilityType::LockedMode => "Locked Mode",
            InstabilityType::Disruption => "DISRUPTION",
            InstabilityType::MinorDisruption => "Minor Disruption",
            InstabilityType::NeoClassicalTearing => "NTM (Neoclassical Tearing)",
            InstabilityType::ResistiveWallMode => "RWM (Resistive Wall Mode)",
        }
    }

    fn severity(&self) -> u8 {
        match self {
            InstabilityType::None => 0,
            InstabilityType::NeoClassicalTearing => 1,
            InstabilityType::SawtoothCrash => 2,
            InstabilityType::LockedMode => 3,
            InstabilityType::MinorDisruption => 4,
            InstabilityType::ResistiveWallMode => 5,
            InstabilityType::EdgeLocalizedMode => 6,
            InstabilityType::VerticalDisplacement => 7,
            InstabilityType::Disruption => 10,
        }
    }
}

impl Default for ControlSystems {
    fn default() -> Self {
        Self {
            pirs_response_time: 1e-7,      // 0.1 μs
            pcs_response_time: 1e-4,       // 100 μs
            deepmind_response_time: 5e-3,  // 5 ms (average)

            current_event_rate: 0.0,
            fastest_event_timescale: 1.0,

            pirs_can_respond: true,
            pcs_can_respond: true,
            deepmind_can_respond: true,

            pirs_failures: 0,
            pcs_failures: 0,
            deepmind_failures: 0,

            instability_type: InstabilityType::None,
        }
    }
}

impl ControlSystems {
    fn update(&mut self, limits: &PhysicsLimits, params: &TokamakParams) {
        // Determine instability based on limit violations
        let old_instability = self.instability_type;

        self.instability_type = if limits.beta_exceeded && limits.q95_too_low {
            InstabilityType::Disruption
        } else if limits.q95_too_low && limits.q95 < 1.5 {
            InstabilityType::VerticalDisplacement
        } else if limits.beta_exceeded {
            if limits.beta_actual > limits.beta_troyon_limit * 1.1 {
                InstabilityType::ResistiveWallMode
            } else {
                InstabilityType::EdgeLocalizedMode
            }
        } else if limits.q95 < 2.5 {
            InstabilityType::SawtoothCrash
        } else if limits.density_exceeded {
            if limits.greenwald_fraction > 1.0 {
                InstabilityType::MinorDisruption
            } else {
                InstabilityType::LockedMode
            }
        } else if limits.greenwald_fraction > 0.7 || limits.beta_actual > limits.beta_troyon_limit * 0.8 {
            InstabilityType::NeoClassicalTearing
        } else {
            InstabilityType::None
        };

        self.fastest_event_timescale = self.instability_type.timescale();
        self.current_event_rate = 1.0 / self.fastest_event_timescale;

        // Check which systems can respond
        self.pirs_can_respond = self.pirs_response_time < self.fastest_event_timescale;
        self.pcs_can_respond = self.pcs_response_time < self.fastest_event_timescale;
        self.deepmind_can_respond = self.deepmind_response_time < self.fastest_event_timescale;

        // Count failures when instability occurs and system can't respond
        if self.instability_type != InstabilityType::None && old_instability != self.instability_type {
            if !self.pirs_can_respond { self.pirs_failures += 1; }
            if !self.pcs_can_respond { self.pcs_failures += 1; }
            if !self.deepmind_can_respond { self.deepmind_failures += 1; }
        }
    }

    fn reset_failures(&mut self) {
        self.pirs_failures = 0;
        self.pcs_failures = 0;
        self.deepmind_failures = 0;
    }
}

// ============================================================================
// HEAT MAP VISUALIZATION
// ============================================================================

#[derive(Resource)]
struct HeatMapState {
    // View settings
    enabled: bool,
    toroidal_position: f32,        // Phi angle [0, 2π] - "pizza slice" position
    toroidal_width: f32,           // Angular width of slice [rad]
    show_cross_section: bool,

    // Temperature profile (simplified parabolic)
    // T(r) = T_core * (1 - (r/a)^α)^β
    core_temperature: f32,         // keV
    edge_temperature: f32,         // keV (pedestal)
    profile_alpha: f32,            // Peaking factor
    profile_beta: f32,             // Profile shape

    // Density profile
    core_density: f32,             // 10²⁰ m⁻³
    edge_density: f32,             // 10²⁰ m⁻³

    // Visualization
    color_map: ColorMapType,
    radial_resolution: u32,
    poloidal_resolution: u32,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ColorMapType {
    #[default]
    Plasma,       // Purple-orange-yellow
    Inferno,      // Black-red-yellow-white
    Viridis,      // Blue-green-yellow
    Temperature,  // Blue-white-red (cold-hot)
}

impl ColorMapType {
    fn to_color(&self, value: f32) -> Color {
        let v = value.clamp(0.0, 1.0);
        match self {
            ColorMapType::Plasma => {
                // Purple -> Magenta -> Orange -> Yellow
                if v < 0.33 {
                    let t = v / 0.33;
                    Color::srgb(0.05 + 0.6 * t, 0.0, 0.5 + 0.3 * t)
                } else if v < 0.66 {
                    let t = (v - 0.33) / 0.33;
                    Color::srgb(0.65 + 0.35 * t, 0.3 * t, 0.8 - 0.8 * t)
                } else {
                    let t = (v - 0.66) / 0.34;
                    Color::srgb(1.0, 0.3 + 0.7 * t, t * 0.3)
                }
            }
            ColorMapType::Inferno => {
                // Black -> Dark red -> Orange -> Yellow -> White
                if v < 0.25 {
                    let t = v / 0.25;
                    Color::srgb(0.1 * t, 0.0, 0.1 * t)
                } else if v < 0.5 {
                    let t = (v - 0.25) / 0.25;
                    Color::srgb(0.1 + 0.6 * t, 0.0, 0.1 - 0.1 * t)
                } else if v < 0.75 {
                    let t = (v - 0.5) / 0.25;
                    Color::srgb(0.7 + 0.3 * t, 0.5 * t, 0.0)
                } else {
                    let t = (v - 0.75) / 0.25;
                    Color::srgb(1.0, 0.5 + 0.5 * t, t)
                }
            }
            ColorMapType::Viridis => {
                // Dark blue -> Teal -> Green -> Yellow
                if v < 0.33 {
                    let t = v / 0.33;
                    Color::srgb(0.27 - 0.1 * t, 0.0 + 0.3 * t, 0.33 + 0.2 * t)
                } else if v < 0.66 {
                    let t = (v - 0.33) / 0.33;
                    Color::srgb(0.17 + 0.2 * t, 0.3 + 0.4 * t, 0.53 - 0.2 * t)
                } else {
                    let t = (v - 0.66) / 0.34;
                    Color::srgb(0.37 + 0.63 * t, 0.7 + 0.3 * t, 0.33 - 0.33 * t)
                }
            }
            ColorMapType::Temperature => {
                // Blue -> White -> Red
                if v < 0.5 {
                    let t = v / 0.5;
                    Color::srgb(t, t, 1.0)
                } else {
                    let t = (v - 0.5) / 0.5;
                    Color::srgb(1.0, 1.0 - t, 1.0 - t)
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ColorMapType::Plasma => "Plasma",
            ColorMapType::Inferno => "Inferno",
            ColorMapType::Viridis => "Viridis",
            ColorMapType::Temperature => "Temperature",
        }
    }
}

impl Default for HeatMapState {
    fn default() -> Self {
        Self {
            enabled: false,
            toroidal_position: 0.0,
            toroidal_width: PI / 8.0,  // 22.5 degrees slice
            show_cross_section: true,

            core_temperature: 20.0,    // keV
            edge_temperature: 0.5,     // keV (pedestal)
            profile_alpha: 2.0,
            profile_beta: 1.5,

            core_density: 2.0,
            edge_density: 0.3,

            color_map: ColorMapType::Plasma,
            radial_resolution: 32,
            poloidal_resolution: 64,
        }
    }
}

impl HeatMapState {
    /// Calculate temperature at normalized radius r/a
    fn temperature_at(&self, rho: f32) -> f32 {
        let rho_clamped = rho.clamp(0.0, 1.0);
        let profile = (1.0 - rho_clamped.powf(self.profile_alpha)).powf(self.profile_beta);
        self.edge_temperature + (self.core_temperature - self.edge_temperature) * profile
    }

    /// Calculate density at normalized radius r/a
    fn density_at(&self, rho: f32) -> f32 {
        let rho_clamped = rho.clamp(0.0, 1.0);
        // Density profile is typically flatter than temperature
        let profile = (1.0 - rho_clamped.powf(1.5)).powf(1.0);
        self.edge_density + (self.core_density - self.edge_density) * profile
    }

    /// Calculate pressure at normalized radius (proxy for intensity)
    fn pressure_at(&self, rho: f32) -> f32 {
        self.temperature_at(rho) * self.density_at(rho)
    }

    /// Normalize temperature to [0,1] for color mapping
    fn normalized_temperature(&self, rho: f32) -> f32 {
        let t = self.temperature_at(rho);
        (t - self.edge_temperature) / (self.core_temperature - self.edge_temperature).max(0.001)
    }
}

#[derive(Component)]
struct HeatMapSlice;

#[derive(Component)]
struct CrossSectionPlane;

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║     TOKASIM-RS Parameter Explorer                                  ║");
    println!("║     Interactive Tokamak Design Optimization                        ║");
    println!("║     Avermex Research Division - January 2026                       ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Adjust parameters with sliders to explore design space.");
    println!("Watch for limit violations (red indicators).");
    println!();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "TOKASIM-RS Parameter Explorer".into(),
                resolution: (1600., 1000.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(WireframePlugin)
        .add_plugins(EguiPlugin)
        .insert_resource(WireframeConfig {
            global: false,
            default_color: Color::srgb(1.0, 0.6, 0.0),
        })
        .insert_resource(TokamakParams::default())
        .insert_resource(PhysicsLimits::default())
        .insert_resource(ControlSystems::default())
        .insert_resource(HeatMapState::default())
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.05)))
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 400.0,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (
            camera_controller,
            ui_system,
            update_physics_limits,
            update_control_systems,
            regenerate_geometry,
            update_heat_map,
            keyboard_input,
        ))
        .run();
}

// ============================================================================
// D-SHAPED TORUS MESH
// ============================================================================

fn create_d_shaped_torus(
    major_radius: f32,
    minor_radius: f32,
    elongation: f32,
    triangularity: f32,
    toroidal_segments: u32,
    poloidal_segments: u32,
) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=toroidal_segments {
        let phi = (i as f32 / toroidal_segments as f32) * TAU;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        for j in 0..=poloidal_segments {
            let theta = (j as f32 / poloidal_segments as f32) * TAU;
            let r_local = minor_radius * (theta.cos() + triangularity * theta.sin().powi(2));
            let z_local = minor_radius * elongation * theta.sin();
            let r_total = major_radius + r_local;

            positions.push([r_total * cos_phi, z_local, r_total * sin_phi]);

            let dr_dtheta = minor_radius * (-theta.sin() + 2.0 * triangularity * theta.sin() * theta.cos());
            let dz_dtheta = minor_radius * elongation * theta.cos();
            let tangent_poloidal = Vec3::new(dr_dtheta * cos_phi, dz_dtheta, dr_dtheta * sin_phi);
            let tangent_toroidal = Vec3::new(-r_total * sin_phi, 0.0, r_total * cos_phi);
            let normal = tangent_poloidal.cross(tangent_toroidal).normalize();
            normals.push([normal.x, normal.y, normal.z]);
            uvs.push([i as f32 / toroidal_segments as f32, j as f32 / poloidal_segments as f32]);
        }
    }

    for i in 0..toroidal_segments {
        for j in 0..poloidal_segments {
            let a = i * (poloidal_segments + 1) + j;
            let b = a + 1;
            let c = (i + 1) * (poloidal_segments + 1) + j;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

// ============================================================================
// SETUP
// ============================================================================

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    params: Res<TokamakParams>,
) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 4.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        OrbitCamera::default(),
    ));

    // Lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 25000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -PI / 3.0, PI / 4.0, 0.0)),
        ..default()
    });

    for i in 0..4 {
        let angle = (i as f32) * TAU / 4.0;
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 1000000.0,
                range: 20.0,
                ..default()
            },
            transform: Transform::from_xyz(angle.cos() * 5.0, 4.0, angle.sin() * 5.0),
            ..default()
        });
    }

    // Spawn initial geometry
    spawn_tokamak_geometry(&mut commands, &mut meshes, &mut materials, &params);

    // Ground plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(20.0, 20.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.08, 0.1),
            ..default()
        }),
        transform: Transform::from_xyz(0.0, -3.0, 0.0),
        ..default()
    });
}

fn spawn_tokamak_geometry(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    params: &TokamakParams,
) {
    let r0 = params.major_radius;
    let a = params.minor_radius;
    let kappa = params.elongation;
    let delta = params.triangularity;

    // Plasma
    let plasma_mesh = create_d_shaped_torus(r0, a * 0.85, kappa, delta, 128, 64);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(plasma_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgba(1.0, 0.5, 0.2, 0.7),
                emissive: LinearRgba::new(10.0, 4.0, 0.8, 1.0),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                ..default()
            }),
            ..default()
        },
        Plasma,
        Wireframe,
        WireframeColor { color: Color::srgba(1.0, 0.8, 0.3, 0.4) },
    ));

    // Vacuum vessel
    let vessel_mesh = create_d_shaped_torus(r0, a * 1.1, kappa * 1.02, delta, 64, 32);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(vessel_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgba(0.5, 0.5, 0.55, 0.3),
                metallic: 0.9,
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            ..default()
        },
        VacuumVessel,
    ));

    // TF Coils (simplified)
    let tf_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.45, 0.2),
        metallic: 1.0,
        perceptual_roughness: 0.2,
        ..default()
    });

    for i in 0..16 {
        let phi = (i as f32 / 16.0) * TAU;
        let coil_mesh = create_d_shaped_torus(r0, a * 1.25, kappa * 1.05, delta, 4, 24);

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(coil_mesh),
                material: tf_material.clone(),
                transform: Transform::from_rotation(Quat::from_rotation_y(phi)),
                ..default()
            },
            TFCoil(i),
        ));
    }
}

// ============================================================================
// UPDATE SYSTEMS
// ============================================================================

fn camera_controller(
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let Ok((mut transform, mut orbit)) = query.get_single_mut() else { return };

    if mouse_button.pressed(MouseButton::Left) {
        for ev in mouse_motion.read() {
            orbit.azimuth -= ev.delta.x * 0.005;
            orbit.elevation = (orbit.elevation - ev.delta.y * 0.005).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
        }
    } else {
        mouse_motion.clear();
    }

    for ev in scroll.read() {
        orbit.radius = (orbit.radius * (1.0 - ev.y * 0.1)).clamp(2.0, 30.0);
    }

    if keys.pressed(KeyCode::ArrowLeft) { orbit.azimuth += 0.02; }
    if keys.pressed(KeyCode::ArrowRight) { orbit.azimuth -= 0.02; }
    if keys.pressed(KeyCode::ArrowUp) { orbit.elevation = (orbit.elevation + 0.02).min(PI / 2.0 - 0.1); }
    if keys.pressed(KeyCode::ArrowDown) { orbit.elevation = (orbit.elevation - 0.02).max(-PI / 2.0 + 0.1); }

    let pos = Vec3::new(
        orbit.radius * orbit.elevation.cos() * orbit.azimuth.sin(),
        orbit.radius * orbit.elevation.sin(),
        orbit.radius * orbit.elevation.cos() * orbit.azimuth.cos(),
    ) + orbit.focus;

    *transform = Transform::from_translation(pos).looking_at(orbit.focus, Vec3::Y);
}

fn update_physics_limits(params: Res<TokamakParams>, mut limits: ResMut<PhysicsLimits>) {
    *limits = PhysicsLimits::calculate(&params);
}

fn update_control_systems(
    params: Res<TokamakParams>,
    limits: Res<PhysicsLimits>,
    mut control: ResMut<ControlSystems>,
) {
    control.update(&limits, &params);
}

fn update_heat_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    params: Res<TokamakParams>,
    heat_map: Res<HeatMapState>,
    slice_query: Query<Entity, With<HeatMapSlice>>,
    cross_section_query: Query<Entity, With<CrossSectionPlane>>,
) {
    // Remove old heat map entities if heat map is disabled
    if !heat_map.enabled {
        for entity in slice_query.iter() {
            commands.entity(entity).despawn();
        }
        for entity in cross_section_query.iter() {
            commands.entity(entity).despawn();
        }
        return;
    }

    // Only update if changed (simple check)
    if heat_map.is_changed() || params.is_changed() {
        // Remove old entities
        for entity in slice_query.iter() {
            commands.entity(entity).despawn();
        }
        for entity in cross_section_query.iter() {
            commands.entity(entity).despawn();
        }

        // Create new heat map slice
        let r0 = params.major_radius;
        let a = params.minor_radius * 0.85;  // Plasma boundary
        let kappa = params.elongation;
        let _delta = params.triangularity;

        if heat_map.show_cross_section {
            // Create cross-section mesh (2D slice through plasma)
            let mesh = create_cross_section_mesh(
                r0, a, kappa,
                &heat_map,
                heat_map.radial_resolution,
                heat_map.poloidal_resolution,
            );

            // Position at the selected toroidal angle
            let phi = heat_map.toroidal_position;

            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(mesh),
                    material: materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        emissive: LinearRgba::new(2.0, 2.0, 2.0, 1.0),
                        unlit: true,
                        double_sided: true,
                        cull_mode: None,
                        ..default()
                    }),
                    transform: Transform::from_rotation(Quat::from_rotation_y(phi)),
                    ..default()
                },
                CrossSectionPlane,
            ));

            // Add slice markers at the toroidal edges
            spawn_slice_markers(&mut commands, &mut meshes, &mut materials, r0, a, kappa, phi, heat_map.toroidal_width);
        }
    }
}

fn create_cross_section_mesh(
    r0: f32,
    a: f32,
    kappa: f32,
    heat_map: &HeatMapState,
    radial_res: u32,
    poloidal_res: u32,
) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Generate concentric rings of the cross-section
    for i in 0..=radial_res {
        let rho = i as f32 / radial_res as f32;  // Normalized radius

        for j in 0..=poloidal_res {
            let theta = (j as f32 / poloidal_res as f32) * TAU;

            // D-shaped cross-section position
            let r_local = a * rho;
            let x = r0 + r_local * theta.cos();
            let z = r_local * kappa * theta.sin();

            positions.push([x, z, 0.0]);  // In the X-Z plane (Y=0)
            normals.push([0.0, 0.0, 1.0]);  // Normal pointing out of plane

            // Color based on temperature profile
            let temp_normalized = heat_map.normalized_temperature(rho);
            let color = heat_map.color_map.to_color(temp_normalized);

            if let Color::Srgba(srgba) = color.into() {
                colors.push([srgba.red, srgba.green, srgba.blue, 0.9]);
            } else {
                colors.push([1.0, 0.5, 0.2, 0.9]);
            }
        }
    }

    // Generate indices for triangles
    for i in 0..radial_res {
        for j in 0..poloidal_res {
            let a_idx = i * (poloidal_res + 1) + j;
            let b_idx = a_idx + 1;
            let c_idx = (i + 1) * (poloidal_res + 1) + j;
            let d_idx = c_idx + 1;

            indices.extend_from_slice(&[a_idx, c_idx, b_idx, b_idx, c_idx, d_idx]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn spawn_slice_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    r0: f32,
    a: f32,
    kappa: f32,
    phi: f32,
    width: f32,
) {
    // Create edge markers for the pizza slice
    let marker_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.0, 1.0, 1.0, 0.5),
        emissive: LinearRgba::new(0.0, 2.0, 2.0, 1.0),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // Create lines showing slice boundaries
    for sign in [-1.0_f32, 1.0] {
        let edge_phi = phi + sign * width / 2.0;
        let cos_phi = edge_phi.cos();
        let sin_phi = edge_phi.sin();

        // Line from center to outer edge
        let inner_r = r0 - a;
        let outer_r = r0 + a;

        let mut positions = Vec::new();
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let r = inner_r + (outer_r - inner_r) * t;
            positions.push([r * cos_phi, 0.0, r * sin_phi]);
        }

        // Create a thin tube along the line
        let line_mesh = create_line_mesh(&positions, 0.02);

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(line_mesh),
                material: marker_material.clone(),
                ..default()
            },
            HeatMapSlice,
        ));
    }
}

fn create_line_mesh(points: &[[f32; 3]], radius: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let segments = 8;

    for (i, point) in points.iter().enumerate() {
        // Get direction
        let dir = if i < points.len() - 1 {
            let next = points[i + 1];
            Vec3::new(next[0] - point[0], next[1] - point[1], next[2] - point[2]).normalize()
        } else if i > 0 {
            let prev = points[i - 1];
            Vec3::new(point[0] - prev[0], point[1] - prev[1], point[2] - prev[2]).normalize()
        } else {
            Vec3::X
        };

        // Create perpendicular vectors
        let up = if dir.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let right = dir.cross(up).normalize();
        let actual_up = right.cross(dir).normalize();

        // Create ring of vertices
        for j in 0..segments {
            let angle = (j as f32 / segments as f32) * TAU;
            let offset = right * angle.cos() * radius + actual_up * angle.sin() * radius;
            let pos = Vec3::new(point[0], point[1], point[2]) + offset;
            let normal = offset.normalize();

            positions.push([pos.x, pos.y, pos.z]);
            normals.push([normal.x, normal.y, normal.z]);
        }
    }

    // Create indices
    for i in 0..(points.len() - 1) as u32 {
        for j in 0..segments as u32 {
            let a = i * segments as u32 + j;
            let b = i * segments as u32 + (j + 1) % segments as u32;
            let c = (i + 1) * segments as u32 + j;
            let d = (i + 1) * segments as u32 + (j + 1) % segments as u32;

            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_indices(Indices::U32(indices))
}

fn regenerate_geometry(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut params: ResMut<TokamakParams>,
    plasma_query: Query<Entity, With<Plasma>>,
    vessel_query: Query<Entity, With<VacuumVessel>>,
    coil_query: Query<Entity, With<TFCoil>>,
) {
    if !params.needs_geometry_update {
        return;
    }
    params.needs_geometry_update = false;

    // Despawn old geometry
    for entity in plasma_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in vessel_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in coil_query.iter() {
        commands.entity(entity).despawn();
    }

    // Spawn new geometry
    spawn_tokamak_geometry(&mut commands, &mut meshes, &mut materials, &params);
}

fn keyboard_input(keys: Res<ButtonInput<KeyCode>>, mut params: ResMut<TokamakParams>) {
    // Presets
    if keys.just_pressed(KeyCode::Digit1) {
        // TS-1
        params.major_radius = 1.5;
        params.minor_radius = 0.6;
        params.elongation = 1.97;
        params.triangularity = 0.54;
        params.toroidal_field = 25.0;
        params.plasma_current = 12.0;
        params.needs_geometry_update = true;
        println!("[INFO] Loaded TS-1 preset");
    }
    if keys.just_pressed(KeyCode::Digit2) {
        // SPARC
        params.major_radius = 1.85;
        params.minor_radius = 0.57;
        params.elongation = 1.8;
        params.triangularity = 0.4;
        params.toroidal_field = 12.2;
        params.plasma_current = 8.7;
        params.needs_geometry_update = true;
        println!("[INFO] Loaded SPARC preset");
    }
    if keys.just_pressed(KeyCode::Digit3) {
        // ITER
        params.major_radius = 6.2;
        params.minor_radius = 2.0;
        params.elongation = 1.7;
        params.triangularity = 0.33;
        params.toroidal_field = 5.3;
        params.plasma_current = 15.0;
        params.needs_geometry_update = true;
        println!("[INFO] Loaded ITER preset");
    }
}

// ============================================================================
// UI SYSTEM
// ============================================================================

fn ui_system(
    mut contexts: EguiContexts,
    mut params: ResMut<TokamakParams>,
    limits: Res<PhysicsLimits>,
    mut control: ResMut<ControlSystems>,
    mut heat_map: ResMut<HeatMapState>,
) {
    // Top panel - Control Systems Comparison
    egui::TopBottomPanel::top("control_panel").min_height(180.0).show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.heading("🎮 Control Systems Comparison");
            ui.separator();

            // Instability status
            let instability_color = match control.instability_type.severity() {
                0 => egui::Color32::GREEN,
                1..=3 => egui::Color32::YELLOW,
                4..=6 => egui::Color32::from_rgb(255, 165, 0),  // Orange
                7..=9 => egui::Color32::RED,
                _ => egui::Color32::from_rgb(139, 0, 0),  // Dark red
            };
            ui.colored_label(instability_color, format!("Event: {}", control.instability_type.name()));

            if control.instability_type != InstabilityType::None {
                ui.label(format!("τ_event = {:.2e} s", control.fastest_event_timescale));
            }
        });

        ui.separator();

        // Control systems grid
        ui.columns(3, |columns| {
            // PIRS Column
            columns[0].vertical_centered(|ui| {
                ui.heading("PIRS");
                ui.label("(Prolog Inference in Rust)");
                ui.label(format!("Response: {:.1} μs", control.pirs_response_time * 1e6));

                let status_color = if control.pirs_can_respond {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                let status_text = if control.pirs_can_respond { "✓ CAN RESPOND" } else { "✗ TOO SLOW" };
                ui.colored_label(status_color, status_text);

                // Response time bar
                let response_ratio = (control.pirs_response_time / control.fastest_event_timescale).min(2.0);
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(format!("{:.0}x margin", control.fastest_event_timescale / control.pirs_response_time))
                    .fill(if control.pirs_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.label(format!("Failures: {}", control.pirs_failures));
            });

            // PCS Column
            columns[1].vertical_centered(|ui| {
                ui.heading("Traditional PCS");
                ui.label("(PID + MPC)");
                ui.label(format!("Response: {:.0} μs", control.pcs_response_time * 1e6));

                let status_color = if control.pcs_can_respond {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                let status_text = if control.pcs_can_respond { "✓ CAN RESPOND" } else { "✗ TOO SLOW" };
                ui.colored_label(status_color, status_text);

                let response_ratio = (control.pcs_response_time / control.fastest_event_timescale).min(2.0);
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(if control.pcs_can_respond {
                        format!("{:.1}x margin", control.fastest_event_timescale / control.pcs_response_time)
                    } else {
                        format!("{:.1}x too slow", control.pcs_response_time / control.fastest_event_timescale)
                    })
                    .fill(if control.pcs_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.label(format!("Failures: {}", control.pcs_failures));
            });

            // DeepMind Column
            columns[2].vertical_centered(|ui| {
                ui.heading("DeepMind AI");
                ui.label("(ML Inference)");
                ui.label(format!("Response: {:.1} ms", control.deepmind_response_time * 1e3));

                let status_color = if control.deepmind_can_respond {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                let status_text = if control.deepmind_can_respond { "✓ CAN RESPOND" } else { "✗ TOO SLOW" };
                ui.colored_label(status_color, status_text);

                let response_ratio = (control.deepmind_response_time / control.fastest_event_timescale).min(2.0);
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(if control.deepmind_can_respond {
                        format!("{:.1}x margin", control.fastest_event_timescale / control.deepmind_response_time)
                    } else {
                        format!("{:.0}x too slow", control.deepmind_response_time / control.fastest_event_timescale)
                    })
                    .fill(if control.deepmind_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.label(format!("Failures: {}", control.deepmind_failures));
            });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Reset Failure Counters").clicked() {
                control.reset_failures();
            }
            ui.separator();
            ui.label("⚠️ Note: Response times are algorithmic latency only.");
            ui.label("Excludes: sensor acquisition, communication, actuator delays.");
        });
    });

    // Left panel - Parameters
    egui::SidePanel::left("params_panel").min_width(280.0).show(contexts.ctx_mut(), |ui| {
        ui.heading("🔧 Geometry Parameters");
        ui.separator();

        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("R₀ [m]:");
            changed |= ui.add(egui::Slider::new(&mut params.major_radius, 0.5..=10.0).step_by(0.1)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("a [m]:");
            changed |= ui.add(egui::Slider::new(&mut params.minor_radius, 0.1..=3.0).step_by(0.05)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("κ (elongation):");
            changed |= ui.add(egui::Slider::new(&mut params.elongation, 1.0..=2.5).step_by(0.05)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("δ (triangularity):");
            changed |= ui.add(egui::Slider::new(&mut params.triangularity, 0.0..=0.8).step_by(0.02)).changed();
        });

        ui.separator();
        ui.heading("🧲 Magnetic Field");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Bt [T]:");
            changed |= ui.add(egui::Slider::new(&mut params.toroidal_field, 1.0..=30.0).step_by(0.5)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Ip [MA]:");
            changed |= ui.add(egui::Slider::new(&mut params.plasma_current, 0.1..=20.0).step_by(0.1)).changed();
        });

        ui.separator();
        ui.heading("🔥 Plasma");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("n̄e [10²⁰/m³]:");
            ui.add(egui::Slider::new(&mut params.density, 0.1..=5.0).step_by(0.1));
        });

        ui.horizontal(|ui| {
            ui.label("T [keV]:");
            ui.add(egui::Slider::new(&mut params.temperature, 1.0..=50.0).step_by(1.0));
        });

        if changed {
            params.needs_geometry_update = true;
        }

        ui.separator();
        ui.heading("⌨️ Presets");
        ui.horizontal(|ui| {
            if ui.button("1: TS-1").clicked() {
                params.major_radius = 1.5;
                params.minor_radius = 0.6;
                params.elongation = 1.97;
                params.triangularity = 0.54;
                params.toroidal_field = 25.0;
                params.plasma_current = 12.0;
                params.needs_geometry_update = true;
            }
            if ui.button("2: SPARC").clicked() {
                params.major_radius = 1.85;
                params.minor_radius = 0.57;
                params.elongation = 1.8;
                params.triangularity = 0.4;
                params.toroidal_field = 12.2;
                params.plasma_current = 8.7;
                params.needs_geometry_update = true;
            }
            if ui.button("3: ITER").clicked() {
                params.major_radius = 6.2;
                params.minor_radius = 2.0;
                params.elongation = 1.7;
                params.triangularity = 0.33;
                params.toroidal_field = 5.3;
                params.plasma_current = 15.0;
                params.needs_geometry_update = true;
            }
        });

        ui.separator();
        ui.heading("🌡️ Heat Map");
        ui.separator();

        ui.checkbox(&mut heat_map.enabled, "Show Heat Map");

        if heat_map.enabled {
            ui.checkbox(&mut heat_map.show_cross_section, "Show Cross-Section");

            // Pizza slice position
            ui.horizontal(|ui| {
                ui.label("Toroidal pos [°]:");
                let mut phi_deg = heat_map.toroidal_position.to_degrees();
                if ui.add(egui::Slider::new(&mut phi_deg, 0.0..=360.0).step_by(5.0)).changed() {
                    heat_map.toroidal_position = phi_deg.to_radians();
                }
            });

            ui.horizontal(|ui| {
                ui.label("Slice width [°]:");
                let mut width_deg = heat_map.toroidal_width.to_degrees();
                if ui.add(egui::Slider::new(&mut width_deg, 5.0..=90.0).step_by(5.0)).changed() {
                    heat_map.toroidal_width = width_deg.to_radians();
                }
            });

            ui.separator();
            ui.label("Temperature Profile:");

            ui.horizontal(|ui| {
                ui.label("T_core [keV]:");
                ui.add(egui::Slider::new(&mut heat_map.core_temperature, 5.0..=50.0).step_by(1.0));
            });

            ui.horizontal(|ui| {
                ui.label("T_edge [keV]:");
                ui.add(egui::Slider::new(&mut heat_map.edge_temperature, 0.1..=5.0).step_by(0.1));
            });

            ui.horizontal(|ui| {
                ui.label("Profile α:");
                ui.add(egui::Slider::new(&mut heat_map.profile_alpha, 1.0..=4.0).step_by(0.1));
            });

            ui.separator();
            ui.label("Color Map:");
            ui.horizontal(|ui| {
                if ui.selectable_label(heat_map.color_map == ColorMapType::Plasma, "Plasma").clicked() {
                    heat_map.color_map = ColorMapType::Plasma;
                }
                if ui.selectable_label(heat_map.color_map == ColorMapType::Inferno, "Inferno").clicked() {
                    heat_map.color_map = ColorMapType::Inferno;
                }
                if ui.selectable_label(heat_map.color_map == ColorMapType::Viridis, "Viridis").clicked() {
                    heat_map.color_map = ColorMapType::Viridis;
                }
                if ui.selectable_label(heat_map.color_map == ColorMapType::Temperature, "Temp").clicked() {
                    heat_map.color_map = ColorMapType::Temperature;
                }
            });

            // Color bar preview
            ui.horizontal(|ui| {
                ui.label("Scale:");
                for i in 0..20 {
                    let v = i as f32 / 19.0;
                    let color = heat_map.color_map.to_color(v);
                    if let Color::Srgba(srgba) = color.into() {
                        let egui_color = egui::Color32::from_rgb(
                            (srgba.red * 255.0) as u8,
                            (srgba.green * 255.0) as u8,
                            (srgba.blue * 255.0) as u8,
                        );
                        ui.colored_label(egui_color, "█");
                    }
                }
            });

            // Temperature readout
            ui.separator();
            ui.label(format!("Core: {:.1} keV", heat_map.core_temperature));
            ui.label(format!("Edge: {:.1} keV", heat_map.edge_temperature));
            ui.label(format!("Gradient: {:.1}x", heat_map.core_temperature / heat_map.edge_temperature.max(0.1)));
        }
    });

    // Right panel - Limits and Analysis
    egui::SidePanel::right("limits_panel").min_width(300.0).show(contexts.ctx_mut(), |ui| {
        ui.heading("📊 Derived Parameters");
        ui.separator();

        ui.label(format!("Aspect ratio A = {:.2}", limits.aspect_ratio));
        ui.label(format!("Plasma volume = {:.1} m³", limits.plasma_volume));

        ui.separator();
        ui.heading("⚠️ MHD Stability (β limit)");

        let beta_color = if limits.beta_exceeded { egui::Color32::RED } else { egui::Color32::GREEN };
        ui.colored_label(beta_color, format!("β = {:.2}% / {:.2}% (Troyon)", limits.beta_actual, limits.beta_troyon_limit));

        ui.add(egui::ProgressBar::new(limits.beta_actual / limits.beta_troyon_limit.max(0.01))
            .text(format!("{:.0}% of limit", 100.0 * limits.beta_actual / limits.beta_troyon_limit.max(0.01)))
            .fill(if limits.beta_exceeded { egui::Color32::RED } else { egui::Color32::DARK_GREEN }));

        ui.separator();
        ui.heading("⚠️ Density Limit (Greenwald)");

        let gw_color = if limits.density_exceeded { egui::Color32::RED } else { egui::Color32::GREEN };
        ui.colored_label(gw_color, format!("n/n_G = {:.2} (n_G = {:.2}×10²⁰/m³)",
            limits.greenwald_fraction, limits.greenwald_density));

        ui.add(egui::ProgressBar::new(limits.greenwald_fraction.min(1.5))
            .text(format!("{:.0}% of Greenwald", 100.0 * limits.greenwald_fraction))
            .fill(if limits.density_exceeded { egui::Color32::RED } else { egui::Color32::DARK_GREEN }));

        ui.separator();
        ui.heading("⚠️ Safety Factor q95");

        let q_color = if limits.q95_too_low { egui::Color32::RED } else { egui::Color32::GREEN };
        ui.colored_label(q_color, format!("q95 = {:.2} (min safe: {:.1})", limits.q95, limits.q95_min_safe));

        if limits.q95_too_low {
            ui.colored_label(egui::Color32::RED, "⚠️ DISRUPTION RISK - Increase q95!");
        }

        ui.separator();
        ui.heading("🔬 Confinement");

        ui.label(format!("τ_E (ITER98) = {:.3} s", limits.tau_e_iter98));
        ui.label(format!("n·T·τ = {:.2}×10²¹ keV·s/m³", limits.triple_product));

        let lawson = 3.0;  // Ignition threshold ~3×10²¹
        let ignition_frac = limits.triple_product / lawson;
        ui.add(egui::ProgressBar::new(ignition_frac.min(2.0) / 2.0)
            .text(format!("{:.0}% of ignition", 100.0 * ignition_frac))
            .fill(if ignition_frac >= 1.0 { egui::Color32::GOLD } else { egui::Color32::DARK_BLUE }));

        ui.separator();
        ui.heading("⚡ Fusion Performance");

        ui.label(format!("P_fusion = {:.0} MW", limits.fusion_power));
        ui.label(format!("Q = {:.1}", limits.q_factor));

        let q_status = if limits.q_factor >= 10.0 { ("✓ IGNITION CAPABLE", egui::Color32::GREEN) }
                       else if limits.q_factor >= 1.0 { ("Net energy gain", egui::Color32::YELLOW) }
                       else { ("Sub-breakeven", egui::Color32::GRAY) };
        ui.colored_label(q_status.1, q_status.0);

        ui.separator();
        ui.heading("🏗️ Structural Loads");

        ui.label(format!("Magnetic pressure = {:.1} MPa", limits.magnetic_pressure));

        let stress_color = if limits.structural_warning { egui::Color32::RED } else { egui::Color32::GREEN };
        ui.colored_label(stress_color, format!("TF coil hoop stress ≈ {:.0} MPa", limits.hoop_stress));

        if limits.structural_warning {
            ui.colored_label(egui::Color32::RED, "⚠️ HIGH STRESS - Strengthen coils!");
        }

        ui.separator();
        ui.heading("📋 Overall Status");

        let all_ok = !limits.beta_exceeded && !limits.density_exceeded && !limits.q95_too_low && !limits.structural_warning;
        if all_ok {
            ui.colored_label(egui::Color32::GREEN, "✅ All limits satisfied");
        } else {
            if limits.beta_exceeded { ui.colored_label(egui::Color32::RED, "❌ β limit exceeded"); }
            if limits.density_exceeded { ui.colored_label(egui::Color32::RED, "❌ Greenwald limit exceeded"); }
            if limits.q95_too_low { ui.colored_label(egui::Color32::RED, "❌ q95 too low (disruption risk)"); }
            if limits.structural_warning { ui.colored_label(egui::Color32::YELLOW, "⚠️ High structural stress"); }
        }
    });

    // Bottom panel
    egui::TopBottomPanel::bottom("help_panel").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.label("TOKASIM-RS Explorer | Mouse=Rotate, Scroll=Zoom | 1=TS-1, 2=SPARC, 3=ITER");
        });
    });
}
