//! # TOKASIM-RS Parameter Explorer v2.0
//!
//! Advanced interactive parameter exploration for tokamak design optimization.
//! Features comprehensive materials system, control system comparison,
//! event simulation, and real-time physics constraints.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin tokasim-explorer-v2 --features bevy-viz --release
//! ```
//!
//! ## Features
//!
//! - Interactive sliders with +/- buttons (0.01 increment)
//! - Dynamic parameter limits based on physics constraints
//! - Comprehensive materials database with selectable components
//! - Control system comparison (PIRS vs PCS vs DeepMind)
//! - Event simulation (ELM, VDE, Sensor Noise, Random)
//! - Parallel controller execution with Rayon
//! - Heat map visualization with cross-sectional cuts
//! - Failure logging with export capability
//! - Thickness calculations based on power/field requirements
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
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

// Import materials from the library
use tokasim_rs::materials::{
    Material, MaterialCategory, TokamakMaterials,
    SUPERCONDUCTORS, PLASMA_FACING, STRUCTURAL, BLANKET_MATERIALS,
    calculate_tf_coil_thickness, calculate_wall_thickness,
    NB3SN, REBCO, YBCO, TUNGSTEN, BERYLLIUM, SS316L, EUROFER97, PBLI,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Format simulation time as HH:MM:SS.mmm
fn format_simulation_time(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let millis = ((seconds - total_secs as f64) * 1000.0) as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
    } else if mins > 0 {
        format!("{:02}:{:02}.{:03}", mins, secs, millis)
    } else {
        format!("{:02}.{:03}s", secs, millis)
    }
}

// ============================================================================
// COMPONENTS
// ============================================================================

#[derive(Component)]
struct Plasma;

#[derive(Component)]
struct VacuumVessel;

#[derive(Component)]
#[allow(dead_code)]
struct TFCoil(usize);

#[derive(Component)]
struct HeatMapSlice;

#[derive(Component)]
struct CrossSectionPlane;

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
// TOKAMAK PARAMETERS
// ============================================================================

#[derive(Resource)]
struct TokamakParams {
    // Geometry
    major_radius: f32,      // R₀ [m]
    minor_radius: f32,      // a [m]
    elongation: f32,        // κ
    triangularity: f32,     // δ

    // Magnetic field
    toroidal_field: f32,    // Bt [T]
    plasma_current: f32,    // Ip [MA]

    // Plasma
    density: f32,           // n̄e [10²⁰ m⁻³]
    temperature: f32,       // T [keV]

    // Operational
    pulse_duration: f32,    // τ_pulse [s]
    heating_power: f32,     // P_heat [MW]

    // Materials configuration
    materials: TokamakMaterials,

    // State flags
    needs_geometry_update: bool,
    needs_thickness_update: bool,
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
            pulse_duration: 10.0,
            heating_power: 50.0,
            materials: TokamakMaterials::ts1_preset(),
            needs_geometry_update: false,
            needs_thickness_update: true,
        }
    }
}

// ============================================================================
// PARAMETER INFO FOR TOOLTIPS
// ============================================================================

struct ParamInfo {
    name: &'static str,
    symbol: &'static str,
    unit: &'static str,
    description: &'static str,
    affects: &'static str,
    limits_reason: &'static str,
}

const PARAM_INFO: &[ParamInfo] = &[
    ParamInfo {
        name: "Major Radius",
        symbol: "R₀",
        unit: "m",
        description: "Distance from torus center to plasma center. Defines overall reactor size.",
        affects: "Plasma volume, confinement time, cost. Larger = better confinement but more expensive.",
        limits_reason: "Min: structural feasibility. Max: practical/economic limits.",
    },
    ParamInfo {
        name: "Minor Radius",
        symbol: "a",
        unit: "m",
        description: "Cross-sectional radius of the plasma column.",
        affects: "Plasma volume, aspect ratio, stability. Must be < R₀/2.5 for stability.",
        limits_reason: "Limited by aspect ratio A = R₀/a > 2.5 for MHD stability.",
    },
    ParamInfo {
        name: "Elongation",
        symbol: "κ",
        unit: "-",
        description: "Ratio of plasma height to width. κ=1 is circular, κ>1 is vertically elongated.",
        affects: "Plasma volume, β limit, vertical stability. Higher κ = more volume but less stable.",
        limits_reason: "Max ~2.2: above this, vertical instability grows faster than control can respond.",
    },
    ParamInfo {
        name: "Triangularity",
        symbol: "δ",
        unit: "-",
        description: "D-shape of plasma cross-section. δ=0 is elliptical, δ>0 is D-shaped.",
        affects: "Edge stability, ELM behavior, divertor geometry. Higher δ improves confinement.",
        limits_reason: "Max ~0.7: higher values cause manufacturing and control difficulties.",
    },
    ParamInfo {
        name: "Toroidal Field",
        symbol: "Bt",
        unit: "T",
        description: "Magnetic field strength in toroidal direction. Main confinement field.",
        affects: "Confinement, β limit, fusion power. Higher field = better confinement.",
        limits_reason: "Limited by superconductor critical field. REBCO: 20T, YBCO: 30T+.",
    },
    ParamInfo {
        name: "Plasma Current",
        symbol: "Ip",
        unit: "MA",
        description: "Toroidal current flowing through plasma. Creates poloidal field for stability.",
        affects: "Confinement, ohmic heating, safety factor q. Higher Ip = better confinement.",
        limits_reason: "Limited by q95 > 2 requirement to avoid disruptions.",
    },
    ParamInfo {
        name: "Density",
        symbol: "n̄e",
        unit: "10²⁰/m³",
        description: "Average electron density in plasma.",
        affects: "Fusion rate, radiation losses, opacity. Must balance for optimal burn.",
        limits_reason: "Max: Greenwald limit n_G = Ip/(πa²). Above this, plasma disrupts.",
    },
    ParamInfo {
        name: "Temperature",
        symbol: "T",
        unit: "keV",
        description: "Core plasma temperature. 1 keV ≈ 11.6 million Kelvin.",
        affects: "Fusion rate (peaks at ~15-20 keV for D-T), radiation, confinement.",
        limits_reason: "Must exceed ~10 keV for significant fusion. Limited by heating power and losses.",
    },
];

fn get_param_info(index: usize) -> Option<&'static ParamInfo> {
    PARAM_INFO.get(index)
}

// ============================================================================
// CALCULATED THICKNESSES
// ============================================================================

#[derive(Resource, Default)]
struct ComponentThicknesses {
    tf_coil: f32,           // [m]
    first_wall: f32,        // [m]
    blanket: f32,           // [m]
    vacuum_vessel: f32,     // [m]
    total_build: f32,       // [m] total radial build
}

impl ComponentThicknesses {
    fn calculate(params: &TokamakParams, limits: &PhysicsLimits) -> Self {
        let tf_coil = calculate_tf_coil_thickness(
            params.toroidal_field,
            params.major_radius,
            &SS316L,  // Structural support
            2.5,
        );

        // Heat flux estimation: P_heat / surface area
        let surface_area = 4.0 * PI * PI * params.major_radius * params.minor_radius * params.elongation;
        let heat_flux = params.heating_power / surface_area;  // MW/m²

        let first_wall = calculate_wall_thickness(
            heat_flux,
            params.pulse_duration,
            params.materials.first_wall,
            500.0,  // Allow 500K temperature rise
        );

        let blanket = 0.5;  // 50 cm typical for TBR > 1.1

        let vacuum_vessel = 0.06;  // 6 cm typical

        let total_build = tf_coil + first_wall + blanket + vacuum_vessel + 0.1;  // +10cm gaps

        Self {
            tf_coil,
            first_wall,
            blanket,
            vacuum_vessel,
            total_build,
        }
    }
}

// ============================================================================
// PHYSICS LIMITS WITH DYNAMIC CONSTRAINTS
// ============================================================================

#[derive(Resource, Default)]
#[allow(dead_code)]
struct PhysicsLimits {
    // Calculated limits
    aspect_ratio: f32,
    plasma_volume: f32,

    // MHD Stability
    beta_actual: f32,
    beta_troyon_limit: f32,
    beta_margin: f32,

    // Density limit
    greenwald_density: f32,
    greenwald_fraction: f32,

    // Safety factor
    q95: f32,
    q95_min_safe: f32,

    // Confinement
    tau_e_iter98: f32,
    triple_product: f32,

    // Fusion performance
    fusion_power: f32,
    net_power: f32,
    q_factor: f32,

    // Structural
    magnetic_pressure: f32,
    hoop_stress: f32,

    // Dynamic parameter limits (based on current state)
    min_minor_radius: f32,
    max_minor_radius: f32,
    max_toroidal_field: f32,
    max_plasma_current: f32,
    max_elongation: f32,
    max_triangularity: f32,

    // Status flags
    beta_exceeded: bool,
    density_exceeded: bool,
    q95_too_low: bool,
    structural_warning: bool,
    geometry_invalid: bool,
}

impl PhysicsLimits {
    fn calculate(params: &TokamakParams) -> Self {
        let r0 = params.major_radius;
        let a = params.minor_radius;
        let kappa = params.elongation;
        let _delta = params.triangularity;
        let bt = params.toroidal_field;
        let ip = params.plasma_current;
        let ne = params.density;
        let te = params.temperature;

        let aspect_ratio = r0 / a;
        let plasma_volume = 2.0 * PI * PI * r0 * a * a * kappa;

        // Beta calculation
        let mu0: f32 = 4.0 * PI * 1e-7;
        let n_si = ne * 1e20;
        let t_si = te * 1.602e-16;
        let pressure = n_si * t_si;
        let beta_actual = 100.0 * 2.0 * mu0 * pressure / (bt * bt);
        let beta_troyon_limit = 2.8 * ip / (a * bt);
        let beta_margin = beta_troyon_limit - beta_actual;

        // Greenwald density limit
        let greenwald_density = ip / (PI * a * a);
        let greenwald_fraction = ne / greenwald_density;

        // Safety factor q95
        let kappa_factor = (1.0 + kappa * kappa) / 2.0;
        let q95 = 5.0 * a * a * bt * kappa_factor / (r0 * ip);
        let q95_min_safe = 2.0;

        // ITER98 H-mode confinement
        let p_heat = params.heating_power;
        let tau_e = 0.0562 * ip.powf(0.93) * bt.powf(0.15) * ne.powf(0.41)
                    * p_heat.powf(-0.69) * r0.powf(1.97) * a.powf(0.58)
                    * kappa.powf(0.78) * 2.5_f32.powf(0.19);

        let triple_product = ne * te * tau_e;

        // Fusion power
        let sigma_v = 4e-22_f32 * (te / 20.0).powf(2.0).min(2.0);
        let e_fus: f32 = 17.6e6 * 1.602e-19;
        let fusion_power = (n_si * n_si) * sigma_v * e_fus * plasma_volume / 4.0 / 1e6;
        let net_power = fusion_power - p_heat;
        let q_factor = fusion_power / p_heat.max(0.1);

        // Structural
        let magnetic_pressure = (bt * bt) / (2.0 * mu0) / 1e6;
        let coil_thickness = 0.3;
        let hoop_stress = (bt * bt * r0) / (2.0 * mu0 * coil_thickness) / 1e6;

        // Dynamic limits based on current configuration
        let min_minor_radius = r0 / 4.0;  // Aspect ratio < 4
        let max_minor_radius = r0 / 2.5;  // Aspect ratio > 2.5
        let max_toroidal_field = params.materials.max_toroidal_field();
        let max_plasma_current = 5.0 * a * a * bt * kappa_factor / (r0 * q95_min_safe);
        let max_elongation = 2.2;
        let max_triangularity = 0.7;

        // Status flags
        let beta_exceeded = beta_actual > beta_troyon_limit * 0.9;
        let density_exceeded = greenwald_fraction > 0.85;
        let q95_too_low = q95 < q95_min_safe * 1.2;
        let structural_warning = hoop_stress > 500.0;
        let geometry_invalid = aspect_ratio < 2.5 || aspect_ratio > 4.5;

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
            net_power,
            q_factor,
            magnetic_pressure,
            hoop_stress,
            min_minor_radius,
            max_minor_radius,
            max_toroidal_field,
            max_plasma_current,
            max_elongation,
            max_triangularity,
            beta_exceeded,
            density_exceeded,
            q95_too_low,
            structural_warning,
            geometry_invalid,
        }
    }
}

// ============================================================================
// CONTROL SYSTEMS
// ============================================================================

#[derive(Clone, Copy, PartialEq, Default, Debug)]
enum InstabilityType {
    #[default]
    None,
    EdgeLocalizedMode,
    SawtoothCrash,
    VerticalDisplacement,
    LockedMode,
    Disruption,
    MinorDisruption,
    NeoClassicalTearing,
    ResistiveWallMode,
    SensorNoise,
}

impl InstabilityType {
    fn timescale(&self) -> f32 {
        match self {
            InstabilityType::None => 1.0,
            InstabilityType::EdgeLocalizedMode => 5e-4,
            InstabilityType::SawtoothCrash => 5e-2,
            InstabilityType::VerticalDisplacement => 2e-3,
            InstabilityType::LockedMode => 5e-2,
            InstabilityType::Disruption => 2e-3,
            InstabilityType::MinorDisruption => 3e-2,
            InstabilityType::NeoClassicalTearing => 5e-1,
            InstabilityType::ResistiveWallMode => 1e-2,
            InstabilityType::SensorNoise => 1e-2,
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
            InstabilityType::SensorNoise => "Sensor Noise Injection",
        }
    }

    fn severity(&self) -> u8 {
        match self {
            InstabilityType::None => 0,
            InstabilityType::NeoClassicalTearing => 1,
            InstabilityType::SawtoothCrash => 2,
            InstabilityType::LockedMode => 3,
            InstabilityType::SensorNoise => 3,
            InstabilityType::MinorDisruption => 4,
            InstabilityType::ResistiveWallMode => 5,
            InstabilityType::EdgeLocalizedMode => 6,
            InstabilityType::VerticalDisplacement => 8,
            InstabilityType::Disruption => 10,
        }
    }

    fn description(&self) -> &'static str {
        match self {
            InstabilityType::None => "Normal plasma operation within all limits.",
            InstabilityType::EdgeLocalizedMode => "Sudden 5% energy loss at plasma edge. Tests controller temperature recovery speed.",
            InstabilityType::SawtoothCrash => "Periodic core temperature crash and recovery. Natural in tokamaks.",
            InstabilityType::VerticalDisplacement => "Plasma moves 2cm upward. CRITICAL: Must correct in <1ms or plasma hits wall.",
            InstabilityType::LockedMode => "Rotating MHD mode locks to wall. Can lead to disruption.",
            InstabilityType::Disruption => "Complete plasma termination. Thermal quench in ~2ms.",
            InstabilityType::MinorDisruption => "Partial energy loss without full termination.",
            InstabilityType::NeoClassicalTearing => "Slowly growing magnetic island. Can degrade confinement.",
            InstabilityType::ResistiveWallMode => "Instability limited by wall resistivity. Needs active control.",
            InstabilityType::SensorNoise => "Gaussian noise added to sensors. Tests controller robustness.",
        }
    }
}

#[derive(Resource)]
struct ControlSystems {
    // Response times (seconds)
    pirs_response_time: f32,
    pcs_response_time: f32,
    deepmind_response_time: f32,

    // Current state
    current_event_rate: f32,
    fastest_event_timescale: f32,
    instability_type: InstabilityType,

    // Manual event triggers
    trigger_elm: bool,
    trigger_vde: bool,
    trigger_noise: bool,
    trigger_random: bool,
    random_seed: u64,

    // Noise parameters
    noise_sigma: f32,  // % of signal

    // System status
    pirs_can_respond: bool,
    pcs_can_respond: bool,
    deepmind_can_respond: bool,

    // Failure tracking
    pirs_failures: u32,
    pcs_failures: u32,
    deepmind_failures: u32,

    // Failure logs
    failure_logs: Arc<Mutex<VecDeque<FailureLogEntry>>>,
    show_log_window: bool,

    // Simulation control
    simulation_running: bool,
    simulation_time: f64,
    simulation_speed: f32,  // 1.0 = real-time, 10.0 = 10x faster

    // Next random event timing
    next_random_event_time: f64,
    random_event_interval: f32,  // seconds between random events

    // Current parameter snapshot for logging
    current_snapshot: ParamSnapshot,
}

#[derive(Clone, Debug)]
struct FailureLogEntry {
    id: u32,
    timestamp: f64,
    controller: String,
    event_type: String,
    event_timescale_ms: f32,
    response_time_ms: f32,
    success: bool,
    margin: f32,
    details: String,
    // Parameter snapshot at event time
    param_r0: f32,
    param_a: f32,
    param_bt: f32,
    param_ip: f32,
    param_kappa: f32,
    param_delta: f32,
    param_density: f32,
    param_beta: f32,
    param_q95: f32,
}

/// Snapshot of parameters for logging
#[derive(Clone, Debug, Default)]
struct ParamSnapshot {
    r0: f32,
    a: f32,
    bt: f32,
    ip: f32,
    kappa: f32,
    delta: f32,
    density: f32,
    beta: f32,
    q95: f32,
}

impl ParamSnapshot {
    fn from_params(params: &TokamakParams, limits: &PhysicsLimits) -> Self {
        Self {
            r0: params.major_radius,
            a: params.minor_radius,
            bt: params.toroidal_field,
            ip: params.plasma_current,
            kappa: params.elongation,
            delta: params.triangularity,
            density: params.density,
            beta: limits.beta_actual,
            q95: limits.q95,
        }
    }
}

impl Default for ControlSystems {
    fn default() -> Self {
        Self {
            pirs_response_time: 1e-7,      // 0.1 μs
            pcs_response_time: 1e-4,       // 100 μs
            deepmind_response_time: 5e-3,  // 5 ms

            current_event_rate: 0.0,
            fastest_event_timescale: 1.0,
            instability_type: InstabilityType::None,

            trigger_elm: false,
            trigger_vde: false,
            trigger_noise: false,
            trigger_random: false,
            random_seed: 12345,

            noise_sigma: 1.0,

            pirs_can_respond: true,
            pcs_can_respond: true,
            deepmind_can_respond: true,

            pirs_failures: 0,
            pcs_failures: 0,
            deepmind_failures: 0,

            failure_logs: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            show_log_window: false,

            simulation_running: false,
            simulation_time: 0.0,
            simulation_speed: 1.0,

            next_random_event_time: 0.0,
            random_event_interval: 5.0,  // Random event every 5 seconds

            current_snapshot: ParamSnapshot::default(),
        }
    }
}

impl ControlSystems {
    fn update_snapshot(&mut self, params: &TokamakParams, limits: &PhysicsLimits) {
        self.current_snapshot = ParamSnapshot::from_params(params, limits);
    }

    fn update(&mut self, limits: &PhysicsLimits, delta_time: f64) {
        // Only advance simulation time when running
        if !self.simulation_running {
            // Still process manual triggers even when paused
            self.process_manual_triggers();
            return;
        }

        // Advance simulation time
        let time_step = delta_time * self.simulation_speed as f64;
        self.simulation_time += time_step;

        let old_instability = self.instability_type;

        // Check for manual triggers first
        if self.process_manual_triggers() {
            // Manual trigger was processed
        } else if self.simulation_time >= self.next_random_event_time && self.random_event_interval > 0.0 {
            // Automatic random event
            self.trigger_random_event();
            self.next_random_event_time = self.simulation_time + self.random_event_interval as f64;
        } else {
            // Determine instability from physics limits
            self.instability_type = self.determine_instability_from_limits(limits);
        }

        self.fastest_event_timescale = self.instability_type.timescale();
        self.current_event_rate = 1.0 / self.fastest_event_timescale;

        // PARALLEL CONTROLLER RESPONSE CHECK WITH RAYON
        self.check_controller_responses_parallel();

        // Log failures when instability changes
        if self.instability_type != InstabilityType::None && old_instability != self.instability_type {
            self.log_controller_responses();
        }
    }

    fn process_manual_triggers(&mut self) -> bool {
        if self.trigger_elm {
            self.instability_type = InstabilityType::EdgeLocalizedMode;
            self.trigger_elm = false;
            true
        } else if self.trigger_vde {
            self.instability_type = InstabilityType::VerticalDisplacement;
            self.trigger_vde = false;
            true
        } else if self.trigger_noise {
            self.instability_type = InstabilityType::SensorNoise;
            self.trigger_noise = false;
            true
        } else if self.trigger_random {
            self.trigger_random_event();
            self.trigger_random = false;
            true
        } else {
            false
        }
    }

    fn trigger_random_event(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_val: f32 = rng.gen();

        self.instability_type = if random_val < 0.25 {
            InstabilityType::EdgeLocalizedMode
        } else if random_val < 0.45 {
            InstabilityType::VerticalDisplacement
        } else if random_val < 0.60 {
            InstabilityType::SensorNoise
        } else if random_val < 0.75 {
            InstabilityType::SawtoothCrash
        } else if random_val < 0.85 {
            InstabilityType::Disruption
        } else if random_val < 0.92 {
            InstabilityType::ResistiveWallMode
        } else {
            InstabilityType::NeoClassicalTearing
        };
    }

    fn determine_instability_from_limits(&self, limits: &PhysicsLimits) -> InstabilityType {
        if limits.beta_exceeded && limits.q95_too_low {
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
        }
    }

    /// Check controller responses in parallel using Rayon
    fn check_controller_responses_parallel(&mut self) {
        use rayon::prelude::*;

        let event_timescale = self.fastest_event_timescale;

        // Define controller data: (name, response_time)
        let controllers = [
            ("PIRS", self.pirs_response_time),
            ("PCS", self.pcs_response_time),
            ("DeepMind", self.deepmind_response_time),
        ];

        // Process in parallel with Rayon
        let results: Vec<(bool, f32)> = controllers
            .par_iter()
            .map(|(_name, response_time)| {
                // Simulate controller processing (deterministic)
                let can_respond = *response_time < event_timescale;
                // Return (can_respond, actual_response_time_with_jitter)
                (can_respond, *response_time)
            })
            .collect();

        // Update state from parallel results
        self.pirs_can_respond = results[0].0;
        self.pcs_can_respond = results[1].0;
        self.deepmind_can_respond = results[2].0;
    }

    fn log_controller_responses(&mut self) {
        let event_ts_ms = self.fastest_event_timescale * 1000.0;
        let sim_time = self.simulation_time;

        if !self.pirs_can_respond {
            self.pirs_failures += 1;
        }
        self.log_failure("PIRS", &self.instability_type, event_ts_ms,
                        self.pirs_response_time * 1000.0, self.pirs_can_respond, sim_time);

        if !self.pcs_can_respond {
            self.pcs_failures += 1;
        }
        self.log_failure("PCS", &self.instability_type, event_ts_ms,
                        self.pcs_response_time * 1000.0, self.pcs_can_respond, sim_time);

        if !self.deepmind_can_respond {
            self.deepmind_failures += 1;
        }
        self.log_failure("DeepMind", &self.instability_type, event_ts_ms,
                        self.deepmind_response_time * 1000.0, self.deepmind_can_respond, sim_time);
    }

    fn log_failure(&self, controller: &str, event: &InstabilityType, event_ts_ms: f32, response_ms: f32, success: bool, time: f64) {
        if let Ok(mut logs) = self.failure_logs.lock() {
            let id = logs.len() as u32 + 1;
            let margin = event_ts_ms / response_ms;
            let details = if success {
                format!("OK margin={:.1}x", margin)
            } else {
                format!("FAIL slow={:.1}x", response_ms / event_ts_ms)
            };

            let snap = &self.current_snapshot;
            logs.push_back(FailureLogEntry {
                id,
                timestamp: time,
                controller: controller.to_string(),
                event_type: event.name().to_string(),
                event_timescale_ms: event_ts_ms,
                response_time_ms: response_ms,
                success,
                margin,
                details,
                param_r0: snap.r0,
                param_a: snap.a,
                param_bt: snap.bt,
                param_ip: snap.ip,
                param_kappa: snap.kappa,
                param_delta: snap.delta,
                param_density: snap.density,
                param_beta: snap.beta,
                param_q95: snap.q95,
            });

            // Keep only last 1000 entries
            while logs.len() > 1000 {
                logs.pop_front();
            }
        }
    }

    fn reset_failures(&mut self) {
        self.pirs_failures = 0;
        self.pcs_failures = 0;
        self.deepmind_failures = 0;
        if let Ok(mut logs) = self.failure_logs.lock() {
            logs.clear();
        }
    }

    fn export_logs(&self) -> String {
        let mut csv = String::from(
            "id,timestamp_s,controller,event,event_timescale_ms,response_time_ms,success,margin,\
             R0_m,a_m,Bt_T,Ip_MA,kappa,delta,n_e20,beta_pct,q95,details\n"
        );
        if let Ok(logs) = self.failure_logs.lock() {
            for entry in logs.iter() {
                csv.push_str(&format!(
                    "{},{:.6},{},{},{:.4},{:.6},{},{:.2},{:.3},{:.3},{:.2},{:.2},{:.2},{:.3},{:.2},{:.3},{:.2},{}\n",
                    entry.id,
                    entry.timestamp,
                    entry.controller,
                    entry.event_type,
                    entry.event_timescale_ms,
                    entry.response_time_ms,
                    entry.success,
                    entry.margin,
                    entry.param_r0,
                    entry.param_a,
                    entry.param_bt,
                    entry.param_ip,
                    entry.param_kappa,
                    entry.param_delta,
                    entry.param_density,
                    entry.param_beta * 100.0,  // Convert to %
                    entry.param_q95,
                    entry.details,
                ));
            }
        }
        csv
    }

    /// Save logs to a file
    fn save_logs_to_file(&self) -> std::io::Result<String> {
        use std::io::Write;
        let timestamp = chrono_lite_timestamp();
        let filename = format!("tokasim_failure_log_{}.csv", timestamp);
        let desktop = std::env::var("USERPROFILE")
            .map(|p| format!("{}\\Desktop\\{}", p, filename))
            .unwrap_or(filename.clone());

        let csv = self.export_logs();
        let mut file = std::fs::File::create(&desktop)?;
        file.write_all(csv.as_bytes())?;
        Ok(desktop)
    }
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs();
    // Convert to readable format (basic)
    format!("{}", secs)
}

// ============================================================================
// HEAT MAP STATE
// ============================================================================

#[derive(Resource)]
#[allow(dead_code)]
struct HeatMapState {
    enabled: bool,
    toroidal_position: f32,
    toroidal_width: f32,
    show_cross_section: bool,
    core_temperature: f32,
    edge_temperature: f32,
    profile_alpha: f32,
    profile_beta: f32,
    color_map: ColorMapType,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ColorMapType {
    #[default]
    Plasma,
    Inferno,
    Viridis,
    Temperature,
}

impl ColorMapType {
    fn to_color(&self, value: f32) -> Color {
        let v = value.clamp(0.0, 1.0);
        match self {
            ColorMapType::Plasma => {
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
}

impl Default for HeatMapState {
    fn default() -> Self {
        Self {
            enabled: false,
            toroidal_position: 0.0,
            toroidal_width: PI / 8.0,
            show_cross_section: true,
            core_temperature: 20.0,
            edge_temperature: 0.5,
            profile_alpha: 2.0,
            profile_beta: 1.5,
            color_map: ColorMapType::Plasma,
        }
    }
}

impl HeatMapState {
    fn normalized_temperature(&self, rho: f32) -> f32 {
        let rho_clamped = rho.clamp(0.0, 1.0);
        let profile = (1.0 - rho_clamped.powf(self.profile_alpha)).powf(self.profile_beta);
        let t = self.edge_temperature + (self.core_temperature - self.edge_temperature) * profile;
        (t - self.edge_temperature) / (self.core_temperature - self.edge_temperature).max(0.001)
    }
}

// ============================================================================
// UI STATE
// ============================================================================

#[derive(Resource, Default)]
struct UiState {
    mouse_over_panel: bool,
    show_param_tooltip: Option<usize>,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║     TOKASIM-RS Parameter Explorer v2.0                             ║");
    println!("║     Advanced Tokamak Design Optimization                           ║");
    println!("║     Avermex Research Division - January 2026                       ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Features:");
    println!("  - Interactive sliders with +/- fine control");
    println!("  - Dynamic physics-based parameter limits");
    println!("  - Materials selection for all components");
    println!("  - Control system comparison (PIRS/PCS/DeepMind)");
    println!("  - Event simulation (ELM, VDE, Noise)");
    println!("  - Failure logging with export");
    println!();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "TOKASIM-RS Explorer v2.0".into(),
                resolution: (1800., 1000.).into(),
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
        .insert_resource(ComponentThicknesses::default())
        .insert_resource(UiState::default())
        .insert_resource(Time::<Virtual>::default())
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
            update_thicknesses,
            regenerate_geometry,
        ))
        .run();
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

    // TF Coils
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
    ui_state: Res<UiState>,
) {
    let Ok((mut transform, mut orbit)) = query.get_single_mut() else { return };

    // Don't rotate camera if mouse is over UI panel
    if !ui_state.mouse_over_panel && mouse_button.pressed(MouseButton::Left) {
        for ev in mouse_motion.read() {
            orbit.azimuth -= ev.delta.x * 0.005;
            orbit.elevation = (orbit.elevation - ev.delta.y * 0.005).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
        }
    } else {
        mouse_motion.clear();
    }

    if !ui_state.mouse_over_panel {
        for ev in scroll.read() {
            orbit.radius = (orbit.radius * (1.0 - ev.y * 0.1)).clamp(2.0, 30.0);
        }
    } else {
        scroll.clear();
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
    time: Res<Time>,
) {
    // Update parameter snapshot for logging
    control.update_snapshot(&params, &limits);
    // Update simulation
    control.update(&limits, time.delta_seconds_f64());
}

fn update_thicknesses(
    params: Res<TokamakParams>,
    limits: Res<PhysicsLimits>,
    mut thicknesses: ResMut<ComponentThicknesses>,
) {
    if params.needs_thickness_update {
        *thicknesses = ComponentThicknesses::calculate(&params, &limits);
    }
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
    params.needs_thickness_update = true;

    for entity in plasma_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in vessel_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in coil_query.iter() {
        commands.entity(entity).despawn();
    }

    spawn_tokamak_geometry(&mut commands, &mut meshes, &mut materials, &params);
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
    thicknesses: Res<ComponentThicknesses>,
    mut ui_state: ResMut<UiState>,
) {
    let ctx = contexts.ctx_mut();

    // Track if mouse is over any panel
    ui_state.mouse_over_panel = false;

    // ========================================================================
    // TOP PANEL - CONTROL SYSTEMS COMPARISON
    // ========================================================================
    egui::TopBottomPanel::top("control_panel").min_height(200.0).show(ctx, |ui| {
        if ui.rect_contains_pointer(ui.min_rect()) {
            ui_state.mouse_over_panel = true;
        }

        // =====================================================================
        // SIMULATION CONTROL BAR
        // =====================================================================
        ui.horizontal(|ui| {
            // Play/Pause button
            let play_text = if control.simulation_running { "⏸ PAUSE" } else { "▶ PLAY" };
            let play_color = if control.simulation_running {
                egui::Color32::from_rgb(255, 165, 0)  // Orange when running
            } else {
                egui::Color32::from_rgb(50, 205, 50)   // Green when paused
            };

            if ui.add(egui::Button::new(egui::RichText::new(play_text).size(16.0).color(egui::Color32::BLACK))
                .fill(play_color)
                .min_size(egui::vec2(80.0, 28.0)))
                .clicked()
            {
                control.simulation_running = !control.simulation_running;
                if control.simulation_running && control.next_random_event_time == 0.0 {
                    control.next_random_event_time = control.simulation_time + control.random_event_interval as f64;
                }
            }

            // Reset button
            if ui.button("⟲ Reset").clicked() {
                control.simulation_time = 0.0;
                control.next_random_event_time = 0.0;
                control.pirs_failures = 0;
                control.pcs_failures = 0;
                control.deepmind_failures = 0;
                if let Ok(mut logs) = control.failure_logs.lock() {
                    logs.clear();
                }
            }

            ui.separator();

            // Simulation time display
            let time_str = format_simulation_time(control.simulation_time);
            ui.label(egui::RichText::new(format!("⏱ {}", time_str)).size(14.0).strong());

            ui.separator();

            // Speed control
            ui.label("Speed:");
            ui.add(egui::Slider::new(&mut control.simulation_speed, 0.1..=100.0)
                .logarithmic(true)
                .suffix("x")
                .max_decimals(1));

            ui.separator();

            // Auto random events
            ui.label("Auto events:");
            ui.add(egui::DragValue::new(&mut control.random_event_interval)
                .speed(0.1)
                .range(0.0..=60.0)
                .suffix(" s"));
            if control.random_event_interval > 0.0 && control.simulation_running {
                let next_in = (control.next_random_event_time - control.simulation_time).max(0.0);
                ui.label(format!("(next: {:.1}s)", next_in));
            }
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.heading("Control Systems Comparison");
            ui.separator();

            let instability_color = match control.instability_type.severity() {
                0 => egui::Color32::GREEN,
                1..=3 => egui::Color32::YELLOW,
                4..=6 => egui::Color32::from_rgb(255, 165, 0),
                7..=9 => egui::Color32::RED,
                _ => egui::Color32::from_rgb(139, 0, 0),
            };
            ui.colored_label(instability_color, format!("Event: {}", control.instability_type.name()));

            if control.instability_type != InstabilityType::None {
                ui.label(format!("| t_event = {:.2e} s", control.fastest_event_timescale));
            }
        });

        ui.separator();

        // Event triggers (manual)
        ui.horizontal(|ui| {
            ui.label("Manual Trigger:");
            if ui.button("ELM Crash").on_hover_text(InstabilityType::EdgeLocalizedMode.description()).clicked() {
                control.trigger_elm = true;
            }
            if ui.button("VDE Kick").on_hover_text(InstabilityType::VerticalDisplacement.description()).clicked() {
                control.trigger_vde = true;
            }
            if ui.button("Sensor Noise").on_hover_text(InstabilityType::SensorNoise.description()).clicked() {
                control.trigger_noise = true;
            }
            ui.separator();
            if ui.button("🎲 RANDOM").on_hover_text("Trigger random instability").clicked() {
                control.trigger_random = true;
            }
        });

        ui.separator();

        // Control systems grid
        ui.columns(3, |columns| {
            // PIRS Column
            columns[0].vertical_centered(|ui| {
                ui.heading("PIRS");
                ui.label("(Prolog Inference in Rust)");
                ui.label(format!("Response: {:.1} us", control.pirs_response_time * 1e6));

                let (status_color, status_text) = if control.pirs_can_respond {
                    (egui::Color32::GREEN, "CAN RESPOND")
                } else {
                    (egui::Color32::RED, "TOO SLOW")
                };
                ui.colored_label(status_color, status_text);

                let response_ratio = (control.pirs_response_time / control.fastest_event_timescale).min(2.0);
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(format!("{:.0}x margin", control.fastest_event_timescale / control.pirs_response_time))
                    .fill(if control.pirs_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.horizontal(|ui| {
                    ui.label(format!("Failures: {}", control.pirs_failures));
                    if control.pirs_failures > 0 {
                        if ui.small_button("Logs").clicked() {
                            control.show_log_window = true;
                        }
                    }
                });
            });

            // PCS Column
            columns[1].vertical_centered(|ui| {
                ui.heading("Traditional PCS");
                ui.label("(PID + MPC)");
                ui.label(format!("Response: {:.0} us", control.pcs_response_time * 1e6));

                let (status_color, status_text) = if control.pcs_can_respond {
                    (egui::Color32::GREEN, "CAN RESPOND")
                } else {
                    (egui::Color32::RED, "TOO SLOW")
                };
                ui.colored_label(status_color, status_text);

                let response_ratio = (control.pcs_response_time / control.fastest_event_timescale).min(2.0);
                let bar_text = if control.pcs_can_respond {
                    format!("{:.1}x margin", control.fastest_event_timescale / control.pcs_response_time)
                } else {
                    format!("{:.1}x too slow", control.pcs_response_time / control.fastest_event_timescale)
                };
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(bar_text)
                    .fill(if control.pcs_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.horizontal(|ui| {
                    ui.label(format!("Failures: {}", control.pcs_failures));
                    if control.pcs_failures > 0 {
                        if ui.small_button("Logs").clicked() {
                            control.show_log_window = true;
                        }
                    }
                });
            });

            // DeepMind Column
            columns[2].vertical_centered(|ui| {
                ui.heading("DeepMind AI");
                ui.label("(ML Inference)");
                ui.label(format!("Response: {:.1} ms", control.deepmind_response_time * 1e3));

                let (status_color, status_text) = if control.deepmind_can_respond {
                    (egui::Color32::GREEN, "CAN RESPOND")
                } else {
                    (egui::Color32::RED, "TOO SLOW")
                };
                ui.colored_label(status_color, status_text);

                let response_ratio = (control.deepmind_response_time / control.fastest_event_timescale).min(2.0);
                let bar_text = if control.deepmind_can_respond {
                    format!("{:.1}x margin", control.fastest_event_timescale / control.deepmind_response_time)
                } else {
                    format!("{:.0}x too slow", control.deepmind_response_time / control.fastest_event_timescale)
                };
                ui.add(egui::ProgressBar::new(1.0 - response_ratio / 2.0)
                    .text(bar_text)
                    .fill(if control.deepmind_can_respond { egui::Color32::DARK_GREEN } else { egui::Color32::DARK_RED }));

                ui.horizontal(|ui| {
                    ui.label(format!("Failures: {}", control.deepmind_failures));
                    if control.deepmind_failures > 0 {
                        if ui.small_button("Logs").clicked() {
                            control.show_log_window = true;
                        }
                    }
                });
            });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Reset Counters").clicked() {
                control.reset_failures();
            }
            ui.separator();
            ui.label("Note: Response times are algorithmic latency only. Excludes sensor, communication, and actuator delays.");
        });
    });

    // ========================================================================
    // LEFT PANEL - PARAMETERS
    // ========================================================================
    egui::SidePanel::left("params_panel").min_width(320.0).show(ctx, |ui| {
        if ui.rect_contains_pointer(ui.min_rect()) {
            ui_state.mouse_over_panel = true;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Geometry Parameters");
            ui.separator();

            let mut changed = false;

            // Helper macro for parameter slider with +/- buttons
            macro_rules! param_slider {
                ($ui:expr, $label:expr, $value:expr, $min:expr, $max:expr, $step:expr, $info_idx:expr) => {{
                    $ui.horizontal(|ui| {
                        // Info button
                        if let Some(info) = get_param_info($info_idx) {
                            ui.label("i").on_hover_ui(|ui| {
                                ui.heading(info.name);
                                ui.label(format!("{} [{}]", info.symbol, info.unit));
                                ui.separator();
                                ui.label(info.description);
                                ui.separator();
                                ui.label(format!("Affects: {}", info.affects));
                                ui.separator();
                                ui.label(format!("Limits: {}", info.limits_reason));
                            });
                        }

                        // Minus button
                        if ui.small_button("-").clicked() {
                            *$value = (*$value - 0.01).max($min);
                            changed = true;
                        }

                        // Label
                        ui.label($label);

                        // Slider
                        if ui.add(egui::Slider::new($value, $min..=$max).step_by($step as f64)).changed() {
                            changed = true;
                        }

                        // Plus button
                        if ui.small_button("+").clicked() {
                            *$value = (*$value + 0.01).min($max);
                            changed = true;
                        }
                    });
                    // Min/Max labels
                    $ui.horizontal(|ui| {
                        ui.label(format!("    {:.2}", $min));
                        ui.add_space(180.0);
                        ui.label(format!("{:.2}", $max));
                    });
                }};
            }

            param_slider!(ui, "R0 [m]:", &mut params.major_radius, 0.5, 10.0, 0.1, 0);
            param_slider!(ui, "a [m]:", &mut params.minor_radius, limits.min_minor_radius, limits.max_minor_radius, 0.05, 1);
            param_slider!(ui, "kappa:", &mut params.elongation, 1.0, limits.max_elongation, 0.05, 2);
            param_slider!(ui, "delta:", &mut params.triangularity, 0.0, limits.max_triangularity, 0.02, 3);

            ui.separator();
            ui.heading("Magnetic Field");
            ui.separator();

            param_slider!(ui, "Bt [T]:", &mut params.toroidal_field, 1.0, limits.max_toroidal_field, 0.5, 4);
            param_slider!(ui, "Ip [MA]:", &mut params.plasma_current, 0.1, limits.max_plasma_current.min(20.0), 0.1, 5);

            ui.separator();
            ui.heading("Plasma");
            ui.separator();

            param_slider!(ui, "ne [10^20/m3]:", &mut params.density, 0.1, limits.greenwald_density * 0.95, 0.1, 6);
            param_slider!(ui, "T [keV]:", &mut params.temperature, 1.0, 50.0, 1.0, 7);

            if changed {
                params.needs_geometry_update = true;
            }

            ui.separator();
            ui.heading("Presets");
            ui.horizontal(|ui| {
                if ui.button("TS-1").clicked() {
                    *params = TokamakParams::default();
                    params.needs_geometry_update = true;
                }
                if ui.button("SPARC").clicked() {
                    params.major_radius = 1.85;
                    params.minor_radius = 0.57;
                    params.elongation = 1.8;
                    params.triangularity = 0.4;
                    params.toroidal_field = 12.2;
                    params.plasma_current = 8.7;
                    params.materials = TokamakMaterials::sparc_preset();
                    params.needs_geometry_update = true;
                }
                if ui.button("ITER").clicked() {
                    params.major_radius = 6.2;
                    params.minor_radius = 2.0;
                    params.elongation = 1.7;
                    params.triangularity = 0.33;
                    params.toroidal_field = 5.3;
                    params.plasma_current = 15.0;
                    params.materials = TokamakMaterials::iter_preset();
                    params.needs_geometry_update = true;
                }
            });

            ui.separator();
            ui.heading("Materials");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("TF Coil:");
                egui::ComboBox::from_id_source("tf_coil_mat")
                    .selected_text(params.materials.tf_coil.name)
                    .show_ui(ui, |ui| {
                        for mat in SUPERCONDUCTORS {
                            if ui.selectable_label(params.materials.tf_coil.id == mat.id, mat.name)
                                .on_hover_text(mat.description)
                                .clicked() {
                                params.materials.tf_coil = mat;
                                params.needs_geometry_update = true;
                            }
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("First Wall:");
                egui::ComboBox::from_id_source("first_wall_mat")
                    .selected_text(params.materials.first_wall.name)
                    .show_ui(ui, |ui| {
                        for mat in PLASMA_FACING {
                            if ui.selectable_label(params.materials.first_wall.id == mat.id, mat.name)
                                .on_hover_text(mat.description)
                                .clicked() {
                                params.materials.first_wall = mat;
                            }
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Structure:");
                egui::ComboBox::from_id_source("structure_mat")
                    .selected_text(params.materials.structure.name)
                    .show_ui(ui, |ui| {
                        for mat in STRUCTURAL {
                            if ui.selectable_label(params.materials.structure.id == mat.id, mat.name)
                                .on_hover_text(mat.description)
                                .clicked() {
                                params.materials.structure = mat;
                            }
                        }
                    });
            });

            ui.separator();
            ui.heading("Component Thicknesses");
            ui.label(format!("TF Coil: {:.2} m", thicknesses.tf_coil));
            ui.label(format!("First Wall: {:.1} mm", thicknesses.first_wall * 1000.0));
            ui.label(format!("Blanket: {:.2} m", thicknesses.blanket));
            ui.label(format!("Vessel: {:.1} mm", thicknesses.vacuum_vessel * 1000.0));
            ui.label(format!("Total Build: {:.2} m", thicknesses.total_build));
        });
    });

    // ========================================================================
    // RIGHT PANEL - LIMITS AND PERFORMANCE
    // ========================================================================
    egui::SidePanel::right("limits_panel").min_width(320.0).show(ctx, |ui| {
        if ui.rect_contains_pointer(ui.min_rect()) {
            ui_state.mouse_over_panel = true;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            // PROMINENT ENERGY DISPLAY
            ui.add_space(10.0);
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(20, 40, 20))
                .stroke(egui::Stroke::new(2.0, egui::Color32::GOLD))
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("FUSION PERFORMANCE");
                        ui.separator();
                        ui.label(egui::RichText::new(format!("P_fusion = {:.0} MW", limits.fusion_power))
                            .size(20.0).color(egui::Color32::WHITE));
                        ui.label(egui::RichText::new(format!("P_net = {:.0} MW", limits.net_power))
                            .size(16.0).color(if limits.net_power > 0.0 { egui::Color32::GREEN } else { egui::Color32::RED }));
                        ui.label(egui::RichText::new(format!("Q = {:.1}", limits.q_factor))
                            .size(24.0).color(
                                if limits.q_factor >= 10.0 { egui::Color32::GOLD }
                                else if limits.q_factor >= 1.0 { egui::Color32::GREEN }
                                else { egui::Color32::GRAY }
                            ));

                        let q_status = if limits.q_factor >= 10.0 { "IGNITION CAPABLE" }
                                       else if limits.q_factor >= 5.0 { "High Gain" }
                                       else if limits.q_factor >= 1.0 { "Net Energy Gain" }
                                       else { "Sub-breakeven" };
                        ui.label(q_status);
                    });
                });
            ui.add_space(10.0);

            ui.separator();
            ui.heading("Derived Parameters");
            ui.label(format!("Aspect ratio A = {:.2}", limits.aspect_ratio));
            ui.label(format!("Plasma volume = {:.1} m3", limits.plasma_volume));

            ui.separator();
            ui.heading("MHD Stability (beta limit)");

            let beta_color = if limits.beta_exceeded { egui::Color32::RED } else { egui::Color32::GREEN };
            ui.colored_label(beta_color, format!("beta = {:.2}% / {:.2}% (Troyon)", limits.beta_actual, limits.beta_troyon_limit));

            let beta_frac = limits.beta_actual / limits.beta_troyon_limit.max(0.01);
            let bar_text_color = if beta_frac > 0.7 { egui::Color32::BLACK } else { egui::Color32::WHITE };
            ui.add(egui::ProgressBar::new(beta_frac.min(1.5))
                .text(egui::RichText::new(format!("{:.0}% of limit", 100.0 * beta_frac)).color(bar_text_color))
                .fill(if limits.beta_exceeded { egui::Color32::RED } else { egui::Color32::DARK_GREEN }));

            ui.separator();
            ui.heading("Density Limit (Greenwald)");

            let gw_color = if limits.density_exceeded { egui::Color32::RED } else { egui::Color32::GREEN };
            ui.colored_label(gw_color, format!("n/n_G = {:.2}", limits.greenwald_fraction));

            let gw_bar_color = if limits.greenwald_fraction > 0.7 { egui::Color32::BLACK } else { egui::Color32::WHITE };
            ui.add(egui::ProgressBar::new(limits.greenwald_fraction.min(1.5))
                .text(egui::RichText::new(format!("{:.0}% of Greenwald", 100.0 * limits.greenwald_fraction)).color(gw_bar_color))
                .fill(if limits.density_exceeded { egui::Color32::RED } else { egui::Color32::DARK_GREEN }));

            ui.separator();
            ui.heading("Safety Factor q95");

            let q_color = if limits.q95_too_low { egui::Color32::RED } else { egui::Color32::GREEN };
            ui.colored_label(q_color, format!("q95 = {:.2} (min safe: {:.1})", limits.q95, limits.q95_min_safe));

            if limits.q95_too_low {
                ui.colored_label(egui::Color32::RED, "DISRUPTION RISK!");
            }

            ui.separator();
            ui.heading("Confinement");

            ui.label(format!("tau_E (ITER98) = {:.3} s", limits.tau_e_iter98));
            ui.label(format!("n*T*tau = {:.2} x 10^21 keV*s/m3", limits.triple_product));

            let lawson = 3.0;
            let ignition_frac = limits.triple_product / lawson;
            let ign_bar_color = if ignition_frac > 0.5 { egui::Color32::BLACK } else { egui::Color32::WHITE };
            ui.add(egui::ProgressBar::new(ignition_frac.min(2.0) / 2.0)
                .text(egui::RichText::new(format!("{:.0}% of ignition", 100.0 * ignition_frac)).color(ign_bar_color))
                .fill(if ignition_frac >= 1.0 { egui::Color32::GOLD } else { egui::Color32::DARK_BLUE }));

            ui.separator();
            ui.heading("Structural Loads");

            ui.label(format!("Magnetic pressure = {:.1} MPa", limits.magnetic_pressure));

            let stress_color = if limits.structural_warning { egui::Color32::RED } else { egui::Color32::GREEN };
            ui.colored_label(stress_color, format!("TF hoop stress ~ {:.0} MPa", limits.hoop_stress));

            if limits.structural_warning {
                ui.colored_label(egui::Color32::RED, "HIGH STRESS!");
            }

            ui.separator();
            ui.heading("Overall Status");

            let all_ok = !limits.beta_exceeded && !limits.density_exceeded && !limits.q95_too_low && !limits.structural_warning && !limits.geometry_invalid;
            if all_ok {
                ui.colored_label(egui::Color32::GREEN, "All limits satisfied");
            } else {
                if limits.geometry_invalid { ui.colored_label(egui::Color32::RED, "Invalid geometry"); }
                if limits.beta_exceeded { ui.colored_label(egui::Color32::RED, "Beta limit exceeded"); }
                if limits.density_exceeded { ui.colored_label(egui::Color32::RED, "Greenwald limit exceeded"); }
                if limits.q95_too_low { ui.colored_label(egui::Color32::RED, "q95 too low"); }
                if limits.structural_warning { ui.colored_label(egui::Color32::YELLOW, "High structural stress"); }
            }
        });
    });

    // ========================================================================
    // FAILURE LOG WINDOW
    // ========================================================================
    if control.show_log_window {
        egui::Window::new("📋 Event & Failure Logs")
            .default_size([900.0, 500.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("💾 Save to Desktop").clicked() {
                        match control.save_logs_to_file() {
                            Ok(path) => println!("✓ Log saved to: {}", path),
                            Err(e) => println!("✗ Error saving log: {}", e),
                        }
                    }
                    if ui.button("📋 Copy CSV").clicked() {
                        let csv = control.export_logs();
                        println!("=== FAILURE LOG EXPORT ===\n{}", csv);
                    }
                    if ui.button("🗑 Clear Logs").clicked() {
                        if let Ok(mut logs) = control.failure_logs.lock() {
                            logs.clear();
                        }
                    }
                    ui.separator();
                    if let Ok(logs) = control.failure_logs.lock() {
                        ui.label(format!("Total entries: {}", logs.len()));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✕ Close").clicked() {
                            control.show_log_window = false;
                        }
                    });
                });

                ui.separator();

                egui::ScrollArea::both().max_height(420.0).show(ui, |ui| {
                    egui::Grid::new("log_grid")
                        .num_columns(11)
                        .striped(true)
                        .min_col_width(50.0)
                        .show(ui, |ui| {
                            // Header row
                            ui.strong("ID");
                            ui.strong("Time [s]");
                            ui.strong("Controller");
                            ui.strong("Event");
                            ui.strong("t_event");
                            ui.strong("t_resp");
                            ui.strong("Status");
                            ui.strong("R₀/a/Bt");
                            ui.strong("Ip/κ/δ");
                            ui.strong("β/q95");
                            ui.strong("Details");
                            ui.end_row();

                            if let Ok(logs) = control.failure_logs.lock() {
                                for entry in logs.iter().rev().take(200) {
                                    ui.label(format!("{}", entry.id));
                                    ui.label(format!("{:.3}", entry.timestamp));

                                    // Controller with color
                                    let ctrl_color = match entry.controller.as_str() {
                                        "PIRS" => egui::Color32::from_rgb(100, 200, 255),
                                        "PCS" => egui::Color32::from_rgb(255, 200, 100),
                                        _ => egui::Color32::from_rgb(255, 150, 200),
                                    };
                                    ui.colored_label(ctrl_color, &entry.controller);

                                    ui.label(&entry.event_type);
                                    ui.label(format!("{:.2}ms", entry.event_timescale_ms));
                                    ui.label(format!("{:.4}ms", entry.response_time_ms));

                                    let status_color = if entry.success {
                                        egui::Color32::GREEN
                                    } else {
                                        egui::Color32::RED
                                    };
                                    ui.colored_label(status_color, if entry.success { "✓" } else { "✗" });

                                    // Parameter snapshot (grouped)
                                    ui.label(format!("{:.1}/{:.1}/{:.0}",
                                        entry.param_r0, entry.param_a, entry.param_bt));
                                    ui.label(format!("{:.1}/{:.2}/{:.2}",
                                        entry.param_ip, entry.param_kappa, entry.param_delta));
                                    ui.label(format!("{:.1}%/{:.1}",
                                        entry.param_beta * 100.0, entry.param_q95));

                                    ui.label(&entry.details);
                                    ui.end_row();
                                }
                            }
                        });
                });
            });
    }

    // ========================================================================
    // BOTTOM PANEL - HELP
    // ========================================================================
    egui::TopBottomPanel::bottom("help_panel").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("TOKASIM-RS Explorer v2.0 | Mouse=Rotate, Scroll=Zoom | Hover (i) for parameter info");
        });
    });
}
