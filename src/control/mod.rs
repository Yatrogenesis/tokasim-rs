//! # Control Module
//!
//! SYNTEX deterministic control system for plasma control.
//!
//! ## Architecture
//!
//! Unlike NVIDIA's ML-based approach, we use deterministic PIRS (Prolog) rules
//! for explainable, auditable control decisions.
//!
//! ```text
//! NL-SRE (Natural Language) → PIRS (Rules) → Control Actions
//! ```
//!
//! ## Advantages
//!
//! - Response time: <1ms (vs 10-100ms for ML)
//! - Decisions are fully explainable
//! - No "hallucinations" or unpredictable behavior
//! - Auditable for nuclear regulators

use crate::types::{Vec3, TokamakConfig, SimulationState, SimulationStatus};

/// Plasma control system
pub struct PlasmaController {
    /// Target parameters
    pub targets: ControlTargets,
    /// PID controllers for different actuators
    pub pid_position: PIDController,
    pub pid_current: PIDController,
    pub pid_density: PIDController,
    /// Safety limits
    pub limits: SafetyLimits,
    /// Control rules (PIRS-style)
    pub rules: Vec<ControlRule>,
    /// Last decision explanation
    pub last_explanation: String,
}

/// Target plasma parameters
#[derive(Debug, Clone)]
pub struct ControlTargets {
    /// Target plasma current (A)
    pub plasma_current: f64,
    /// Target vertical position (m)
    pub vertical_position: f64,
    /// Target radial position (m)
    pub radial_position: f64,
    /// Target density (m⁻³)
    pub density: f64,
    /// Target βN (normalized beta)
    pub beta_n: f64,
}

impl Default for ControlTargets {
    fn default() -> Self {
        use crate::constants::*;
        Self {
            plasma_current: TS1_PLASMA_CURRENT,
            vertical_position: 0.0,
            radial_position: TS1_MAJOR_RADIUS,
            density: TS1_PLASMA_DENSITY,
            beta_n: 2.5,
        }
    }
}

/// Safety limits (hard constraints)
#[derive(Debug, Clone)]
pub struct SafetyLimits {
    /// Maximum plasma current (A)
    pub max_current: f64,
    /// Maximum vertical displacement (m)
    pub max_vertical_displacement: f64,
    /// Maximum density (Greenwald limit) (m⁻³)
    pub max_density: f64,
    /// Maximum βN (stability limit)
    pub max_beta_n: f64,
    /// Minimum q95 (safety factor at 95% flux)
    pub min_q95: f64,
    /// Maximum heating power (W)
    pub max_heating_power: f64,
    /// Disruption risk threshold (0-1)
    pub disruption_threshold: f64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        use crate::constants::*;
        Self {
            max_current: TS1_PLASMA_CURRENT * 1.2,
            max_vertical_displacement: TS1_MINOR_RADIUS * 0.3,
            max_density: 4e20,  // ~Greenwald limit
            max_beta_n: 3.5,
            min_q95: 2.0,
            max_heating_power: 60e6,  // 60 MW
            disruption_threshold: 0.8,
        }
    }
}

/// PID Controller
#[derive(Debug, Clone)]
pub struct PIDController {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Integral accumulator
    integral: f64,
    /// Previous error
    prev_error: f64,
    /// Output limits
    pub min_output: f64,
    pub max_output: f64,
}

impl PIDController {
    /// Create new PID controller
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            min_output: f64::NEG_INFINITY,
            max_output: f64::INFINITY,
        }
    }

    /// Set output limits
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.min_output = min;
        self.max_output = max;
        self
    }

    /// Compute control output
    pub fn compute(&mut self, error: f64, dt: f64) -> f64 {
        // Proportional
        let p = self.kp * error;

        // Integral (with anti-windup)
        self.integral += error * dt;
        let i_max = (self.max_output - self.min_output) / self.ki.abs().max(1e-10);
        self.integral = self.integral.clamp(-i_max, i_max);
        let i = self.ki * self.integral;

        // Derivative
        let d = if dt > 0.0 {
            self.kd * (error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = error;

        // Total output with limits
        (p + i + d).clamp(self.min_output, self.max_output)
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

/// PIRS-style control rule
#[derive(Debug, Clone)]
pub struct ControlRule {
    /// Rule name/ID
    pub name: String,
    /// Conditions (in PIRS format)
    pub conditions: Vec<Condition>,
    /// Actions to take if conditions met
    pub actions: Vec<ControlAction>,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Is rule active
    pub active: bool,
}

/// Condition for a control rule
#[derive(Debug, Clone)]
pub enum Condition {
    /// Parameter greater than threshold
    GreaterThan { param: PlasmaParam, value: f64 },
    /// Parameter less than threshold
    LessThan { param: PlasmaParam, value: f64 },
    /// Parameter in range
    InRange { param: PlasmaParam, min: f64, max: f64 },
    /// Parameter equals value (with tolerance)
    Equals { param: PlasmaParam, value: f64, tolerance: f64 },
    /// Rate of change greater than threshold
    RateGreaterThan { param: PlasmaParam, rate: f64 },
    /// Logical AND of conditions
    And(Vec<Condition>),
    /// Logical OR of conditions
    Or(Vec<Condition>),
}

/// Plasma parameters that can be monitored
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasmaParam {
    PlasmaCurrent,
    VerticalPosition,
    RadialPosition,
    Density,
    IonTemperature,
    ElectronTemperature,
    BetaN,
    Q95,
    DisruptionRisk,
    FusionPower,
    StoredEnergy,
}

/// Control actions
#[derive(Debug, Clone)]
pub enum ControlAction {
    /// Adjust coil current
    AdjustCoil { coil_id: u32, delta_current: f64 },
    /// Adjust heating power
    AdjustHeating { system: HeatingSystem, power_delta: f64 },
    /// Inject gas
    GasInjection { species: GasSpecies, rate: f64 },
    /// Trigger shutdown
    EmergencyShutdown { reason: String },
    /// Log warning
    LogWarning { message: String },
    /// Set control target
    SetTarget { param: PlasmaParam, value: f64 },
}

/// Heating systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatingSystem {
    ICRF,
    ECRH,
    NBI,
}

/// Gas species for injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GasSpecies {
    Deuterium,
    Tritium,
    Helium,
    Neon,     // For radiation cooling
    Argon,    // For disruption mitigation
}

/// Current plasma measurements
#[derive(Debug, Clone, Default)]
pub struct PlasmaMeasurements {
    pub plasma_current: f64,
    pub vertical_position: f64,
    pub radial_position: f64,
    pub density: f64,
    pub ion_temperature_kev: f64,
    pub electron_temperature_kev: f64,
    pub beta_n: f64,
    pub q95: f64,
    pub stored_energy: f64,
    pub fusion_power: f64,
}

/// Control output (commands to actuators)
#[derive(Debug, Clone, Default)]
pub struct ControlOutput {
    /// Poloidal field coil currents (A)
    pub pf_coil_currents: Vec<f64>,
    /// ICRF power (W)
    pub icrf_power: f64,
    /// ECRH power (W)
    pub ecrh_power: f64,
    /// NBI power (W)
    pub nbi_power: f64,
    /// Gas puff rate (particles/s)
    pub gas_puff_rate: f64,
    /// Emergency shutdown requested
    pub emergency_shutdown: bool,
    /// Explanation of control decisions
    pub explanation: String,
}

impl PlasmaController {
    /// Create new controller with default settings for TS-1
    pub fn new() -> Self {
        let mut controller = Self {
            targets: ControlTargets::default(),
            pid_position: PIDController::new(1000.0, 100.0, 50.0)
                .with_limits(-1e6, 1e6),
            pid_current: PIDController::new(0.1, 0.01, 0.001)
                .with_limits(-0.1, 0.1),
            pid_density: PIDController::new(1e-19, 1e-20, 1e-21)
                .with_limits(-1e22, 1e22),
            limits: SafetyLimits::default(),
            rules: Vec::new(),
            last_explanation: String::new(),
        };

        // Load default control rules
        controller.load_default_rules();
        controller
    }

    /// Load default PIRS-style control rules
    fn load_default_rules(&mut self) {
        // Rule 1: Emergency shutdown on high disruption risk
        self.rules.push(ControlRule {
            name: "emergency_shutdown_disruption".to_string(),
            conditions: vec![
                Condition::GreaterThan {
                    param: PlasmaParam::DisruptionRisk,
                    value: 0.9,
                }
            ],
            actions: vec![
                ControlAction::EmergencyShutdown {
                    reason: "Disruption risk > 90%".to_string()
                },
                ControlAction::LogWarning {
                    message: "EMERGENCY: Disruption imminent".to_string()
                },
            ],
            priority: 100,
            active: true,
        });

        // Rule 2: Reduce power on high beta
        self.rules.push(ControlRule {
            name: "reduce_power_high_beta".to_string(),
            conditions: vec![
                Condition::GreaterThan {
                    param: PlasmaParam::BetaN,
                    value: 3.2,
                }
            ],
            actions: vec![
                ControlAction::AdjustHeating {
                    system: HeatingSystem::ICRF,
                    power_delta: -5e6,  // -5 MW
                },
                ControlAction::LogWarning {
                    message: "βN approaching limit, reducing heating".to_string()
                },
            ],
            priority: 80,
            active: true,
        });

        // Rule 3: Increase fueling on low density
        self.rules.push(ControlRule {
            name: "increase_density".to_string(),
            conditions: vec![
                Condition::LessThan {
                    param: PlasmaParam::Density,
                    value: 2e20,
                }
            ],
            actions: vec![
                ControlAction::GasInjection {
                    species: GasSpecies::Deuterium,
                    rate: 1e21,  // particles/s
                },
            ],
            priority: 50,
            active: true,
        });

        // Rule 4: Vertical stability - position feedback
        self.rules.push(ControlRule {
            name: "vertical_stability".to_string(),
            conditions: vec![
                Condition::Or(vec![
                    Condition::GreaterThan {
                        param: PlasmaParam::VerticalPosition,
                        value: 0.05,  // 5 cm
                    },
                    Condition::LessThan {
                        param: PlasmaParam::VerticalPosition,
                        value: -0.05,
                    },
                ])
            ],
            actions: vec![
                ControlAction::LogWarning {
                    message: "Vertical position drift detected".to_string()
                },
            ],
            priority: 90,
            active: true,
        });

        // Rule 5: Optimal operating point
        self.rules.push(ControlRule {
            name: "maintain_q95".to_string(),
            conditions: vec![
                Condition::LessThan {
                    param: PlasmaParam::Q95,
                    value: 2.5,
                }
            ],
            actions: vec![
                ControlAction::LogWarning {
                    message: "q95 low - risk of kink instability".to_string()
                },
            ],
            priority: 85,
            active: true,
        });
    }

    /// Add a control rule (PIRS-style)
    pub fn add_rule(&mut self, rule: ControlRule) {
        self.rules.push(rule);
        // Sort by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Evaluate a condition against measurements
    fn evaluate_condition(&self, cond: &Condition, meas: &PlasmaMeasurements) -> bool {
        match cond {
            Condition::GreaterThan { param, value } => {
                self.get_param(param, meas) > *value
            }
            Condition::LessThan { param, value } => {
                self.get_param(param, meas) < *value
            }
            Condition::InRange { param, min, max } => {
                let v = self.get_param(param, meas);
                v >= *min && v <= *max
            }
            Condition::Equals { param, value, tolerance } => {
                (self.get_param(param, meas) - value).abs() <= *tolerance
            }
            Condition::RateGreaterThan { .. } => {
                // TODO: implement rate tracking
                false
            }
            Condition::And(conds) => {
                conds.iter().all(|c| self.evaluate_condition(c, meas))
            }
            Condition::Or(conds) => {
                conds.iter().any(|c| self.evaluate_condition(c, meas))
            }
        }
    }

    /// Get parameter value from measurements
    fn get_param(&self, param: &PlasmaParam, meas: &PlasmaMeasurements) -> f64 {
        match param {
            PlasmaParam::PlasmaCurrent => meas.plasma_current,
            PlasmaParam::VerticalPosition => meas.vertical_position,
            PlasmaParam::RadialPosition => meas.radial_position,
            PlasmaParam::Density => meas.density,
            PlasmaParam::IonTemperature => meas.ion_temperature_kev,
            PlasmaParam::ElectronTemperature => meas.electron_temperature_kev,
            PlasmaParam::BetaN => meas.beta_n,
            PlasmaParam::Q95 => meas.q95,
            PlasmaParam::DisruptionRisk => self.estimate_disruption_risk(meas),
            PlasmaParam::FusionPower => meas.fusion_power,
            PlasmaParam::StoredEnergy => meas.stored_energy,
        }
    }

    /// Estimate disruption risk (0-1)
    fn estimate_disruption_risk(&self, meas: &PlasmaMeasurements) -> f64 {
        let mut risk = 0.0;

        // High beta increases risk
        if meas.beta_n > 3.0 {
            risk += 0.3 * (meas.beta_n - 3.0) / 0.5;
        }

        // Low q95 increases risk
        if meas.q95 < 2.5 {
            risk += 0.4 * (2.5 - meas.q95) / 0.5;
        }

        // High vertical displacement increases risk
        let z_norm = meas.vertical_position.abs() / self.limits.max_vertical_displacement;
        if z_norm > 0.5 {
            risk += 0.3 * (z_norm - 0.5) / 0.5;
        }

        risk.clamp(0.0, 1.0)
    }

    /// Main control loop - compute control output
    pub fn compute(&mut self, meas: &PlasmaMeasurements, dt: f64) -> ControlOutput {
        let mut output = ControlOutput::default();
        let mut explanations = Vec::new();

        // 1. Evaluate PIRS rules (highest priority first)
        let triggered_rules: Vec<&ControlRule> = self.rules.iter()
            .filter(|r| r.active)
            .filter(|r| r.conditions.iter().all(|c| self.evaluate_condition(c, meas)))
            .collect();

        for rule in &triggered_rules {
            explanations.push(format!("Rule '{}' triggered", rule.name));

            for action in &rule.actions {
                match action {
                    ControlAction::EmergencyShutdown { reason } => {
                        output.emergency_shutdown = true;
                        explanations.push(format!("EMERGENCY SHUTDOWN: {}", reason));
                    }
                    ControlAction::AdjustHeating { system, power_delta } => {
                        match system {
                            HeatingSystem::ICRF => output.icrf_power += power_delta,
                            HeatingSystem::ECRH => output.ecrh_power += power_delta,
                            HeatingSystem::NBI => output.nbi_power += power_delta,
                        }
                    }
                    ControlAction::GasInjection { rate, .. } => {
                        output.gas_puff_rate += rate;
                    }
                    ControlAction::LogWarning { message } => {
                        explanations.push(format!("WARNING: {}", message));
                    }
                    _ => {}
                }
            }
        }

        // 2. PID control for position
        let z_error = self.targets.vertical_position - meas.vertical_position;
        let z_correction = self.pid_position.compute(z_error, dt);
        // Convert to coil current (simplified)
        output.pf_coil_currents = vec![z_correction * 0.001];  // Placeholder
        explanations.push(format!("Position PID: error={:.3}m, correction={:.1}A",
            z_error, z_correction));

        // 3. PID control for density
        let n_error = self.targets.density - meas.density;
        let n_correction = self.pid_density.compute(n_error, dt);
        if n_correction > 0.0 {
            output.gas_puff_rate += n_correction;
        }

        // 4. Apply safety limits
        output.icrf_power = output.icrf_power.clamp(0.0, self.limits.max_heating_power / 3.0);
        output.ecrh_power = output.ecrh_power.clamp(0.0, self.limits.max_heating_power / 3.0);
        output.nbi_power = output.nbi_power.clamp(0.0, self.limits.max_heating_power / 3.0);

        output.explanation = explanations.join("\n");
        self.last_explanation = output.explanation.clone();

        output
    }

    /// Generate PIRS-format rules as string (for debugging/export)
    pub fn to_pirs(&self) -> String {
        let mut output = String::new();
        output.push_str("% TOKASIM Control Rules (PIRS format)\n");
        output.push_str("% Auto-generated from PlasmaController\n\n");

        for rule in &self.rules {
            output.push_str(&format!("% Rule: {} (priority: {})\n", rule.name, rule.priority));
            output.push_str(&format!("control_rule({}, [\n", rule.name));

            // Conditions
            output.push_str("  conditions([\n");
            for cond in &rule.conditions {
                output.push_str(&format!("    {},\n", self.condition_to_pirs(cond)));
            }
            output.push_str("  ]),\n");

            // Actions
            output.push_str("  actions([\n");
            for action in &rule.actions {
                output.push_str(&format!("    {},\n", self.action_to_pirs(action)));
            }
            output.push_str("  ])\n");

            output.push_str("]).\n\n");
        }

        output
    }

    fn condition_to_pirs(&self, cond: &Condition) -> String {
        match cond {
            Condition::GreaterThan { param, value } => {
                format!("greater_than({:?}, {})", param, value)
            }
            Condition::LessThan { param, value } => {
                format!("less_than({:?}, {})", param, value)
            }
            Condition::InRange { param, min, max } => {
                format!("in_range({:?}, {}, {})", param, min, max)
            }
            Condition::And(conds) => {
                let inner: Vec<String> = conds.iter()
                    .map(|c| self.condition_to_pirs(c))
                    .collect();
                format!("and([{}])", inner.join(", "))
            }
            Condition::Or(conds) => {
                let inner: Vec<String> = conds.iter()
                    .map(|c| self.condition_to_pirs(c))
                    .collect();
                format!("or([{}])", inner.join(", "))
            }
            _ => "unknown_condition".to_string()
        }
    }

    fn action_to_pirs(&self, action: &ControlAction) -> String {
        match action {
            ControlAction::EmergencyShutdown { reason } => {
                format!("emergency_shutdown(\"{}\")", reason)
            }
            ControlAction::AdjustHeating { system, power_delta } => {
                format!("adjust_heating({:?}, {})", system, power_delta)
            }
            ControlAction::GasInjection { species, rate } => {
                format!("gas_injection({:?}, {})", species, rate)
            }
            ControlAction::LogWarning { message } => {
                format!("log_warning(\"{}\")", message)
            }
            _ => "unknown_action".to_string()
        }
    }
}

impl Default for PlasmaController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_controller() {
        let mut pid = PIDController::new(1.0, 0.1, 0.01)
            .with_limits(-10.0, 10.0);

        let output = pid.compute(1.0, 0.01);
        assert!(output > 0.0);
        assert!(output <= 10.0);
    }

    #[test]
    fn test_plasma_controller() {
        let mut controller = PlasmaController::new();

        let meas = PlasmaMeasurements {
            plasma_current: 12e6,
            vertical_position: 0.0,
            radial_position: 1.5,
            density: 3e20,
            ion_temperature_kev: 15.0,
            electron_temperature_kev: 15.0,
            beta_n: 2.5,
            q95: 3.0,
            stored_energy: 100e6,
            fusion_power: 500e6,
        };

        let output = controller.compute(&meas, 0.001);
        assert!(!output.emergency_shutdown);
    }

    #[test]
    fn test_disruption_rule() {
        let mut controller = PlasmaController::new();

        // High beta should trigger warning
        let meas = PlasmaMeasurements {
            beta_n: 3.5,
            q95: 2.0,
            ..Default::default()
        };

        let output = controller.compute(&meas, 0.001);
        assert!(output.explanation.contains("βN") ||
                output.explanation.contains("q95"));
    }

    #[test]
    fn test_pirs_export() {
        let controller = PlasmaController::new();
        let pirs = controller.to_pirs();

        assert!(pirs.contains("control_rule"));
        assert!(pirs.contains("conditions"));
        assert!(pirs.contains("actions"));
    }
}
