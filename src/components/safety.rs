//! # Safety Systems
//!
//! Comprehensive safety mechanisms for all tokamak components.
//! Defense-in-depth approach with multiple independent barriers.

use super::*;

/// Safety Integrity Level (IEC 61508)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SIL {
    /// Safety Integrity Level 1 (PFD 10^-1 to 10^-2)
    SIL1,
    /// Safety Integrity Level 2 (PFD 10^-2 to 10^-3)
    SIL2,
    /// Safety Integrity Level 3 (PFD 10^-3 to 10^-4)
    SIL3,
    /// Safety Integrity Level 4 (PFD 10^-4 to 10^-5)
    SIL4,
}

/// Safety function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyFunctionType {
    /// Prevent hazardous event
    Prevention,
    /// Mitigate consequences
    Mitigation,
    /// Detect hazardous condition
    Detection,
    /// Isolate affected systems
    Isolation,
    /// Emergency shutdown
    EmergencyShutdown,
}

/// Safety function definition
#[derive(Debug, Clone)]
pub struct SafetyFunction {
    pub id: String,
    pub name: String,
    pub description: String,
    pub function_type: SafetyFunctionType,
    pub sil: SIL,
    pub response_time_ms: TimeMs,
    pub protected_components: Vec<String>,
    pub triggering_conditions: Vec<String>,
    pub actions: Vec<String>,
    pub redundancy: RedundancyLevel,
    pub test_interval_hours: f64,
    pub health: ComponentHealth,
}

/// Redundancy level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RedundancyLevel {
    /// No redundancy (1oo1)
    None,
    /// Dual redundancy with voting (1oo2)
    Dual1oo2,
    /// Dual redundancy both required (2oo2)
    Dual2oo2,
    /// Triple redundancy (2oo3)
    Triple,
    /// Quad redundancy (2oo4)
    Quad,
}

impl RedundancyLevel {
    /// Probability of failure on demand (approximate)
    pub fn pfd(&self) -> f64 {
        match self {
            Self::None => 1e-2,
            Self::Dual1oo2 => 1e-4,
            Self::Dual2oo2 => 2e-2,  // Higher - both must work
            Self::Triple => 1e-6,
            Self::Quad => 1e-8,
        }
    }

    /// Can tolerate N failures
    pub fn fault_tolerance(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Dual1oo2 => 1,
            Self::Dual2oo2 => 0,
            Self::Triple => 1,
            Self::Quad => 2,
        }
    }
}

/// Complete safety system
#[derive(Debug, Clone)]
pub struct SafetySystem {
    pub functions: Vec<SafetyFunction>,
    pub interlock_matrix: Vec<Interlock>,
}

/// Interlock definition
#[derive(Debug, Clone)]
pub struct Interlock {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub action: String,
    pub priority: u8,  // 1 = highest
    pub can_be_bypassed: bool,
    pub bypass_requires_dual_authorization: bool,
}

impl SafetySystem {
    /// Create TS-1 safety system
    pub fn ts1() -> Self {
        let functions = vec![
            // ============================================================
            // PLASMA SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-PLASMA-01".to_string(),
                name: "Plasma Emergency Shutdown".to_string(),
                description: "Immediate plasma termination via controlled ramp-down or MGI".to_string(),
                function_type: SafetyFunctionType::EmergencyShutdown,
                sil: SIL::SIL3,
                response_time_ms: 10,
                protected_components: vec![
                    "First Wall".to_string(),
                    "Divertor".to_string(),
                    "Vacuum Vessel".to_string(),
                ],
                triggering_conditions: vec![
                    "DRI > 0.95".to_string(),
                    "Vertical position > 10 cm".to_string(),
                    "Beta_N > Troyon limit".to_string(),
                    "Runaway electron current > 100 kA".to_string(),
                ],
                actions: vec![
                    "Trigger MGI valves".to_string(),
                    "Fast current ramp (10 MA/s)".to_string(),
                    "ECCD cutoff".to_string(),
                    "NBI shutdown".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 168.0,  // Weekly
                health: ComponentHealth::default(),
            },

            SafetyFunction {
                id: "SF-PLASMA-02".to_string(),
                name: "Vertical Displacement Control".to_string(),
                description: "Fast vertical position feedback control".to_string(),
                function_type: SafetyFunctionType::Prevention,
                sil: SIL::SIL3,
                response_time_ms: 1,
                protected_components: vec!["First Wall".to_string(), "Vacuum Vessel".to_string()],
                triggering_conditions: vec![
                    "|z| > 2 cm".to_string(),
                    "dz/dt > 50 m/s".to_string(),
                ],
                actions: vec![
                    "VS coil current adjustment".to_string(),
                    "Proportional feedback loop".to_string(),
                ],
                redundancy: RedundancyLevel::Dual1oo2,
                test_interval_hours: 24.0,
                health: ComponentHealth::default(),
            },

            SafetyFunction {
                id: "SF-PLASMA-03".to_string(),
                name: "Runaway Electron Mitigation".to_string(),
                description: "Detect and suppress runaway electron generation".to_string(),
                function_type: SafetyFunctionType::Mitigation,
                sil: SIL::SIL3,
                response_time_ms: 5,
                protected_components: vec!["First Wall".to_string(), "Diagnostics".to_string()],
                triggering_conditions: vec![
                    "Hard X-ray spike > 10x baseline".to_string(),
                    "Synchrotron emission detected".to_string(),
                    "I_RE > 10 kA".to_string(),
                ],
                actions: vec![
                    "Massive Gas Injection (Ar/Ne)".to_string(),
                    "Shattered Pellet Injection".to_string(),
                    "Controlled dissipation via 3D fields".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 168.0,
                health: ComponentHealth::default(),
            },

            // ============================================================
            // MAGNET SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-MAG-01".to_string(),
                name: "Quench Detection and Protection".to_string(),
                description: "Detect superconductor quench and safely dissipate energy".to_string(),
                function_type: SafetyFunctionType::Detection,
                sil: SIL::SIL4,
                response_time_ms: 5,
                protected_components: vec![
                    "TF Coils".to_string(),
                    "PF Coils".to_string(),
                    "CS Coils".to_string(),
                ],
                triggering_conditions: vec![
                    "dV/dt > 100 mV/ms (resistive voltage)".to_string(),
                    "Local temperature > T_critical + 0.5K".to_string(),
                    "Helium flow anomaly".to_string(),
                ],
                actions: vec![
                    "Open quench protection heaters".to_string(),
                    "Engage dump resistors".to_string(),
                    "Switch to flywheel power".to_string(),
                    "Fast energy extraction".to_string(),
                ],
                redundancy: RedundancyLevel::Quad,
                test_interval_hours: 720.0,  // Monthly
                health: ComponentHealth::default(),
            },

            SafetyFunction {
                id: "SF-MAG-02".to_string(),
                name: "Magnet Current Limit".to_string(),
                description: "Prevent overcurrent in superconducting coils".to_string(),
                function_type: SafetyFunctionType::Prevention,
                sil: SIL::SIL3,
                response_time_ms: 10,
                protected_components: vec!["All Magnets".to_string()],
                triggering_conditions: vec![
                    "I > I_critical * 0.95".to_string(),
                    "dI/dt > rated ramp rate * 1.2".to_string(),
                ],
                actions: vec![
                    "Current reduction command".to_string(),
                    "Power supply trip if limit exceeded".to_string(),
                ],
                redundancy: RedundancyLevel::Dual1oo2,
                test_interval_hours: 168.0,
                health: ComponentHealth::default(),
            },

            // ============================================================
            // CRYOGENIC SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-CRYO-01".to_string(),
                name: "Helium Leak Detection".to_string(),
                description: "Detect helium leaks in cryogenic system".to_string(),
                function_type: SafetyFunctionType::Detection,
                sil: SIL::SIL2,
                response_time_ms: 100,
                protected_components: vec!["Cryostat".to_string(), "Magnet System".to_string()],
                triggering_conditions: vec![
                    "Helium concentration > 1% in cryostat vacuum".to_string(),
                    "Pressure rise rate > 1 mbar/hour".to_string(),
                ],
                actions: vec![
                    "Isolate affected section".to_string(),
                    "Activate backup cooling".to_string(),
                    "Alarm to control room".to_string(),
                ],
                redundancy: RedundancyLevel::Dual1oo2,
                test_interval_hours: 720.0,
                health: ComponentHealth::default(),
            },

            SafetyFunction {
                id: "SF-CRYO-02".to_string(),
                name: "Cryogenic Overpressure Protection".to_string(),
                description: "Prevent overpressure in helium circuits".to_string(),
                function_type: SafetyFunctionType::Prevention,
                sil: SIL::SIL3,
                response_time_ms: 50,
                protected_components: vec!["Cryogenic System".to_string()],
                triggering_conditions: vec![
                    "Pressure > design limit * 0.9".to_string(),
                ],
                actions: vec![
                    "Open relief valves".to_string(),
                    "Controlled venting to recovery system".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 8760.0,  // Annually
                health: ComponentHealth::default(),
            },

            // ============================================================
            // VACUUM SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-VAC-01".to_string(),
                name: "Vacuum Loss Protection".to_string(),
                description: "Detect and respond to vacuum vessel breach".to_string(),
                function_type: SafetyFunctionType::Detection,
                sil: SIL::SIL2,
                response_time_ms: 100,
                protected_components: vec!["Vacuum Vessel".to_string(), "First Wall".to_string()],
                triggering_conditions: vec![
                    "Pressure > 10^-4 mbar".to_string(),
                    "Pressure rise rate > 10 mbar/s".to_string(),
                ],
                actions: vec![
                    "Plasma emergency shutdown".to_string(),
                    "Close vacuum valves".to_string(),
                    "Trigger gas purge if air ingress".to_string(),
                ],
                redundancy: RedundancyLevel::Dual1oo2,
                test_interval_hours: 168.0,
                health: ComponentHealth::default(),
            },

            // ============================================================
            // POWER SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-PWR-01".to_string(),
                name: "Grid Loss Response".to_string(),
                description: "Automatic transfer to backup power on grid loss".to_string(),
                function_type: SafetyFunctionType::Mitigation,
                sil: SIL::SIL3,
                response_time_ms: 0,  // Zero gap with online UPS
                protected_components: vec!["Control System".to_string(), "Safety Systems".to_string()],
                triggering_conditions: vec![
                    "Grid voltage < 85% nominal".to_string(),
                    "Grid frequency outside 49.5-50.5 Hz".to_string(),
                ],
                actions: vec![
                    "UPS takes load (0 ms)".to_string(),
                    "Start diesel generators".to_string(),
                    "Initiate controlled plasma shutdown".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 720.0,
                health: ComponentHealth::default(),
            },

            // ============================================================
            // RADIATION SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-RAD-01".to_string(),
                name: "Radiation Monitoring".to_string(),
                description: "Continuous radiation level monitoring".to_string(),
                function_type: SafetyFunctionType::Detection,
                sil: SIL::SIL2,
                response_time_ms: 1000,
                protected_components: vec!["Personnel".to_string(), "Public".to_string()],
                triggering_conditions: vec![
                    "Neutron flux > alarm threshold".to_string(),
                    "Gamma dose rate > occupational limit".to_string(),
                ],
                actions: vec![
                    "Area alarm activation".to_string(),
                    "Access restriction".to_string(),
                    "Automatic door locks".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 24.0,
                health: ComponentHealth::default(),
            },

            // ============================================================
            // CONTROL SYSTEM SAFETY FUNCTIONS
            // ============================================================
            SafetyFunction {
                id: "SF-CTRL-01".to_string(),
                name: "Control System Watchdog".to_string(),
                description: "Detect control system failure and transfer to backup".to_string(),
                function_type: SafetyFunctionType::Detection,
                sil: SIL::SIL3,
                response_time_ms: 10,
                protected_components: vec!["Control System".to_string()],
                triggering_conditions: vec![
                    "Heartbeat timeout > 100 ms".to_string(),
                    "Command queue overflow".to_string(),
                    "Memory corruption detected".to_string(),
                ],
                actions: vec![
                    "Switch to backup controller".to_string(),
                    "Initiate safe state if no backup".to_string(),
                    "Log failure for analysis".to_string(),
                ],
                redundancy: RedundancyLevel::Triple,
                test_interval_hours: 1.0,  // Continuous self-test
                health: ComponentHealth::default(),
            },

            SafetyFunction {
                id: "SF-CTRL-02".to_string(),
                name: "PIRS Rule Verification".to_string(),
                description: "Verify control rules are valid before execution".to_string(),
                function_type: SafetyFunctionType::Prevention,
                sil: SIL::SIL2,
                response_time_ms: 1,
                protected_components: vec!["Plasma".to_string()],
                triggering_conditions: vec![
                    "Rule conflicts detected".to_string(),
                    "Invalid parameter range".to_string(),
                ],
                actions: vec![
                    "Reject invalid command".to_string(),
                    "Use default safe action".to_string(),
                    "Alert operator".to_string(),
                ],
                redundancy: RedundancyLevel::Dual1oo2,
                test_interval_hours: 0.001,  // Every rule execution
                health: ComponentHealth::default(),
            },
        ];

        let interlock_matrix = vec![
            Interlock {
                id: "IL-01".to_string(),
                name: "Magnet Energization Interlock".to_string(),
                condition: "Cryogenic system at operating temperature".to_string(),
                action: "Block magnet power-up".to_string(),
                priority: 1,
                can_be_bypassed: false,
                bypass_requires_dual_authorization: false,
            },
            Interlock {
                id: "IL-02".to_string(),
                name: "Plasma Startup Interlock".to_string(),
                condition: "Vacuum < 10^-6 mbar AND Magnets at field AND Control online".to_string(),
                action: "Block plasma initiation".to_string(),
                priority: 1,
                can_be_bypassed: false,
                bypass_requires_dual_authorization: false,
            },
            Interlock {
                id: "IL-03".to_string(),
                name: "Heating Power Interlock".to_string(),
                condition: "Plasma current > 2 MA AND beta_N < limit".to_string(),
                action: "Block heating power increase".to_string(),
                priority: 2,
                can_be_bypassed: true,
                bypass_requires_dual_authorization: true,
            },
            Interlock {
                id: "IL-04".to_string(),
                name: "Personnel Access Interlock".to_string(),
                condition: "Radiation level safe AND Magnets de-energized".to_string(),
                action: "Block access door opening".to_string(),
                priority: 1,
                can_be_bypassed: true,
                bypass_requires_dual_authorization: true,
            },
            Interlock {
                id: "IL-05".to_string(),
                name: "Tritium Handling Interlock".to_string(),
                condition: "Containment verified AND Atmosphere monitors online".to_string(),
                action: "Block tritium operations".to_string(),
                priority: 1,
                can_be_bypassed: false,
                bypass_requires_dual_authorization: false,
            },
        ];

        Self {
            functions,
            interlock_matrix,
        }
    }

    /// Get all SIL3+ functions
    pub fn critical_functions(&self) -> Vec<&SafetyFunction> {
        self.functions.iter()
            .filter(|f| f.sil >= SIL::SIL3)
            .collect()
    }

    /// Calculate overall system availability
    pub fn system_availability(&self) -> f64 {
        // Product of individual function availabilities
        self.functions.iter()
            .map(|f| 1.0 - f.redundancy.pfd())
            .product()
    }

    /// Generate safety report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== TOKASIM-RS SAFETY SYSTEM REPORT ===\n\n");

        report.push_str("SAFETY FUNCTIONS:\n");
        report.push_str("-----------------\n");
        for func in &self.functions {
            report.push_str(&format!(
                "{} - {} ({:?})\n  Response: {} ms\n  Redundancy: {:?}\n  PFD: {:.2e}\n\n",
                func.id, func.name, func.sil,
                func.response_time_ms, func.redundancy, func.redundancy.pfd()
            ));
        }

        report.push_str("\nINTERLOCKS:\n");
        report.push_str("-----------\n");
        for il in &self.interlock_matrix {
            report.push_str(&format!(
                "{}: {} (Priority {})\n  Condition: {}\n  Bypassable: {}\n\n",
                il.id, il.name, il.priority, il.condition, il.can_be_bypassed
            ));
        }

        report.push_str(&format!(
            "\nSYSTEM AVAILABILITY: {:.6}\n",
            self.system_availability()
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_system() {
        let safety = SafetySystem::ts1();
        assert!(!safety.functions.is_empty());
        assert!(!safety.interlock_matrix.is_empty());
    }

    #[test]
    fn test_critical_functions() {
        let safety = SafetySystem::ts1();
        let critical = safety.critical_functions();
        assert!(!critical.is_empty());
        for f in critical {
            assert!(f.sil >= SIL::SIL3);
        }
    }

    #[test]
    fn test_availability() {
        let safety = SafetySystem::ts1();
        let avail = safety.system_availability();
        assert!(avail > 0.99);  // Should be very high
    }
}
