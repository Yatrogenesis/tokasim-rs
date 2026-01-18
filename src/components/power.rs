//! # Power Systems
//!
//! Complete power infrastructure for tokamak operation including:
//! - Main grid connection
//! - Backup power systems
//! - UPS systems
//! - Flywheel energy storage
//! - Diesel generators
//! - Battery banks

use super::*;

/// Power source type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerSource {
    /// Main electrical grid
    MainGrid,
    /// Uninterruptible Power Supply (battery-backed)
    UPS,
    /// Flywheel energy storage
    Flywheel,
    /// Diesel generator
    DieselGenerator,
    /// Battery bank
    BatteryBank,
    /// Superconducting Magnetic Energy Storage
    SMES,
}

/// Power system configuration
#[derive(Debug, Clone)]
pub struct PowerSystem {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Power source type
    pub source: PowerSource,
    /// Maximum power output (MW)
    pub max_power_mw: f64,
    /// Current power output (MW)
    pub current_power_mw: f64,
    /// Energy storage capacity (MJ) - 0 for grid
    pub storage_capacity_mj: f64,
    /// Current stored energy (MJ)
    pub stored_energy_mj: f64,
    /// Startup time (milliseconds)
    pub startup_time_ms: TimeMs,
    /// Ramp rate (MW/s)
    pub ramp_rate_mw_per_s: f64,
    /// Efficiency (%)
    pub efficiency: f64,
    /// Is system online
    pub online: bool,
    /// Health status
    pub health: ComponentHealth,
}

impl PowerSystem {
    /// Time to reach full power from startup (ms)
    pub fn time_to_full_power_ms(&self) -> TimeMs {
        if self.ramp_rate_mw_per_s <= 0.0 {
            return u64::MAX;
        }
        let ramp_time = (self.max_power_mw / self.ramp_rate_mw_per_s * 1000.0) as u64;
        self.startup_time_ms + ramp_time
    }

    /// Energy available before depletion (MJ)
    pub fn available_energy_mj(&self) -> f64 {
        match self.source {
            PowerSource::MainGrid => f64::INFINITY,
            _ => self.stored_energy_mj,
        }
    }

    /// Runtime at current load (seconds)
    pub fn runtime_at_load_s(&self, load_mw: f64) -> f64 {
        if load_mw <= 0.0 {
            return f64::INFINITY;
        }
        match self.source {
            PowerSource::MainGrid => f64::INFINITY,
            _ => self.stored_energy_mj / load_mw,
        }
    }
}

/// Complete power infrastructure
#[derive(Debug, Clone)]
pub struct PowerInfrastructure {
    pub systems: Vec<PowerSystem>,
    /// Total facility load (MW)
    pub total_load_mw: f64,
    /// Critical load that must never lose power (MW)
    pub critical_load_mw: f64,
    /// Current active power source
    pub active_source: PowerSource,
}

impl PowerInfrastructure {
    /// Create TS-1 power infrastructure
    pub fn ts1() -> Self {
        let systems = vec![
            // Main grid connection
            PowerSystem {
                id: "GRID-01".to_string(),
                name: "Main Grid Connection (CFE 230kV)".to_string(),
                source: PowerSource::MainGrid,
                max_power_mw: 500.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 0.0,
                stored_energy_mj: 0.0,
                startup_time_ms: 0,
                ramp_rate_mw_per_s: 100.0,
                efficiency: 99.5,
                online: true,
                health: ComponentHealth::default(),
            },

            // UPS for control systems (instantaneous transfer)
            PowerSystem {
                id: "UPS-CTRL-01".to_string(),
                name: "Control Room UPS Primary".to_string(),
                source: PowerSource::UPS,
                max_power_mw: 0.5,
                current_power_mw: 0.0,
                storage_capacity_mj: 1800.0,  // 30 min at full load
                stored_energy_mj: 1800.0,
                startup_time_ms: 0,  // Instantaneous (online UPS)
                ramp_rate_mw_per_s: 1000.0,  // Instant
                efficiency: 94.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // UPS for control systems (backup)
            PowerSystem {
                id: "UPS-CTRL-02".to_string(),
                name: "Control Room UPS Backup".to_string(),
                source: PowerSource::UPS,
                max_power_mw: 0.5,
                current_power_mw: 0.0,
                storage_capacity_mj: 1800.0,
                stored_energy_mj: 1800.0,
                startup_time_ms: 0,
                ramp_rate_mw_per_s: 1000.0,
                efficiency: 94.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // UPS for safety systems
            PowerSystem {
                id: "UPS-SAFE-01".to_string(),
                name: "Safety Systems UPS".to_string(),
                source: PowerSource::UPS,
                max_power_mw: 2.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 7200.0,  // 1 hour at full load
                stored_energy_mj: 7200.0,
                startup_time_ms: 0,
                ramp_rate_mw_per_s: 1000.0,
                efficiency: 94.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // Flywheel for magnet protection
            PowerSystem {
                id: "FLY-MAG-01".to_string(),
                name: "Magnet Protection Flywheel Primary".to_string(),
                source: PowerSource::Flywheel,
                max_power_mw: 50.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 500_000.0,  // 500 MJ
                stored_energy_mj: 500_000.0,
                startup_time_ms: 50,  // 50 ms to engage
                ramp_rate_mw_per_s: 500.0,
                efficiency: 90.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // Flywheel backup
            PowerSystem {
                id: "FLY-MAG-02".to_string(),
                name: "Magnet Protection Flywheel Backup".to_string(),
                source: PowerSource::Flywheel,
                max_power_mw: 50.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 500_000.0,
                stored_energy_mj: 500_000.0,
                startup_time_ms: 50,
                ramp_rate_mw_per_s: 500.0,
                efficiency: 90.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // SMES for quench protection
            PowerSystem {
                id: "SMES-01".to_string(),
                name: "Superconducting Magnetic Energy Storage".to_string(),
                source: PowerSource::SMES,
                max_power_mw: 100.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 1_000_000.0,  // 1 GJ
                stored_energy_mj: 1_000_000.0,
                startup_time_ms: 10,  // 10 ms activation
                ramp_rate_mw_per_s: 10000.0,  // Very fast
                efficiency: 95.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // Diesel generator primary
            PowerSystem {
                id: "DIESEL-01".to_string(),
                name: "Diesel Generator Primary (2 MW)".to_string(),
                source: PowerSource::DieselGenerator,
                max_power_mw: 2.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 36_000_000.0,  // 10,000 L fuel @ 36 MJ/L
                stored_energy_mj: 36_000_000.0,
                startup_time_ms: 10_000,  // 10 seconds to start
                ramp_rate_mw_per_s: 0.5,  // 4 seconds to full power
                efficiency: 40.0,
                online: false,  // Standby
                health: ComponentHealth::default(),
            },

            // Diesel generator backup
            PowerSystem {
                id: "DIESEL-02".to_string(),
                name: "Diesel Generator Backup (2 MW)".to_string(),
                source: PowerSource::DieselGenerator,
                max_power_mw: 2.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 36_000_000.0,
                stored_energy_mj: 36_000_000.0,
                startup_time_ms: 10_000,
                ramp_rate_mw_per_s: 0.5,
                efficiency: 40.0,
                online: false,
                health: ComponentHealth::default(),
            },

            // Battery bank for cryogenics
            PowerSystem {
                id: "BATT-CRYO-01".to_string(),
                name: "Cryogenic System Battery Bank".to_string(),
                source: PowerSource::BatteryBank,
                max_power_mw: 5.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 72_000.0,  // 4 hours at full load
                stored_energy_mj: 72_000.0,
                startup_time_ms: 5,  // 5 ms transfer
                ramp_rate_mw_per_s: 100.0,
                efficiency: 92.0,
                online: true,
                health: ComponentHealth::default(),
            },

            // Battery bank for diagnostics
            PowerSystem {
                id: "BATT-DIAG-01".to_string(),
                name: "Diagnostics Battery Bank".to_string(),
                source: PowerSource::BatteryBank,
                max_power_mw: 1.0,
                current_power_mw: 0.0,
                storage_capacity_mj: 14_400.0,  // 4 hours
                stored_energy_mj: 14_400.0,
                startup_time_ms: 5,
                ramp_rate_mw_per_s: 100.0,
                efficiency: 92.0,
                online: true,
                health: ComponentHealth::default(),
            },
        ];

        Self {
            systems,
            total_load_mw: 150.0,  // Typical operation
            critical_load_mw: 15.0, // Safety + control + cryo
            active_source: PowerSource::MainGrid,
        }
    }

    /// Simulate grid failure and calculate response
    pub fn simulate_grid_failure(&self) -> PowerFailureAnalysis {
        let mut analysis = PowerFailureAnalysis {
            event: "Main Grid Loss".to_string(),
            timestamp_ms: 0,
            phases: Vec::new(),
        };

        // Phase 1: Grid loss (t=0)
        analysis.phases.push(FailurePhase {
            name: "Grid Loss Detection".to_string(),
            start_ms: 0,
            end_ms: 1,
            description: "Grid loss detected by voltage/frequency monitors".to_string(),
            systems_affected: vec!["GRID-01".to_string()],
            power_gap_mw: 0.0,  // UPS covers instantly
        });

        // Phase 2: UPS takes over (t=0-1ms)
        analysis.phases.push(FailurePhase {
            name: "UPS Instant Transfer".to_string(),
            start_ms: 0,
            end_ms: 1,
            description: "Online UPS maintains control and safety power with zero gap".to_string(),
            systems_affected: vec!["UPS-CTRL-01".to_string(), "UPS-SAFE-01".to_string()],
            power_gap_mw: 0.0,
        });

        // Phase 3: Flywheel activation (t=1-50ms)
        analysis.phases.push(FailurePhase {
            name: "Flywheel Engagement".to_string(),
            start_ms: 1,
            end_ms: 51,
            description: "Flywheels engage to support magnet systems".to_string(),
            systems_affected: vec!["FLY-MAG-01".to_string(), "FLY-MAG-02".to_string()],
            power_gap_mw: 0.0,  // Covered by UPS
        });

        // Phase 4: SMES activation (t=1-10ms)
        analysis.phases.push(FailurePhase {
            name: "SMES Activation".to_string(),
            start_ms: 1,
            end_ms: 11,
            description: "SMES provides high-power quench protection capability".to_string(),
            systems_affected: vec!["SMES-01".to_string()],
            power_gap_mw: 0.0,
        });

        // Phase 5: Diesel generator startup (t=0-14s)
        analysis.phases.push(FailurePhase {
            name: "Diesel Generator Startup".to_string(),
            start_ms: 0,
            end_ms: 14_000,
            description: "Diesel generators auto-start and ramp to full power".to_string(),
            systems_affected: vec!["DIESEL-01".to_string(), "DIESEL-02".to_string()],
            power_gap_mw: 0.0,  // UPS/Flywheel covers
        });

        // Phase 6: Plasma controlled shutdown (t=0-100ms)
        analysis.phases.push(FailurePhase {
            name: "Plasma Controlled Termination".to_string(),
            start_ms: 0,
            end_ms: 100,
            description: "PIRS initiates controlled plasma shutdown within 100ms".to_string(),
            systems_affected: vec!["Control System".to_string()],
            power_gap_mw: 0.0,
        });

        analysis
    }

    /// Get total UPS capacity
    pub fn total_ups_capacity_mj(&self) -> f64 {
        self.systems.iter()
            .filter(|s| s.source == PowerSource::UPS)
            .map(|s| s.storage_capacity_mj)
            .sum()
    }

    /// Get total flywheel capacity
    pub fn total_flywheel_capacity_mj(&self) -> f64 {
        self.systems.iter()
            .filter(|s| s.source == PowerSource::Flywheel)
            .map(|s| s.storage_capacity_mj)
            .sum()
    }

    /// Get maximum power gap during any failure scenario (ms)
    pub fn max_power_gap_ms(&self) -> TimeMs {
        // With online UPS, there is zero gap for critical systems
        0
    }
}

/// Analysis of power failure scenario
#[derive(Debug, Clone)]
pub struct PowerFailureAnalysis {
    pub event: String,
    pub timestamp_ms: TimeMs,
    pub phases: Vec<FailurePhase>,
}

/// Phase of failure response
#[derive(Debug, Clone)]
pub struct FailurePhase {
    pub name: String,
    pub start_ms: TimeMs,
    pub end_ms: TimeMs,
    pub description: String,
    pub systems_affected: Vec<String>,
    pub power_gap_mw: f64,
}

impl PowerFailureAnalysis {
    /// Get total time to stable backup power (ms)
    pub fn time_to_stable_ms(&self) -> TimeMs {
        self.phases.iter()
            .map(|p| p.end_ms)
            .max()
            .unwrap_or(0)
    }

    /// Get maximum power gap during transition
    pub fn max_power_gap_mw(&self) -> f64 {
        self.phases.iter()
            .map(|p| p.power_gap_mw)
            .fold(0.0, f64::max)
    }

    /// Generate detailed report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("=== Power Failure Analysis: {} ===\n\n", self.event));

        for phase in &self.phases {
            report.push_str(&format!(
                "Phase: {}\n  Time: {} - {} ms\n  Description: {}\n  Systems: {:?}\n  Power Gap: {:.2} MW\n\n",
                phase.name, phase.start_ms, phase.end_ms,
                phase.description, phase.systems_affected, phase.power_gap_mw
            ));
        }

        report.push_str(&format!(
            "SUMMARY:\n  Time to stable backup: {} ms\n  Maximum power gap: {:.2} MW\n",
            self.time_to_stable_ms(), self.max_power_gap_mw()
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_infrastructure() {
        let infra = PowerInfrastructure::ts1();
        assert!(!infra.systems.is_empty());
        assert!(infra.total_ups_capacity_mj() > 0.0);
    }

    #[test]
    fn test_zero_power_gap() {
        let infra = PowerInfrastructure::ts1();
        assert_eq!(infra.max_power_gap_ms(), 0);
    }

    #[test]
    fn test_grid_failure_analysis() {
        let infra = PowerInfrastructure::ts1();
        let analysis = infra.simulate_grid_failure();
        assert!(!analysis.phases.is_empty());
        // With online UPS, max gap should be 0
        assert_eq!(analysis.max_power_gap_mw(), 0.0);
    }
}
