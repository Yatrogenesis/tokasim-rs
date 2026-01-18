//! # Complete Tokamak Component System
//!
//! Comprehensive definition of ALL tokamak components including:
//! - Power systems and redundancy
//! - Safety mechanisms
//! - Failure mode analysis
//! - Backup systems and timing

pub mod power;
pub mod safety;
pub mod redundancy;
pub mod inventory;

pub use power::*;
pub use safety::*;
pub use redundancy::*;
pub use inventory::*;

use std::collections::HashMap;

/// System-wide component status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentStatus {
    /// Fully operational
    Operational,
    /// Degraded but functional
    Degraded,
    /// In standby mode
    Standby,
    /// Running on backup power
    BackupPower,
    /// Failed - requires attention
    Failed,
    /// Under maintenance
    Maintenance,
    /// Emergency shutdown
    EmergencyShutdown,
}

/// Time duration in milliseconds
pub type TimeMs = u64;

/// Component health metrics
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub status: ComponentStatus,
    pub uptime_hours: f64,
    pub last_maintenance_hours: f64,
    pub temperature_celsius: f64,
    pub power_draw_kw: f64,
    pub efficiency_percent: f64,
}

impl Default for ComponentHealth {
    fn default() -> Self {
        Self {
            status: ComponentStatus::Operational,
            uptime_hours: 0.0,
            last_maintenance_hours: 0.0,
            temperature_celsius: 20.0,
            power_draw_kw: 0.0,
            efficiency_percent: 100.0,
        }
    }
}
