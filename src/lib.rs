//! # TOKASIM-RS
//!
//! Tokamak Fusion Reactor Hyperrealistic Simulator
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                          TOKASIM-RS                                         │
//! │                  Tokamak Simulator in Rust                                  │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  LEVEL 1: PARTICLES (Particle-In-Cell)                                      │
//! │  LEVEL 2: FIELDS (Maxwell Equations - FDTD)                                 │
//! │  LEVEL 3: MHD (Magnetohydrodynamics)                                        │
//! │  LEVEL 4: NUCLEAR (Fusion Reactions)                                        │
//! │  LEVEL 5: CONTROL (SYNTEX Deterministic)                                    │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Competitive Advantage vs NVIDIA Omniverse
//!
//! | Feature           | NVIDIA Omniverse    | TOKASIM-RS + YatroNet           |
//! |-------------------|---------------------|----------------------------------|
//! | Physics           | Approximate (RT)    | First-principles (exact)         |
//! | Particles         | Visual only         | 10^9-10^14 simulated            |
//! | Control           | ML/black box        | SYNTEX deterministic             |
//! | Hardware          | RTX GPUs ($$$)      | CPUs existing ($0)              |
//! | Determinism       | No                  | Yes (reproducible)               |
//! | Auditability      | Difficult           | Total (nuclear regulators)       |
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## Date
//! January 2026

pub mod constants;
pub mod types;
pub mod particle;
pub mod field;
pub mod mhd;
pub mod nuclear;
pub mod control;
pub mod geometry;
pub mod simulator;
pub mod optimizer;
pub mod visualization;
pub mod components;

// Re-exports
pub use constants::*;
pub use types::*;
pub use simulator::TokamakSimulator;
pub use optimizer::{
    ReactorParameterSpace, ReactorDesign, NSGA2Optimizer,
    CostModel, ScalingLaws, InfrastructureCalculator, ReportGenerator
};

/// TOKASIM version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Information about the simulator
pub fn info() -> String {
    format!(
        "TOKASIM-RS v{}\n\
         Tokamak Fusion Reactor Hyperrealistic Simulator\n\
         Deterministic Physics Engine in Rust\n\
         Author: Francisco Molina-Burgos, Avermex Research Division\n\
         Zero dependencies core - Pure Rust",
        VERSION
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info() {
        let info = info();
        assert!(info.contains("TOKASIM"));
        assert!(info.contains("Molina-Burgos"));
    }
}
