//! # TOKASIM-RS
//!
//! Tokamak Fusion Reactor Hyperrealistic Simulator
//!
//! A deterministic physics engine competing with NVIDIA Omniverse + SPARC.

use tokasim_rs::*;
use tokasim_rs::simulator::{TokamakSimulator, SimulationParams};

fn main() {
    println!("{}", info());
    println!();

    // Create TS-1 simulation with moderate parameters
    let params = SimulationParams {
        n_particles: 10_000,
        grid_resolution: 20,
        dt: 1e-11,
        total_time: 1e-8,
        output_interval: 1e-9,
        seed: 42,
    };

    println!("Initializing simulation...");
    let mut sim = TokamakSimulator::ts1(params);

    println!("Configuration: {}", sim.config.name);
    println!("  Major radius: {:.3} m", sim.config.major_radius);
    println!("  Minor radius: {:.3} m", sim.config.minor_radius);
    println!("  Toroidal field: {:.1} T", sim.config.toroidal_field);
    println!("  Plasma current: {:.1} MA", sim.config.plasma_current / 1e6);
    println!("  Temperature: {:.1} keV", sim.config.ion_temperature_kev);
    println!("  Density: {:.2e} m⁻³", sim.config.density);
    println!();

    println!("Running simulation...");
    let start = std::time::Instant::now();

    // Run for specified time
    sim.run(1e-8);

    let elapsed = start.elapsed();

    println!();
    println!("{}", sim.summary());
    println!();
    println!("Wall-clock time: {:.3} s", elapsed.as_secs_f64());
    println!("Performance: {:.2e} particle-steps/s",
             sim.state.particle_count as f64 * sim.state.step as f64 / elapsed.as_secs_f64());

    // Export control rules in PIRS format
    println!();
    println!("Control Rules (PIRS format):");
    println!("----------------------------");
    println!("{}", sim.controller.to_pirs());
}
