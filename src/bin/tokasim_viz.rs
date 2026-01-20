//! # TOKASIM-RS 3D Visualization
//!
//! Real-time 3D visualization of tokamak fusion reactor using Bevy.
//!
//! ## Usage
//!
//! ```bash
//! # TS-1 (default)
//! cargo run --bin tokasim-viz --features bevy-viz --release
//!
//! # SPARC
//! cargo run --bin tokasim-viz --features bevy-viz --release -- sparc
//!
//! # ITER
//! cargo run --bin tokasim-viz --features bevy-viz --release -- iter
//! ```
//!
//! ## Controls
//!
//! - Mouse drag: Rotate camera
//! - Scroll: Zoom
//! - Space: Play/Pause
//! - R: Reset
//! - W: Toggle wireframe
//! - +/-: Speed control
//!
//! ## Author
//!
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use tokasim_rs::visualization::{TokasimBridge, run_visualization};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let bridge = if args.len() > 1 {
        match args[1].to_lowercase().as_str() {
            "sparc" => {
                println!("Loading SPARC configuration...");
                TokasimBridge::sparc()
            }
            "iter" => {
                println!("Loading ITER configuration...");
                TokasimBridge::iter()
            }
            "ts1" | "ts-1" | _ => {
                println!("Loading TS-1 configuration...");
                TokasimBridge::ts1()
            }
        }
    } else {
        println!("Loading TS-1 configuration (default)...");
        println!("Usage: tokasim-viz [ts1|sparc|iter]");
        println!();
        TokasimBridge::ts1()
    };

    run_visualization(bridge);
}
