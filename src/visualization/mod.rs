//! # Visualization Module
//!
//! High-fidelity visualization system for tokamak simulation.
//!
//! ## Features
//!
//! - SVG rendering for vector graphics output
//! - 3D to 2D projection (isometric, cross-section, top-down)
//! - Component decomposition views
//! - Particle trajectory visualization
//! - Field line rendering
//! - Real-time status dashboard

pub mod renderer;
pub mod projections;
pub mod components;
pub mod particles;
pub mod fields;
pub mod dashboard;
pub mod animation;

pub use renderer::*;
pub use projections::project_to_2d;
pub use components::*;
pub use particles::{ParticleType, ParticleTrajectory, ParticleRenderer};
pub use dashboard::{StatusIndicator, Dashboard};
pub use animation::{AnimationConfig, AnimationGenerator, AnimationFrame};

#[allow(unused_imports)]
use crate::types::Vec3;
use std::f64::consts::PI;

/// Color representation (RGB)
#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: f64,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    pub fn to_rgba_string(&self) -> String {
        format!("rgba({},{},{},{:.2})", self.r, self.g, self.b, self.a)
    }

    // Predefined colors for tokamak visualization
    pub const PLASMA_CORE: Color = Color::rgb(255, 100, 50);      // Hot orange-red
    pub const PLASMA_EDGE: Color = Color::rgb(255, 200, 100);     // Cooler yellow
    pub const MAGNETIC_FIELD: Color = Color::rgb(50, 150, 255);   // Blue
    pub const TOROIDAL_COIL: Color = Color::rgb(100, 100, 120);   // Steel gray
    pub const POLOIDAL_COIL: Color = Color::rgb(180, 100, 50);    // Copper
    pub const VACUUM_VESSEL: Color = Color::rgb(80, 80, 90);      // Dark steel
    pub const FIRST_WALL: Color = Color::rgb(60, 60, 70);         // Carbon/tungsten
    pub const DIVERTOR: Color = Color::rgb(40, 40, 50);           // Dark
    pub const CRYOSTAT: Color = Color::rgb(200, 200, 220);        // Light gray
    pub const NEUTRON: Color = Color::rgb(0, 255, 0);             // Green
    pub const ALPHA: Color = Color::rgb(255, 0, 255);             // Magenta
    pub const ELECTRON: Color = Color::rgb(0, 200, 255);          // Cyan
    pub const DEUTERIUM: Color = Color::rgb(255, 255, 0);         // Yellow
    pub const TRITIUM: Color = Color::rgb(255, 150, 0);           // Orange
}

/// Temperature to color mapping (for plasma visualization)
pub fn temperature_to_color(temp_kev: f64) -> Color {
    // Map 0-20 keV to color gradient
    let t = (temp_kev / 20.0).clamp(0.0, 1.0);

    if t < 0.25 {
        // Blue to cyan (cold edge)
        let f = t / 0.25;
        Color::rgb(0, (100.0 * f) as u8, (255.0 - 55.0 * f) as u8)
    } else if t < 0.5 {
        // Cyan to yellow (warm)
        let f = (t - 0.25) / 0.25;
        Color::rgb((255.0 * f) as u8, (100.0 + 155.0 * f) as u8, (200.0 * (1.0 - f)) as u8)
    } else if t < 0.75 {
        // Yellow to orange (hot)
        let f = (t - 0.5) / 0.25;
        Color::rgb(255, (255.0 - 105.0 * f) as u8, 0)
    } else {
        // Orange to white-hot (core)
        let f = (t - 0.75) / 0.25;
        Color::rgb(255, (150.0 + 105.0 * f) as u8, (255.0 * f) as u8)
    }
}

/// Density to opacity mapping
pub fn density_to_opacity(density: f64, max_density: f64) -> f64 {
    (density / max_density).clamp(0.1, 0.9)
}

/// 2D point for rendering
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Bounding box for viewports
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        Self { min_x, max_x, min_y, max_y }
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

/// View type for different visualization perspectives
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewType {
    /// Top-down view (looking along Z axis)
    TopDown,
    /// Poloidal cross-section (R-Z plane at fixed phi)
    PoloidalCrossSection,
    /// Toroidal cross-section (R-phi plane at Z=0)
    ToroidalCrossSection,
    /// Isometric 3D projection
    Isometric,
    /// Custom 3D view angles
    Custom3D { theta: f64, phi: f64 },
    /// Exploded component view
    Exploded,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisConfig {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// View type
    pub view: ViewType,
    /// Show magnetic field lines
    pub show_field_lines: bool,
    /// Show particle trajectories
    pub show_particles: bool,
    /// Show temperature profile
    pub show_temperature: bool,
    /// Show component labels
    pub show_labels: bool,
    /// Animation frame rate (0 for static)
    pub frame_rate: u32,
    /// Scale factor
    pub scale: f64,
    /// Background color
    pub background: Color,
}

impl Default for VisConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            view: ViewType::PoloidalCrossSection,
            show_field_lines: true,
            show_particles: true,
            show_temperature: true,
            show_labels: true,
            frame_rate: 0,
            scale: 100.0,  // pixels per meter
            background: Color::rgb(10, 10, 20),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_conversion() {
        let c = Color::rgb(255, 128, 64);
        assert_eq!(c.to_hex(), "#ff8040");
    }

    #[test]
    fn test_temperature_color() {
        let cold = temperature_to_color(1.0);
        let hot = temperature_to_color(15.0);
        // Cold should be more blue
        assert!(cold.b > cold.r);
        // Hot should be more red
        assert!(hot.r > hot.b);
    }
}
