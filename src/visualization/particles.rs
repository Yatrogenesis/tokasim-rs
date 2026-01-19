//! # Visualización de Partículas
//!
//! Renderizado de trayectorias de partículas.

use super::{Color, ViewType};
use super::projections::project_to_2d;
use crate::types::Vec3;

/// Tipos de partículas para visualización
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParticleType {
    Deuterium,
    Tritium,
    Alpha,
    Neutron,
    Electron,
}

impl ParticleType {
    /// Color asociado a cada tipo de partícula
    pub fn color(&self) -> Color {
        match self {
            ParticleType::Deuterium => Color::DEUTERIUM,
            ParticleType::Tritium => Color::TRITIUM,
            ParticleType::Alpha => Color::ALPHA,
            ParticleType::Neutron => Color::NEUTRON,
            ParticleType::Electron => Color::ELECTRON,
        }
    }

    /// Radio de visualización
    pub fn radius(&self) -> f64 {
        match self {
            ParticleType::Electron => 1.0,
            ParticleType::Neutron => 2.0,
            _ => 3.0,
        }
    }
}

/// Trayectoria de partícula para visualización
#[derive(Debug, Clone)]
pub struct ParticleTrajectory {
    pub particle_type: ParticleType,
    pub positions: Vec<Vec3>,
    pub times: Vec<f64>,
}

impl ParticleTrajectory {
    pub fn new(particle_type: ParticleType) -> Self {
        Self {
            particle_type,
            positions: Vec::new(),
            times: Vec::new(),
        }
    }

    pub fn add_point(&mut self, position: Vec3, time: f64) {
        self.positions.push(position);
        self.times.push(time);
    }

    /// Genera SVG path para la trayectoria
    pub fn to_svg_path(
        &self,
        view: ViewType,
        scale: f64,
        offset_x: f64,
        offset_y: f64,
    ) -> String {
        if self.positions.is_empty() {
            return String::new();
        }

        let color = self.particle_type.color();
        let mut path = String::new();

        // Inicio del path
        let first = project_to_2d(self.positions[0], view, scale, offset_x, offset_y);
        path.push_str(&format!(
            "<path d=\"M {:.2} {:.2}",
            first.x, first.y
        ));

        // Resto de puntos
        for pos in self.positions.iter().skip(1) {
            let p = project_to_2d(*pos, view, scale, offset_x, offset_y);
            path.push_str(&format!(" L {:.2} {:.2}", p.x, p.y));
        }

        path.push_str(&format!(
            "\" fill=\"none\" stroke=\"{}\" stroke-width=\"1\" stroke-opacity=\"0.7\"/>",
            color.to_hex()
        ));

        // Punto final (partícula actual)
        if let Some(last_pos) = self.positions.last() {
            let p = project_to_2d(*last_pos, view, scale, offset_x, offset_y);
            let r = self.particle_type.radius();
            path.push_str(&format!(
                "\n<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.1}\" fill=\"{}\"/>",
                p.x, p.y, r, color.to_hex()
            ));
        }

        path
    }
}

/// Renderizador de múltiples partículas
pub struct ParticleRenderer {
    pub trajectories: Vec<ParticleTrajectory>,
    pub max_trail_length: usize,
}

impl ParticleRenderer {
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
            max_trail_length: 100,
        }
    }

    pub fn add_trajectory(&mut self, trajectory: ParticleTrajectory) {
        self.trajectories.push(trajectory);
    }

    pub fn render_all(
        &self,
        view: ViewType,
        scale: f64,
        offset_x: f64,
        offset_y: f64,
    ) -> String {
        let mut svg = String::from("<g id=\"particles\">\n");

        for traj in &self.trajectories {
            svg.push_str(&traj.to_svg_path(view, scale, offset_x, offset_y));
            svg.push('\n');
        }

        svg.push_str("</g>");
        svg
    }
}

impl Default for ParticleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_trajectory() {
        let mut traj = ParticleTrajectory::new(ParticleType::Deuterium);
        traj.add_point(Vec3::new(0.0, 0.0, 0.0), 0.0);
        traj.add_point(Vec3::new(1.0, 0.0, 0.0), 1.0);

        let svg = traj.to_svg_path(ViewType::TopDown, 100.0, 500.0, 500.0);
        assert!(svg.contains("path"));
        assert!(svg.contains("circle"));
    }
}
