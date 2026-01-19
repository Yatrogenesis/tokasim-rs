//! # Proyecciones 3D a 2D
//!
//! Funciones de proyección para visualización.

use crate::types::Vec3;
use super::{Point2D, ViewType};

/// Proyecta un punto 3D a 2D según el tipo de vista
pub fn project_to_2d(point: Vec3, view: ViewType, scale: f64, offset_x: f64, offset_y: f64) -> Point2D {
    match view {
        ViewType::TopDown => {
            // Vista desde arriba (X-Y)
            Point2D::new(
                offset_x + point.x * scale,
                offset_y - point.y * scale,
            )
        }
        ViewType::PoloidalCrossSection => {
            // Corte poloidal (R-Z)
            let r = (point.x * point.x + point.y * point.y).sqrt();
            Point2D::new(
                offset_x + r * scale,
                offset_y - point.z * scale,
            )
        }
        ViewType::ToroidalCrossSection => {
            // Corte toroidal (R-phi)
            let r = (point.x * point.x + point.y * point.y).sqrt();
            let phi = point.y.atan2(point.x);
            Point2D::new(
                offset_x + phi * scale * 10.0, // Escalado angular
                offset_y - r * scale,
            )
        }
        ViewType::Isometric => {
            // Proyección isométrica
            let iso_angle = std::f64::consts::PI / 6.0; // 30 grados
            let x_proj = (point.x - point.y) * iso_angle.cos();
            let y_proj = (point.x + point.y) * iso_angle.sin() - point.z;
            Point2D::new(
                offset_x + x_proj * scale,
                offset_y + y_proj * scale,
            )
        }
        ViewType::Custom3D { theta, phi } => {
            // Rotación arbitraria
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let sin_p = phi.sin();
            let cos_p = phi.cos();

            let x_rot = point.x * cos_p - point.y * sin_p;
            let y_rot = point.x * sin_p * cos_t + point.y * cos_p * cos_t - point.z * sin_t;

            Point2D::new(
                offset_x + x_rot * scale,
                offset_y - y_rot * scale,
            )
        }
        ViewType::Exploded => {
            // Vista explosionada (igual que isométrica por ahora)
            let iso_angle = std::f64::consts::PI / 6.0;
            let x_proj = (point.x - point.y) * iso_angle.cos();
            let y_proj = (point.x + point.y) * iso_angle.sin() - point.z;
            Point2D::new(
                offset_x + x_proj * scale,
                offset_y + y_proj * scale,
            )
        }
    }
}

/// Transforma coordenadas cilíndricas (R, Z, phi) a cartesianas
pub fn cylindrical_to_cartesian(r: f64, z: f64, phi: f64) -> Vec3 {
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Genera puntos en un círculo toroidal
pub fn toroidal_circle(major_r: f64, minor_r: f64, phi: f64, n_points: usize) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_points as f64);
        let r = major_r + minor_r * theta.cos();
        let z = minor_r * theta.sin();
        points.push(cylindrical_to_cartesian(r, z, phi));
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let projected = project_to_2d(point, ViewType::TopDown, 100.0, 500.0, 500.0);
        assert!((projected.x - 600.0).abs() < 1e-10);
    }
}
