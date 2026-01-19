//! # Utilidades para el Optimizador
//!
//! Funciones auxiliares, conversiones y herramientas de análisis.

use std::collections::HashMap;

/// Constantes físicas fundamentales
pub mod constants {
    /// Masa del electrón (kg)
    pub const M_E: f64 = 9.109e-31;
    /// Masa del protón (kg)
    pub const M_P: f64 = 1.673e-27;
    /// Masa del deuterón (kg)
    pub const M_D: f64 = 3.344e-27;
    /// Masa del tritón (kg)
    pub const M_T: f64 = 5.008e-27;
    /// Carga elemental (C)
    pub const E_CHARGE: f64 = 1.602e-19;
    /// Permitividad del vacío (F/m)
    pub const EPSILON_0: f64 = 8.854e-12;
    /// Permeabilidad del vacío (H/m)
    pub const MU_0: f64 = 1.257e-6;
    /// Constante de Boltzmann (J/K)
    pub const K_B: f64 = 1.381e-23;
    /// Velocidad de la luz (m/s)
    pub const C: f64 = 2.998e8;
    /// Energía de fusión D-T (MeV)
    pub const E_FUSION_DT: f64 = 17.6;
    /// Pi
    pub const PI: f64 = std::f64::consts::PI;
}

/// Conversiones de unidades
pub mod conversions {
    /// keV a Joules
    pub fn kev_to_joules(kev: f64) -> f64 {
        kev * 1.602e-16
    }

    /// Joules a keV
    pub fn joules_to_kev(j: f64) -> f64 {
        j / 1.602e-16
    }

    /// keV a Kelvin
    pub fn kev_to_kelvin(kev: f64) -> f64 {
        kev * 1.16e7
    }

    /// Kelvin a keV
    pub fn kelvin_to_kev(k: f64) -> f64 {
        k / 1.16e7
    }

    /// Tesla a Gauss
    pub fn tesla_to_gauss(t: f64) -> f64 {
        t * 1e4
    }

    /// Gauss a Tesla
    pub fn gauss_to_tesla(g: f64) -> f64 {
        g * 1e-4
    }

    /// m³ a litros
    pub fn m3_to_liters(m3: f64) -> f64 {
        m3 * 1000.0
    }

    /// MW a W
    pub fn mw_to_w(mw: f64) -> f64 {
        mw * 1e6
    }

    /// W a MW
    pub fn w_to_mw(w: f64) -> f64 {
        w * 1e-6
    }
}

/// Funciones estadísticas
pub mod statistics {
    /// Media aritmética
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Desviación estándar
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let m = mean(values);
        let variance = values.iter()
            .map(|v| (v - m).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Mediana
    pub fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Percentil
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() || p < 0.0 || p > 100.0 {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Coeficiente de variación
    pub fn coefficient_of_variation(values: &[f64]) -> f64 {
        let m = mean(values);
        if m.abs() < 1e-10 {
            return 0.0;
        }
        std_dev(values) / m
    }

    /// Rango intercuartílico
    pub fn iqr(values: &[f64]) -> f64 {
        percentile(values, 75.0) - percentile(values, 25.0)
    }
}

/// Interpolación y ajuste
pub mod interpolation {
    /// Interpolación lineal entre dos puntos
    pub fn linear(x: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        if (x1 - x0).abs() < 1e-10 {
            return y0;
        }
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    /// Interpolación lineal en tabla
    pub fn table_lookup(x: f64, x_values: &[f64], y_values: &[f64]) -> Option<f64> {
        if x_values.len() != y_values.len() || x_values.is_empty() {
            return None;
        }

        // Encontrar intervalo
        for i in 0..x_values.len() - 1 {
            if x >= x_values[i] && x <= x_values[i + 1] {
                return Some(linear(x, x_values[i], y_values[i], x_values[i + 1], y_values[i + 1]));
            }
        }

        // Extrapolación si está fuera de rango
        if x < x_values[0] {
            Some(linear(x, x_values[0], y_values[0], x_values[1], y_values[1]))
        } else {
            let n = x_values.len();
            Some(linear(x, x_values[n - 2], y_values[n - 2], x_values[n - 1], y_values[n - 1]))
        }
    }

    /// Interpolación bilineal
    pub fn bilinear(
        x: f64, y: f64,
        x0: f64, x1: f64,
        y0: f64, y1: f64,
        f00: f64, f10: f64, f01: f64, f11: f64
    ) -> f64 {
        let dx = x1 - x0;
        let dy = y1 - y0;

        if dx.abs() < 1e-10 || dy.abs() < 1e-10 {
            return f00;
        }

        let tx = (x - x0) / dx;
        let ty = (y - y0) / dy;

        f00 * (1.0 - tx) * (1.0 - ty)
            + f10 * tx * (1.0 - ty)
            + f01 * (1.0 - tx) * ty
            + f11 * tx * ty
    }
}

/// Métodos numéricos
pub mod numerical {
    /// Integración trapezoidal
    pub fn trapezoid(y: &[f64], dx: f64) -> f64 {
        if y.len() < 2 {
            return 0.0;
        }
        let mut sum = 0.5 * (y[0] + y[y.len() - 1]);
        for i in 1..y.len() - 1 {
            sum += y[i];
        }
        sum * dx
    }

    /// Derivada numérica (diferencias centradas)
    pub fn derivative(y: &[f64], dx: f64) -> Vec<f64> {
        if y.len() < 3 {
            return vec![0.0; y.len()];
        }

        let mut dy = vec![0.0; y.len()];

        // Forward difference at start
        dy[0] = (y[1] - y[0]) / dx;

        // Central differences
        for i in 1..y.len() - 1 {
            dy[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx);
        }

        // Backward difference at end
        dy[y.len() - 1] = (y[y.len() - 1] - y[y.len() - 2]) / dx;

        dy
    }

    /// Root finding by bisection
    pub fn bisection<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64>
    where
        F: Fn(f64) -> f64
    {
        let mut fa = f(a);
        let fb = f(b);

        if fa * fb > 0.0 {
            return None; // No root in interval
        }

        for _ in 0..max_iter {
            let c = (a + b) / 2.0;
            let fc = f(c);

            if fc.abs() < tol || (b - a) / 2.0 < tol {
                return Some(c);
            }

            if fa * fc < 0.0 {
                b = c;
            } else {
                a = c;
                fa = fc;
            }
        }

        Some((a + b) / 2.0)
    }

    /// Newton-Raphson con derivada numérica
    pub fn newton_raphson<F>(f: F, x0: f64, tol: f64, max_iter: usize) -> Option<f64>
    where
        F: Fn(f64) -> f64
    {
        let h = 1e-8;
        let mut x = x0;

        for _ in 0..max_iter {
            let fx = f(x);
            if fx.abs() < tol {
                return Some(x);
            }

            let dfx = (f(x + h) - f(x - h)) / (2.0 * h);
            if dfx.abs() < 1e-15 {
                return None; // Division by zero
            }

            x = x - fx / dfx;
        }

        Some(x)
    }
}

/// Normalización de Pareto
pub fn normalize_pareto(values: &[f64], minimize: bool) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range.abs() < 1e-10 {
        return vec![0.5; values.len()];
    }

    values.iter()
        .map(|v| {
            let normalized = (v - min_val) / range;
            if minimize { normalized } else { 1.0 - normalized }
        })
        .collect()
}

/// Calcula distancia de crowding para NSGA-II
pub fn crowding_distance(objectives: &[HashMap<String, f64>]) -> Vec<f64> {
    let n = objectives.len();
    if n < 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0; n];

    // Obtener nombres de objetivos del primer elemento
    if let Some(first) = objectives.first() {
        for obj_name in first.keys() {
            // Extraer valores y ordenar
            let mut indexed: Vec<(usize, f64)> = objectives.iter()
                .enumerate()
                .filter_map(|(i, obj)| obj.get(obj_name).map(|&v| (i, v)))
                .collect();

            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Extremos tienen distancia infinita
            if let Some(&(first_idx, _)) = indexed.first() {
                distances[first_idx] = f64::INFINITY;
            }
            if let Some(&(last_idx, _)) = indexed.last() {
                distances[last_idx] = f64::INFINITY;
            }

            // Calcular distancias intermedias
            let range = if indexed.len() >= 2 {
                indexed.last().unwrap().1 - indexed.first().unwrap().1
            } else {
                1.0
            };

            if range > 1e-10 {
                for i in 1..indexed.len() - 1 {
                    let (idx, _) = indexed[i];
                    let prev_val = indexed[i - 1].1;
                    let next_val = indexed[i + 1].1;
                    distances[idx] += (next_val - prev_val) / range;
                }
            }
        }
    }

    distances
}

/// Hipervolumen indicator (2D simplificado)
pub fn hypervolume_2d(points: &[(f64, f64)], reference: (f64, f64)) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Ordenar por primera coordenada
    let mut sorted: Vec<_> = points.iter()
        .filter(|p| p.0 < reference.0 && p.1 < reference.1)
        .cloned()
        .collect();

    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut volume = 0.0;
    let mut prev_y = reference.1;

    for (x, y) in sorted {
        if y < prev_y {
            volume += (reference.0 - x) * (prev_y - y);
            prev_y = y;
        }
    }

    volume
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((statistics::mean(&values) - 3.0).abs() < 1e-10);
        assert!((statistics::median(&values) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation() {
        let y = interpolation::linear(0.5, 0.0, 0.0, 1.0, 1.0);
        assert!((y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bisection() {
        let root = numerical::bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100);
        assert!(root.is_some());
        assert!((root.unwrap() - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_conversions() {
        let kev = 1.0;
        let joules = conversions::kev_to_joules(kev);
        let back = conversions::joules_to_kev(joules);
        assert!((kev - back).abs() < 1e-10);
    }
}
