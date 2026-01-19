//! # Dashboard de Estado
//!
//! Panel de información del simulador.

use super::Color;

/// Indicador de estado
#[derive(Debug, Clone)]
pub struct StatusIndicator {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub min_value: f64,
    pub max_value: f64,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

impl StatusIndicator {
    pub fn new(name: &str, unit: &str, min: f64, max: f64) -> Self {
        Self {
            name: name.to_string(),
            value: 0.0,
            unit: unit.to_string(),
            min_value: min,
            max_value: max,
            warning_threshold: max * 0.8,
            critical_threshold: max * 0.95,
        }
    }

    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    pub fn status_color(&self) -> Color {
        if self.value >= self.critical_threshold {
            Color::rgb(255, 50, 50) // Rojo crítico
        } else if self.value >= self.warning_threshold {
            Color::rgb(255, 200, 50) // Amarillo advertencia
        } else {
            Color::rgb(50, 255, 50) // Verde normal
        }
    }

    pub fn percentage(&self) -> f64 {
        ((self.value - self.min_value) / (self.max_value - self.min_value) * 100.0).clamp(0.0, 100.0)
    }

    pub fn to_svg(&self, x: f64, y: f64, width: f64, height: f64) -> String {
        let bar_width = width * self.percentage() / 100.0;
        let color = self.status_color();

        format!(
            "<g transform=\"translate({}, {})\">\n\
  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#333\" stroke=\"#555\" stroke-width=\"1\"/>\n\
  <rect x=\"2\" y=\"2\" width=\"{}\" height=\"{}\" fill=\"{}\"/>\n\
  <text x=\"5\" y=\"{}\" font-size=\"12\" fill=\"white\">{}</text>\n\
  <text x=\"{}\" y=\"{}\" font-size=\"12\" fill=\"white\" text-anchor=\"end\">{:.2} {}</text>\n\
</g>",
            x, y,
            width, height,
            bar_width - 4.0, height - 4.0, color.to_hex(),
            height - 5.0, self.name,
            width - 5.0, height - 5.0, self.value, self.unit
        )
    }
}

/// Dashboard completo
pub struct Dashboard {
    pub indicators: Vec<StatusIndicator>,
    pub title: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
}

impl Dashboard {
    pub fn new(title: &str, x: f64, y: f64, width: f64) -> Self {
        Self {
            indicators: Vec::new(),
            title: title.to_string(),
            x,
            y,
            width,
        }
    }

    pub fn add_indicator(&mut self, indicator: StatusIndicator) {
        self.indicators.push(indicator);
    }

    /// Crea dashboard predefinido para tokamak
    pub fn tokamak_default(x: f64, y: f64) -> Self {
        let mut dashboard = Self::new("Tokamak Status", x, y, 250.0);

        dashboard.add_indicator(StatusIndicator::new("Plasma Current", "MA", 0.0, 20.0));
        dashboard.add_indicator(StatusIndicator::new("B Toroidal", "T", 0.0, 15.0));
        dashboard.add_indicator(StatusIndicator::new("Density", "10²⁰/m³", 0.0, 3.0));
        dashboard.add_indicator(StatusIndicator::new("Temperature", "keV", 0.0, 30.0));
        dashboard.add_indicator(StatusIndicator::new("Fusion Power", "MW", 0.0, 1000.0));
        dashboard.add_indicator(StatusIndicator::new("Q Factor", "", 0.0, 50.0));
        dashboard.add_indicator(StatusIndicator::new("β_N", "", 0.0, 5.0));
        dashboard.add_indicator(StatusIndicator::new("Wall Load", "MW/m²", 0.0, 5.0));

        dashboard
    }

    pub fn update_values(&mut self, values: &[(&str, f64)]) {
        for (name, value) in values {
            if let Some(ind) = self.indicators.iter_mut().find(|i| i.name == *name) {
                ind.set_value(*value);
            }
        }
    }

    pub fn to_svg(&self) -> String {
        let bar_height = 25.0;
        let spacing = 5.0;
        let total_height = self.indicators.len() as f64 * (bar_height + spacing) + 30.0;

        let mut svg = format!(
            "<g id=\"dashboard\" transform=\"translate({}, {})\">\n\
  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#1a1a2e\" stroke=\"#444\" stroke-width=\"2\" rx=\"5\"/>\n\
  <text x=\"{}\" y=\"20\" font-size=\"14\" font-weight=\"bold\" fill=\"white\" text-anchor=\"middle\">{}</text>\n",
            self.x, self.y,
            self.width, total_height,
            self.width / 2.0, self.title
        );

        for (i, indicator) in self.indicators.iter().enumerate() {
            let ind_y = 30.0 + i as f64 * (bar_height + spacing);
            svg.push_str(&indicator.to_svg(5.0, ind_y, self.width - 10.0, bar_height));
            svg.push('\n');
        }

        svg.push_str("</g>");
        svg
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::tokamak_default(10.0, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_indicator() {
        let mut ind = StatusIndicator::new("Test", "MW", 0.0, 100.0);
        ind.set_value(50.0);
        assert!((ind.percentage() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_dashboard() {
        let dashboard = Dashboard::tokamak_default(10.0, 10.0);
        let svg = dashboard.to_svg();
        assert!(svg.contains("dashboard"));
    }
}
