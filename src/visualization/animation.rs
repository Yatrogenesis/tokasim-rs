//! # Sistema de Animación
//!
//! Generación de frames para animación del simulador.

use super::{VisConfig, ViewType};

/// Frame de animación
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    pub time: f64,
    pub svg_content: String,
}

/// Configuración de animación
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    pub start_time: f64,
    pub end_time: f64,
    pub fps: u32,
    pub loop_animation: bool,
    pub view_config: VisConfig,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            start_time: 0.0,
            end_time: 1.0,
            fps: 30,
            loop_animation: true,
            view_config: VisConfig::default(),
        }
    }
}

impl AnimationConfig {
    pub fn total_frames(&self) -> usize {
        ((self.end_time - self.start_time) * self.fps as f64).ceil() as usize
    }

    pub fn frame_duration(&self) -> f64 {
        1.0 / self.fps as f64
    }
}

/// Generador de animaciones
pub struct AnimationGenerator {
    pub config: AnimationConfig,
    pub frames: Vec<AnimationFrame>,
}

impl AnimationGenerator {
    pub fn new(config: AnimationConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
        }
    }

    /// Agrega un frame
    pub fn add_frame(&mut self, time: f64, svg_content: String) {
        self.frames.push(AnimationFrame { time, svg_content });
    }

    /// Genera SVG animado con SMIL
    pub fn to_animated_svg(&self) -> String {
        if self.frames.is_empty() {
            return String::new();
        }

        let width = self.config.view_config.width;
        let height = self.config.view_config.height;
        let bg = self.config.view_config.background.to_hex();
        let _duration = self.config.end_time - self.config.start_time;
        let repeat = if self.config.loop_animation { "indefinite" } else { "1" };

        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{}" height="{}"
     viewBox="0 0 {} {}">
  <rect width="100%" height="100%" fill="{}"/>
  <defs>
    <style>
      .frame {{ opacity: 0; }}
      .frame.active {{ opacity: 1; }}
    </style>
  </defs>
"#,
            width, height, width, height, bg
        );

        // Generar frames con animación de visibilidad
        for (i, frame) in self.frames.iter().enumerate() {
            let begin = frame.time - self.config.start_time;
            let frame_dur = self.config.frame_duration();

            svg.push_str(&format!(
                r#"  <g class="frame" id="frame-{}">
    {}
    <animate attributeName="opacity"
             values="0;1;1;0"
             keyTimes="0;0.01;0.99;1"
             dur="{}s"
             begin="{}s"
             repeatCount="{}"
             fill="freeze"/>
  </g>
"#,
                i,
                frame.svg_content,
                frame_dur,
                begin,
                repeat
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Exporta frames individuales
    pub fn export_frames(&self) -> Vec<(String, String)> {
        self.frames
            .iter()
            .enumerate()
            .map(|(i, frame)| {
                let filename = format!("frame_{:05}.svg", i);
                let content = self.frame_to_standalone_svg(frame);
                (filename, content)
            })
            .collect()
    }

    fn frame_to_standalone_svg(&self, frame: &AnimationFrame) -> String {
        let width = self.config.view_config.width;
        let height = self.config.view_config.height;
        let bg = self.config.view_config.background.to_hex();

        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{}" height="{}"
     viewBox="0 0 {} {}">
  <rect width="100%" height="100%" fill="{}"/>
  {}
</svg>"#,
            width, height, width, height, bg, frame.svg_content
        )
    }
}

/// Generador de rotación 3D
pub struct RotationAnimation {
    pub center: (f64, f64, f64),
    pub radius: f64,
    pub elevation: f64,
}

impl RotationAnimation {
    pub fn new(center: (f64, f64, f64), radius: f64, elevation: f64) -> Self {
        Self { center, radius, elevation }
    }

    /// Genera vista para un ángulo dado
    pub fn view_at_angle(&self, angle: f64) -> ViewType {
        ViewType::Custom3D {
            theta: self.elevation,
            phi: angle,
        }
    }

    /// Genera secuencia de vistas para rotación completa
    pub fn rotation_sequence(&self, n_frames: usize) -> Vec<ViewType> {
        (0..n_frames)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_frames as f64);
                self.view_at_angle(angle)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_config() {
        let config = AnimationConfig {
            start_time: 0.0,
            end_time: 2.0,
            fps: 30,
            ..Default::default()
        };
        assert_eq!(config.total_frames(), 60);
    }

    #[test]
    fn test_rotation_animation() {
        let rot = RotationAnimation::new((0.0, 0.0, 0.0), 10.0, 0.5);
        let views = rot.rotation_sequence(4);
        assert_eq!(views.len(), 4);
    }
}
