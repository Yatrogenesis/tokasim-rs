//! # SVG Renderer
//!
//! High-quality SVG generation for tokamak visualization.

use super::*;
use std::fmt::Write;

/// SVG document builder
pub struct SvgRenderer {
    pub width: u32,
    pub height: u32,
    pub content: String,
    pub defs: String,
    pub scale: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl SvgRenderer {
    /// Create new SVG renderer
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            content: String::with_capacity(100_000),
            defs: String::new(),
            scale: 100.0,
            offset_x: width as f64 / 2.0,
            offset_y: height as f64 / 2.0,
        }
    }

    /// Set scale (pixels per meter)
    pub fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }

    /// Set center offset
    pub fn set_offset(&mut self, x: f64, y: f64) {
        self.offset_x = x;
        self.offset_y = y;
    }

    /// Convert world coordinates to screen coordinates
    fn world_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.offset_x + x * self.scale,
            self.offset_y - y * self.scale,  // Y flipped for SVG
        )
    }

    /// Add gradient definition for plasma
    pub fn add_plasma_gradient(&mut self, id: &str) {
        write!(
            self.defs,
            r#"<radialGradient id="{}" cx="50%" cy="50%" r="50%">
                <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.9"/>
                <stop offset="20%" style="stop-color:#ff6432;stop-opacity:0.8"/>
                <stop offset="50%" style="stop-color:#ff9632;stop-opacity:0.6"/>
                <stop offset="80%" style="stop-color:#ffdc64;stop-opacity:0.3"/>
                <stop offset="100%" style="stop-color:#6496ff;stop-opacity:0.1"/>
            </radialGradient>"#,
            id
        ).unwrap();
    }

    /// Add gradient for magnetic field
    pub fn add_field_gradient(&mut self, id: &str) {
        write!(
            self.defs,
            r#"<linearGradient id="{}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#3296ff;stop-opacity:0.8"/>
                <stop offset="50%" style="stop-color:#64c8ff;stop-opacity:0.5"/>
                <stop offset="100%" style="stop-color:#3296ff;stop-opacity:0.8"/>
            </linearGradient>"#,
            id
        ).unwrap();
    }

    /// Add metal gradient for coils
    pub fn add_metal_gradient(&mut self, id: &str, color: Color) {
        write!(
            self.defs,
            r#"<linearGradient id="{}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:rgb({},{},{});stop-opacity:1"/>
                <stop offset="50%" style="stop-color:rgb({},{},{});stop-opacity:1"/>
                <stop offset="100%" style="stop-color:rgb({},{},{});stop-opacity:1"/>
            </linearGradient>"#,
            id,
            (color.r as f32 * 1.3).min(255.0) as u8,
            (color.g as f32 * 1.3).min(255.0) as u8,
            (color.b as f32 * 1.3).min(255.0) as u8,
            color.r, color.g, color.b,
            (color.r as f32 * 0.7) as u8,
            (color.g as f32 * 0.7) as u8,
            (color.b as f32 * 0.7) as u8,
        ).unwrap();
    }

    /// Add glow filter
    pub fn add_glow_filter(&mut self, id: &str, color: Color) {
        write!(
            self.defs,
            r#"<filter id="{}" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
                <feFlood flood-color="{}" result="color"/>
                <feComposite in="color" in2="blur" operator="in" result="glow"/>
                <feMerge>
                    <feMergeNode in="glow"/>
                    <feMergeNode in="glow"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>"#,
            id, color.to_hex()
        ).unwrap();
    }

    /// Draw circle
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str, stroke_width: f64) {
        let (sx, sy) = self.world_to_screen(cx, cy);
        let sr = r * self.scale;
        write!(
            self.content,
            r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            sx, sy, sr, fill, stroke, stroke_width
        ).unwrap();
    }

    /// Draw ellipse
    pub fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, fill: &str, stroke: &str, stroke_width: f64) {
        let (sx, sy) = self.world_to_screen(cx, cy);
        let srx = rx * self.scale;
        let sry = ry * self.scale;
        write!(
            self.content,
            r#"<ellipse cx="{:.2}" cy="{:.2}" rx="{:.2}" ry="{:.2}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            sx, sy, srx, sry, fill, stroke, stroke_width
        ).unwrap();
    }

    /// Draw rectangle
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, stroke: &str, stroke_width: f64) {
        let (sx, sy) = self.world_to_screen(x, y + h);
        let sw = w * self.scale;
        let sh = h * self.scale;
        write!(
            self.content,
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            sx, sy, sw, sh, fill, stroke, stroke_width
        ).unwrap();
    }

    /// Draw line
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) {
        let (sx1, sy1) = self.world_to_screen(x1, y1);
        let (sx2, sy2) = self.world_to_screen(x2, y2);
        write!(
            self.content,
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="{:.1}"/>"#,
            sx1, sy1, sx2, sy2, stroke, stroke_width
        ).unwrap();
    }

    /// Draw path (SVG path data)
    pub fn path(&mut self, d: &str, fill: &str, stroke: &str, stroke_width: f64) {
        write!(
            self.content,
            r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            d, fill, stroke, stroke_width
        ).unwrap();
    }

    /// Draw path with filter
    pub fn path_filtered(&mut self, d: &str, fill: &str, stroke: &str, stroke_width: f64, filter: &str) {
        write!(
            self.content,
            r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{:.1}" filter="url(#{})"/>"#,
            d, fill, stroke, stroke_width, filter
        ).unwrap();
    }

    /// Draw D-shaped tokamak cross-section
    pub fn d_shape(&mut self, r0: f64, a: f64, kappa: f64, delta: f64, fill: &str, stroke: &str, stroke_width: f64) {
        let n_points = 100;
        let mut path = String::new();

        for i in 0..=n_points {
            let theta = 2.0 * PI * (i as f64) / (n_points as f64);
            let r = r0 + a * (theta.cos() + delta * (2.0 * theta).cos());
            let z = a * kappa * theta.sin();

            let (sx, sy) = self.world_to_screen(r, z);
            if i == 0 {
                write!(path, "M {:.2} {:.2}", sx, sy).unwrap();
            } else {
                write!(path, " L {:.2} {:.2}", sx, sy).unwrap();
            }
        }
        path.push_str(" Z");

        write!(
            self.content,
            r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            path, fill, stroke, stroke_width
        ).unwrap();
    }

    /// Draw text
    pub fn text(&mut self, x: f64, y: f64, content: &str, font_size: f64, fill: &str) {
        let (sx, sy) = self.world_to_screen(x, y);
        write!(
            self.content,
            r#"<text x="{:.2}" y="{:.2}" font-family="monospace" font-size="{:.1}" fill="{}">{}</text>"#,
            sx, sy, font_size, fill, content
        ).unwrap();
    }

    /// Draw text with anchor
    pub fn text_anchored(&mut self, x: f64, y: f64, content: &str, font_size: f64, fill: &str, anchor: &str) {
        let (sx, sy) = self.world_to_screen(x, y);
        write!(
            self.content,
            r#"<text x="{:.2}" y="{:.2}" font-family="monospace" font-size="{:.1}" fill="{}" text-anchor="{}">{}</text>"#,
            sx, sy, font_size, fill, anchor, content
        ).unwrap();
    }

    /// Start a group with optional transform
    pub fn group_start(&mut self, id: Option<&str>, class: Option<&str>, transform: Option<&str>) {
        write!(self.content, "<g").unwrap();
        if let Some(id) = id {
            write!(self.content, r#" id="{}""#, id).unwrap();
        }
        if let Some(class) = class {
            write!(self.content, r#" class="{}""#, class).unwrap();
        }
        if let Some(transform) = transform {
            write!(self.content, r#" transform="{}""#, transform).unwrap();
        }
        write!(self.content, ">").unwrap();
    }

    /// End a group
    pub fn group_end(&mut self) {
        write!(self.content, "</g>").unwrap();
    }

    /// Add raw SVG content
    pub fn raw(&mut self, content: &str) {
        self.content.push_str(content);
    }

    /// Generate final SVG string
    pub fn to_svg(&self, background: Color) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {} {}"
     width="{}" height="{}">
<rect width="100%" height="100%" fill="{}"/>
<defs>
{}
</defs>
{}
</svg>"#,
            self.width, self.height,
            self.width, self.height,
            background.to_hex(),
            self.defs,
            self.content
        )
    }

    /// Save to file
    pub fn save(&self, path: &str, background: Color) -> std::io::Result<()> {
        std::fs::write(path, self.to_svg(background))
    }
}

/// Draw tokamak flux surfaces
pub fn draw_flux_surfaces(renderer: &mut SvgRenderer, r0: f64, a: f64, kappa: f64, delta: f64, n_surfaces: usize) {
    for i in 1..=n_surfaces {
        let rho = i as f64 / n_surfaces as f64;
        let color = temperature_to_color(15.0 * (1.0 - rho * rho));  // Parabolic profile
        let opacity = 0.3 + 0.4 * (1.0 - rho);

        let n_points = 60;
        let mut path = String::new();

        for j in 0..=n_points {
            let theta = 2.0 * PI * (j as f64) / (n_points as f64);
            let r = r0 + rho * a * (theta.cos() + delta * (2.0 * theta).cos());
            let z = rho * a * kappa * theta.sin();

            let (sx, sy) = (renderer.offset_x + r * renderer.scale, renderer.offset_y - z * renderer.scale);
            if j == 0 {
                write!(path, "M {:.2} {:.2}", sx, sy).unwrap();
            } else {
                write!(path, " L {:.2} {:.2}", sx, sy).unwrap();
            }
        }
        path.push_str(" Z");

        write!(
            renderer.content,
            r#"<path d="{}" fill="{}" fill-opacity="{:.2}" stroke="none"/>"#,
            path, color.to_hex(), opacity
        ).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = SvgRenderer::new(800, 600);
        assert_eq!(renderer.width, 800);
        assert_eq!(renderer.height, 600);
    }

    #[test]
    fn test_svg_generation() {
        let mut renderer = SvgRenderer::new(400, 400);
        renderer.circle(0.0, 0.0, 1.0, "red", "black", 2.0);
        let svg = renderer.to_svg(Color::rgb(255, 255, 255));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("fill=\"red\""));
    }
}
