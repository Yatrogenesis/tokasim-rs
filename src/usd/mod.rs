//! # USD Exporter Module
//!
//! Universal Scene Description (USD) exporter for NVIDIA Omniverse integration.
//!
//! ## Overview
//!
//! USD is Pixar's 3D scene exchange format, adopted by NVIDIA for Omniverse.
//! This module exports tokamak geometry and simulation data to USD format for:
//! - Visualization in Omniverse
//! - Digital twin applications
//! - VR/AR training systems
//!
//! ## File Formats
//!
//! - `.usda` - Human-readable ASCII format
//! - `.usdc` - Binary compiled format (faster loading)
//! - `.usdz` - Compressed package with assets
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] Pixar USD Documentation: https://graphics.pixar.com/usd/docs/
//! [2] NVIDIA Omniverse USD: https://docs.omniverse.nvidia.com/

use std::fmt::Write;
use crate::types::Vec3;

/// USD file writer (ASCII format)
pub struct UsdWriter {
    content: String,
    indent_level: usize,
}

impl UsdWriter {
    /// Create new USD writer
    pub fn new() -> Self {
        let mut writer = Self {
            content: String::with_capacity(65536),
            indent_level: 0,
        };

        // USD file header
        writer.writeln("#usda 1.0");
        writer.writeln("(");
        writer.indent();
        writer.writeln("defaultPrim = \"Tokamak\"");
        writer.writeln("metersPerUnit = 1.0");
        writer.writeln("upAxis = \"Z\"");
        writer.dedent();
        writer.writeln(")");
        writer.writeln("");

        writer
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent_level {
            self.content.push_str("    ");
        }
    }

    fn writeln(&mut self, line: &str) {
        self.write_indent();
        self.content.push_str(line);
        self.content.push('\n');
    }

    /// Begin a prim definition (USD primitive/node)
    pub fn begin_prim(&mut self, prim_type: &str, name: &str) {
        self.write_indent();
        let _ = write!(self.content, "def {} \"{}\"", prim_type, name);
        self.content.push('\n');
        self.writeln("{");
        self.indent();
    }

    /// Begin a scope (group without transform)
    pub fn begin_scope(&mut self, name: &str) {
        self.begin_prim("Scope", name);
    }

    /// Begin Xform (transform node)
    pub fn begin_xform(&mut self, name: &str) {
        self.begin_prim("Xform", name);
    }

    /// End current prim
    pub fn end_prim(&mut self) {
        self.dedent();
        self.writeln("}");
        self.writeln("");
    }

    /// Write attribute
    pub fn attribute(&mut self, attr_type: &str, name: &str, value: &str) {
        self.write_indent();
        let _ = write!(self.content, "{} {} = {}", attr_type, name, value);
        self.content.push('\n');
    }

    /// Write float attribute
    pub fn attr_float(&mut self, name: &str, value: f64) {
        self.attribute("float", name, &format!("{}", value));
    }

    /// Write float3 attribute
    pub fn attr_float3(&mut self, name: &str, v: &Vec3) {
        self.attribute("float3", name, &format!("({}, {}, {})", v.x, v.y, v.z));
    }

    /// Write color attribute
    pub fn attr_color(&mut self, name: &str, r: f64, g: f64, b: f64) {
        self.attribute("color3f", name, &format!("({}, {}, {})", r, g, b));
    }

    /// Write transform
    pub fn transform(&mut self, translate: &Vec3, rotate: &Vec3, scale: &Vec3) {
        self.attr_float3("xformOp:translate", translate);
        self.attr_float3("xformOp:rotateXYZ", rotate);
        self.attr_float3("xformOp:scale", scale);
        self.attribute("uniform token[]", "xformOpOrder",
            "[\"xformOp:translate\", \"xformOp:rotateXYZ\", \"xformOp:scale\"]");
    }

    /// Add mesh prim
    pub fn mesh(&mut self, name: &str, points: &[Vec3], face_vertex_counts: &[i32], face_vertex_indices: &[i32]) {
        self.begin_prim("Mesh", name);

        // Points
        self.write_indent();
        self.content.push_str("point3f[] points = [");
        for (i, p) in points.iter().enumerate() {
            if i > 0 { self.content.push_str(", "); }
            let _ = write!(self.content, "({}, {}, {})", p.x, p.y, p.z);
        }
        self.content.push_str("]\n");

        // Face vertex counts
        self.write_indent();
        self.content.push_str("int[] faceVertexCounts = [");
        for (i, &c) in face_vertex_counts.iter().enumerate() {
            if i > 0 { self.content.push_str(", "); }
            let _ = write!(self.content, "{}", c);
        }
        self.content.push_str("]\n");

        // Face vertex indices
        self.write_indent();
        self.content.push_str("int[] faceVertexIndices = [");
        for (i, &idx) in face_vertex_indices.iter().enumerate() {
            if i > 0 { self.content.push_str(", "); }
            let _ = write!(self.content, "{}", idx);
        }
        self.content.push_str("]\n");

        self.end_prim();
    }

    /// Add cylinder prim (for coils)
    pub fn cylinder(&mut self, name: &str, radius: f64, height: f64) {
        self.begin_prim("Cylinder", name);
        self.attr_float("radius", radius);
        self.attr_float("height", height);
        self.attribute("token", "axis", "\"Z\"");
        self.end_prim();
    }

    /// Add torus prim (for plasma/vacuum vessel)
    pub fn torus(&mut self, name: &str, major_radius: f64, minor_radius: f64) {
        // USD doesn't have native torus, so we use a mesh approximation
        // Or reference a procedural asset
        self.begin_prim("Xform", name);

        // Custom attributes for torus parameters
        self.write_indent();
        self.content.push_str("custom double majorRadius = ");
        let _ = write!(self.content, "{}\n", major_radius);

        self.write_indent();
        self.content.push_str("custom double minorRadius = ");
        let _ = write!(self.content, "{}\n", minor_radius);

        // Add procedural torus mesh
        self.add_torus_mesh("TorusMesh", major_radius, minor_radius, 64, 32);

        self.end_prim();
    }

    /// Generate torus mesh
    fn add_torus_mesh(&mut self, name: &str, r_major: f64, r_minor: f64, n_major: usize, n_minor: usize) {
        use std::f64::consts::PI;

        let mut points = Vec::new();
        let mut face_counts = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices
        for i in 0..n_major {
            let theta = 2.0 * PI * i as f64 / n_major as f64;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            for j in 0..n_minor {
                let phi = 2.0 * PI * j as f64 / n_minor as f64;
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();

                let x = (r_major + r_minor * cos_phi) * cos_theta;
                let y = (r_major + r_minor * cos_phi) * sin_theta;
                let z = r_minor * sin_phi;

                points.push(Vec3::new(x, y, z));
            }
        }

        // Generate faces (quads)
        for i in 0..n_major {
            for j in 0..n_minor {
                let i_next = (i + 1) % n_major;
                let j_next = (j + 1) % n_minor;

                let v0 = (i * n_minor + j) as i32;
                let v1 = (i_next * n_minor + j) as i32;
                let v2 = (i_next * n_minor + j_next) as i32;
                let v3 = (i * n_minor + j_next) as i32;

                face_counts.push(4);
                indices.extend_from_slice(&[v0, v1, v2, v3]);
            }
        }

        self.mesh(name, &points, &face_counts, &indices);
    }

    /// Add material binding
    pub fn material(&mut self, name: &str, base_color: (f64, f64, f64), metallic: f64, roughness: f64) {
        self.begin_prim("Material", name);

        self.write_indent();
        self.content.push_str("token outputs:surface.connect = </Materials/");
        self.content.push_str(name);
        self.content.push_str("/PBRShader.outputs:surface>\n");

        // PBR Shader
        self.begin_prim("Shader", "PBRShader");
        self.attribute("uniform token", "info:id", "\"UsdPreviewSurface\"");
        self.attr_color("inputs:diffuseColor", base_color.0, base_color.1, base_color.2);
        self.attr_float("inputs:metallic", metallic);
        self.attr_float("inputs:roughness", roughness);
        self.attribute("token", "outputs:surface", "");
        self.end_prim();

        self.end_prim();
    }

    /// Get final USD content
    pub fn finish(self) -> String {
        self.content
    }

    /// Write to file
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, &self.content)
    }
}

impl Default for UsdWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokamak geometry exporter
pub struct TokamakExporter {
    writer: UsdWriter,
}

impl TokamakExporter {
    /// Create new exporter
    pub fn new() -> Self {
        Self {
            writer: UsdWriter::new(),
        }
    }

    /// Export complete tokamak assembly
    pub fn export_tokamak(
        &mut self,
        config: &TokamakGeometry,
    ) {
        // Root node
        self.writer.begin_xform("Tokamak");
        self.writer.attribute("kind", "kind", "\"assembly\"");

        // Materials
        self.export_materials();

        // Vacuum Vessel
        self.export_vacuum_vessel(config);

        // TF Coils
        self.export_tf_coils(config);

        // PF Coils
        self.export_pf_coils(config);

        // Central Solenoid
        self.export_central_solenoid(config);

        // Plasma (visualization)
        self.export_plasma(config);

        // Divertor
        self.export_divertor(config);

        self.writer.end_prim();
    }

    fn export_materials(&mut self) {
        self.writer.begin_scope("Materials");

        // Tungsten (gray metallic)
        self.writer.material("Tungsten", (0.5, 0.5, 0.5), 0.9, 0.3);

        // Steel (bluish gray)
        self.writer.material("Steel", (0.4, 0.45, 0.5), 0.8, 0.4);

        // Copper (orange)
        self.writer.material("Copper", (0.85, 0.5, 0.2), 0.95, 0.2);

        // Plasma (hot orange/white)
        self.writer.material("Plasma", (1.0, 0.7, 0.3), 0.0, 0.0);

        // REBCO (dark gray)
        self.writer.material("REBCO", (0.2, 0.2, 0.2), 0.7, 0.5);

        self.writer.end_prim();
    }

    fn export_vacuum_vessel(&mut self, config: &TokamakGeometry) {
        self.writer.begin_xform("VacuumVessel");

        // Main vessel (torus)
        self.writer.torus(
            "OuterShell",
            config.major_radius,
            config.minor_radius + config.wall_thickness,
        );

        self.writer.end_prim();
    }

    fn export_tf_coils(&mut self, config: &TokamakGeometry) {
        use std::f64::consts::PI;

        self.writer.begin_scope("TFCoils");

        for i in 0..config.n_tf_coils {
            let angle = 2.0 * PI * i as f64 / config.n_tf_coils as f64;
            let angle_deg = angle.to_degrees();

            let coil_name = format!("TFCoil_{:02}", i + 1);
            self.writer.begin_xform(&coil_name);

            // Position at R=0, rotate around Z
            self.writer.transform(
                &Vec3::zero(),
                &Vec3::new(0.0, 0.0, angle_deg),
                &Vec3::new(1.0, 1.0, 1.0),
            );

            // D-shaped coil approximation (simplified as cylinder for now)
            // In full implementation, would use actual D-shape mesh
            self.writer.cylinder(
                "Winding",
                config.tf_coil_width / 2.0,
                config.tf_coil_height,
            );

            self.writer.end_prim();
        }

        self.writer.end_prim();
    }

    fn export_pf_coils(&mut self, config: &TokamakGeometry) {
        self.writer.begin_scope("PFCoils");

        for (i, coil) in config.pf_coils.iter().enumerate() {
            let coil_name = format!("PFCoil_{}", i + 1);
            self.writer.begin_xform(&coil_name);

            self.writer.transform(
                &Vec3::new(coil.r, 0.0, coil.z),
                &Vec3::zero(),
                &Vec3::new(1.0, 1.0, 1.0),
            );

            // Torus for each PF coil
            self.writer.torus("Winding", coil.r, coil.dr / 2.0);

            self.writer.end_prim();
        }

        self.writer.end_prim();
    }

    fn export_central_solenoid(&mut self, config: &TokamakGeometry) {
        self.writer.begin_xform("CentralSolenoid");

        self.writer.cylinder(
            "Solenoid",
            config.cs_inner_radius,
            config.cs_height,
        );

        self.writer.end_prim();
    }

    fn export_plasma(&mut self, config: &TokamakGeometry) {
        self.writer.begin_xform("Plasma");

        // Plasma as a torus (D-shaped would need custom mesh)
        self.writer.torus(
            "PlasmaVolume",
            config.major_radius,
            config.minor_radius * config.elongation,
        );

        // Add plasma properties as custom attributes
        self.writer.write_indent();
        let _ = write!(self.writer.content, "custom double temperature_keV = {}\n", config.temperature_kev);

        self.writer.write_indent();
        let _ = write!(self.writer.content, "custom double density_m3 = {:.3e}\n", config.density);

        self.writer.end_prim();
    }

    fn export_divertor(&mut self, config: &TokamakGeometry) {
        use std::f64::consts::PI;

        self.writer.begin_scope("Divertor");

        // Simplified divertor as a ring at bottom
        for i in 0..config.n_divertor_cassettes {
            let angle = 2.0 * PI * i as f64 / config.n_divertor_cassettes as f64;
            let angle_deg = angle.to_degrees();

            let cassette_name = format!("Cassette_{:02}", i + 1);
            self.writer.begin_xform(&cassette_name);

            self.writer.transform(
                &Vec3::new(0.0, 0.0, -config.minor_radius * 0.8),
                &Vec3::new(0.0, 0.0, angle_deg),
                &Vec3::new(1.0, 1.0, 1.0),
            );

            // Simplified cassette geometry
            self.writer.cylinder(
                "Target",
                config.major_radius * 0.1,
                config.minor_radius * 0.3,
            );

            self.writer.end_prim();
        }

        self.writer.end_prim();
    }

    /// Get final USD content
    pub fn finish(self) -> String {
        self.writer.finish()
    }

    /// Write to file
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        self.writer.write_to_file(path)
    }
}

impl Default for TokamakExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokamak geometry specification for export
#[derive(Debug, Clone)]
pub struct TokamakGeometry {
    /// Major radius R₀ (m)
    pub major_radius: f64,
    /// Minor radius a (m)
    pub minor_radius: f64,
    /// Plasma elongation κ
    pub elongation: f64,
    /// Plasma triangularity δ
    pub triangularity: f64,
    /// Wall thickness (m)
    pub wall_thickness: f64,
    /// Number of TF coils
    pub n_tf_coils: usize,
    /// TF coil width (m)
    pub tf_coil_width: f64,
    /// TF coil height (m)
    pub tf_coil_height: f64,
    /// PF coil positions
    pub pf_coils: Vec<PFCoilGeometry>,
    /// Central solenoid inner radius (m)
    pub cs_inner_radius: f64,
    /// Central solenoid height (m)
    pub cs_height: f64,
    /// Number of divertor cassettes
    pub n_divertor_cassettes: usize,
    /// Plasma temperature (keV)
    pub temperature_kev: f64,
    /// Plasma density (m⁻³)
    pub density: f64,
}

/// PF coil geometry
#[derive(Debug, Clone)]
pub struct PFCoilGeometry {
    /// Radial position (m)
    pub r: f64,
    /// Vertical position (m)
    pub z: f64,
    /// Radial extent (m)
    pub dr: f64,
    /// Vertical extent (m)
    pub dz: f64,
}

impl TokamakGeometry {
    /// Create geometry for TS-1 design
    pub fn ts1() -> Self {
        Self {
            major_radius: 1.5,
            minor_radius: 0.6,
            elongation: 1.97,
            triangularity: 0.54,
            wall_thickness: 0.05,
            n_tf_coils: 18,
            tf_coil_width: 0.3,
            tf_coil_height: 2.5,
            pf_coils: vec![
                PFCoilGeometry { r: 0.5, z: 1.5, dr: 0.2, dz: 0.3 },
                PFCoilGeometry { r: 0.5, z: -1.5, dr: 0.2, dz: 0.3 },
                PFCoilGeometry { r: 2.0, z: 1.2, dr: 0.3, dz: 0.2 },
                PFCoilGeometry { r: 2.0, z: -1.2, dr: 0.3, dz: 0.2 },
                PFCoilGeometry { r: 2.5, z: 0.5, dr: 0.25, dz: 0.25 },
                PFCoilGeometry { r: 2.5, z: -0.5, dr: 0.25, dz: 0.25 },
            ],
            cs_inner_radius: 0.3,
            cs_height: 3.0,
            n_divertor_cassettes: 54,
            temperature_kev: 15.0,
            density: 3e20,
        }
    }

    /// Create geometry for SPARC-class device
    pub fn sparc() -> Self {
        Self {
            major_radius: 1.85,
            minor_radius: 0.57,
            elongation: 1.8,
            triangularity: 0.4,
            wall_thickness: 0.03,
            n_tf_coils: 18,
            tf_coil_width: 0.25,
            tf_coil_height: 2.0,
            pf_coils: vec![
                PFCoilGeometry { r: 0.4, z: 1.2, dr: 0.15, dz: 0.25 },
                PFCoilGeometry { r: 0.4, z: -1.2, dr: 0.15, dz: 0.25 },
                PFCoilGeometry { r: 2.2, z: 1.0, dr: 0.2, dz: 0.15 },
                PFCoilGeometry { r: 2.2, z: -1.0, dr: 0.2, dz: 0.15 },
            ],
            cs_inner_radius: 0.25,
            cs_height: 2.4,
            n_divertor_cassettes: 32,
            temperature_kev: 10.0,
            density: 1.8e20,
        }
    }
}

/// Time-varying data exporter for simulation playback
pub struct TimeVaryingExporter {
    base_writer: UsdWriter,
    time_samples: Vec<f64>,
}

impl TimeVaryingExporter {
    /// Create new time-varying exporter
    pub fn new() -> Self {
        let mut writer = UsdWriter::new();

        // Add time metadata
        writer.writeln("(");
        writer.indent();
        writer.writeln("startTimeCode = 0");
        writer.writeln("endTimeCode = 100");
        writer.writeln("framesPerSecond = 24");
        writer.dedent();
        writer.writeln(")");

        Self {
            base_writer: writer,
            time_samples: Vec::new(),
        }
    }

    /// Add time sample
    pub fn add_time_sample(&mut self, time: f64) {
        self.time_samples.push(time);
    }

    /// Export plasma parameters over time
    pub fn export_plasma_evolution(
        &mut self,
        _times: &[f64],
        _temperatures: &[f64],
        _densities: &[f64],
        _positions: &[(f64, f64)],  // (R, Z) of plasma center
    ) {
        // Time-sampled attributes would be written here
        // Format: float temperature.timeSamples = { 0: 10.0, 1: 10.5, 2: 11.0 }

        self.base_writer.begin_xform("AnimatedPlasma");
        // ... time-sampled attributes
        self.base_writer.end_prim();
    }
}

impl Default for TimeVaryingExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usd_writer_basic() {
        let mut writer = UsdWriter::new();

        writer.begin_xform("TestObject");
        writer.attr_float("radius", 1.5);
        writer.end_prim();

        let content = writer.finish();
        assert!(content.contains("TestObject"));
        assert!(content.contains("radius"));
    }

    #[test]
    fn test_tokamak_export() {
        let mut exporter = TokamakExporter::new();
        let geometry = TokamakGeometry::ts1();

        exporter.export_tokamak(&geometry);
        let usd = exporter.finish();

        // Check that major components are present
        assert!(usd.contains("Tokamak"));
        assert!(usd.contains("VacuumVessel"));
        assert!(usd.contains("TFCoils"));
        assert!(usd.contains("PFCoils"));
        assert!(usd.contains("Plasma"));

        // Should be valid USD syntax
        assert!(usd.contains("#usda 1.0"));
    }

    #[test]
    fn test_torus_mesh() {
        let mut writer = UsdWriter::new();
        writer.torus("TestTorus", 1.0, 0.3);
        let content = writer.finish();

        // Should have mesh data
        assert!(content.contains("points"));
        assert!(content.contains("faceVertexCounts"));
    }
}
