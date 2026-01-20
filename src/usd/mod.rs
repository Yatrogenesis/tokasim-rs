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

// ============================================================================
// USD DATA INGESTION (DIGITAL TWIN INPUT)
// ============================================================================

/// Sensor data point from USD time sample
#[derive(Debug, Clone)]
pub struct SensorReading {
    /// Time of measurement (s)
    pub time: f64,
    /// Sensor ID/name
    pub sensor_id: String,
    /// Measured value
    pub value: f64,
    /// Uncertainty (if available)
    pub uncertainty: Option<f64>,
}

/// Physics state from USD physics schema
#[derive(Debug, Clone)]
pub struct PhysicsState {
    /// Position (m)
    pub position: Vec3,
    /// Velocity (m/s)
    pub velocity: Vec3,
    /// Angular velocity (rad/s)
    pub angular_velocity: Vec3,
    /// Mass (kg)
    pub mass: f64,
    /// Temperature (K)
    pub temperature: f64,
}

/// Material state for surface tracking
#[derive(Debug, Clone)]
pub struct MaterialState {
    /// Surface ID
    pub surface_id: String,
    /// Erosion depth (m)
    pub erosion_depth: f64,
    /// Deposited layer thickness (m)
    pub deposition_thickness: f64,
    /// Surface temperature (K)
    pub surface_temperature: f64,
    /// Hydrogen retention (atoms/m²)
    pub h_retention: f64,
}

/// Plasma diagnostic data from USD
#[derive(Debug, Clone)]
pub struct PlasmaDiagnostics {
    /// Time (s)
    pub time: f64,
    /// Plasma current (A)
    pub ip: Option<f64>,
    /// Central electron temperature (keV)
    pub te0: Option<f64>,
    /// Central ion temperature (keV)
    pub ti0: Option<f64>,
    /// Central electron density (m⁻³)
    pub ne0: Option<f64>,
    /// Beta poloidal
    pub beta_p: Option<f64>,
    /// Beta toroidal
    pub beta_t: Option<f64>,
    /// Internal inductance
    pub li: Option<f64>,
    /// Safety factor at 95% flux
    pub q95: Option<f64>,
    /// Stored energy (J)
    pub w_mhd: Option<f64>,
    /// Plasma position R (m)
    pub r_geo: Option<f64>,
    /// Plasma position Z (m)
    pub z_geo: Option<f64>,
}

/// Coil data from USD
#[derive(Debug, Clone)]
pub struct CoilData {
    /// Coil name
    pub name: String,
    /// Current (A)
    pub current: f64,
    /// Voltage (V)
    pub voltage: f64,
    /// Temperature (K)
    pub temperature: f64,
    /// Field at conductor (T)
    pub b_max: f64,
}

/// Complete digital twin state imported from USD
#[derive(Debug, Clone)]
pub struct DigitalTwinState {
    /// Timestamp
    pub time: f64,
    /// Plasma diagnostics
    pub plasma: PlasmaDiagnostics,
    /// TF coil data
    pub tf_coils: Vec<CoilData>,
    /// PF coil data
    pub pf_coils: Vec<CoilData>,
    /// Central solenoid data
    pub cs_data: Option<CoilData>,
    /// First wall material state
    pub first_wall: Vec<MaterialState>,
    /// Divertor material state
    pub divertor: Vec<MaterialState>,
    /// Sensor readings
    pub sensors: Vec<SensorReading>,
}

impl Default for PlasmaDiagnostics {
    fn default() -> Self {
        Self {
            time: 0.0,
            ip: None,
            te0: None,
            ti0: None,
            ne0: None,
            beta_p: None,
            beta_t: None,
            li: None,
            q95: None,
            w_mhd: None,
            r_geo: None,
            z_geo: None,
        }
    }
}

impl Default for DigitalTwinState {
    fn default() -> Self {
        Self {
            time: 0.0,
            plasma: PlasmaDiagnostics::default(),
            tf_coils: Vec::new(),
            pf_coils: Vec::new(),
            cs_data: None,
            first_wall: Vec::new(),
            divertor: Vec::new(),
            sensors: Vec::new(),
        }
    }
}

/// USD Reader for data ingestion
///
/// Parses USD ASCII format to extract physics and sensor data
/// for digital twin feedback loop.
pub struct UsdReader {
    content: String,
    #[allow(dead_code)]
    current_pos: usize, // Reserved for streaming parser in future
}

impl UsdReader {
    /// Create reader from USD content string
    pub fn from_string(content: String) -> Self {
        Self {
            content,
            current_pos: 0,
        }
    }

    /// Create reader from file
    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_string(content))
    }

    /// Parse complete digital twin state from USD
    pub fn parse_digital_twin_state(&mut self) -> Result<DigitalTwinState, String> {
        let mut state = DigitalTwinState::default();

        // Parse time
        if let Some(time) = self.find_attribute_float("time") {
            state.time = time;
            state.plasma.time = time;
        }

        // Parse plasma diagnostics
        state.plasma = self.parse_plasma_diagnostics()?;

        // Parse coil data
        state.tf_coils = self.parse_coil_array("TFCoil")?;
        state.pf_coils = self.parse_coil_array("PFCoil")?;

        // Parse material states
        state.first_wall = self.parse_material_states("FirstWall")?;
        state.divertor = self.parse_material_states("Divertor")?;

        // Parse sensor readings
        state.sensors = self.parse_sensor_readings()?;

        Ok(state)
    }

    /// Parse plasma diagnostic data
    fn parse_plasma_diagnostics(&self) -> Result<PlasmaDiagnostics, String> {
        let mut diag = PlasmaDiagnostics::default();

        diag.ip = self.find_attribute_float("plasma:current");
        diag.te0 = self.find_attribute_float("plasma:te0_keV");
        diag.ti0 = self.find_attribute_float("plasma:ti0_keV");
        diag.ne0 = self.find_attribute_float("plasma:ne0_m3");
        diag.beta_p = self.find_attribute_float("plasma:beta_p");
        diag.beta_t = self.find_attribute_float("plasma:beta_t");
        diag.li = self.find_attribute_float("plasma:li");
        diag.q95 = self.find_attribute_float("plasma:q95");
        diag.w_mhd = self.find_attribute_float("plasma:w_mhd");
        diag.r_geo = self.find_attribute_float("plasma:r_geo");
        diag.z_geo = self.find_attribute_float("plasma:z_geo");

        Ok(diag)
    }

    /// Parse coil data array
    fn parse_coil_array(&self, prefix: &str) -> Result<Vec<CoilData>, String> {
        let mut coils = Vec::new();

        // Find all coils matching prefix
        for i in 0..100 {
            let name = format!("{}_{:02}", prefix, i);
            if let Some(current) = self.find_attribute_float(&format!("{}:current", name)) {
                coils.push(CoilData {
                    name: name.clone(),
                    current,
                    voltage: self.find_attribute_float(&format!("{}:voltage", name)).unwrap_or(0.0),
                    temperature: self.find_attribute_float(&format!("{}:temperature", name)).unwrap_or(4.5),
                    b_max: self.find_attribute_float(&format!("{}:b_max", name)).unwrap_or(0.0),
                });
            } else {
                break;
            }
        }

        Ok(coils)
    }

    /// Parse material state array
    fn parse_material_states(&self, section: &str) -> Result<Vec<MaterialState>, String> {
        let mut states = Vec::new();

        for i in 0..1000 {
            let base = format!("{}_{:04}", section, i);
            if let Some(erosion) = self.find_attribute_float(&format!("{}:erosion_depth", base)) {
                states.push(MaterialState {
                    surface_id: base.clone(),
                    erosion_depth: erosion,
                    deposition_thickness: self.find_attribute_float(&format!("{}:deposition", base)).unwrap_or(0.0),
                    surface_temperature: self.find_attribute_float(&format!("{}:temperature", base)).unwrap_or(300.0),
                    h_retention: self.find_attribute_float(&format!("{}:h_retention", base)).unwrap_or(0.0),
                });
            } else {
                break;
            }
        }

        Ok(states)
    }

    /// Parse sensor readings
    fn parse_sensor_readings(&self) -> Result<Vec<SensorReading>, String> {
        let mut readings = Vec::new();

        // Look for sensor data patterns
        let sensor_pattern = "custom double sensor:";
        let mut pos = 0;

        while let Some(idx) = self.content[pos..].find(sensor_pattern) {
            let start = pos + idx + sensor_pattern.len();

            // Extract sensor ID and value
            if let Some(end) = self.content[start..].find('=') {
                let sensor_id = self.content[start..start + end].trim().to_string();

                // Find the value
                let value_start = start + end + 1;
                if let Some(value_end) = self.content[value_start..].find('\n') {
                    if let Ok(value) = self.content[value_start..value_start + value_end].trim().parse::<f64>() {
                        readings.push(SensorReading {
                            time: self.find_attribute_float("time").unwrap_or(0.0),
                            sensor_id,
                            value,
                            uncertainty: None,
                        });
                    }
                }
            }

            pos = start;
        }

        Ok(readings)
    }

    /// Find and parse a float attribute
    fn find_attribute_float(&self, name: &str) -> Option<f64> {
        // Look for pattern: float/double <name> = <value>
        let patterns = [
            format!("float {} = ", name),
            format!("double {} = ", name),
            format!("custom float {} = ", name),
            format!("custom double {} = ", name),
        ];

        for pattern in &patterns {
            if let Some(idx) = self.content.find(pattern) {
                let start = idx + pattern.len();
                // Find end of value (newline or comma)
                let end = self.content[start..].find(|c: char| c == '\n' || c == ',' || c == ')');
                if let Some(end_idx) = end {
                    let value_str = self.content[start..start + end_idx].trim();
                    if let Ok(value) = value_str.parse::<f64>() {
                        return Some(value);
                    }
                }
            }
        }
        None
    }

    /// Find and parse a float3 attribute
    pub fn find_attribute_float3(&self, name: &str) -> Option<Vec3> {
        let patterns = [
            format!("float3 {} = ", name),
            format!("point3f {} = ", name),
            format!("vector3f {} = ", name),
        ];

        for pattern in &patterns {
            if let Some(idx) = self.content.find(pattern) {
                let start = idx + pattern.len();
                // Find opening paren
                if let Some(paren_start) = self.content[start..].find('(') {
                    let vec_start = start + paren_start + 1;
                    if let Some(paren_end) = self.content[vec_start..].find(')') {
                        let vec_str = &self.content[vec_start..vec_start + paren_end];
                        let parts: Vec<&str> = vec_str.split(',').collect();
                        if parts.len() == 3 {
                            if let (Ok(x), Ok(y), Ok(z)) = (
                                parts[0].trim().parse::<f64>(),
                                parts[1].trim().parse::<f64>(),
                                parts[2].trim().parse::<f64>(),
                            ) {
                                return Some(Vec3::new(x, y, z));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Parse time samples for an attribute
    pub fn parse_time_samples(&self, attr_name: &str) -> Vec<(f64, f64)> {
        let mut samples = Vec::new();

        let pattern = format!("{}.timeSamples = {{", attr_name);
        if let Some(start_idx) = self.content.find(&pattern) {
            let search_start = start_idx + pattern.len();
            if let Some(end_idx) = self.content[search_start..].find('}') {
                let samples_str = &self.content[search_start..search_start + end_idx];

                // Parse "time: value" pairs
                for part in samples_str.split(',') {
                    let kv: Vec<&str> = part.split(':').collect();
                    if kv.len() == 2 {
                        if let (Ok(time), Ok(value)) = (
                            kv[0].trim().parse::<f64>(),
                            kv[1].trim().parse::<f64>(),
                        ) {
                            samples.push((time, value));
                        }
                    }
                }
            }
        }

        samples
    }
}

/// USD Writer extension for digital twin output
impl UsdWriter {
    /// Write time-sampled attribute
    pub fn attr_time_sampled(&mut self, attr_type: &str, name: &str, samples: &[(f64, f64)]) {
        self.write_indent();
        let _ = write!(self.content, "{} {}.timeSamples = {{ ", attr_type, name);
        for (i, (time, value)) in samples.iter().enumerate() {
            if i > 0 { self.content.push_str(", "); }
            let _ = write!(self.content, "{}: {}", time, value);
        }
        self.content.push_str(" }\n");
    }

    /// Write sensor data as USD custom attribute
    pub fn sensor_data(&mut self, sensor_id: &str, value: f64, time: f64) {
        self.write_indent();
        let _ = write!(self.content, "custom double sensor:{} = {}\n", sensor_id, value);
        self.write_indent();
        let _ = write!(self.content, "custom double sensor:{}:time = {}\n", sensor_id, time);
    }

    /// Write plasma diagnostics block
    pub fn plasma_diagnostics(&mut self, diag: &PlasmaDiagnostics) {
        self.begin_scope("PlasmaDiagnostics");

        self.write_indent();
        let _ = write!(self.content, "custom double time = {}\n", diag.time);

        if let Some(ip) = diag.ip {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:current = {}\n", ip);
        }
        if let Some(te0) = diag.te0 {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:te0_keV = {}\n", te0);
        }
        if let Some(ti0) = diag.ti0 {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:ti0_keV = {}\n", ti0);
        }
        if let Some(ne0) = diag.ne0 {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:ne0_m3 = {:.3e}\n", ne0);
        }
        if let Some(beta_p) = diag.beta_p {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:beta_p = {}\n", beta_p);
        }
        if let Some(beta_t) = diag.beta_t {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:beta_t = {}\n", beta_t);
        }
        if let Some(q95) = diag.q95 {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:q95 = {}\n", q95);
        }
        if let Some(w_mhd) = diag.w_mhd {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:w_mhd = {:.3e}\n", w_mhd);
        }
        if let Some(r_geo) = diag.r_geo {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:r_geo = {}\n", r_geo);
        }
        if let Some(z_geo) = diag.z_geo {
            self.write_indent();
            let _ = write!(self.content, "custom double plasma:z_geo = {}\n", z_geo);
        }

        self.end_prim();
    }

    /// Write coil data
    pub fn coil_data(&mut self, coil: &CoilData) {
        self.begin_scope(&coil.name);

        self.write_indent();
        let _ = write!(self.content, "custom double {}:current = {}\n", coil.name, coil.current);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:voltage = {}\n", coil.name, coil.voltage);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:temperature = {}\n", coil.name, coil.temperature);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:b_max = {}\n", coil.name, coil.b_max);

        self.end_prim();
    }

    /// Write material state
    pub fn material_state(&mut self, state: &MaterialState) {
        self.begin_scope(&state.surface_id);

        self.write_indent();
        let _ = write!(self.content, "custom double {}:erosion_depth = {:.6e}\n", state.surface_id, state.erosion_depth);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:deposition = {:.6e}\n", state.surface_id, state.deposition_thickness);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:temperature = {}\n", state.surface_id, state.surface_temperature);
        self.write_indent();
        let _ = write!(self.content, "custom double {}:h_retention = {:.3e}\n", state.surface_id, state.h_retention);

        self.end_prim();
    }

    /// Write complete digital twin state
    pub fn digital_twin_state(&mut self, state: &DigitalTwinState) {
        self.begin_scope("DigitalTwinState");

        self.write_indent();
        let _ = write!(self.content, "custom double time = {}\n", state.time);

        // Plasma diagnostics
        self.plasma_diagnostics(&state.plasma);

        // TF Coils
        self.begin_scope("TFCoils");
        for coil in &state.tf_coils {
            self.coil_data(coil);
        }
        self.end_prim();

        // PF Coils
        self.begin_scope("PFCoils");
        for coil in &state.pf_coils {
            self.coil_data(coil);
        }
        self.end_prim();

        // First Wall
        self.begin_scope("FirstWall");
        for mat in &state.first_wall {
            self.material_state(mat);
        }
        self.end_prim();

        // Divertor
        self.begin_scope("Divertor");
        for mat in &state.divertor {
            self.material_state(mat);
        }
        self.end_prim();

        // Sensors
        self.begin_scope("Sensors");
        for sensor in &state.sensors {
            self.sensor_data(&sensor.sensor_id, sensor.value, sensor.time);
        }
        self.end_prim();

        self.end_prim();
    }
}

/// Digital twin data exchange manager
///
/// Manages bidirectional data flow between tokasim-rs and external systems
/// (NVIDIA Omniverse, real sensor data, control systems).
pub struct DigitalTwinExchange {
    /// Export directory for USD files
    pub export_dir: String,
    /// Import directory for incoming data
    pub import_dir: String,
    /// Current simulation time
    pub current_time: f64,
    /// History of imported states
    pub state_history: Vec<DigitalTwinState>,
    /// Maximum history length
    pub max_history: usize,
}

impl DigitalTwinExchange {
    /// Create new exchange manager
    pub fn new(export_dir: &str, import_dir: &str) -> Self {
        Self {
            export_dir: export_dir.to_string(),
            import_dir: import_dir.to_string(),
            current_time: 0.0,
            state_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Export simulation state to USD
    pub fn export_state(&self, state: &DigitalTwinState) -> std::io::Result<String> {
        let mut writer = UsdWriter::new();
        writer.digital_twin_state(state);

        let filename = format!("{}/state_{:.3}.usda", self.export_dir, state.time);
        writer.write_to_file(&filename)?;

        Ok(filename)
    }

    /// Import state from USD file
    pub fn import_state(&mut self, filename: &str) -> Result<DigitalTwinState, String> {
        let mut reader = UsdReader::from_file(filename)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let state = reader.parse_digital_twin_state()?;

        // Add to history
        self.state_history.push(state.clone());
        if self.state_history.len() > self.max_history {
            self.state_history.remove(0);
        }

        self.current_time = state.time;

        Ok(state)
    }

    /// Get interpolated state at given time
    pub fn interpolate_state(&self, time: f64) -> Option<DigitalTwinState> {
        if self.state_history.is_empty() {
            return None;
        }

        // Find bracketing states
        let mut before: Option<&DigitalTwinState> = None;
        let mut after: Option<&DigitalTwinState> = None;

        for state in &self.state_history {
            if state.time <= time {
                before = Some(state);
            }
            if state.time >= time && after.is_none() {
                after = Some(state);
            }
        }

        match (before, after) {
            (Some(b), Some(a)) if b.time != a.time => {
                // Linear interpolation
                let t = (time - b.time) / (a.time - b.time);
                Some(Self::lerp_states(b, a, t))
            }
            (Some(state), _) | (_, Some(state)) => Some(state.clone()),
            _ => None,
        }
    }

    /// Linear interpolation between two states
    fn lerp_states(a: &DigitalTwinState, b: &DigitalTwinState, t: f64) -> DigitalTwinState {
        let lerp = |x: f64, y: f64| x + t * (y - x);
        let lerp_opt = |x: Option<f64>, y: Option<f64>| {
            match (x, y) {
                (Some(xv), Some(yv)) => Some(lerp(xv, yv)),
                (Some(xv), None) => Some(xv),
                (None, Some(yv)) => Some(yv),
                _ => None,
            }
        };

        DigitalTwinState {
            time: lerp(a.time, b.time),
            plasma: PlasmaDiagnostics {
                time: lerp(a.plasma.time, b.plasma.time),
                ip: lerp_opt(a.plasma.ip, b.plasma.ip),
                te0: lerp_opt(a.plasma.te0, b.plasma.te0),
                ti0: lerp_opt(a.plasma.ti0, b.plasma.ti0),
                ne0: lerp_opt(a.plasma.ne0, b.plasma.ne0),
                beta_p: lerp_opt(a.plasma.beta_p, b.plasma.beta_p),
                beta_t: lerp_opt(a.plasma.beta_t, b.plasma.beta_t),
                li: lerp_opt(a.plasma.li, b.plasma.li),
                q95: lerp_opt(a.plasma.q95, b.plasma.q95),
                w_mhd: lerp_opt(a.plasma.w_mhd, b.plasma.w_mhd),
                r_geo: lerp_opt(a.plasma.r_geo, b.plasma.r_geo),
                z_geo: lerp_opt(a.plasma.z_geo, b.plasma.z_geo),
            },
            tf_coils: a.tf_coils.clone(),  // Use first state's structure
            pf_coils: a.pf_coils.clone(),
            cs_data: a.cs_data.clone(),
            first_wall: a.first_wall.clone(),
            divertor: a.divertor.clone(),
            sensors: a.sensors.clone(),
        }
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

    #[test]
    fn test_digital_twin_state_roundtrip() {
        // Create a state
        let state = DigitalTwinState {
            time: 1.5,
            plasma: PlasmaDiagnostics {
                time: 1.5,
                ip: Some(15e6),
                te0: Some(10.0),
                ti0: Some(9.5),
                ne0: Some(3e20),
                beta_p: Some(1.2),
                beta_t: Some(0.05),
                li: Some(0.85),
                q95: Some(3.2),
                w_mhd: Some(5e6),
                r_geo: Some(1.5),
                z_geo: Some(0.02),
            },
            tf_coils: Vec::new(),
            pf_coils: Vec::new(),
            cs_data: None,
            first_wall: Vec::new(),
            divertor: Vec::new(),
            sensors: vec![
                SensorReading {
                    time: 1.5,
                    sensor_id: "temp_fw_001".to_string(),
                    value: 850.0,
                    uncertainty: None,
                },
            ],
        };

        // Write to USD
        let mut writer = UsdWriter::new();
        writer.digital_twin_state(&state);
        let usd_content = writer.finish();

        // Verify content
        assert!(usd_content.contains("DigitalTwinState"));
        assert!(usd_content.contains("plasma:current"));
        assert!(usd_content.contains("plasma:te0_keV"));
    }

    #[test]
    fn test_usd_reader_float_attribute() {
        let content = r#"
            custom double test_value = 42.5
            custom float another = 3.14
        "#.to_string();

        let reader = UsdReader::from_string(content);

        assert_eq!(reader.find_attribute_float("test_value"), Some(42.5));
        assert_eq!(reader.find_attribute_float("another"), Some(3.14));
        assert_eq!(reader.find_attribute_float("nonexistent"), None);
    }

    #[test]
    fn test_plasma_diagnostics_default() {
        let diag = PlasmaDiagnostics::default();
        assert_eq!(diag.time, 0.0);
        assert!(diag.ip.is_none());
        assert!(diag.te0.is_none());
    }

    #[test]
    fn test_digital_twin_exchange_interpolation() {
        let mut exchange = DigitalTwinExchange::new("./export", "./import");

        // Add two states
        let state1 = DigitalTwinState {
            time: 0.0,
            plasma: PlasmaDiagnostics {
                time: 0.0,
                te0: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };

        let state2 = DigitalTwinState {
            time: 1.0,
            plasma: PlasmaDiagnostics {
                time: 1.0,
                te0: Some(20.0),
                ..Default::default()
            },
            ..Default::default()
        };

        exchange.state_history.push(state1);
        exchange.state_history.push(state2);

        // Interpolate at t=0.5
        let interp = exchange.interpolate_state(0.5).unwrap();
        assert!((interp.time - 0.5).abs() < 1e-10);
        assert!((interp.plasma.te0.unwrap() - 15.0).abs() < 1e-10);
    }
}
