//! # Tokamak Component Visualization
//!
//! Detailed visualization of individual tokamak components with decomposition capability.

use super::*;
use std::fmt::Write;

/// Tokamak component identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokamakComponent {
    // Magnetic system
    ToroidalFieldCoil(u8),      // TF coil number (0-17 for 18 coils)
    PoloidalFieldCoil(u8),      // PF coil number
    CentralSolenoid,            // CS stack
    VerticalStabilityCoil,      // VS coils

    // Vacuum system
    VacuumVessel,
    VacuumVesselPort(u8),       // Port number
    Cryostat,

    // Plasma-facing components
    FirstWall,
    Blanket,
    DivertorInner,
    DivertorOuter,
    DivertorDome,

    // Support structures
    GravitySupport,
    TFCoilSupport,
    ThermalShield,

    // Plasma regions
    PlasmaCore,
    PlasmaEdge,
    Scrapeoff,
    PrivateFluxRegion,

    // Heating systems
    ICRFAntenna(u8),
    ECRHLauncher(u8),
    NBIPort(u8),
}

impl TokamakComponent {
    /// Get display name for component
    pub fn name(&self) -> &'static str {
        match self {
            Self::ToroidalFieldCoil(_) => "Toroidal Field Coil",
            Self::PoloidalFieldCoil(_) => "Poloidal Field Coil",
            Self::CentralSolenoid => "Central Solenoid",
            Self::VerticalStabilityCoil => "Vertical Stability Coil",
            Self::VacuumVessel => "Vacuum Vessel",
            Self::VacuumVesselPort(_) => "Vacuum Vessel Port",
            Self::Cryostat => "Cryostat",
            Self::FirstWall => "First Wall",
            Self::Blanket => "Blanket Module",
            Self::DivertorInner => "Inner Divertor",
            Self::DivertorOuter => "Outer Divertor",
            Self::DivertorDome => "Divertor Dome",
            Self::GravitySupport => "Gravity Support",
            Self::TFCoilSupport => "TF Coil Support",
            Self::ThermalShield => "Thermal Shield",
            Self::PlasmaCore => "Plasma Core",
            Self::PlasmaEdge => "Plasma Edge",
            Self::Scrapeoff => "Scrape-off Layer",
            Self::PrivateFluxRegion => "Private Flux Region",
            Self::ICRFAntenna(_) => "ICRF Antenna",
            Self::ECRHLauncher(_) => "ECRH Launcher",
            Self::NBIPort(_) => "NBI Port",
        }
    }

    /// Get component color
    pub fn color(&self) -> Color {
        match self {
            Self::ToroidalFieldCoil(_) => Color::TOROIDAL_COIL,
            Self::PoloidalFieldCoil(_) => Color::POLOIDAL_COIL,
            Self::CentralSolenoid => Color::rgb(150, 80, 40),
            Self::VerticalStabilityCoil => Color::rgb(200, 100, 50),
            Self::VacuumVessel => Color::VACUUM_VESSEL,
            Self::VacuumVesselPort(_) => Color::rgb(100, 100, 110),
            Self::Cryostat => Color::CRYOSTAT,
            Self::FirstWall => Color::FIRST_WALL,
            Self::Blanket => Color::rgb(80, 120, 80),
            Self::DivertorInner | Self::DivertorOuter | Self::DivertorDome => Color::DIVERTOR,
            Self::GravitySupport => Color::rgb(120, 120, 130),
            Self::TFCoilSupport => Color::rgb(100, 100, 110),
            Self::ThermalShield => Color::rgb(180, 180, 200),
            Self::PlasmaCore => Color::PLASMA_CORE,
            Self::PlasmaEdge => Color::PLASMA_EDGE,
            Self::Scrapeoff => Color::rgb(100, 150, 200),
            Self::PrivateFluxRegion => Color::rgb(150, 100, 150),
            Self::ICRFAntenna(_) => Color::rgb(200, 150, 50),
            Self::ECRHLauncher(_) => Color::rgb(150, 200, 50),
            Self::NBIPort(_) => Color::rgb(50, 150, 200),
        }
    }

    /// Get component description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ToroidalFieldCoil(_) => "Superconducting HTS REBCO magnet generating toroidal magnetic field (25T)",
            Self::PoloidalFieldCoil(_) => "Superconducting magnet for plasma shaping and position control",
            Self::CentralSolenoid => "Pulsed magnet stack for plasma current induction",
            Self::VerticalStabilityCoil => "Fast-response coils for vertical position control",
            Self::VacuumVessel => "Double-walled stainless steel vessel (10^-8 mbar)",
            Self::VacuumVesselPort(_) => "Access port for diagnostics and heating systems",
            Self::Cryostat => "Thermal insulation vessel maintaining 4K environment",
            Self::FirstWall => "Plasma-facing beryllium/tungsten armor tiles",
            Self::Blanket => "Tritium breeding and neutron shielding module",
            Self::DivertorInner => "Inner target for exhaust particle handling",
            Self::DivertorOuter => "Outer target for exhaust particle handling",
            Self::DivertorDome => "Dome structure protecting from radiation",
            Self::GravitySupport => "Load-bearing structure for tokamak weight",
            Self::TFCoilSupport => "Inter-coil structure maintaining magnet alignment",
            Self::ThermalShield => "80K shield reducing heat load to magnets",
            Self::PlasmaCore => "Hot core region (T > 10 keV, fusion active)",
            Self::PlasmaEdge => "Edge region (T ~ 1-5 keV, transport zone)",
            Self::Scrapeoff => "Open field line region outside separatrix",
            Self::PrivateFluxRegion => "Closed flux region near X-point",
            Self::ICRFAntenna(_) => "Ion Cyclotron Resonance Heating antenna (40-50 MHz)",
            Self::ECRHLauncher(_) => "Electron Cyclotron Resonance Heating launcher (170 GHz)",
            Self::NBIPort(_) => "Neutral Beam Injection port (1 MeV deuterium)",
        }
    }
}

/// Component geometry for rendering
pub struct ComponentGeometry {
    /// Component identifier
    pub component: TokamakComponent,
    /// Poloidal cross-section points (R, Z pairs in meters)
    pub poloidal_outline: Vec<(f64, f64)>,
    /// Is this component visible
    pub visible: bool,
    /// Exploded offset (for decomposition view)
    pub explode_offset: (f64, f64),
}

impl ComponentGeometry {
    pub fn new(component: TokamakComponent) -> Self {
        Self {
            component,
            poloidal_outline: Vec::new(),
            visible: true,
            explode_offset: (0.0, 0.0),
        }
    }

    /// Generate D-shaped outline
    pub fn d_shape(&mut self, r0: f64, a: f64, kappa: f64, delta: f64, n_points: usize) {
        self.poloidal_outline.clear();
        for i in 0..=n_points {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_points as f64);
            let r = r0 + a * (theta.cos() + delta * (2.0 * theta).cos());
            let z = a * kappa * theta.sin();
            self.poloidal_outline.push((r, z));
        }
    }

    /// Generate rectangular outline
    pub fn rectangle(&mut self, r_min: f64, r_max: f64, z_min: f64, z_max: f64) {
        self.poloidal_outline = vec![
            (r_min, z_min),
            (r_max, z_min),
            (r_max, z_max),
            (r_min, z_max),
            (r_min, z_min),
        ];
    }

    /// Generate TF coil D-shape
    pub fn tf_coil_shape(&mut self, r_inner: f64, r_outer: f64, z_max: f64, thickness: f64) {
        self.poloidal_outline.clear();
        let n_points = 30;

        // Outer edge (top half)
        for i in 0..=n_points {
            let t = i as f64 / n_points as f64;
            let angle = std::f64::consts::PI * t;
            let r = r_inner + (r_outer - r_inner) * (1.0 - angle.cos()) / 2.0;
            let z = z_max * angle.sin();
            self.poloidal_outline.push((r + thickness / 2.0, z));
        }

        // Outer edge (bottom half)
        for i in 0..=n_points {
            let t = i as f64 / n_points as f64;
            let angle = std::f64::consts::PI * (1.0 + t);
            let r = r_inner + (r_outer - r_inner) * (1.0 - angle.cos()) / 2.0;
            let z = z_max * angle.sin();
            self.poloidal_outline.push((r + thickness / 2.0, z));
        }

        // Inner edge (going back up)
        for i in (0..=n_points).rev() {
            let t = i as f64 / n_points as f64;
            let angle = std::f64::consts::PI * (1.0 + t);
            let r = r_inner + (r_outer - r_inner) * (1.0 - angle.cos()) / 2.0;
            let z = z_max * angle.sin();
            self.poloidal_outline.push((r - thickness / 2.0, z));
        }

        for i in (0..=n_points).rev() {
            let t = i as f64 / n_points as f64;
            let angle = std::f64::consts::PI * t;
            let r = r_inner + (r_outer - r_inner) * (1.0 - angle.cos()) / 2.0;
            let z = z_max * angle.sin();
            self.poloidal_outline.push((r - thickness / 2.0, z));
        }
    }
}

/// Complete tokamak assembly for visualization
pub struct TokamakAssembly {
    /// Major radius (m)
    pub r0: f64,
    /// Minor radius (m)
    pub a: f64,
    /// Elongation
    pub kappa: f64,
    /// Triangularity
    pub delta: f64,
    /// All component geometries
    pub components: Vec<ComponentGeometry>,
    /// Number of TF coils
    pub n_tf_coils: u8,
}

impl TokamakAssembly {
    /// Create TS-1 assembly
    pub fn ts1() -> Self {
        let mut assembly = Self {
            r0: 1.5,
            a: 0.6,
            kappa: 1.97,
            delta: 0.54,
            components: Vec::new(),
            n_tf_coils: 18,
        };
        assembly.build_components();
        assembly
    }

    /// Build all component geometries
    fn build_components(&mut self) {
        // Plasma core (rho < 0.5)
        let mut plasma_core = ComponentGeometry::new(TokamakComponent::PlasmaCore);
        plasma_core.d_shape(self.r0, self.a * 0.5, self.kappa, self.delta, 60);
        self.components.push(plasma_core);

        // Plasma edge (0.5 < rho < 1.0)
        let mut plasma_edge = ComponentGeometry::new(TokamakComponent::PlasmaEdge);
        plasma_edge.d_shape(self.r0, self.a, self.kappa, self.delta, 60);
        self.components.push(plasma_edge);

        // First wall (just outside plasma)
        let mut first_wall = ComponentGeometry::new(TokamakComponent::FirstWall);
        first_wall.d_shape(self.r0, self.a + 0.05, self.kappa, self.delta, 60);
        self.components.push(first_wall);

        // Vacuum vessel
        let mut vacuum_vessel = ComponentGeometry::new(TokamakComponent::VacuumVessel);
        vacuum_vessel.d_shape(self.r0, self.a + 0.15, self.kappa * 1.1, self.delta, 60);
        self.components.push(vacuum_vessel);

        // Inner divertor
        let mut div_inner = ComponentGeometry::new(TokamakComponent::DivertorInner);
        div_inner.rectangle(self.r0 - self.a - 0.1, self.r0 - self.a + 0.1,
                           -self.a * self.kappa - 0.2, -self.a * self.kappa);
        div_inner.explode_offset = (0.0, -0.3);
        self.components.push(div_inner);

        // Outer divertor
        let mut div_outer = ComponentGeometry::new(TokamakComponent::DivertorOuter);
        div_outer.rectangle(self.r0 + self.a - 0.1, self.r0 + self.a + 0.2,
                           -self.a * self.kappa - 0.2, -self.a * self.kappa);
        div_outer.explode_offset = (0.0, -0.3);
        self.components.push(div_outer);

        // TF coil (single representative)
        let mut tf_coil = ComponentGeometry::new(TokamakComponent::ToroidalFieldCoil(0));
        tf_coil.tf_coil_shape(0.3, self.r0 + self.a + 0.5, self.a * self.kappa + 0.6, 0.15);
        tf_coil.explode_offset = (0.5, 0.0);
        self.components.push(tf_coil);

        // PF coils
        let pf_positions = [
            (0.4, 1.5, 0.15, 0.1),   // PF1 - top inner
            (0.4, -1.5, 0.15, 0.1),  // PF2 - bottom inner
            (2.2, 1.2, 0.2, 0.12),   // PF3 - top outer
            (2.2, -1.2, 0.2, 0.12),  // PF4 - bottom outer
            (2.5, 0.5, 0.15, 0.1),   // PF5 - mid outer top
            (2.5, -0.5, 0.15, 0.1),  // PF6 - mid outer bottom
        ];

        for (i, &(r, z, w, h)) in pf_positions.iter().enumerate() {
            let mut pf_coil = ComponentGeometry::new(TokamakComponent::PoloidalFieldCoil(i as u8));
            pf_coil.rectangle(r - w/2.0, r + w/2.0, z - h/2.0, z + h/2.0);
            pf_coil.explode_offset = if z > 0.0 { (0.0, 0.4) } else { (0.0, -0.4) };
            self.components.push(pf_coil);
        }

        // Central solenoid
        let mut cs = ComponentGeometry::new(TokamakComponent::CentralSolenoid);
        cs.rectangle(0.15, 0.35, -1.4, 1.4);
        cs.explode_offset = (-0.3, 0.0);
        self.components.push(cs);

        // Cryostat
        let mut cryostat = ComponentGeometry::new(TokamakComponent::Cryostat);
        cryostat.rectangle(0.0, 3.0, -2.0, 2.0);
        cryostat.explode_offset = (0.8, 0.0);
        self.components.push(cryostat);
    }

    /// Render to SVG
    pub fn render_poloidal(&self, config: &VisConfig, explode_factor: f64) -> String {
        let mut renderer = SvgRenderer::new(config.width, config.height);
        renderer.set_scale(config.scale);
        renderer.set_offset(config.width as f64 * 0.35, config.height as f64 / 2.0);

        // Add gradients
        renderer.add_plasma_gradient("plasma_gradient");
        renderer.add_glow_filter("plasma_glow", Color::PLASMA_CORE);

        // Draw components from outermost to innermost
        let draw_order = [
            TokamakComponent::Cryostat,
            TokamakComponent::ToroidalFieldCoil(0),
            TokamakComponent::PoloidalFieldCoil(0),
            TokamakComponent::PoloidalFieldCoil(1),
            TokamakComponent::PoloidalFieldCoil(2),
            TokamakComponent::PoloidalFieldCoil(3),
            TokamakComponent::PoloidalFieldCoil(4),
            TokamakComponent::PoloidalFieldCoil(5),
            TokamakComponent::CentralSolenoid,
            TokamakComponent::VacuumVessel,
            TokamakComponent::FirstWall,
            TokamakComponent::DivertorInner,
            TokamakComponent::DivertorOuter,
            TokamakComponent::PlasmaEdge,
            TokamakComponent::PlasmaCore,
        ];

        for target in &draw_order {
            for comp in &self.components {
                let matches = match (target, &comp.component) {
                    (TokamakComponent::PoloidalFieldCoil(_), TokamakComponent::PoloidalFieldCoil(_)) => true,
                    (a, b) if std::mem::discriminant(a) == std::mem::discriminant(b) => true,
                    _ => false,
                };

                if matches && comp.visible && !comp.poloidal_outline.is_empty() {
                    let color = comp.component.color();
                    let offset_x = comp.explode_offset.0 * explode_factor;
                    let offset_y = comp.explode_offset.1 * explode_factor;

                    let mut path = String::new();
                    for (i, &(r, z)) in comp.poloidal_outline.iter().enumerate() {
                        let r_adj = r + offset_x;
                        let z_adj = z + offset_y;
                        let (sx, sy) = (
                            renderer.offset_x + r_adj * renderer.scale,
                            renderer.offset_y - z_adj * renderer.scale
                        );
                        if i == 0 {
                            write!(path, "M {:.2} {:.2}", sx, sy).unwrap();
                        } else {
                            write!(path, " L {:.2} {:.2}", sx, sy).unwrap();
                        }
                    }
                    path.push_str(" Z");

                    let fill = if matches!(comp.component, TokamakComponent::PlasmaCore) {
                        "url(#plasma_gradient)".to_string()
                    } else {
                        color.to_hex()
                    };

                    let filter = if matches!(comp.component, TokamakComponent::PlasmaCore | TokamakComponent::PlasmaEdge) {
                        Some("plasma_glow")
                    } else {
                        None
                    };

                    if let Some(f) = filter {
                        renderer.path_filtered(&path, &fill, &color.to_hex(), 1.0, f);
                    } else {
                        renderer.path(&path, &fill, "#000000", 0.5);
                    }

                    // Add label if enabled
                    if config.show_labels && explode_factor > 0.5 {
                        if let Some(&(r, z)) = comp.poloidal_outline.first() {
                            let r_adj = r + offset_x + 0.1;
                            let z_adj = z + offset_y;
                            renderer.text(r_adj, z_adj, comp.component.name(), 10.0, "#ffffff");
                        }
                    }
                }
            }
        }

        // Add title and legend
        renderer.text_anchored(-0.5, 1.8, "TOKASIM-RS: TS-1 Tokamak Cross-Section", 16.0, "#ffffff", "start");
        renderer.text_anchored(-0.5, 1.65, &format!("R₀ = {:.2} m, a = {:.2} m, κ = {:.2}, δ = {:.2}",
            self.r0, self.a, self.kappa, self.delta), 12.0, "#aaaaaa", "start");

        // Add scale bar
        let scale_len = 0.5;  // 0.5 m
        renderer.line(2.3, -1.7, 2.3 + scale_len, -1.7, "#ffffff", 2.0);
        renderer.line(2.3, -1.65, 2.3, -1.75, "#ffffff", 2.0);
        renderer.line(2.3 + scale_len, -1.65, 2.3 + scale_len, -1.75, "#ffffff", 2.0);
        renderer.text_anchored(2.3 + scale_len / 2.0, -1.8, "0.5 m", 10.0, "#ffffff", "middle");

        renderer.to_svg(config.background)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembly_creation() {
        let assembly = TokamakAssembly::ts1();
        assert_eq!(assembly.r0, 1.5);
        assert!(!assembly.components.is_empty());
    }

    #[test]
    fn test_component_names() {
        let comp = TokamakComponent::ToroidalFieldCoil(0);
        assert_eq!(comp.name(), "Toroidal Field Coil");
    }
}
