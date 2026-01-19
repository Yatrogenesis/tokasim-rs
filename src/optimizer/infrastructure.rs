//! # Calculador de Infraestructura
//!
//! Dimensionamiento de edificios, equipos y terreno.

use crate::optimizer::design::ReactorDesign;

/// Especificación completa de infraestructura
#[derive(Debug, Clone)]
pub struct InfrastructureSpec {
    pub cryostat_radius: f64,
    pub cryostat_height: f64,
    pub building_diameter: f64,
    pub building_height: f64,
    pub hot_cell_area: f64,
    pub control_room_area: f64,
    pub cryo_plant_area: f64,
    pub electrical_area: f64,
    pub total_site_area: f64,
    pub crane_capacity_tons: f64,
}

/// Calculador de dimensiones de infraestructura
#[derive(Debug, Clone)]
pub struct InfrastructureCalculator {
    /// Factor de seguridad para espacios
    pub safety_margin: f64,
    /// Densidad de concreto (kg/m³)
    pub concrete_density: f64,
}

impl Default for InfrastructureCalculator {
    fn default() -> Self {
        Self {
            safety_margin: 1.3,
            concrete_density: 2400.0,
        }
    }
}

impl InfrastructureCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calcula todas las especificaciones de infraestructura
    pub fn calculate_all(&self, design: &ReactorDesign) -> InfrastructureSpec {
        InfrastructureSpec {
            cryostat_radius: self.cryostat_outer_radius(design),
            cryostat_height: self.cryostat_height(design),
            building_diameter: self.tokamak_building_diameter(design),
            building_height: self.tokamak_building_height(design),
            hot_cell_area: self.hot_cell_area(design),
            control_room_area: 200.0,
            cryo_plant_area: self.cryo_plant_area(design),
            electrical_area: self.electrical_area(design),
            total_site_area: self.total_site_area(design),
            crane_capacity_tons: self.crane_capacity(design),
        }
    }

    /// Radio exterior del criostato (m)
    pub fn cryostat_outer_radius(&self, design: &ReactorDesign) -> f64 {
        design.major_radius
            + design.minor_radius
            + design.blanket_thickness()
            + design.shield_thickness
            + design.tf_coil_radial_build
            + design.cryostat_margin
    }

    /// Altura del criostato (m)
    pub fn cryostat_height(&self, design: &ReactorDesign) -> f64 {
        2.0 * (design.minor_radius * design.elongation
            + design.blanket_thickness()
            + design.shield_thickness
            + design.tf_coil_radial_build
            + design.cryostat_margin)
    }

    /// Diámetro del edificio tokamak (m)
    pub fn tokamak_building_diameter(&self, design: &ReactorDesign) -> f64 {
        2.0 * self.cryostat_outer_radius(design) * self.safety_margin + 10.0
    }

    /// Altura del edificio tokamak (m)
    pub fn tokamak_building_height(&self, design: &ReactorDesign) -> f64 {
        self.cryostat_height(design) + design.crane_height + 10.0
    }

    /// Volumen total del edificio tokamak (m³)
    pub fn tokamak_building_volume(&self, design: &ReactorDesign) -> f64 {
        let diameter = self.tokamak_building_diameter(design);
        let height = self.tokamak_building_height(design);
        std::f64::consts::PI * (diameter / 2.0).powi(2) * height
    }

    /// Área de planta del edificio tokamak (m²)
    pub fn tokamak_building_footprint(&self, design: &ReactorDesign) -> f64 {
        let diameter = self.tokamak_building_diameter(design);
        std::f64::consts::PI * (diameter / 2.0).powi(2)
    }

    /// Área de hot cell (m²)
    pub fn hot_cell_area(&self, design: &ReactorDesign) -> f64 {
        // Hot cell debe poder albergar componentes más grandes
        let max_component_length = self.cryostat_height(design) / 2.0;
        max_component_length.powi(2) * 2.0
    }

    /// Área de planta criogénica (m²)
    pub fn cryo_plant_area(&self, design: &ReactorDesign) -> f64 {
        // Escala con masa de imanes
        let magnet_mass = design.major_radius * design.toroidal_field.powi(2) * 100.0;
        500.0 + magnet_mass * 0.5
    }

    /// Área eléctrica (m²)
    pub fn electrical_area(&self, design: &ReactorDesign) -> f64 {
        // Escala con potencia de calentamiento
        300.0 + design.total_heating_power() * 5.0
    }

    /// Capacidad de grúa requerida (toneladas)
    pub fn crane_capacity(&self, design: &ReactorDesign) -> f64 {
        // Componente más pesado típicamente es la bobina TF
        let tf_mass_per_coil = design.major_radius * design.toroidal_field * 50.0;
        tf_mass_per_coil * 1.5 // Factor de seguridad
    }

    /// Área total del sitio (m²)
    pub fn total_site_area(&self, design: &ReactorDesign) -> f64 {
        let tokamak = self.tokamak_building_footprint(design);
        let hot_cell = self.hot_cell_area(design);
        let cryo = self.cryo_plant_area(design);
        let electrical = self.electrical_area(design);
        let control = 200.0;
        let storage = 1000.0;
        let parking = 500.0;

        let buildings = tokamak + hot_cell + cryo + electrical + control + storage + parking;
        let roads = buildings * 0.2;
        let buffer = buildings * 0.5; // Zona de exclusión

        (buildings + roads) * self.safety_margin + buffer
    }

    /// Volumen total de todos los edificios (m³)
    pub fn total_building_volume(&self, design: &ReactorDesign) -> f64 {
        let tokamak = self.tokamak_building_volume(design);
        let hot_cell = self.hot_cell_area(design) * 15.0;
        let cryo = self.cryo_plant_area(design) * 8.0;
        let electrical = self.electrical_area(design) * 6.0;
        let control = 200.0 * 4.0;

        tokamak + hot_cell + cryo + electrical + control
    }

    /// Potencia eléctrica pico del sitio (MW)
    pub fn peak_electrical_load(&self, design: &ReactorDesign) -> f64 {
        let heating = design.total_heating_power();
        let magnets = design.toroidal_field * design.major_radius * 2.0;
        let cryo = magnets * 0.05;
        let auxiliaries = 20.0;

        heating + magnets + cryo + auxiliaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cryostat_dimensions() {
        let calc = InfrastructureCalculator::new();
        let design = ReactorDesign::default();

        let radius = calc.cryostat_outer_radius(&design);
        let height = calc.cryostat_height(&design);

        println!("Cryostat: R={:.2}m, H={:.2}m", radius, height);
        assert!(radius > design.major_radius);
        assert!(height > 2.0 * design.minor_radius);
    }

    #[test]
    fn test_site_area() {
        let calc = InfrastructureCalculator::new();
        let design = ReactorDesign::default();

        let area = calc.total_site_area(&design);
        println!("Site area: {:.0} m² ({:.2} hectares)", area, area / 10000.0);
        assert!(area > 0.0);
    }
}
