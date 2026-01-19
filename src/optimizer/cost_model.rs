//! # Modelo de Costos Paramétrico
//!
//! Estimación de CAPEX, OPEX y LCOE para reactores de fusión.
//! Basado en escalamiento de ITER y estudios de plantas comerciales.

use crate::optimizer::design::ReactorDesign;
use crate::optimizer::scaling_laws::ScalingLaws;

/// Modelo de costos para reactores de fusión
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Año base para costos (2024 USD)
    pub base_year: u32,
    /// Tasa de descuento para LCOE
    pub discount_rate: f64,
    /// Vida útil de la planta (años)
    pub plant_lifetime: u32,
    /// Factor de capacidad
    pub capacity_factor: f64,
    /// Costo base de referencia (ITER ~ $25B)
    pub reference_cost: f64,
    /// Radio mayor de referencia (ITER = 6.2m)
    pub reference_radius: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            base_year: 2024,
            discount_rate: 0.08,
            plant_lifetime: 40,
            capacity_factor: 0.75,
            reference_cost: 25e9, // $25 billion (ITER)
            reference_radius: 6.2, // meters
        }
    }
}

impl CostModel {
    /// Crea modelo basado en escalamiento de ITER
    pub fn iter_scaling() -> Self {
        Self::default()
    }

    /// Estima costo de capital (CAPEX) en USD
    pub fn estimate_capex(&self, design: &ReactorDesign) -> f64 {
        // Escalamiento aproximado: CAPEX ~ R^2.5 * B^0.8
        let r_ratio = design.major_radius / self.reference_radius;
        let b_ratio = design.toroidal_field / 5.3; // ITER B = 5.3T

        let scale_factor = r_ratio.powf(2.5) * b_ratio.powf(0.8);

        // Ajuste por tecnología de imanes
        let magnet_factor = design.magnet_technology.relative_cost();

        // Costo base escalado
        let base_cost = self.reference_cost * scale_factor * (magnet_factor / 2.5);

        // Componentes adicionales
        let heating_cost = self.heating_system_cost(design);
        let blanket_cost = self.blanket_cost(design);
        let building_cost = self.building_cost(design);
        let electrical_cost = self.electrical_cost(design);

        base_cost + heating_cost + blanket_cost + building_cost + electrical_cost
    }

    /// Costo del sistema de calentamiento
    fn heating_system_cost(&self, design: &ReactorDesign) -> f64 {
        // ~$5M/MW para ICRF, ~$8M/MW para NBI, ~$6M/MW para ECRH
        design.icrf_power_mw * 5e6 + design.nbi_power_mw * 8e6 + design.ecrh_power_mw * 6e6
    }

    /// Costo del blanket
    fn blanket_cost(&self, design: &ReactorDesign) -> f64 {
        // ~$500/kg, masa escala con superficie
        let mass_tons = design.plasma_surface() * design.blanket_thickness() * 5000.0 / 1000.0;
        mass_tons * 500_000.0
    }

    /// Costo del edificio
    fn building_cost(&self, design: &ReactorDesign) -> f64 {
        // ~$5000/m³ para edificio nuclear
        let volume = (design.major_radius * 4.0).powi(2) * design.major_radius * 6.0;
        volume * 5000.0
    }

    /// Costo del sistema eléctrico
    fn electrical_cost(&self, design: &ReactorDesign) -> f64 {
        // ~$200/kW de capacidad
        design.total_heating_power() * 1000.0 * 200.0
    }

    /// Estima costo operativo anual (OPEX) en USD/año
    pub fn estimate_opex(&self, design: &ReactorDesign) -> f64 {
        let capex = self.estimate_capex(design);

        // OPEX típicamente 2-4% de CAPEX para plantas nucleares
        let fixed_opex = capex * 0.025;

        // Costo de combustible (tritio + deuterio)
        let p_fusion = ScalingLaws::fusion_power_mw(design);
        let tritium_consumption = p_fusion * 0.055 / 17.6; // kg/year at full power
        let tritium_cost = tritium_consumption * 30_000.0 * 365.0 * 24.0 * self.capacity_factor;

        // Personal (~500 personas * $150k/año)
        let labor_cost = 500.0 * 150_000.0;

        // Mantenimiento y reemplazo de componentes
        let maintenance = capex * 0.01;

        fixed_opex + tritium_cost + labor_cost + maintenance
    }

    /// Calcula LCOE (Levelized Cost of Electricity) en $/MWh
    pub fn calculate_lcoe(&self, design: &ReactorDesign) -> f64 {
        let capex = self.estimate_capex(design);
        let opex = self.estimate_opex(design);

        // Potencia eléctrica neta (asumiendo 33% eficiencia térmica)
        let p_fusion = ScalingLaws::fusion_power_mw(design);
        let p_electric = p_fusion * 0.33 - design.total_heating_power() * 0.5; // MW netos

        if p_electric <= 0.0 {
            return f64::INFINITY;
        }

        // Factor de recuperación de capital
        let crf = self.discount_rate * (1.0 + self.discount_rate).powi(self.plant_lifetime as i32)
            / ((1.0 + self.discount_rate).powi(self.plant_lifetime as i32) - 1.0);

        // Costo anualizado de capital
        let annual_capital = capex * crf;

        // Energía anual generada (MWh)
        let annual_energy = p_electric * 8760.0 * self.capacity_factor;

        // LCOE = (Capital anualizado + OPEX) / Energía
        (annual_capital + opex) / annual_energy
    }

    /// Estima tiempo de construcción (años)
    pub fn construction_time(&self, design: &ReactorDesign) -> f64 {
        // Escalamiento basado en tamaño
        let base_time = 8.0; // años para reactor pequeño
        let scale = (design.major_radius / 2.0).powf(0.5);

        base_time * scale
    }

    /// Desglose de costos detallado
    pub fn cost_breakdown(&self, design: &ReactorDesign) -> CostBreakdown {
        let magnet_cost = self.reference_cost * 0.4
            * (design.major_radius / self.reference_radius).powf(2.0)
            * (design.toroidal_field / 5.3).powf(1.5)
            * design.magnet_technology.relative_cost() / 2.5;

        CostBreakdown {
            magnets: magnet_cost,
            vacuum_vessel: self.reference_cost * 0.15 * (design.major_radius / self.reference_radius).powf(2.5),
            blanket: self.blanket_cost(design),
            heating: self.heating_system_cost(design),
            cryogenics: magnet_cost * 0.1,
            power_supplies: self.electrical_cost(design),
            buildings: self.building_cost(design),
            engineering: self.estimate_capex(design) * 0.15,
            contingency: self.estimate_capex(design) * 0.20,
        }
    }
}

/// Desglose detallado de costos
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub magnets: f64,
    pub vacuum_vessel: f64,
    pub blanket: f64,
    pub heating: f64,
    pub cryogenics: f64,
    pub power_supplies: f64,
    pub buildings: f64,
    pub engineering: f64,
    pub contingency: f64,
}

impl CostBreakdown {
    pub fn total(&self) -> f64 {
        self.magnets + self.vacuum_vessel + self.blanket + self.heating
            + self.cryogenics + self.power_supplies + self.buildings
            + self.engineering + self.contingency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capex_scaling() {
        let model = CostModel::default();
        let design = ReactorDesign::default();

        let capex = model.estimate_capex(&design);
        println!("CAPEX: ${:.2}B", capex / 1e9);
        assert!(capex > 0.0);
    }

    #[test]
    fn test_lcoe() {
        let model = CostModel::default();
        let mut design = ReactorDesign::default();
        design.density = 1e20;
        design.ion_temperature_kev = 15.0;

        let lcoe = model.calculate_lcoe(&design);
        println!("LCOE: ${:.2}/MWh", lcoe);
    }
}
