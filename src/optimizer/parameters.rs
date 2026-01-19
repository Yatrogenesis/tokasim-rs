//! # Definición de Parámetros del Reactor
//!
//! Espacio de parámetros completo desde plasma hasta edificio.
//! Cada parámetro tiene rangos físicamente válidos y unidades SI.

use std::collections::HashMap;

/// Definición de un parámetro optimizable
#[derive(Debug, Clone)]
pub struct ParameterDef {
    /// Nombre descriptivo
    pub name: String,
    /// Símbolo matemático
    pub symbol: String,
    /// Unidad SI
    pub unit: String,
    /// Valor mínimo permitido
    pub min: f64,
    /// Valor máximo permitido
    pub max: f64,
    /// Valor por defecto
    pub default: f64,
    /// ¿Es optimizable?
    pub is_optimizable: bool,
    /// ¿Es entero?
    pub is_integer: bool,
    /// Descripción
    pub description: String,
}

impl ParameterDef {
    /// Crea un nuevo parámetro
    pub fn new(
        name: &str,
        symbol: &str,
        unit: &str,
        min: f64,
        max: f64,
        default: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            symbol: symbol.to_string(),
            unit: unit.to_string(),
            min,
            max,
            default,
            is_optimizable: true,
            is_integer: false,
            description: String::new(),
        }
    }

    /// Marca como entero
    pub fn integer(mut self) -> Self {
        self.is_integer = true;
        self
    }

    /// Marca como no optimizable (fijo)
    pub fn fixed(mut self) -> Self {
        self.is_optimizable = false;
        self
    }

    /// Añade descripción
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Valida que un valor esté en rango
    pub fn validate(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    /// Clampea un valor al rango válido
    pub fn clamp(&self, value: f64) -> f64 {
        value.max(self.min).min(self.max)
    }

    /// Normaliza valor a [0, 1]
    pub fn normalize(&self, value: f64) -> f64 {
        if self.max == self.min {
            0.5
        } else {
            (value - self.min) / (self.max - self.min)
        }
    }

    /// Desnormaliza de [0, 1] a rango real
    pub fn denormalize(&self, normalized: f64) -> f64 {
        self.min + normalized * (self.max - self.min)
    }
}

/// Tipo de tecnología de imanes superconductores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MagnetType {
    /// Baja temperatura - NbTi (9 T máx)
    LtsNbTi,
    /// Baja temperatura - Nb3Sn (13 T máx)
    LtsNb3Sn,
    /// Alta temperatura - REBCO (23 T conservador)
    HtsRebco,
    /// Alta temperatura - REBCO avanzado (30 T)
    HtsRebcoAdvanced,
}

impl MagnetType {
    /// Campo máximo en conductor para esta tecnología
    pub fn max_field(&self) -> f64 {
        match self {
            MagnetType::LtsNbTi => 9.0,
            MagnetType::LtsNb3Sn => 13.0,
            MagnetType::HtsRebco => 23.0,
            MagnetType::HtsRebcoAdvanced => 30.0,
        }
    }

    /// Temperatura de operación (K)
    pub fn operating_temperature(&self) -> f64 {
        match self {
            MagnetType::LtsNbTi | MagnetType::LtsNb3Sn => 4.2,
            MagnetType::HtsRebco | MagnetType::HtsRebcoAdvanced => 20.0,
        }
    }

    /// Costo relativo (normalizado a NbTi = 1.0)
    pub fn relative_cost(&self) -> f64 {
        match self {
            MagnetType::LtsNbTi => 1.0,
            MagnetType::LtsNb3Sn => 2.5,
            MagnetType::HtsRebco => 8.0,
            MagnetType::HtsRebcoAdvanced => 12.0,
        }
    }
}

/// Material de primera pared
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallMaterial {
    /// Tungsteno (alta temperatura, alta erosión)
    Tungsten,
    /// Berilio (baja Z, tóxico)
    Beryllium,
    /// EUROFER (acero reducido en activación)
    Eurofer,
}

impl WallMaterial {
    /// Carga de pared máxima permitida (MW/m²)
    pub fn max_wall_load(&self) -> f64 {
        match self {
            WallMaterial::Tungsten => 2.0,
            WallMaterial::Beryllium => 1.0,
            WallMaterial::Eurofer => 1.5,
        }
    }

    /// Temperatura máxima de operación (K)
    pub fn max_temperature(&self) -> f64 {
        match self {
            WallMaterial::Tungsten => 2000.0,
            WallMaterial::Beryllium => 700.0,
            WallMaterial::Eurofer => 823.0,
        }
    }
}

/// Tipo de blanket para breeding de tritio
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlanketType {
    /// Helium Cooled Pebble Bed
    Hcpb,
    /// Water Cooled Lithium Lead
    Wcll,
    /// Dual Coolant Lithium Lead
    Dcll,
}

impl BlanketType {
    /// TBR esperado
    pub fn expected_tbr(&self) -> f64 {
        match self {
            BlanketType::Hcpb => 1.12,
            BlanketType::Wcll => 1.14,
            BlanketType::Dcll => 1.15,
        }
    }

    /// Espesor típico (m)
    pub fn typical_thickness(&self) -> f64 {
        match self {
            BlanketType::Hcpb => 0.80,
            BlanketType::Wcll => 0.70,
            BlanketType::Dcll => 0.75,
        }
    }
}

/// Escala del reactor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactorScale {
    /// Micro-tokamak (R < 1m)
    Micro,
    /// Pequeño (1-2m)
    Small,
    /// Mediano (2-4m)
    Medium,
    /// Grande (4-7m)
    Large,
    /// Escala ITER (>7m)
    IterScale,
}

/// Espacio de parámetros completo del reactor
#[derive(Debug, Clone)]
pub struct ReactorParameterSpace {
    // ========== CAPA 1: PLASMA ==========
    /// Densidad electrónica (m⁻³)
    pub density: ParameterDef,
    /// Temperatura iónica (keV)
    pub ion_temperature: ParameterDef,
    /// Temperatura electrónica (keV)
    pub electron_temperature: ParameterDef,
    /// Zeff (impurezas)
    pub z_effective: ParameterDef,
    /// Fracción de deuterio
    pub deuterium_fraction: ParameterDef,

    // ========== CAPA 2: GEOMETRÍA ==========
    /// Radio mayor R₀ (m)
    pub major_radius: ParameterDef,
    /// Radio menor a (m)
    pub minor_radius: ParameterDef,
    /// Elongación κ
    pub elongation: ParameterDef,
    /// Triangularidad δ
    pub triangularity: ParameterDef,

    // ========== CAPA 3: MAGNÉTICO ==========
    /// Campo toroidal B_t (T)
    pub toroidal_field: ParameterDef,
    /// Corriente de plasma I_p (MA)
    pub plasma_current: ParameterDef,
    /// Tipo de superconductor
    pub magnet_technology: MagnetType,
    /// Número de bobinas TF
    pub n_tf_coils: ParameterDef,

    // ========== CAPA 4: CALENTAMIENTO ==========
    /// Potencia ICRF (MW)
    pub icrf_power: ParameterDef,
    /// Potencia ECRH (MW)
    pub ecrh_power: ParameterDef,
    /// Potencia NBI (MW)
    pub nbi_power: ParameterDef,

    // ========== CAPA 5: BLINDAJE ==========
    /// Espesor de blindaje (m)
    pub shield_thickness: ParameterDef,
    /// Material primera pared
    pub first_wall_material: WallMaterial,
    /// Tipo de blanket
    pub blanket_type: BlanketType,

    // ========== CAPA 6: INFRAESTRUCTURA ==========
    /// Margen de criostato sobre TF (m)
    pub cryostat_margin: ParameterDef,
    /// Altura de grúa sobre criostato (m)
    pub crane_height: ParameterDef,

    // ========== METADATOS ==========
    /// Escala del reactor
    pub scale: ReactorScale,
    /// Parámetros adicionales
    pub extra: HashMap<String, ParameterDef>,
}

impl ReactorParameterSpace {
    /// Crea espacio de parámetros por defecto (escala media)
    pub fn default() -> Self {
        Self::medium_reactor()
    }

    /// Micro-tokamak (R < 1m) - para investigación y educación
    pub fn micro_tokamak() -> Self {
        Self {
            // Plasma
            density: ParameterDef::new("Densidad", "n", "m⁻³", 1e18, 1e20, 5e19),
            ion_temperature: ParameterDef::new("Temp. iónica", "T_i", "keV", 0.1, 5.0, 1.0),
            electron_temperature: ParameterDef::new("Temp. electrónica", "T_e", "keV", 0.1, 5.0, 1.0),
            z_effective: ParameterDef::new("Z efectivo", "Z_eff", "", 1.0, 3.0, 1.5),
            deuterium_fraction: ParameterDef::new("Fracción D", "f_D", "", 0.4, 0.6, 0.5),

            // Geometría
            major_radius: ParameterDef::new("Radio mayor", "R₀", "m", 0.2, 1.0, 0.5),
            minor_radius: ParameterDef::new("Radio menor", "a", "m", 0.05, 0.3, 0.15),
            elongation: ParameterDef::new("Elongación", "κ", "", 1.0, 1.8, 1.3),
            triangularity: ParameterDef::new("Triangularidad", "δ", "", 0.0, 0.5, 0.2),

            // Magnético
            toroidal_field: ParameterDef::new("Campo toroidal", "B_t", "T", 0.5, 3.0, 1.5),
            plasma_current: ParameterDef::new("Corriente plasma", "I_p", "MA", 0.01, 0.5, 0.1),
            magnet_technology: MagnetType::LtsNbTi,
            n_tf_coils: ParameterDef::new("Bobinas TF", "N_TF", "", 12.0, 18.0, 16.0).integer(),

            // Calentamiento
            icrf_power: ParameterDef::new("Potencia ICRF", "P_ICRF", "MW", 0.0, 1.0, 0.2),
            ecrh_power: ParameterDef::new("Potencia ECRH", "P_ECRH", "MW", 0.0, 1.0, 0.3),
            nbi_power: ParameterDef::new("Potencia NBI", "P_NBI", "MW", 0.0, 0.5, 0.0),

            // Blindaje
            shield_thickness: ParameterDef::new("Espesor blindaje", "Δ_sh", "m", 0.1, 0.3, 0.2),
            first_wall_material: WallMaterial::Eurofer,
            blanket_type: BlanketType::Hcpb,

            // Infraestructura
            cryostat_margin: ParameterDef::new("Margen criostato", "Δ_cryo", "m", 0.3, 0.8, 0.5),
            crane_height: ParameterDef::new("Altura grúa", "H_crane", "m", 3.0, 8.0, 5.0),

            scale: ReactorScale::Micro,
            extra: HashMap::new(),
        }
    }

    /// Reactor pequeño (1-2m) - tipo SPARC
    pub fn small_reactor() -> Self {
        Self {
            // Plasma
            density: ParameterDef::new("Densidad", "n", "m⁻³", 1e19, 5e20, 2e20),
            ion_temperature: ParameterDef::new("Temp. iónica", "T_i", "keV", 1.0, 25.0, 12.0),
            electron_temperature: ParameterDef::new("Temp. electrónica", "T_e", "keV", 1.0, 25.0, 12.0),
            z_effective: ParameterDef::new("Z efectivo", "Z_eff", "", 1.0, 3.0, 1.8),
            deuterium_fraction: ParameterDef::new("Fracción D", "f_D", "", 0.4, 0.6, 0.5),

            // Geometría
            major_radius: ParameterDef::new("Radio mayor", "R₀", "m", 1.0, 2.0, 1.85),
            minor_radius: ParameterDef::new("Radio menor", "a", "m", 0.3, 0.7, 0.57),
            elongation: ParameterDef::new("Elongación", "κ", "", 1.4, 2.0, 1.75),
            triangularity: ParameterDef::new("Triangularidad", "δ", "", 0.2, 0.6, 0.4),

            // Magnético (HTS para alto campo)
            toroidal_field: ParameterDef::new("Campo toroidal", "B_t", "T", 8.0, 20.0, 12.2),
            plasma_current: ParameterDef::new("Corriente plasma", "I_p", "MA", 1.0, 10.0, 8.7),
            magnet_technology: MagnetType::HtsRebco,
            n_tf_coils: ParameterDef::new("Bobinas TF", "N_TF", "", 16.0, 20.0, 18.0).integer(),

            // Calentamiento
            icrf_power: ParameterDef::new("Potencia ICRF", "P_ICRF", "MW", 0.0, 30.0, 11.0),
            ecrh_power: ParameterDef::new("Potencia ECRH", "P_ECRH", "MW", 0.0, 20.0, 0.0),
            nbi_power: ParameterDef::new("Potencia NBI", "P_NBI", "MW", 0.0, 20.0, 0.0),

            // Blindaje
            shield_thickness: ParameterDef::new("Espesor blindaje", "Δ_sh", "m", 0.3, 0.6, 0.4),
            first_wall_material: WallMaterial::Tungsten,
            blanket_type: BlanketType::Hcpb,

            // Infraestructura
            cryostat_margin: ParameterDef::new("Margen criostato", "Δ_cryo", "m", 0.5, 1.5, 1.0),
            crane_height: ParameterDef::new("Altura grúa", "H_crane", "m", 8.0, 15.0, 12.0),

            scale: ReactorScale::Small,
            extra: HashMap::new(),
        }
    }

    /// Reactor mediano (2-4m) - demostración comercial
    pub fn medium_reactor() -> Self {
        Self {
            // Plasma
            density: ParameterDef::new("Densidad", "n", "m⁻³", 5e19, 2e21, 1e20),
            ion_temperature: ParameterDef::new("Temp. iónica", "T_i", "keV", 5.0, 30.0, 15.0),
            electron_temperature: ParameterDef::new("Temp. electrónica", "T_e", "keV", 5.0, 30.0, 15.0),
            z_effective: ParameterDef::new("Z efectivo", "Z_eff", "", 1.0, 3.5, 1.7),
            deuterium_fraction: ParameterDef::new("Fracción D", "f_D", "", 0.45, 0.55, 0.5),

            // Geometría
            major_radius: ParameterDef::new("Radio mayor", "R₀", "m", 2.0, 4.0, 3.0),
            minor_radius: ParameterDef::new("Radio menor", "a", "m", 0.5, 1.3, 1.0),
            elongation: ParameterDef::new("Elongación", "κ", "", 1.5, 2.2, 1.8),
            triangularity: ParameterDef::new("Triangularidad", "δ", "", 0.2, 0.6, 0.4),

            // Magnético
            toroidal_field: ParameterDef::new("Campo toroidal", "B_t", "T", 4.0, 15.0, 8.0),
            plasma_current: ParameterDef::new("Corriente plasma", "I_p", "MA", 2.0, 15.0, 8.0),
            magnet_technology: MagnetType::HtsRebco,
            n_tf_coils: ParameterDef::new("Bobinas TF", "N_TF", "", 16.0, 20.0, 18.0).integer(),

            // Calentamiento
            icrf_power: ParameterDef::new("Potencia ICRF", "P_ICRF", "MW", 0.0, 50.0, 20.0),
            ecrh_power: ParameterDef::new("Potencia ECRH", "P_ECRH", "MW", 0.0, 30.0, 10.0),
            nbi_power: ParameterDef::new("Potencia NBI", "P_NBI", "MW", 0.0, 40.0, 20.0),

            // Blindaje
            shield_thickness: ParameterDef::new("Espesor blindaje", "Δ_sh", "m", 0.4, 1.0, 0.6),
            first_wall_material: WallMaterial::Tungsten,
            blanket_type: BlanketType::Wcll,

            // Infraestructura
            cryostat_margin: ParameterDef::new("Margen criostato", "Δ_cryo", "m", 0.8, 2.0, 1.2),
            crane_height: ParameterDef::new("Altura grúa", "H_crane", "m", 12.0, 25.0, 18.0),

            scale: ReactorScale::Medium,
            extra: HashMap::new(),
        }
    }

    /// Reactor grande (4-7m) - planta comercial
    pub fn large_reactor() -> Self {
        Self {
            // Plasma
            density: ParameterDef::new("Densidad", "n", "m⁻³", 5e19, 2e21, 1e20),
            ion_temperature: ParameterDef::new("Temp. iónica", "T_i", "keV", 8.0, 35.0, 18.0),
            electron_temperature: ParameterDef::new("Temp. electrónica", "T_e", "keV", 8.0, 35.0, 18.0),
            z_effective: ParameterDef::new("Z efectivo", "Z_eff", "", 1.0, 3.0, 1.6),
            deuterium_fraction: ParameterDef::new("Fracción D", "f_D", "", 0.45, 0.55, 0.5),

            // Geometría
            major_radius: ParameterDef::new("Radio mayor", "R₀", "m", 4.0, 7.0, 5.5),
            minor_radius: ParameterDef::new("Radio menor", "a", "m", 1.0, 2.2, 1.7),
            elongation: ParameterDef::new("Elongación", "κ", "", 1.6, 2.2, 1.85),
            triangularity: ParameterDef::new("Triangularidad", "δ", "", 0.25, 0.55, 0.4),

            // Magnético
            toroidal_field: ParameterDef::new("Campo toroidal", "B_t", "T", 4.0, 12.0, 6.0),
            plasma_current: ParameterDef::new("Corriente plasma", "I_p", "MA", 5.0, 18.0, 12.0),
            magnet_technology: MagnetType::LtsNb3Sn,
            n_tf_coils: ParameterDef::new("Bobinas TF", "N_TF", "", 16.0, 20.0, 18.0).integer(),

            // Calentamiento
            icrf_power: ParameterDef::new("Potencia ICRF", "P_ICRF", "MW", 0.0, 80.0, 30.0),
            ecrh_power: ParameterDef::new("Potencia ECRH", "P_ECRH", "MW", 0.0, 40.0, 20.0),
            nbi_power: ParameterDef::new("Potencia NBI", "P_NBI", "MW", 0.0, 60.0, 30.0),

            // Blindaje
            shield_thickness: ParameterDef::new("Espesor blindaje", "Δ_sh", "m", 0.5, 1.5, 0.8),
            first_wall_material: WallMaterial::Tungsten,
            blanket_type: BlanketType::Dcll,

            // Infraestructura
            cryostat_margin: ParameterDef::new("Margen criostato", "Δ_cryo", "m", 1.0, 3.0, 2.0),
            crane_height: ParameterDef::new("Altura grúa", "H_crane", "m", 18.0, 35.0, 25.0),

            scale: ReactorScale::Large,
            extra: HashMap::new(),
        }
    }

    /// Escala ITER (>7m) - reactor de referencia internacional
    pub fn iter_scale() -> Self {
        Self {
            // Plasma (valores ITER)
            density: ParameterDef::new("Densidad", "n", "m⁻³", 8e19, 1.2e20, 1e20),
            ion_temperature: ParameterDef::new("Temp. iónica", "T_i", "keV", 10.0, 25.0, 18.0),
            electron_temperature: ParameterDef::new("Temp. electrónica", "T_e", "keV", 10.0, 25.0, 18.0),
            z_effective: ParameterDef::new("Z efectivo", "Z_eff", "", 1.5, 2.5, 1.65),
            deuterium_fraction: ParameterDef::new("Fracción D", "f_D", "", 0.45, 0.55, 0.5),

            // Geometría (ITER: R=6.2m, a=2.0m)
            major_radius: ParameterDef::new("Radio mayor", "R₀", "m", 5.5, 8.0, 6.2),
            minor_radius: ParameterDef::new("Radio menor", "a", "m", 1.5, 2.5, 2.0),
            elongation: ParameterDef::new("Elongación", "κ", "", 1.7, 2.0, 1.85),
            triangularity: ParameterDef::new("Triangularidad", "δ", "", 0.33, 0.5, 0.4),

            // Magnético (ITER: B=5.3T, Ip=15MA)
            toroidal_field: ParameterDef::new("Campo toroidal", "B_t", "T", 4.5, 6.5, 5.3),
            plasma_current: ParameterDef::new("Corriente plasma", "I_p", "MA", 10.0, 17.0, 15.0),
            magnet_technology: MagnetType::LtsNb3Sn,
            n_tf_coils: ParameterDef::new("Bobinas TF", "N_TF", "", 18.0, 18.0, 18.0).integer().fixed(),

            // Calentamiento (ITER: 73MW total)
            icrf_power: ParameterDef::new("Potencia ICRF", "P_ICRF", "MW", 15.0, 25.0, 20.0),
            ecrh_power: ParameterDef::new("Potencia ECRH", "P_ECRH", "MW", 15.0, 25.0, 20.0),
            nbi_power: ParameterDef::new("Potencia NBI", "P_NBI", "MW", 25.0, 40.0, 33.0),

            // Blindaje
            shield_thickness: ParameterDef::new("Espesor blindaje", "Δ_sh", "m", 0.8, 1.2, 1.0),
            first_wall_material: WallMaterial::Beryllium,
            blanket_type: BlanketType::Hcpb,

            // Infraestructura
            cryostat_margin: ParameterDef::new("Margen criostato", "Δ_cryo", "m", 2.0, 4.0, 3.0),
            crane_height: ParameterDef::new("Altura grúa", "H_crane", "m", 25.0, 40.0, 30.0),

            scale: ReactorScale::IterScale,
            extra: HashMap::new(),
        }
    }

    /// Obtiene todos los parámetros como vector
    pub fn all_parameters(&self) -> Vec<&ParameterDef> {
        vec![
            &self.density,
            &self.ion_temperature,
            &self.electron_temperature,
            &self.z_effective,
            &self.deuterium_fraction,
            &self.major_radius,
            &self.minor_radius,
            &self.elongation,
            &self.triangularity,
            &self.toroidal_field,
            &self.plasma_current,
            &self.n_tf_coils,
            &self.icrf_power,
            &self.ecrh_power,
            &self.nbi_power,
            &self.shield_thickness,
            &self.cryostat_margin,
            &self.crane_height,
        ]
    }

    /// Obtiene solo parámetros optimizables
    pub fn optimizable_parameters(&self) -> Vec<&ParameterDef> {
        self.all_parameters()
            .into_iter()
            .filter(|p| p.is_optimizable)
            .collect()
    }

    /// Número de dimensiones optimizables
    pub fn n_dimensions(&self) -> usize {
        self.optimizable_parameters().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_validation() {
        let param = ParameterDef::new("Test", "t", "m", 0.0, 10.0, 5.0);
        assert!(param.validate(5.0));
        assert!(param.validate(0.0));
        assert!(param.validate(10.0));
        assert!(!param.validate(-1.0));
        assert!(!param.validate(11.0));
    }

    #[test]
    fn test_parameter_normalization() {
        let param = ParameterDef::new("Test", "t", "m", 0.0, 10.0, 5.0);
        assert!((param.normalize(0.0) - 0.0).abs() < 1e-10);
        assert!((param.normalize(5.0) - 0.5).abs() < 1e-10);
        assert!((param.normalize(10.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_iter_scale_parameters() {
        let space = ReactorParameterSpace::iter_scale();
        assert!((space.major_radius.default - 6.2).abs() < 0.1);
        assert!((space.toroidal_field.default - 5.3).abs() < 0.1);
        assert!((space.plasma_current.default - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_magnet_types() {
        assert!(MagnetType::HtsRebco.max_field() > MagnetType::LtsNb3Sn.max_field());
        assert!(MagnetType::LtsNb3Sn.max_field() > MagnetType::LtsNbTi.max_field());
    }
}
