//! # Complete Equipment Inventory
//!
//! Comprehensive inventory of ALL components in the TS-1 tokamak facility.

use super::*;

/// Equipment category
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquipmentCategory {
    /// Core tokamak structure
    TokamakCore,
    /// Magnet systems
    MagnetSystem,
    /// Power supply systems
    PowerSupply,
    /// Cryogenic systems
    Cryogenics,
    /// Vacuum systems
    Vacuum,
    /// Heating systems
    Heating,
    /// Diagnostic systems
    Diagnostics,
    /// Control systems
    Control,
    /// Safety systems
    Safety,
    /// Auxiliary systems
    Auxiliary,
    /// Cooling systems
    Cooling,
    /// Building infrastructure
    Infrastructure,
}

/// Individual equipment item
#[derive(Debug, Clone)]
pub struct Equipment {
    pub id: String,
    pub name: String,
    pub category: EquipmentCategory,
    pub description: String,
    pub location: String,
    pub manufacturer: String,
    pub model: String,
    pub quantity: u32,
    pub weight_kg: f64,
    pub power_kw: f64,
    pub voltage_v: f64,
    pub cooling_kw: f64,
    pub sil_rating: Option<u8>,
    pub redundancy: String,
    pub mtbf_hours: f64,
    pub health: ComponentHealth,
}

impl Equipment {
    pub fn new(id: &str, name: &str, category: EquipmentCategory) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            category,
            description: String::new(),
            location: String::new(),
            manufacturer: String::new(),
            model: String::new(),
            quantity: 1,
            weight_kg: 0.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        }
    }
}

/// Complete facility inventory
#[derive(Debug, Clone)]
pub struct FacilityInventory {
    pub equipment: Vec<Equipment>,
    pub total_weight_tonnes: f64,
    pub total_power_mw: f64,
    pub total_cooling_mw: f64,
}

impl FacilityInventory {
    /// Create complete TS-1 inventory
    pub fn ts1() -> Self {
        let mut equipment = Vec::new();

        // ========== TOKAMAK CORE ==========
        equipment.push(Equipment {
            id: "TK-VV-01".to_string(),
            name: "Vacuum Vessel".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "D-shaped double-wall vacuum vessel with water cooling channels".to_string(),
            location: "Tokamak Hall, Center".to_string(),
            manufacturer: "Mitsubishi Heavy Industries".to_string(),
            model: "VV-TS1-D".to_string(),
            quantity: 1,
            weight_kg: 180_000.0,
            power_kw: 0.0,  // Passive
            voltage_v: 0.0,
            cooling_kw: 5_000.0,
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: f64::INFINITY,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-FW-01".to_string(),
            name: "First Wall Panels".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Beryllium-coated first wall panels with active cooling".to_string(),
            location: "Vacuum Vessel Interior".to_string(),
            manufacturer: "Framatome".to_string(),
            model: "FW-Be-440".to_string(),
            quantity: 440,
            weight_kg: 25_000.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 50_000.0,
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-DIV-IN".to_string(),
            name: "Inner Divertor".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Tungsten monoblock inner divertor with CuCrZr heat sink".to_string(),
            location: "Vacuum Vessel Bottom, Inboard".to_string(),
            manufacturer: "Plansee".to_string(),
            model: "DIV-W-IN-54".to_string(),
            quantity: 54,
            weight_kg: 8_000.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 100_000.0,  // 100 MW peak
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 20_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-DIV-OUT".to_string(),
            name: "Outer Divertor".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Tungsten monoblock outer divertor with CuCrZr heat sink".to_string(),
            location: "Vacuum Vessel Bottom, Outboard".to_string(),
            manufacturer: "Plansee".to_string(),
            model: "DIV-W-OUT-54".to_string(),
            quantity: 54,
            weight_kg: 12_000.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 150_000.0,  // 150 MW peak
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 20_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-CRYO-01".to_string(),
            name: "Cryostat".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Stainless steel cryostat providing thermal insulation for SC magnets".to_string(),
            location: "Tokamak Hall, Enclosing VV".to_string(),
            manufacturer: "CNIM".to_string(),
            model: "CRYO-TS1".to_string(),
            quantity: 1,
            weight_kg: 400_000.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 1_000.0,
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: f64::INFINITY,
            health: ComponentHealth::default(),
        });

        // ========== BREEDING BLANKET ==========
        // Tritium Breeding Ratio (TBR) > 1.1 required for self-sufficiency
        equipment.push(Equipment {
            id: "TK-BB-IB".to_string(),
            name: "Inboard Breeding Blanket".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "HCPB (Helium-Cooled Pebble Bed) breeding blanket with Li4SiO4 + Be pebbles, inboard modules".to_string(),
            location: "Vacuum Vessel Interior, Inboard".to_string(),
            manufacturer: "KIT / Framatome".to_string(),
            model: "BB-HCPB-IB-TS1".to_string(),
            quantity: 16,  // Inboard modules
            weight_kg: 80_000.0,  // Total inboard
            power_kw: 0.0,  // Passive - generates heat
            voltage_v: 0.0,
            cooling_kw: 120_000.0,  // 120 MW thermal from neutrons
            sil_rating: None,
            redundancy: "Modular replacement".to_string(),
            mtbf_hours: 30_000.0,  // ~3.5 years before replacement
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-BB-OB".to_string(),
            name: "Outboard Breeding Blanket".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "HCPB breeding blanket with Li4SiO4 + Be pebbles, outboard modules with higher TBR".to_string(),
            location: "Vacuum Vessel Interior, Outboard".to_string(),
            manufacturer: "KIT / Framatome".to_string(),
            model: "BB-HCPB-OB-TS1".to_string(),
            quantity: 32,  // Outboard modules (more area)
            weight_kg: 200_000.0,  // Total outboard
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 280_000.0,  // 280 MW thermal
            sil_rating: None,
            redundancy: "Modular replacement".to_string(),
            mtbf_hours: 30_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-BB-TES".to_string(),
            name: "Tritium Extraction System".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Helium purge gas system for tritium extraction from breeding blanket pebbles".to_string(),
            location: "Tritium Building, BB Interface".to_string(),
            manufacturer: "AECL / Tritium Systems".to_string(),
            model: "TES-HCPB-TS1".to_string(),
            quantity: 1,
            weight_kg: 15_000.0,
            power_kw: 100.0,
            voltage_v: 400.0,
            cooling_kw: 50.0,
            sil_rating: Some(2),
            redundancy: "Dual extraction trains".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "TK-BB-MULT".to_string(),
            name: "Neutron Multiplier Assemblies".to_string(),
            category: EquipmentCategory::TokamakCore,
            description: "Beryllium pebble beds for neutron multiplication (n,2n) to achieve TBR > 1.1".to_string(),
            location: "Inside Breeding Blanket Modules".to_string(),
            manufacturer: "Materion".to_string(),
            model: "Be-MULT-12mm".to_string(),
            quantity: 48,  // Integrated in all BB modules
            weight_kg: 25_000.0,  // Total Be inventory
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 0.0,  // Cooled via blanket He
            sil_rating: None,
            redundancy: "Distributed in all modules".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        // ========== MAGNET SYSTEMS ==========
        for i in 1..=18 {
            equipment.push(Equipment {
                id: format!("TF-{:02}", i),
                name: format!("Toroidal Field Coil {}", i),
                category: EquipmentCategory::MagnetSystem,
                description: "Nb3Sn superconducting toroidal field coil, D-shaped".to_string(),
                location: format!("Tokamak Hall, Sector {}", i),
                manufacturer: "ASG Superconductors".to_string(),
                model: "TF-Nb3Sn-12T".to_string(),
                quantity: 1,
                weight_kg: 280_000.0 / 18.0,  // ~15.5 tonnes each
                power_kw: 0.0,  // Superconducting - no resistive power
                voltage_v: 0.0,
                cooling_kw: 100.0,  // Cryogenic load
                sil_rating: None,
                redundancy: format!("18 coils, operates with up to 1 failure"),
                mtbf_hours: 500_000.0,
                health: ComponentHealth::default(),
            });
        }

        for i in 1..=6 {
            equipment.push(Equipment {
                id: format!("PF-{}", i),
                name: format!("Poloidal Field Coil {}", i),
                category: EquipmentCategory::MagnetSystem,
                description: "NbTi superconducting poloidal field coil for plasma shaping".to_string(),
                location: format!("Tokamak Hall, Ring {}", i),
                manufacturer: "ASG Superconductors".to_string(),
                model: "PF-NbTi-6T".to_string(),
                quantity: 1,
                weight_kg: 350_000.0 / 6.0,  // ~58 tonnes each
                power_kw: 0.0,
                voltage_v: 0.0,
                cooling_kw: 50.0,
                sil_rating: None,
                redundancy: "6oo8 with 2 spare channels".to_string(),
                mtbf_hours: 400_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "CS-01".to_string(),
            name: "Central Solenoid".to_string(),
            category: EquipmentCategory::MagnetSystem,
            description: "Nb3Sn superconducting central solenoid for plasma initiation and current drive".to_string(),
            location: "Tokamak Hall, Center Column".to_string(),
            manufacturer: "General Atomics".to_string(),
            model: "CS-Nb3Sn-13T".to_string(),
            quantity: 1,
            weight_kg: 140_000.0,
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 200.0,
            sil_rating: None,
            redundancy: "6 modules, can operate with 1 failed".to_string(),
            mtbf_hours: 300_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=4 {
            equipment.push(Equipment {
                id: format!("VS-{}", i),
                name: format!("Vertical Stability Coil {}", i),
                category: EquipmentCategory::MagnetSystem,
                description: "Fast copper coil for vertical stability control".to_string(),
                location: format!("Inside Vacuum Vessel, Position {}", i),
                manufacturer: "Tesla Engineering".to_string(),
                model: "VS-Cu-Fast".to_string(),
                quantity: 1,
                weight_kg: 500.0,
                power_kw: 500.0,  // Active power
                voltage_v: 1000.0,
                cooling_kw: 600.0,
                sil_rating: Some(3),
                redundancy: "2oo4 voting logic".to_string(),
                mtbf_hours: 100_000.0,
                health: ComponentHealth::default(),
            });
        }

        // ========== POWER SUPPLIES ==========
        equipment.push(Equipment {
            id: "PS-TF-01".to_string(),
            name: "TF Coil Power Supply".to_string(),
            category: EquipmentCategory::PowerSupply,
            description: "68 kA superconducting magnet power supply for TF coils".to_string(),
            location: "Power Supply Building, Bay 1".to_string(),
            manufacturer: "ABB".to_string(),
            model: "SC-PS-68k-18V".to_string(),
            quantity: 1,
            weight_kg: 50_000.0,
            power_kw: 1_500.0,
            voltage_v: 18.0,
            cooling_kw: 500.0,
            sil_rating: Some(2),
            redundancy: "1oo2 with hot standby".to_string(),
            mtbf_hours: 200_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=8 {
            equipment.push(Equipment {
                id: format!("PS-PF-{:02}", i),
                name: format!("PF Coil Power Supply {}", i),
                category: EquipmentCategory::PowerSupply,
                description: "45 kA 4-quadrant power supply for PF coil control".to_string(),
                location: format!("Power Supply Building, Bay {}", 2 + i),
                manufacturer: "Siemens".to_string(),
                model: "4Q-PS-45k-1kV".to_string(),
                quantity: 1,
                weight_kg: 20_000.0,
                power_kw: 45_000.0,
                voltage_v: 1000.0,
                cooling_kw: 15_000.0,
                sil_rating: Some(2),
                redundancy: "6oo8 voting".to_string(),
                mtbf_hours: 150_000.0,
                health: ComponentHealth::default(),
            });
        }

        for i in 1..=4 {
            equipment.push(Equipment {
                id: format!("PS-VS-{}", i),
                name: format!("VS Coil Power Supply {}", i),
                category: EquipmentCategory::PowerSupply,
                description: "Fast-response power supply for vertical stability".to_string(),
                location: format!("Power Supply Building, Bay VS-{}", i),
                manufacturer: "Hitachi".to_string(),
                model: "FAST-PS-10k-1kV".to_string(),
                quantity: 1,
                weight_kg: 5_000.0,
                power_kw: 10_000.0,
                voltage_v: 1000.0,
                cooling_kw: 3_000.0,
                sil_rating: Some(3),
                redundancy: "2oo4 voting".to_string(),
                mtbf_hours: 100_000.0,
                health: ComponentHealth::default(),
            });
        }

        // ========== CRYOGENIC SYSTEMS ==========
        equipment.push(Equipment {
            id: "CRYO-LHe-01".to_string(),
            name: "Helium Liquefier".to_string(),
            category: EquipmentCategory::Cryogenics,
            description: "Large helium liquefaction plant for magnet cooling".to_string(),
            location: "Cryogenic Building".to_string(),
            manufacturer: "Linde".to_string(),
            model: "L1610".to_string(),
            quantity: 1,
            weight_kg: 100_000.0,
            power_kw: 3_000.0,
            voltage_v: 6600.0,
            cooling_kw: 0.0,  // Provides cooling
            sil_rating: Some(2),
            redundancy: "N+1 with 2 compressors".to_string(),
            mtbf_hours: 80_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=3 {
            equipment.push(Equipment {
                id: format!("CRYO-COMP-{:02}", i),
                name: format!("Helium Compressor {}", i),
                category: EquipmentCategory::Cryogenics,
                description: "Screw compressor for helium recirculation".to_string(),
                location: "Cryogenic Building, Compressor Hall".to_string(),
                manufacturer: "Linde".to_string(),
                model: "HC-1000".to_string(),
                quantity: 1,
                weight_kg: 15_000.0,
                power_kw: 1_000.0,
                voltage_v: 6600.0,
                cooling_kw: 0.0,
                sil_rating: Some(2),
                redundancy: "2oo3 operation".to_string(),
                mtbf_hours: 60_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "CRYO-CB-01".to_string(),
            name: "Cold Box Primary".to_string(),
            category: EquipmentCategory::Cryogenics,
            description: "Primary cold box for 4.5K helium distribution".to_string(),
            location: "Cryogenic Building".to_string(),
            manufacturer: "Air Liquide".to_string(),
            model: "CB-4500-TS1".to_string(),
            quantity: 1,
            weight_kg: 30_000.0,
            power_kw: 500.0,
            voltage_v: 400.0,
            cooling_kw: 0.0,
            sil_rating: Some(2),
            redundancy: "1oo2 with backup".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "CRYO-LN2-01".to_string(),
            name: "Liquid Nitrogen System".to_string(),
            category: EquipmentCategory::Cryogenics,
            description: "LN2 pre-cooling and thermal shield system".to_string(),
            location: "Cryogenic Building, LN2 Area".to_string(),
            manufacturer: "Chart Industries".to_string(),
            model: "LN2-TS1-500".to_string(),
            quantity: 1,
            weight_kg: 20_000.0,
            power_kw: 200.0,
            voltage_v: 400.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "Bulk storage tank backup".to_string(),
            mtbf_hours: 200_000.0,
            health: ComponentHealth::default(),
        });

        // ========== VACUUM SYSTEMS ==========
        for i in 1..=6 {
            equipment.push(Equipment {
                id: format!("VAC-TP-{:02}", i),
                name: format!("Turbo-Molecular Pump {}", i),
                category: EquipmentCategory::Vacuum,
                description: "High-vacuum turbo-molecular pump for vessel evacuation".to_string(),
                location: format!("Tokamak Hall, Port {}", i),
                manufacturer: "Edwards".to_string(),
                model: "STP-XL5003".to_string(),
                quantity: 1,
                weight_kg: 200.0,
                power_kw: 5.0,
                voltage_v: 400.0,
                cooling_kw: 2.0,
                sil_rating: None,
                redundancy: "4oo6 operation".to_string(),
                mtbf_hours: 40_000.0,
                health: ComponentHealth::default(),
            });
        }

        for i in 1..=12 {
            equipment.push(Equipment {
                id: format!("VAC-CRYO-{:02}", i),
                name: format!("Cryopump {}", i),
                category: EquipmentCategory::Vacuum,
                description: "Cryogenic pump for ultra-high vacuum and helium pumping".to_string(),
                location: format!("Tokamak Hall, Sector {}", (i - 1) * 30),
                manufacturer: "SHI Cryogenics".to_string(),
                model: "RP-502".to_string(),
                quantity: 1,
                weight_kg: 300.0,
                power_kw: 10.0,
                voltage_v: 400.0,
                cooling_kw: 5.0,
                sil_rating: None,
                redundancy: "8oo12 operation".to_string(),
                mtbf_hours: 50_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "VAC-ROUGH-01".to_string(),
            name: "Roughing Pump System".to_string(),
            category: EquipmentCategory::Vacuum,
            description: "Scroll pump array for initial vessel evacuation".to_string(),
            location: "Tokamak Hall, Basement".to_string(),
            manufacturer: "Busch".to_string(),
            model: "FOSSA-FO-5000".to_string(),
            quantity: 4,
            weight_kg: 500.0,
            power_kw: 20.0,
            voltage_v: 400.0,
            cooling_kw: 10.0,
            sil_rating: None,
            redundancy: "2oo4".to_string(),
            mtbf_hours: 30_000.0,
            health: ComponentHealth::default(),
        });

        // ========== HEATING SYSTEMS ==========
        for i in 1..=5 {
            equipment.push(Equipment {
                id: format!("ICRF-{:02}", i),
                name: format!("ICRF Generator {}", i),
                category: EquipmentCategory::Heating,
                description: "Ion Cyclotron Resonance Frequency heating generator (40-55 MHz)".to_string(),
                location: format!("RF Building, Bay {}", i),
                manufacturer: "Thomson Broadcast".to_string(),
                model: "IOT-10MW".to_string(),
                quantity: 1,
                weight_kg: 8_000.0,
                power_kw: 10_000.0,
                voltage_v: 6600.0,
                cooling_kw: 6_000.0,
                sil_rating: Some(1),
                redundancy: "4oo5 operation".to_string(),
                mtbf_hours: 20_000.0,
                health: ComponentHealth::default(),
            });

            equipment.push(Equipment {
                id: format!("ICRF-ANT-{:02}", i),
                name: format!("ICRF Antenna {}", i),
                category: EquipmentCategory::Heating,
                description: "Four-strap ICRF antenna with Faraday screen".to_string(),
                location: format!("Tokamak Hall, Port ICRF-{}", i),
                manufacturer: "CEA".to_string(),
                model: "ICRF-4S-TS1".to_string(),
                quantity: 1,
                weight_kg: 2_000.0,
                power_kw: 0.0,  // RF power passthrough
                voltage_v: 0.0,
                cooling_kw: 1_000.0,
                sil_rating: None,
                redundancy: "N/A".to_string(),
                mtbf_hours: 50_000.0,
                health: ComponentHealth::default(),
            });
        }

        for i in 1..=8 {
            equipment.push(Equipment {
                id: format!("ECRH-GYR-{:02}", i),
                name: format!("ECRH Gyrotron {}", i),
                category: EquipmentCategory::Heating,
                description: "170 GHz gyrotron for electron cyclotron resonance heating".to_string(),
                location: format!("ECRH Building, Bay {}", i),
                manufacturer: "GYCOM".to_string(),
                model: "GYR-170-1MW".to_string(),
                quantity: 1,
                weight_kg: 3_000.0,
                power_kw: 1_500.0,  // Electrical input for 1 MW output
                voltage_v: 80_000.0,  // Beam voltage
                cooling_kw: 1_200.0,
                sil_rating: Some(1),
                redundancy: "6oo8 operation".to_string(),
                mtbf_hours: 15_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "ECRH-WG-01".to_string(),
            name: "ECRH Waveguide System".to_string(),
            category: EquipmentCategory::Heating,
            description: "Corrugated waveguide transmission line (63.5 mm ID)".to_string(),
            location: "ECRH Building to Tokamak Hall".to_string(),
            manufacturer: "General Atomics".to_string(),
            model: "WG-63.5-HE11".to_string(),
            quantity: 8,
            weight_kg: 500.0,  // Per line
            power_kw: 0.0,
            voltage_v: 0.0,
            cooling_kw: 50.0,  // Per line
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 200_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=3 {
            equipment.push(Equipment {
                id: format!("NBI-{:02}", i),
                name: format!("Neutral Beam Injector {}", i),
                category: EquipmentCategory::Heating,
                description: "1 MeV D0 neutral beam injector for heating and current drive".to_string(),
                location: format!("NBI Building, Bay {}", i),
                manufacturer: "Consorzio RFX".to_string(),
                model: "NBI-1MeV-16.7MW".to_string(),
                quantity: 1,
                weight_kg: 150_000.0,
                power_kw: 25_000.0,
                voltage_v: 1_000_000.0,
                cooling_kw: 20_000.0,
                sil_rating: Some(2),
                redundancy: "2oo3 operation".to_string(),
                mtbf_hours: 10_000.0,
                health: ComponentHealth::default(),
            });
        }

        // ========== DIAGNOSTIC SYSTEMS ==========
        equipment.push(Equipment {
            id: "DIAG-TS-01".to_string(),
            name: "Thomson Scattering System".to_string(),
            category: EquipmentCategory::Diagnostics,
            description: "Core Thomson scattering for Te and ne measurement".to_string(),
            location: "Tokamak Hall, Port TS-1".to_string(),
            manufacturer: "UKAEA".to_string(),
            model: "TS-Core-200".to_string(),
            quantity: 1,
            weight_kg: 5_000.0,
            power_kw: 100.0,
            voltage_v: 400.0,
            cooling_kw: 50.0,
            sil_rating: None,
            redundancy: "1oo1 (non-safety)".to_string(),
            mtbf_hours: 30_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "DIAG-TS-02".to_string(),
            name: "Edge Thomson Scattering".to_string(),
            category: EquipmentCategory::Diagnostics,
            description: "Edge Thomson scattering for pedestal measurement".to_string(),
            location: "Tokamak Hall, Port TS-2".to_string(),
            manufacturer: "UKAEA".to_string(),
            model: "TS-Edge-100".to_string(),
            quantity: 1,
            weight_kg: 3_000.0,
            power_kw: 50.0,
            voltage_v: 400.0,
            cooling_kw: 25.0,
            sil_rating: None,
            redundancy: "1oo2 with backup".to_string(),
            mtbf_hours: 30_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=4 {
            equipment.push(Equipment {
                id: format!("DIAG-ECE-{:02}", i),
                name: format!("ECE Radiometer {}", i),
                category: EquipmentCategory::Diagnostics,
                description: "Electron Cyclotron Emission radiometer for Te profile".to_string(),
                location: format!("Tokamak Hall, Port ECE-{}", i),
                manufacturer: "IPP Garching".to_string(),
                model: "ECE-HET-32".to_string(),
                quantity: 1,
                weight_kg: 200.0,
                power_kw: 2.0,
                voltage_v: 24.0,
                cooling_kw: 0.5,
                sil_rating: None,
                redundancy: "2oo4".to_string(),
                mtbf_hours: 50_000.0,
                health: ComponentHealth::default(),
            });
        }

        for i in 1..=6 {
            equipment.push(Equipment {
                id: format!("DIAG-INTER-{:02}", i),
                name: format!("Interferometer Channel {}", i),
                category: EquipmentCategory::Diagnostics,
                description: "Far-infrared interferometer for line-integrated density".to_string(),
                location: format!("Tokamak Hall, Port INT-{}", i),
                manufacturer: "ENEA".to_string(),
                model: "FIR-INT-TS1".to_string(),
                quantity: 1,
                weight_kg: 100.0,
                power_kw: 5.0,
                voltage_v: 24.0,
                cooling_kw: 1.0,
                sil_rating: Some(3),  // Used for density limit protection
                redundancy: "4oo6 (safety critical)".to_string(),
                mtbf_hours: 40_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "DIAG-MAG-01".to_string(),
            name: "Magnetic Diagnostics System".to_string(),
            category: EquipmentCategory::Diagnostics,
            description: "Complete set of magnetic probes, flux loops, and Rogowski coils".to_string(),
            location: "Inside Vacuum Vessel".to_string(),
            manufacturer: "IPP Garching".to_string(),
            model: "MAGDIAG-TS1".to_string(),
            quantity: 1,
            weight_kg: 500.0,  // All sensors combined
            power_kw: 1.0,
            voltage_v: 24.0,
            cooling_kw: 0.0,
            sil_rating: Some(3),
            redundancy: "1oo2 per measurement type".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "DIAG-BOLO-01".to_string(),
            name: "Bolometer System".to_string(),
            category: EquipmentCategory::Diagnostics,
            description: "Multi-channel bolometer for radiated power measurement".to_string(),
            location: "Tokamak Hall, Multiple Ports".to_string(),
            manufacturer: "ITER Organization".to_string(),
            model: "BOLO-Au-256".to_string(),
            quantity: 1,
            weight_kg: 100.0,
            power_kw: 5.0,
            voltage_v: 24.0,
            cooling_kw: 1.0,
            sil_rating: None,
            redundancy: "Multiple views".to_string(),
            mtbf_hours: 80_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "DIAG-IR-01".to_string(),
            name: "Infrared Thermography".to_string(),
            category: EquipmentCategory::Diagnostics,
            description: "IR camera system for first wall and divertor monitoring".to_string(),
            location: "Tokamak Hall, Upper and Equatorial Ports".to_string(),
            manufacturer: "FLIR".to_string(),
            model: "SC7600-MW".to_string(),
            quantity: 8,
            weight_kg: 50.0,  // Per camera
            power_kw: 1.0,
            voltage_v: 24.0,
            cooling_kw: 0.5,
            sil_rating: Some(2),
            redundancy: "Multiple views per component".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        // ========== CONTROL SYSTEMS ==========
        equipment.push(Equipment {
            id: "CTRL-MCC-01".to_string(),
            name: "Main Control Computer Primary".to_string(),
            category: EquipmentCategory::Control,
            description: "Real-time control computer running PIRS rule engine".to_string(),
            location: "Control Room, Rack CR-01".to_string(),
            manufacturer: "Concurrent Real-Time".to_string(),
            model: "RedHawk-RT-8000".to_string(),
            quantity: 1,
            weight_kg: 50.0,
            power_kw: 2.0,
            voltage_v: 400.0,
            cooling_kw: 3.0,
            sil_rating: Some(3),
            redundancy: "2oo3 voting".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "CTRL-MCC-02".to_string(),
            name: "Main Control Computer Hot Standby".to_string(),
            category: EquipmentCategory::Control,
            description: "Hot standby control computer, synchronized".to_string(),
            location: "Control Room, Rack CR-02".to_string(),
            manufacturer: "Concurrent Real-Time".to_string(),
            model: "RedHawk-RT-8000".to_string(),
            quantity: 1,
            weight_kg: 50.0,
            power_kw: 2.0,
            voltage_v: 400.0,
            cooling_kw: 3.0,
            sil_rating: Some(3),
            redundancy: "2oo3 voting".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "CTRL-MCC-03".to_string(),
            name: "Main Control Computer Arbiter".to_string(),
            category: EquipmentCategory::Control,
            description: "Arbiter control computer for 2oo3 voting".to_string(),
            location: "Control Room, Rack CR-03".to_string(),
            manufacturer: "Concurrent Real-Time".to_string(),
            model: "RedHawk-RT-8000".to_string(),
            quantity: 1,
            weight_kg: 50.0,
            power_kw: 2.0,
            voltage_v: 400.0,
            cooling_kw: 3.0,
            sil_rating: Some(3),
            redundancy: "2oo3 voting".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=4 {
            equipment.push(Equipment {
                id: format!("CTRL-PLC-{:02}", i),
                name: format!("Safety PLC {}", i),
                category: EquipmentCategory::Control,
                description: "SIL3-certified PLC for safety interlock functions".to_string(),
                location: format!("Control Room, Safety Rack S-{}", i),
                manufacturer: "Siemens".to_string(),
                model: "S7-1500F".to_string(),
                quantity: 1,
                weight_kg: 10.0,
                power_kw: 0.2,
                voltage_v: 24.0,
                cooling_kw: 0.1,
                sil_rating: Some(3),
                redundancy: "2oo4 voting".to_string(),
                mtbf_hours: 200_000.0,
                health: ComponentHealth::default(),
            });
        }

        equipment.push(Equipment {
            id: "CTRL-NET-01".to_string(),
            name: "Control Network Primary".to_string(),
            category: EquipmentCategory::Control,
            description: "Deterministic Ethernet network for real-time control".to_string(),
            location: "Control Room, Network Rack".to_string(),
            manufacturer: "Cisco".to_string(),
            model: "IE-4000-RT".to_string(),
            quantity: 1,
            weight_kg: 20.0,
            power_kw: 0.5,
            voltage_v: 48.0,
            cooling_kw: 0.3,
            sil_rating: Some(2),
            redundancy: "1oo2 with hot standby".to_string(),
            mtbf_hours: 500_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "CTRL-NET-02".to_string(),
            name: "Safety Network".to_string(),
            category: EquipmentCategory::Control,
            description: "Isolated safety network (PROFIsafe)".to_string(),
            location: "Control Room, Safety Rack".to_string(),
            manufacturer: "Siemens".to_string(),
            model: "SCALANCE-X308".to_string(),
            quantity: 4,  // Redundant ring
            weight_kg: 5.0,
            power_kw: 0.1,
            voltage_v: 24.0,
            cooling_kw: 0.05,
            sil_rating: Some(3),
            redundancy: "2oo4 ring topology".to_string(),
            mtbf_hours: 300_000.0,
            health: ComponentHealth::default(),
        });

        // ========== SAFETY SYSTEMS ==========
        equipment.push(Equipment {
            id: "SAFE-QDS-01".to_string(),
            name: "Quench Detection System".to_string(),
            category: EquipmentCategory::Safety,
            description: "Fast quench detection using voltage/current monitoring".to_string(),
            location: "Throughout Magnet System".to_string(),
            manufacturer: "CERN".to_string(),
            model: "QDS-SC-TS1".to_string(),
            quantity: 1,  // System-wide
            weight_kg: 100.0,
            power_kw: 1.0,
            voltage_v: 24.0,
            cooling_kw: 0.5,
            sil_rating: Some(4),
            redundancy: "2oo4 per coil".to_string(),
            mtbf_hours: 200_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "SAFE-DU-01".to_string(),
            name: "Energy Dump System".to_string(),
            category: EquipmentCategory::Safety,
            description: "Fast discharge resistors for magnet energy extraction".to_string(),
            location: "Energy Dump Building".to_string(),
            manufacturer: "Post Glover".to_string(),
            model: "RES-ED-TS1".to_string(),
            quantity: 24,  // One per coil circuit
            weight_kg: 2_000.0,  // Per resistor
            power_kw: 0.0,  // Passive
            voltage_v: 0.0,
            cooling_kw: 0.0,  // Natural convection
            sil_rating: Some(4),
            redundancy: "Parallel paths".to_string(),
            mtbf_hours: f64::INFINITY,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "SAFE-FAST-01".to_string(),
            name: "Fast Plasma Shutdown System".to_string(),
            category: EquipmentCategory::Safety,
            description: "Massive gas injection system for controlled plasma termination".to_string(),
            location: "Tokamak Hall, Multiple Ports".to_string(),
            manufacturer: "JET/UKAEA".to_string(),
            model: "MGI-TS1".to_string(),
            quantity: 6,  // Multiple injectors
            weight_kg: 500.0,
            power_kw: 0.5,
            voltage_v: 24.0,
            cooling_kw: 0.0,
            sil_rating: Some(3),
            redundancy: "3oo6".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "SAFE-RAD-01".to_string(),
            name: "Radiation Monitoring System".to_string(),
            category: EquipmentCategory::Safety,
            description: "Neutron and gamma monitoring throughout facility".to_string(),
            location: "Facility-wide".to_string(),
            manufacturer: "Mirion".to_string(),
            model: "RMS-TS1-100".to_string(),
            quantity: 100,  // Sensors
            weight_kg: 1.0,  // Per sensor
            power_kw: 0.01,
            voltage_v: 24.0,
            cooling_kw: 0.0,
            sil_rating: Some(2),
            redundancy: "Multiple overlapping coverage".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "SAFE-TRIT-01".to_string(),
            name: "Tritium Monitoring System".to_string(),
            category: EquipmentCategory::Safety,
            description: "Continuous tritium air monitoring".to_string(),
            location: "Tritium areas".to_string(),
            manufacturer: "Overhoff".to_string(),
            model: "TRAM-3".to_string(),
            quantity: 20,
            weight_kg: 30.0,
            power_kw: 0.1,
            voltage_v: 120.0,
            cooling_kw: 0.05,
            sil_rating: Some(2),
            redundancy: "Multiple redundant monitors".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        // ========== COOLING SYSTEMS ==========
        equipment.push(Equipment {
            id: "COOL-CT-01".to_string(),
            name: "Primary Cooling Tower".to_string(),
            category: EquipmentCategory::Cooling,
            description: "Evaporative cooling tower for heat rejection".to_string(),
            location: "Cooling Tower Area".to_string(),
            manufacturer: "SPX Cooling".to_string(),
            model: "CT-500MW".to_string(),
            quantity: 4,
            weight_kg: 100_000.0,
            power_kw: 500.0,  // Fans
            voltage_v: 6600.0,
            cooling_kw: 0.0,  // Provides cooling
            sil_rating: None,
            redundancy: "N+1".to_string(),
            mtbf_hours: 80_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "COOL-HX-01".to_string(),
            name: "Component Cooling Heat Exchanger".to_string(),
            category: EquipmentCategory::Cooling,
            description: "Plate heat exchanger for component cooling water".to_string(),
            location: "Cooling Building".to_string(),
            manufacturer: "Alfa Laval".to_string(),
            model: "M20-FM-TS1".to_string(),
            quantity: 4,
            weight_kg: 5_000.0,
            power_kw: 0.0,  // Passive
            voltage_v: 0.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "N+1".to_string(),
            mtbf_hours: 200_000.0,
            health: ComponentHealth::default(),
        });

        for i in 1..=6 {
            equipment.push(Equipment {
                id: format!("COOL-PUMP-{:02}", i),
                name: format!("Primary Cooling Pump {}", i),
                category: EquipmentCategory::Cooling,
                description: "High-capacity centrifugal pump for cooling water circulation".to_string(),
                location: "Cooling Building, Pump Room".to_string(),
                manufacturer: "KSB".to_string(),
                model: "RDLO-250-400".to_string(),
                quantity: 1,
                weight_kg: 3_000.0,
                power_kw: 500.0,
                voltage_v: 6600.0,
                cooling_kw: 0.0,
                sil_rating: None,
                redundancy: "4oo6".to_string(),
                mtbf_hours: 60_000.0,
                health: ComponentHealth::default(),
            });
        }

        // ========== AUXILIARY SYSTEMS ==========
        equipment.push(Equipment {
            id: "AUX-FUEL-01".to_string(),
            name: "Fuel Injection System".to_string(),
            category: EquipmentCategory::Auxiliary,
            description: "Pellet injector for fuel replenishment".to_string(),
            location: "Tokamak Hall, Port FI-1".to_string(),
            manufacturer: "ORNL".to_string(),
            model: "PI-TS1-30Hz".to_string(),
            quantity: 1,
            weight_kg: 2_000.0,
            power_kw: 50.0,
            voltage_v: 400.0,
            cooling_kw: 20.0,
            sil_rating: None,
            redundancy: "1oo2".to_string(),
            mtbf_hours: 20_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "AUX-GAS-01".to_string(),
            name: "Gas Injection System".to_string(),
            category: EquipmentCategory::Auxiliary,
            description: "Piezo valve gas injection for fueling and impurity seeding".to_string(),
            location: "Tokamak Hall, Multiple Ports".to_string(),
            manufacturer: "MKS".to_string(),
            model: "GI-TS1-Multi".to_string(),
            quantity: 1,
            weight_kg: 500.0,
            power_kw: 5.0,
            voltage_v: 24.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "Multiple valves per circuit".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "AUX-RH-01".to_string(),
            name: "Remote Handling System".to_string(),
            category: EquipmentCategory::Auxiliary,
            description: "Robotic system for in-vessel maintenance".to_string(),
            location: "Remote Handling Building".to_string(),
            manufacturer: "Oxford Technologies".to_string(),
            model: "MASCOT-TS1".to_string(),
            quantity: 2,
            weight_kg: 5_000.0,
            power_kw: 100.0,
            voltage_v: 400.0,
            cooling_kw: 20.0,
            sil_rating: None,
            redundancy: "1oo2".to_string(),
            mtbf_hours: 30_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "AUX-TRIT-01".to_string(),
            name: "Tritium Processing System".to_string(),
            category: EquipmentCategory::Auxiliary,
            description: "Tritium handling, storage, and recovery system".to_string(),
            location: "Tritium Building".to_string(),
            manufacturer: "AECL".to_string(),
            model: "TPS-TS1".to_string(),
            quantity: 1,
            weight_kg: 50_000.0,
            power_kw: 200.0,
            voltage_v: 400.0,
            cooling_kw: 100.0,
            sil_rating: Some(2),
            redundancy: "Double containment".to_string(),
            mtbf_hours: 80_000.0,
            health: ComponentHealth::default(),
        });

        // ========== INFRASTRUCTURE ==========
        equipment.push(Equipment {
            id: "INF-CRANE-01".to_string(),
            name: "Tokamak Hall Crane".to_string(),
            category: EquipmentCategory::Infrastructure,
            description: "200-tonne overhead crane for component handling".to_string(),
            location: "Tokamak Hall".to_string(),
            manufacturer: "Konecranes".to_string(),
            model: "CXT-200".to_string(),
            quantity: 1,
            weight_kg: 80_000.0,
            power_kw: 300.0,
            voltage_v: 6600.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "N/A".to_string(),
            mtbf_hours: 100_000.0,
            health: ComponentHealth::default(),
        });

        equipment.push(Equipment {
            id: "INF-HVAC-01".to_string(),
            name: "HVAC System".to_string(),
            category: EquipmentCategory::Infrastructure,
            description: "Heating, ventilation, and air conditioning".to_string(),
            location: "Facility-wide".to_string(),
            manufacturer: "Carrier".to_string(),
            model: "HVAC-TS1".to_string(),
            quantity: 1,  // System
            weight_kg: 20_000.0,
            power_kw: 2_000.0,
            voltage_v: 400.0,
            cooling_kw: 0.0,
            sil_rating: None,
            redundancy: "N+1 per zone".to_string(),
            mtbf_hours: 50_000.0,
            health: ComponentHealth::default(),
        });

        // Calculate totals
        let total_weight_tonnes = equipment.iter().map(|e| e.weight_kg * e.quantity as f64).sum::<f64>() / 1000.0;
        let total_power_mw = equipment.iter().map(|e| e.power_kw * e.quantity as f64).sum::<f64>() / 1000.0;
        let total_cooling_mw = equipment.iter().map(|e| e.cooling_kw * e.quantity as f64).sum::<f64>() / 1000.0;

        Self {
            equipment,
            total_weight_tonnes,
            total_power_mw,
            total_cooling_mw,
        }
    }

    /// Get equipment by category
    pub fn by_category(&self, category: EquipmentCategory) -> Vec<&Equipment> {
        self.equipment.iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Get all safety-critical equipment (SIL rated)
    pub fn safety_critical(&self) -> Vec<&Equipment> {
        self.equipment.iter()
            .filter(|e| e.sil_rating.is_some())
            .collect()
    }

    /// Generate inventory report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== TS-1 FACILITY EQUIPMENT INVENTORY ===\n\n");

        let categories = [
            EquipmentCategory::TokamakCore,
            EquipmentCategory::MagnetSystem,
            EquipmentCategory::PowerSupply,
            EquipmentCategory::Cryogenics,
            EquipmentCategory::Vacuum,
            EquipmentCategory::Heating,
            EquipmentCategory::Diagnostics,
            EquipmentCategory::Control,
            EquipmentCategory::Safety,
            EquipmentCategory::Cooling,
            EquipmentCategory::Auxiliary,
            EquipmentCategory::Infrastructure,
        ];

        for cat in categories {
            let items = self.by_category(cat);
            if !items.is_empty() {
                report.push_str(&format!("\n{:?} ({} items):\n", cat, items.len()));
                report.push_str(&"-".repeat(60));
                report.push_str("\n");
                for item in items {
                    report.push_str(&format!(
                        "  {} - {}\n    Qty: {}, Power: {:.1} kW, Weight: {:.0} kg\n",
                        item.id, item.name, item.quantity, item.power_kw, item.weight_kg
                    ));
                    if let Some(sil) = item.sil_rating {
                        report.push_str(&format!("    SIL: {}, Redundancy: {}\n", sil, item.redundancy));
                    }
                }
            }
        }

        report.push_str(&format!(
            "\n=== TOTALS ===\nTotal Equipment Items: {}\nTotal Weight: {:.1} tonnes\nTotal Power: {:.1} MW\nTotal Cooling: {:.1} MW\n",
            self.equipment.len(), self.total_weight_tonnes, self.total_power_mw, self.total_cooling_mw
        ));

        report
    }

    /// Count equipment by category
    pub fn count_by_category(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for item in &self.equipment {
            *counts.entry(format!("{:?}", item.category)).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_creation() {
        let inv = FacilityInventory::ts1();
        assert!(!inv.equipment.is_empty());
        assert!(inv.total_weight_tonnes > 0.0);
    }

    #[test]
    fn test_safety_critical() {
        let inv = FacilityInventory::ts1();
        let safety = inv.safety_critical();
        assert!(!safety.is_empty());
        // All should have SIL rating
        for item in safety {
            assert!(item.sil_rating.is_some());
        }
    }

    #[test]
    fn test_category_filter() {
        let inv = FacilityInventory::ts1();
        let magnets = inv.by_category(EquipmentCategory::MagnetSystem);
        assert!(!magnets.is_empty());
        // Should include TF coils
        assert!(magnets.iter().any(|e| e.id.starts_with("TF-")));
    }
}
