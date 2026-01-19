//! # Generador de Reportes
//!
//! Genera reportes detallados de optimización y diseño de reactores.

use crate::optimizer::design::ReactorDesign;
use crate::optimizer::constraints::ConstraintEvaluator;
use crate::optimizer::cost_model::CostModel;
use crate::optimizer::infrastructure::InfrastructureCalculator;
use crate::optimizer::scaling_laws::ScalingLaws;

/// Generador de reportes
pub struct ReportGenerator;

impl ReportGenerator {
    /// Genera reporte completo de un diseño
    pub fn full_report(design: &ReactorDesign) -> String {
        let mut report = String::new();

        report.push_str(&Self::header(design));
        report.push_str(&Self::plasma_parameters(design));
        report.push_str(&Self::geometry_section(design));
        report.push_str(&Self::magnetic_system(design));
        report.push_str(&Self::heating_systems(design));
        report.push_str(&Self::performance_metrics(design));
        report.push_str(&Self::constraint_analysis(design));
        report.push_str(&Self::cost_analysis(design));
        report.push_str(&Self::infrastructure_section(design));

        report
    }

    fn header(design: &ReactorDesign) -> String {
        format!(
            r#"
================================================================================
                    TOKASIM-RS REACTOR DESIGN REPORT
================================================================================
Design ID: {}
Generation: {}
Feasible: {}
================================================================================

"#,
            design.id,
            design.generation,
            if design.feasible { "YES" } else { "NO" }
        )
    }

    fn plasma_parameters(design: &ReactorDesign) -> String {
        format!(
            r#"PLASMA PARAMETERS
-----------------
  Density (n):           {:.2e} m⁻³
  Ion Temperature:       {:.2} keV
  Electron Temperature:  {:.2} keV
  Plasma Current:        {:.2} MA
  Beta Normalized:       {:.3}
  q95:                   {:.2}
  Greenwald Fraction:    {:.1}%

"#,
            design.density,
            design.ion_temperature_kev,
            design.electron_temperature_kev,
            design.plasma_current_ma,
            design.beta_n,
            design.q95(),
            design.greenwald_fraction() * 100.0
        )
    }

    fn geometry_section(design: &ReactorDesign) -> String {
        format!(
            r#"GEOMETRY
--------
  Major Radius (R):      {:.2} m
  Minor Radius (a):      {:.2} m
  Aspect Ratio (A):      {:.2}
  Elongation (κ):        {:.2}
  Triangularity (δ):     {:.2}
  Plasma Volume:         {:.1} m³
  Plasma Surface:        {:.1} m²

"#,
            design.major_radius,
            design.minor_radius,
            design.aspect_ratio(),
            design.elongation,
            design.triangularity,
            design.plasma_volume(),
            design.plasma_surface()
        )
    }

    fn magnetic_system(design: &ReactorDesign) -> String {
        let b_max = design.max_field_at_conductor();
        let b_limit = design.magnet_technology.max_field();

        format!(
            r#"MAGNETIC SYSTEM
---------------
  Toroidal Field (B₀):   {:.2} T
  Max Field at Coil:     {:.2} T
  Technology:            {:?}
  Field Limit:           {:.1} T
  Field Margin:          {:.1}%
  TF Radial Build:       {:.2} m
  Cryostat Margin:       {:.2} m

"#,
            design.toroidal_field,
            b_max,
            design.magnet_technology,
            b_limit,
            (1.0 - b_max / b_limit) * 100.0,
            design.tf_coil_radial_build,
            design.cryostat_margin
        )
    }

    fn heating_systems(design: &ReactorDesign) -> String {
        format!(
            r#"HEATING SYSTEMS
---------------
  ICRF Power:            {:.1} MW
  ECRH Power:            {:.1} MW
  NBI Power:             {:.1} MW
  Total Heating:         {:.1} MW

"#,
            design.icrf_power_mw,
            design.ecrh_power_mw,
            design.nbi_power_mw,
            design.total_heating_power()
        )
    }

    fn performance_metrics(design: &ReactorDesign) -> String {
        let p_fusion = ScalingLaws::fusion_power_mw(design);
        let q = ScalingLaws::q_factor(design);
        let tau_e = ScalingLaws::confinement_time_ipb98(design);
        let triple = ScalingLaws::triple_product(design);
        let wall_load = ScalingLaws::neutron_wall_load(design);

        format!(
            r#"PERFORMANCE METRICS
-------------------
  Fusion Power:          {:.1} MW
  Q Factor:              {:.2}
  Confinement Time:      {:.3} s
  Triple Product:        {:.2e} m⁻³·keV·s
  Neutron Wall Load:     {:.2} MW/m²
  Net Electric Power:    {:.1} MW (est.)

"#,
            p_fusion,
            q,
            tau_e,
            triple,
            wall_load,
            p_fusion * 0.33 - design.total_heating_power() * 0.5
        )
    }

    fn constraint_analysis(design: &ReactorDesign) -> String {
        let evaluator = ConstraintEvaluator::new();
        let result = evaluator.evaluate(design);

        let mut section = String::from(
            r#"CONSTRAINT ANALYSIS
-------------------
"#
        );

        // Margins
        section.push_str("  Margins:\n");
        for (name, margin) in &result.margins {
            let status = if *margin > 0.2 {
                "OK"
            } else if *margin > 0.0 {
                "MARGINAL"
            } else {
                "VIOLATED"
            };
            section.push_str(&format!("    {:<20} {:>6.1}%  [{}]\n", name, margin * 100.0, status));
        }

        // Violations
        if !result.violations.is_empty() {
            section.push_str("\n  Violations:\n");
            for v in &result.violations {
                section.push_str(&format!("    - {:?}\n", v));
            }
        }

        // Warnings
        if !result.warnings.is_empty() {
            section.push_str("\n  Warnings:\n");
            for w in &result.warnings {
                section.push_str(&format!("    - {:?}\n", w));
            }
        }

        section.push('\n');
        section
    }

    fn cost_analysis(design: &ReactorDesign) -> String {
        let model = CostModel::default();
        let capex = model.estimate_capex(design);
        let opex = model.estimate_opex(design);
        let lcoe = model.calculate_lcoe(design);
        let breakdown = model.cost_breakdown(design);

        format!(
            r#"COST ANALYSIS
-------------
  CAPEX:                 ${:.2}B
  OPEX (annual):         ${:.0}M/year
  LCOE:                  ${:.2}/MWh
  Construction Time:     {:.1} years

  Cost Breakdown:
    Magnets:             ${:.2}B ({:.1}%)
    Vacuum Vessel:       ${:.2}B ({:.1}%)
    Blanket:             ${:.0}M ({:.1}%)
    Heating Systems:     ${:.0}M ({:.1}%)
    Cryogenics:          ${:.0}M ({:.1}%)
    Power Supplies:      ${:.0}M ({:.1}%)
    Buildings:           ${:.0}M ({:.1}%)
    Engineering:         ${:.2}B ({:.1}%)
    Contingency:         ${:.2}B ({:.1}%)

"#,
            capex / 1e9,
            opex / 1e6,
            lcoe,
            model.construction_time(design),
            breakdown.magnets / 1e9, breakdown.magnets / capex * 100.0,
            breakdown.vacuum_vessel / 1e9, breakdown.vacuum_vessel / capex * 100.0,
            breakdown.blanket / 1e6, breakdown.blanket / capex * 100.0,
            breakdown.heating / 1e6, breakdown.heating / capex * 100.0,
            breakdown.cryogenics / 1e6, breakdown.cryogenics / capex * 100.0,
            breakdown.power_supplies / 1e6, breakdown.power_supplies / capex * 100.0,
            breakdown.buildings / 1e6, breakdown.buildings / capex * 100.0,
            breakdown.engineering / 1e9, breakdown.engineering / capex * 100.0,
            breakdown.contingency / 1e9, breakdown.contingency / capex * 100.0
        )
    }

    fn infrastructure_section(design: &ReactorDesign) -> String {
        let calc = InfrastructureCalculator::new();
        let spec = calc.calculate_all(design);

        format!(
            r#"INFRASTRUCTURE
--------------
  Cryostat:
    Outer Radius:        {:.2} m
    Height:              {:.2} m

  Tokamak Building:
    Diameter:            {:.1} m
    Height:              {:.1} m
    Footprint:           {:.0} m²

  Support Facilities:
    Hot Cell:            {:.0} m²
    Control Room:        {:.0} m²
    Cryo Plant:          {:.0} m²
    Electrical:          {:.0} m²

  Site:
    Total Area:          {:.0} m² ({:.2} hectares)
    Crane Capacity:      {:.0} tons

================================================================================
"#,
            spec.cryostat_radius,
            spec.cryostat_height,
            spec.building_diameter,
            spec.building_height,
            calc.tokamak_building_footprint(design),
            spec.hot_cell_area,
            spec.control_room_area,
            spec.cryo_plant_area,
            spec.electrical_area,
            spec.total_site_area,
            spec.total_site_area / 10000.0,
            spec.crane_capacity_tons
        )
    }

    /// Genera reporte de comparación entre múltiples diseños
    pub fn comparison_report(designs: &[ReactorDesign]) -> String {
        let mut report = String::from(
            r#"
================================================================================
                    DESIGN COMPARISON REPORT
================================================================================

"#
        );

        // Header row
        report.push_str(&format!("{:<20}", "Parameter"));
        for (i, _d) in designs.iter().enumerate() {
            report.push_str(&format!("{:>15}", format!("Design {}", i + 1)));
        }
        report.push('\n');
        report.push_str(&"-".repeat(20 + designs.len() * 15));
        report.push('\n');

        // Parameters
        let params: Vec<(&str, Box<dyn Fn(&ReactorDesign) -> f64>)> = vec![
            ("R (m)", Box::new(|d: &ReactorDesign| d.major_radius)),
            ("a (m)", Box::new(|d: &ReactorDesign| d.minor_radius)),
            ("B₀ (T)", Box::new(|d: &ReactorDesign| d.toroidal_field)),
            ("Ip (MA)", Box::new(|d: &ReactorDesign| d.plasma_current_ma)),
            ("n (10²⁰/m³)", Box::new(|d: &ReactorDesign| d.density / 1e20)),
            ("Ti (keV)", Box::new(|d: &ReactorDesign| d.ion_temperature_kev)),
            ("P_fus (MW)", Box::new(|d: &ReactorDesign| ScalingLaws::fusion_power_mw(d))),
            ("Q", Box::new(|d: &ReactorDesign| ScalingLaws::q_factor(d))),
            ("CAPEX ($B)", Box::new(|d: &ReactorDesign| CostModel::default().estimate_capex(d) / 1e9)),
            ("LCOE ($/MWh)", Box::new(|d: &ReactorDesign| CostModel::default().calculate_lcoe(d))),
        ];

        for (name, func) in params {
            report.push_str(&format!("{:<20}", name));
            for d in designs {
                let value = func(d);
                if value.is_finite() {
                    report.push_str(&format!("{:>15.2}", value));
                } else {
                    report.push_str(&format!("{:>15}", "N/A"));
                }
            }
            report.push('\n');
        }

        report.push_str("\n================================================================================\n");
        report
    }

    /// Genera reporte del frente de Pareto
    pub fn pareto_report(pareto_front: &[ReactorDesign]) -> String {
        let mut report = String::from(
            r#"
================================================================================
                    PARETO FRONT ANALYSIS
================================================================================

"#
        );

        report.push_str(&format!("Total solutions in Pareto front: {}\n\n", pareto_front.len()));

        // Summary statistics
        if !pareto_front.is_empty() {
            let q_values: Vec<f64> = pareto_front.iter()
                .map(|d| ScalingLaws::q_factor(d))
                .collect();
            let capex_values: Vec<f64> = pareto_front.iter()
                .map(|d| CostModel::default().estimate_capex(d))
                .collect();

            let q_min = q_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let q_max = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let capex_min = capex_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let capex_max = capex_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            report.push_str("Objective Ranges:\n");
            report.push_str(&format!("  Q:     {:.2} - {:.2}\n", q_min, q_max));
            report.push_str(&format!("  CAPEX: ${:.2}B - ${:.2}B\n\n", capex_min / 1e9, capex_max / 1e9));
        }

        // List top solutions
        report.push_str("Top Solutions by Q:\n");
        report.push_str("-".repeat(80).as_str());
        report.push('\n');

        let mut sorted: Vec<_> = pareto_front.iter().collect();
        sorted.sort_by(|a, b| {
            ScalingLaws::q_factor(b)
                .partial_cmp(&ScalingLaws::q_factor(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, design) in sorted.iter().take(10).enumerate() {
            let q = ScalingLaws::q_factor(design);
            let capex = CostModel::default().estimate_capex(design);
            report.push_str(&format!(
                "{:>3}. ID: {:>10}  Q={:>6.2}  CAPEX=${:>6.2}B  R={:.2}m  B={:.1}T\n",
                i + 1,
                &design.id[..design.id.len().min(10)],
                q,
                capex / 1e9,
                design.major_radius,
                design.toroidal_field
            ));
        }

        report.push_str("\n================================================================================\n");
        report
    }
}

/// Exportador a diferentes formatos
pub struct ReportExporter;

impl ReportExporter {
    /// Exporta diseños a CSV
    pub fn to_csv(designs: &[ReactorDesign]) -> String {
        let mut csv = String::from(
            "id,R,a,B0,Ip,n,Ti,Te,kappa,delta,P_fus,Q,CAPEX,LCOE,feasible\n"
        );

        for d in designs {
            let p_fus = ScalingLaws::fusion_power_mw(d);
            let q = ScalingLaws::q_factor(d);
            let model = CostModel::default();
            let capex = model.estimate_capex(d);
            let lcoe = model.calculate_lcoe(d);

            csv.push_str(&format!(
                "{},{:.3},{:.3},{:.2},{:.2},{:.2e},{:.2},{:.2},{:.3},{:.3},{:.2},{:.3},{:.2e},{:.2},{}\n",
                d.id,
                d.major_radius,
                d.minor_radius,
                d.toroidal_field,
                d.plasma_current_ma,
                d.density,
                d.ion_temperature_kev,
                d.electron_temperature_kev,
                d.elongation,
                d.triangularity,
                p_fus,
                q,
                capex,
                lcoe,
                d.feasible
            ));
        }

        csv
    }

    /// Exporta a JSON
    pub fn to_json(designs: &[ReactorDesign]) -> String {
        let mut json = String::from("[\n");

        for (i, d) in designs.iter().enumerate() {
            let p_fus = ScalingLaws::fusion_power_mw(d);
            let q = ScalingLaws::q_factor(d);
            let model = CostModel::default();
            let capex = model.estimate_capex(d);
            let lcoe = model.calculate_lcoe(d);

            json.push_str(&format!(
                r#"  {{
    "id": "{}",
    "major_radius": {:.3},
    "minor_radius": {:.3},
    "toroidal_field": {:.2},
    "plasma_current_ma": {:.2},
    "density": {:.2e},
    "ion_temperature_kev": {:.2},
    "electron_temperature_kev": {:.2},
    "elongation": {:.3},
    "triangularity": {:.3},
    "fusion_power_mw": {:.2},
    "q_factor": {:.3},
    "capex_usd": {:.2e},
    "lcoe_usd_mwh": {:.2},
    "feasible": {}
  }}"#,
                d.id,
                d.major_radius,
                d.minor_radius,
                d.toroidal_field,
                d.plasma_current_ma,
                d.density,
                d.ion_temperature_kev,
                d.electron_temperature_kev,
                d.elongation,
                d.triangularity,
                p_fus,
                q,
                capex,
                lcoe,
                d.feasible
            ));

            if i < designs.len() - 1 {
                json.push_str(",\n");
            } else {
                json.push('\n');
            }
        }

        json.push_str("]\n");
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_report() {
        let design = ReactorDesign::default();
        let report = ReportGenerator::full_report(&design);

        println!("{}", report);
        assert!(report.contains("TOKASIM-RS"));
        assert!(report.contains("PLASMA PARAMETERS"));
    }

    #[test]
    fn test_csv_export() {
        let designs = vec![ReactorDesign::default()];
        let csv = ReportExporter::to_csv(&designs);

        println!("{}", csv);
        assert!(csv.contains("id,R,a"));
    }
}
