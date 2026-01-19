//! Genera visualización SVG del tokamak TS-1

use tokasim_rs::visualization::*;
use tokasim_rs::constants::*;

fn main() {
    println!("Generando visualización del tokamak TS-1...");

    // Crear renderizador SVG
    let mut renderer = SvgRenderer::new(1200, 900);
    renderer.set_scale(150.0); // pixels per meter
    renderer.set_offset(600.0, 450.0); // center

    // Agregar gradientes y filtros
    renderer.add_plasma_gradient("plasma_grad");
    renderer.add_field_gradient("field_grad");
    renderer.add_metal_gradient("coil_grad", Color::TOROIDAL_COIL);
    renderer.add_glow_filter("plasma_glow", Color::PLASMA_CORE);

    // Parámetros TS-1
    let r0 = TS1_MAJOR_RADIUS;
    let a = TS1_MINOR_RADIUS;
    let kappa = TS1_ELONGATION;
    let delta = TS1_TRIANGULARITY;

    // Grupo: Criostato (capa externa)
    renderer.group_start(Some("cryostat"), None, None);
    let cryo_r = r0 + a + 0.8;
    let cryo_h = a * kappa + 0.8;
    renderer.ellipse(r0, 0.0, cryo_r, cryo_h,
        &Color::CRYOSTAT.to_hex(), "#888", 3.0);
    renderer.group_end();

    // Grupo: Vacuum Vessel
    renderer.group_start(Some("vacuum_vessel"), None, None);
    let vv_r = r0 + a + 0.3;
    let vv_h = a * kappa + 0.3;
    renderer.ellipse(r0, 0.0, vv_r, vv_h,
        &Color::VACUUM_VESSEL.to_hex(), "#555", 2.0);
    renderer.group_end();

    // Grupo: First Wall
    renderer.group_start(Some("first_wall"), None, None);
    renderer.d_shape(r0, a + 0.1, kappa, delta,
        &Color::FIRST_WALL.to_hex(), "#333", 1.5);
    renderer.group_end();

    // Grupo: Superficies de flujo (plasma)
    renderer.group_start(Some("flux_surfaces"), None, None);
    draw_flux_surfaces(&mut renderer, r0, a, kappa, delta, 8);
    renderer.group_end();

    // Grupo: Núcleo del plasma (con glow)
    renderer.group_start(Some("plasma_core"), None, None);
    renderer.d_shape(r0, a * 0.3, kappa * 0.9, delta * 0.5,
        "url(#plasma_grad)", "none", 0.0);
    renderer.group_end();

    // Bobinas TF (simplificadas como rectángulos)
    renderer.group_start(Some("tf_coils"), None, None);
    let n_coils = 18;
    for i in 0..n_coils {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_coils as f64);
        // Solo mostrar las que están en el plano visible (ángulos cercanos a 0 o PI)
        if angle.cos().abs() > 0.7 {
            let coil_r = r0 + a + 0.5;
            let coil_z = (a * kappa + 0.6) * angle.sin().signum() * 0.9;
            renderer.rect(coil_r - 0.15, coil_z - 0.3, 0.3, 0.6,
                "url(#coil_grad)", "#444", 1.0);
        }
    }
    renderer.group_end();

    // Divertor (parte inferior)
    renderer.group_start(Some("divertor"), None, None);
    let div_path = format!(
        "M {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} Z",
        (r0 - a * 0.6) * 150.0 + 600.0, (a * kappa + 0.15) * 150.0 + 450.0,
        (r0 - a * 0.3) * 150.0 + 600.0, (a * kappa + 0.25) * 150.0 + 450.0,
        (r0 + a * 0.3) * 150.0 + 600.0, (a * kappa + 0.25) * 150.0 + 450.0,
        (r0 + a * 0.6) * 150.0 + 600.0, (a * kappa + 0.15) * 150.0 + 450.0,
    );
    renderer.path(&div_path, &Color::DIVERTOR.to_hex(), "#222", 1.0);
    renderer.group_end();

    // Etiquetas
    renderer.group_start(Some("labels"), None, None);
    renderer.text_anchored(r0, -a * kappa - 0.5, "TS-1 Tokamak", 24.0, "white", "middle");
    renderer.text_anchored(r0, a * kappa + 0.8, "Poloidal Cross-Section", 14.0, "#aaa", "middle");

    // Parámetros
    renderer.text(0.2, -2.5, &format!("R₀ = {:.2} m", r0), 12.0, "#ccc");
    renderer.text(0.2, -2.3, &format!("a = {:.2} m", a), 12.0, "#ccc");
    renderer.text(0.2, -2.1, &format!("κ = {:.2}", kappa), 12.0, "#ccc");
    renderer.text(0.2, -1.9, &format!("δ = {:.2}", delta), 12.0, "#ccc");
    renderer.text(0.2, -1.7, &format!("B₀ = {:.1} T", TS1_TOROIDAL_FIELD), 12.0, "#ccc");
    renderer.text(0.2, -1.5, &format!("Iₚ = {:.1} MA", TS1_PLASMA_CURRENT_MA), 12.0, "#ccc");
    renderer.text(0.2, -1.3, &format!("T = {:.1} keV", TS1_TEMPERATURE_KEV), 12.0, "#ccc");
    renderer.group_end();

    // Escala
    renderer.group_start(Some("scale"), None, None);
    renderer.line(3.5, 2.5, 4.5, 2.5, "white", 2.0);
    renderer.text_anchored(4.0, 2.7, "1 m", 12.0, "white", "middle");
    renderer.group_end();

    // Guardar SVG
    let output_path = "tokasim_ts1_visualization.svg";
    renderer.save(output_path, Color::rgb(10, 10, 30)).expect("Failed to save SVG");

    println!("Visualización guardada en: {}", output_path);
    println!();
    println!("Componentes renderizados:");
    println!("  - Criostato (capa externa)");
    println!("  - Vacuum Vessel");
    println!("  - First Wall");
    println!("  - 8 superficies de flujo magnético");
    println!("  - Núcleo del plasma con gradiente térmico");
    println!("  - Bobinas TF");
    println!("  - Divertor");
    println!("  - Etiquetas con parámetros");
}
