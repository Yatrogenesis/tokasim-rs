//! # Bevy 3D Visualization Module
//!
//! Real-time 3D visualization using Bevy game engine.
//!
//! This module provides high-fidelity 3D rendering of the tokamak,
//! including plasma dynamics, magnetic fields, and component geometry.
//!
//! ## Usage
//!
//! Enable with feature flag:
//! ```bash
//! cargo run --bin tokasim-viz --features bevy-viz --release
//! ```

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::pbr::wireframe::{WireframePlugin, WireframeConfig, Wireframe, WireframeColor};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use std::f32::consts::{PI, TAU};

use crate::types::TokamakConfig;

// ============================================================================
// COMPONENTS
// ============================================================================

#[derive(Component)]
pub struct Plasma;

#[derive(Component)]
pub struct VacuumVessel;

#[derive(Component)]
pub struct TFCoil(pub usize);

#[derive(Component)]
pub struct PFCoil(pub usize);

#[derive(Component)]
pub struct CentralSolenoid;

#[derive(Component)]
pub struct Divertor;

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub azimuth: f32,
    pub elevation: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            radius: 6.0,
            azimuth: PI / 4.0,
            elevation: PI / 5.0,
        }
    }
}

// ============================================================================
// RESOURCES
// ============================================================================

/// Bridge between tokasim-rs physics and Bevy visualization
#[derive(Resource)]
pub struct TokasimBridge {
    pub config: TokamakConfig,
    pub running: bool,
    pub time: f64,
    pub speed: f64,
    pub plasma_current: f64,
    pub electron_temp: f64,
    pub ion_temp: f64,
    pub density: f64,
    pub beta_n: f64,
    pub toroidal_field: f64,
    pub fusion_power: f64,
    pub q_factor: f64,
    pub confinement_time: f64,
}

impl TokasimBridge {
    pub fn from_config(config: TokamakConfig) -> Self {
        Self {
            plasma_current: config.plasma_current / 1e6,  // Convert to MA
            electron_temp: config.ion_temperature_kev,
            ion_temp: config.ion_temperature_kev,
            density: config.density,
            toroidal_field: config.toroidal_field,
            running: false,
            time: 0.0,
            speed: 1.0,
            beta_n: 2.8,
            fusion_power: 500.0,
            q_factor: 10.0,
            confinement_time: 3.0,
            config,
        }
    }

    pub fn ts1() -> Self {
        Self::from_config(TokamakConfig::ts1())
    }

    pub fn sparc() -> Self {
        Self::from_config(TokamakConfig::sparc())
    }

    pub fn iter() -> Self {
        Self::from_config(TokamakConfig::iter())
    }
}

#[derive(Resource)]
pub struct UiState {
    pub show_plasma: bool,
    pub show_vessel: bool,
    pub show_coils: bool,
    pub show_wireframe: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            show_plasma: true,
            show_vessel: true,
            show_coils: true,
            show_wireframe: false,
        }
    }
}

// ============================================================================
// D-SHAPED TORUS MESH GENERATOR
// ============================================================================

/// Generate a D-shaped torus mesh using parametric equations:
///   R(θ) = R₀ + a*(cos(θ) + δ*sin²(θ))
///   Z(θ) = a*κ*sin(θ)
pub fn create_d_shaped_torus(
    major_radius: f32,
    minor_radius: f32,
    elongation: f32,
    triangularity: f32,
    toroidal_segments: u32,
    poloidal_segments: u32,
) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=toroidal_segments {
        let phi = (i as f32 / toroidal_segments as f32) * TAU;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        for j in 0..=poloidal_segments {
            let theta = (j as f32 / poloidal_segments as f32) * TAU;

            // D-shape parametric equations
            let r_local = minor_radius * (theta.cos() + triangularity * theta.sin().powi(2));
            let z_local = minor_radius * elongation * theta.sin();

            // Position in 3D (Y is up, torus lies in XZ plane)
            let r_total = major_radius + r_local;
            let x = r_total * cos_phi;
            let z = r_total * sin_phi;
            let y = z_local;

            positions.push([x, y, z]);

            // Calculate normal
            let dr_dtheta = minor_radius * (-theta.sin() + 2.0 * triangularity * theta.sin() * theta.cos());
            let dz_dtheta = minor_radius * elongation * theta.cos();

            let tangent_poloidal = Vec3::new(dr_dtheta * cos_phi, dz_dtheta, dr_dtheta * sin_phi);
            let tangent_toroidal = Vec3::new(-r_total * sin_phi, 0.0, r_total * cos_phi);
            let normal = tangent_poloidal.cross(tangent_toroidal).normalize();
            normals.push([normal.x, normal.y, normal.z]);

            uvs.push([i as f32 / toroidal_segments as f32, j as f32 / poloidal_segments as f32]);
        }
    }

    for i in 0..toroidal_segments {
        for j in 0..poloidal_segments {
            let a = i * (poloidal_segments + 1) + j;
            let b = a + 1;
            let c = (i + 1) * (poloidal_segments + 1) + j;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

/// Create a simple torus (circular cross-section)
pub fn create_torus(major_radius: f32, minor_radius: f32, segments: u32, tube_segments: u32) -> Mesh {
    create_d_shaped_torus(major_radius, minor_radius, 1.0, 0.0, segments, tube_segments)
}

/// Create a D-shaped coil (single poloidal ring for TF coils)
pub fn create_d_coil_mesh(
    major_radius: f32,
    minor_radius: f32,
    elongation: f32,
    triangularity: f32,
    tube_radius: f32,
) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let path_segments = 48u32;
    let tube_segments = 12u32;

    let mut path_points: Vec<Vec3> = Vec::new();
    let mut path_tangents: Vec<Vec3> = Vec::new();

    for i in 0..=path_segments {
        let theta = (i as f32 / path_segments as f32) * TAU;
        let r_local = minor_radius * (theta.cos() + triangularity * theta.sin().powi(2));
        let z_local = minor_radius * elongation * theta.sin();
        let x = major_radius + r_local;
        let y = z_local;

        path_points.push(Vec3::new(x, y, 0.0));

        let dr = minor_radius * (-theta.sin() + 2.0 * triangularity * theta.sin() * theta.cos());
        let dz = minor_radius * elongation * theta.cos();
        path_tangents.push(Vec3::new(dr, dz, 0.0).normalize());
    }

    for i in 0..=path_segments {
        let idx = i.min(path_segments - 1) as usize;
        let center = path_points[idx];
        let tangent = path_tangents[idx];

        let up = Vec3::Y;
        let right = tangent.cross(up).normalize();
        let actual_up = right.cross(tangent).normalize();

        for j in 0..=tube_segments {
            let angle = (j as f32 / tube_segments as f32) * TAU;
            let offset = right * (angle.cos() * tube_radius) + actual_up * (angle.sin() * tube_radius);
            let pos = center + offset;

            positions.push([pos.x, pos.y, pos.z]);
            let normal = offset.normalize();
            normals.push([normal.x, normal.y, normal.z]);
            uvs.push([i as f32 / path_segments as f32, j as f32 / tube_segments as f32]);
        }
    }

    for i in 0..path_segments {
        for j in 0..tube_segments {
            let a = i * (tube_segments + 1) + j;
            let b = a + 1;
            let c = (i + 1) * (tube_segments + 1) + j;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

// ============================================================================
// BEVY APP BUILDER
// ============================================================================

/// Build and run the Bevy visualization app
pub fn run_visualization(bridge: TokasimBridge) {
    let config_name = bridge.config.name.clone();

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║     TOKASIM-RS 3D Visualization (Bevy)                             ║");
    println!("║     {} Configuration                                    ║", config_name);
    println!("║     Avermex Research Division                                      ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Parameters:");
    println!("  R₀ = {:.2} m (major radius)", bridge.config.major_radius);
    println!("  a  = {:.2} m (minor radius)", bridge.config.minor_radius);
    println!("  κ  = {:.2} (elongation)", bridge.config.elongation);
    println!("  δ  = {:.2} (triangularity)", bridge.config.triangularity);
    println!("  Bt = {:.1} T, Ip = {:.1} MA", bridge.config.toroidal_field, bridge.plasma_current);
    println!();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("TOKASIM-RS - {} Visualization", config_name),
                resolution: (1400., 900.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(WireframePlugin)
        .add_plugins(EguiPlugin)
        .insert_resource(WireframeConfig {
            global: false,
            default_color: Color::srgb(1.0, 0.6, 0.0),
        })
        .insert_resource(bridge)
        .insert_resource(UiState::default())
        .insert_resource(ClearColor(Color::srgb(0.01, 0.01, 0.03)))
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (
            camera_controller,
            simulation_update,
            plasma_animation,
            ui_system,
            keyboard_input,
        ))
        .run();
}

// ============================================================================
// SETUP
// ============================================================================

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    bridge: Res<TokasimBridge>,
) {
    let r0 = bridge.config.major_radius as f32;
    let a = bridge.config.minor_radius as f32;
    let kappa = bridge.config.elongation as f32;
    let delta = bridge.config.triangularity as f32;
    let plasma_height = 2.0 * a * kappa;
    let num_tf_coils = 16;

    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(4.0, 3.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        OrbitCamera::default(),
    ));

    // Lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 20000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -PI / 3.0, PI / 4.0, 0.0)),
        ..default()
    });

    for i in 0..4 {
        let angle = (i as f32) * TAU / 4.0;
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 800000.0,
                range: 15.0,
                ..default()
            },
            transform: Transform::from_xyz(angle.cos() * 4.0, 3.0, angle.sin() * 4.0),
            ..default()
        });
    }

    // PLASMA - D-shaped torus with high resolution and wireframe
    let plasma_mesh = create_d_shaped_torus(r0, a * 0.85, kappa, delta, 256, 128);
    let plasma_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.5, 0.2, 0.7),
        emissive: LinearRgba::new(10.0, 4.0, 0.8, 1.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(plasma_mesh),
            material: plasma_material,
            ..default()
        },
        Plasma,
        Wireframe,
        WireframeColor { color: Color::srgba(1.0, 0.8, 0.3, 0.5) },
    ));

    // FIRST WALL
    let first_wall_mesh = create_d_shaped_torus(r0, a * 0.95, kappa, delta, 64, 32);
    commands.spawn(PbrBundle {
        mesh: meshes.add(first_wall_mesh),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.55, 0.5),
            metallic: 0.8,
            perceptual_roughness: 0.4,
            ..default()
        }),
        ..default()
    });

    // VACUUM VESSEL
    let vessel_mesh = create_d_shaped_torus(r0, a * 1.1, kappa * 1.05, delta, 64, 32);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(vessel_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgba(0.5, 0.5, 0.55, 0.4),
                metallic: 0.9,
                perceptual_roughness: 0.3,
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            ..default()
        },
        VacuumVessel,
    ));

    // TF COILS
    let tf_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.45, 0.2),
        metallic: 1.0,
        perceptual_roughness: 0.2,
        ..default()
    });

    for i in 0..num_tf_coils {
        let phi = (i as f32 / num_tf_coils as f32) * TAU;
        let coil_mesh = create_d_coil_mesh(r0, a * 1.3, kappa * 1.1, delta, 0.08);

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(coil_mesh),
                material: tf_material.clone(),
                transform: Transform::from_rotation(Quat::from_rotation_y(phi)),
                ..default()
            },
            TFCoil(i),
        ));
    }

    // PF COILS
    let pf_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.4, 0.15),
        metallic: 1.0,
        perceptual_roughness: 0.25,
        ..default()
    });

    let pf_positions = [
        (r0 * 0.5, plasma_height * 0.9),
        (r0 * 0.5, -plasma_height * 0.9),
        (r0 * 1.4, plasma_height * 0.6),
        (r0 * 1.4, -plasma_height * 0.6),
        (r0 * 1.6, plasma_height * 0.2),
        (r0 * 1.6, -plasma_height * 0.2),
    ];

    for (i, (r, y)) in pf_positions.iter().enumerate() {
        let pf_mesh = create_torus(*r, 0.06, 48, 12);
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(pf_mesh),
                material: pf_material.clone(),
                transform: Transform::from_xyz(0.0, *y, 0.0),
                ..default()
            },
            PFCoil(i),
        ));
    }

    // CENTRAL SOLENOID
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cylinder::new(r0 * 0.25, plasma_height * 1.5)),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.35, 0.5),
                metallic: 0.9,
                perceptual_roughness: 0.3,
                ..default()
            }),
            ..default()
        },
        CentralSolenoid,
    ));

    // DIVERTOR
    let divertor_mesh = create_torus(r0, a * 0.3, 48, 8);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(divertor_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.4, 0.4, 0.45),
                metallic: 0.95,
                perceptual_roughness: 0.2,
                ..default()
            }),
            transform: Transform::from_xyz(0.0, -plasma_height * 0.5 - 0.1, 0.0)
                .with_scale(Vec3::new(1.0, 0.3, 1.0)),
            ..default()
        },
        Divertor,
    ));

    // CRYOSTAT
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cylinder::new(r0 * 2.0, plasma_height * 2.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgba(0.6, 0.65, 0.7, 0.1),
            metallic: 0.5,
            perceptual_roughness: 0.6,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            double_sided: true,
            ..default()
        }),
        ..default()
    });

    // Ground plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(10.0, 10.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.12),
            ..default()
        }),
        transform: Transform::from_xyz(0.0, -plasma_height * 1.2, 0.0),
        ..default()
    });

    println!("[INFO] Tokamak assembly spawned with D-shaped geometry");
}

// ============================================================================
// UPDATE SYSTEMS
// ============================================================================

fn camera_controller(
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let Ok((mut transform, mut orbit)) = query.get_single_mut() else { return };

    if mouse_button.pressed(MouseButton::Left) {
        for ev in mouse_motion.read() {
            orbit.azimuth -= ev.delta.x * 0.005;
            orbit.elevation = (orbit.elevation - ev.delta.y * 0.005)
                .clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
        }
    } else {
        mouse_motion.clear();
    }

    for ev in scroll.read() {
        orbit.radius = (orbit.radius * (1.0 - ev.y * 0.1)).clamp(2.0, 20.0);
    }

    if keys.pressed(KeyCode::ArrowLeft) { orbit.azimuth += 0.02; }
    if keys.pressed(KeyCode::ArrowRight) { orbit.azimuth -= 0.02; }
    if keys.pressed(KeyCode::ArrowUp) { orbit.elevation = (orbit.elevation + 0.02).min(PI / 2.0 - 0.1); }
    if keys.pressed(KeyCode::ArrowDown) { orbit.elevation = (orbit.elevation - 0.02).max(-PI / 2.0 + 0.1); }

    let pos = Vec3::new(
        orbit.radius * orbit.elevation.cos() * orbit.azimuth.sin(),
        orbit.radius * orbit.elevation.sin(),
        orbit.radius * orbit.elevation.cos() * orbit.azimuth.cos(),
    ) + orbit.focus;

    *transform = Transform::from_translation(pos).looking_at(orbit.focus, Vec3::Y);
}

fn simulation_update(time: Res<Time>, mut bridge: ResMut<TokasimBridge>) {
    if !bridge.running { return; }

    let dt = time.delta_seconds() as f64 * bridge.speed;
    bridge.time += dt;

    let t = bridge.time;
    bridge.plasma_current = (bridge.config.plasma_current / 1e6) + 0.5 * (t * 0.3).sin();
    bridge.electron_temp = bridge.config.ion_temperature_kev + 3.0 * (t * 0.2).sin();
    bridge.ion_temp = bridge.config.ion_temperature_kev + 2.5 * (t * 0.22).sin();
    bridge.density = bridge.config.density + bridge.config.density * 0.15 * (t * 0.15).sin();
    bridge.beta_n = 2.8 + 0.3 * (t * 0.18).sin();
    bridge.fusion_power = 500.0 + 50.0 * (t * 0.25).sin();
    bridge.q_factor = bridge.fusion_power / 50.0;
}

fn plasma_animation(
    time: Res<Time>,
    bridge: Res<TokasimBridge>,
    mut query: Query<(&mut Transform, &Handle<StandardMaterial>), With<Plasma>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (mut transform, material_handle) in &mut query {
        if !bridge.running { continue; }

        let pulse = 1.0 + 0.02 * (time.elapsed_seconds() * 10.0).sin();
        transform.scale = Vec3::splat(pulse);

        if let Some(mat) = materials.get_mut(material_handle) {
            let temp_factor = (bridge.electron_temp / 25.0).clamp(0.5, 1.5) as f32;
            mat.emissive = LinearRgba::new(10.0 * temp_factor, 4.0 * temp_factor, 0.8, 1.0);
        }
    }
}

fn keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut bridge: ResMut<TokasimBridge>,
    mut wireframe_config: ResMut<WireframeConfig>,
) {
    if keys.just_pressed(KeyCode::Space) {
        bridge.running = !bridge.running;
        println!("[INFO] Simulation {}", if bridge.running { "STARTED" } else { "PAUSED" });
    }
    if keys.just_pressed(KeyCode::KeyR) {
        bridge.time = 0.0;
        bridge.running = false;
        println!("[INFO] Simulation RESET");
    }
    if keys.just_pressed(KeyCode::KeyW) {
        wireframe_config.global = !wireframe_config.global;
        println!("[INFO] Global wireframe: {}", wireframe_config.global);
    }
    if keys.just_pressed(KeyCode::Equal) { bridge.speed = (bridge.speed * 2.0).min(100.0); }
    if keys.just_pressed(KeyCode::Minus) { bridge.speed = (bridge.speed / 2.0).max(0.01); }
}

fn ui_system(
    mut contexts: EguiContexts,
    mut bridge: ResMut<TokasimBridge>,
    mut ui_state: ResMut<UiState>,
    mut plasma_query: Query<&mut Visibility, With<Plasma>>,
) {
    // Clone config data to avoid borrow issues with closures
    let config_name = bridge.config.name.clone();
    let major_radius = bridge.config.major_radius;
    let minor_radius = bridge.config.minor_radius;
    let elongation = bridge.config.elongation;
    let triangularity = bridge.config.triangularity;

    egui::TopBottomPanel::top("top_panel").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.heading(format!("{} MONITOR", config_name.to_uppercase()));
            ui.separator();

            let btn_text = if bridge.running { "⏸ PAUSE" } else { "▶ START" };
            if ui.button(egui::RichText::new(btn_text).size(16.0)).clicked() {
                bridge.running = !bridge.running;
            }

            if ui.button("⟲ RESET").clicked() {
                bridge.time = 0.0;
                bridge.running = false;
            }

            ui.separator();
            if ui.button("-").clicked() { bridge.speed = (bridge.speed / 2.0).max(0.01); }
            ui.label(format!("{:.1}x", bridge.speed));
            if ui.button("+").clicked() { bridge.speed = (bridge.speed * 2.0).min(100.0); }

            ui.separator();
            let status = if bridge.running { "● RUNNING" } else { "○ STOPPED" };
            ui.label(egui::RichText::new(status).color(
                if bridge.running { egui::Color32::GREEN } else { egui::Color32::GRAY }
            ));
            ui.label(format!("t = {:.2}s", bridge.time));
        });
    });

    egui::SidePanel::left("status_panel").min_width(220.0).show(contexts.ctx_mut(), |ui| {
        ui.heading(format!("{} Parameters", config_name));
        ui.label(format!("R₀ = {:.2} m", major_radius));
        ui.label(format!("a = {:.2} m", minor_radius));
        ui.label(format!("κ = {:.2}", elongation));
        ui.label(format!("δ = {:.2}", triangularity));

        ui.separator();
        ui.heading("Plasma State");
        ui.label(format!("Ip = {:.1} MA", bridge.plasma_current));
        ui.label(format!("Te = {:.1} keV", bridge.electron_temp));
        ui.label(format!("Ti = {:.1} keV", bridge.ion_temp));
        ui.label(format!("n̄e = {:.2e} m⁻³", bridge.density));
        ui.label(format!("Bt = {:.1} T", bridge.toroidal_field));
        ui.label(format!("βN = {:.2} %", bridge.beta_n));

        ui.separator();
        ui.heading("Performance");
        ui.label(format!("Pfus = {:.0} MW", bridge.fusion_power));
        ui.label(format!("Q = {:.1}", bridge.q_factor));

        let q_color = if bridge.q_factor >= 10.0 { egui::Color32::GREEN }
                      else if bridge.q_factor >= 1.0 { egui::Color32::YELLOW }
                      else { egui::Color32::RED };
        ui.colored_label(q_color, if bridge.q_factor >= 10.0 { "✓ Ignition capable!" }
                                   else if bridge.q_factor >= 1.0 { "Net energy gain" }
                                   else { "Sub-breakeven" });

        ui.separator();
        ui.heading("View Options");
        if ui.checkbox(&mut ui_state.show_plasma, "Show Plasma").changed() {
            for mut vis in &mut plasma_query {
                *vis = if ui_state.show_plasma { Visibility::Visible } else { Visibility::Hidden };
            }
        }
        ui.checkbox(&mut ui_state.show_vessel, "Show Vessel");
        ui.checkbox(&mut ui_state.show_coils, "Show Coils");
    });

    egui::TopBottomPanel::bottom("bottom_panel").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.label("TOKASIM-RS | SPACE=Play/Pause, R=Reset, Mouse=Rotate, Scroll=Zoom, W=Wireframe");
        });
    });
}
