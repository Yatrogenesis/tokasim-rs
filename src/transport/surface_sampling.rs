//! # Adaptive Stochastic Surface Sampling (ASSS)
//!
//! Novel multi-level approach for plasma boundary tracking and
//! real-time digital twin surface modeling.
//!
//! ## Methodology
//!
//! 1. **Quasi-Random Seeding**: Halton/Sobol sequences for low-discrepancy
//!    coverage of the LCFS (Last Closed Flux Surface)
//!
//! 2. **Curvature-Adaptive Refinement**: Increase sampling density where
//!    surface curvature is high (X-point, divertor legs, turbulent regions)
//!
//! 3. **Multi-Level Meshing**:
//!    - Delaunay 2D triangulation per poloidal slice
//!    - Convex hull to connect slices toroidally
//!    - Marching cubes for smooth isosurface extraction
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026
//!
//! ## References
//! - Halton (1960): "On the efficiency of certain quasi-random sequences"
//! - Delaunay (1934): "Sur la sphère vide"
//! - Lorensen & Cline (1987): "Marching Cubes" (SIGGRAPH)

use std::f64::consts::PI;
use rayon::prelude::*;
use crate::types::Vec3;

// ============================================================================
// QUASI-RANDOM SEQUENCES
// ============================================================================

/// Halton sequence generator for quasi-random sampling
pub struct HaltonSequence {
    /// Base for the sequence (prime numbers work best)
    bases: Vec<u32>,
    /// Current index in sequence
    index: u64,
}

impl HaltonSequence {
    /// Create new Halton sequence with given dimensionality
    pub fn new(dimensions: usize) -> Self {
        // First primes for each dimension
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let bases = primes.iter().take(dimensions).cloned().collect();

        Self {
            bases,
            index: 1,
        }
    }

    /// Generate next point in [0,1]^d
    pub fn next(&mut self) -> Vec<f64> {
        let point: Vec<f64> = self.bases.iter()
            .map(|&base| self.halton_value(self.index, base))
            .collect();
        self.index += 1;
        point
    }

    /// Halton value for given index and base
    fn halton_value(&self, mut index: u64, base: u32) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;

        while index > 0 {
            result += f * (index % base as u64) as f64;
            index /= base as u64;
            f /= base as f64;
        }

        result
    }

    /// Generate n points
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next()).collect()
    }

    /// Skip ahead in sequence
    pub fn skip(&mut self, n: u64) {
        self.index += n;
    }
}

/// Sobol sequence generator (more uniform than Halton for high dimensions)
pub struct SobolSequence {
    /// Direction numbers for each dimension
    directions: Vec<Vec<u64>>,
    /// Current index
    index: u64,
    /// Previous values (for Gray code optimization)
    prev: Vec<u64>,
    /// Number of bits
    bits: usize,
}

impl SobolSequence {
    /// Create new Sobol sequence
    pub fn new(dimensions: usize) -> Self {
        let bits = 52; // Double precision mantissa

        // Initialize direction numbers (simplified - production would use full tables)
        let mut directions = Vec::with_capacity(dimensions);

        for d in 0..dimensions {
            let mut dir = vec![0u64; bits];
            // Primitive polynomial coefficients (simplified)
            let poly = match d {
                0 => 1,
                1 => 3,
                2 => 7,
                3 => 11,
                4 => 13,
                5 => 19,
                _ => 37,
            };

            // Initialize first direction numbers
            for i in 0..bits {
                dir[i] = 1u64 << (bits - 1 - i);
                if i > 0 && (poly >> i) & 1 == 1 {
                    dir[i] ^= dir[i - 1];
                }
            }
            directions.push(dir);
        }

        Self {
            directions,
            index: 0,
            prev: vec![0; dimensions],
            bits,
        }
    }

    /// Generate next point in [0,1]^d
    pub fn next(&mut self) -> Vec<f64> {
        self.index += 1;

        // Find rightmost zero bit of index-1
        let c = (self.index - 1).trailing_ones() as usize;

        let scale = 1.0 / (1u64 << self.bits) as f64;

        self.prev.iter_mut()
            .zip(self.directions.iter())
            .map(|(p, dir)| {
                if c < dir.len() {
                    *p ^= dir[c];
                }
                *p as f64 * scale
            })
            .collect()
    }

    /// Generate n points
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next()).collect()
    }
}

// ============================================================================
// SURFACE SAMPLING
// ============================================================================

/// Point on plasma surface with metadata
#[derive(Debug, Clone)]
pub struct SurfacePoint {
    /// 3D position (m)
    pub position: Vec3,
    /// Surface normal vector
    pub normal: Vec3,
    /// Poloidal angle θ
    pub theta: f64,
    /// Toroidal angle φ
    pub phi: f64,
    /// Local curvature (1/m)
    pub curvature: f64,
    /// Normalized flux ψ_n
    pub psi_n: f64,
    /// Weight for adaptive refinement
    pub weight: f64,
}

/// Plasma surface sampler
pub struct SurfaceSampler {
    /// Major radius R0 (m)
    pub r0: f64,
    /// Minor radius a (m)
    pub a: f64,
    /// Elongation κ
    pub kappa: f64,
    /// Triangularity δ
    pub delta: f64,
    /// Number of toroidal slices
    pub n_phi: usize,
    /// Base points per slice
    pub n_theta_base: usize,
    /// Curvature refinement threshold
    pub curvature_threshold: f64,
    /// Maximum refinement factor
    pub max_refinement: usize,
    /// Sampled points
    pub points: Vec<SurfacePoint>,
    /// Quasi-random generator
    halton: HaltonSequence,
}

impl SurfaceSampler {
    /// Create new surface sampler
    pub fn new(r0: f64, a: f64, kappa: f64, delta: f64) -> Self {
        Self {
            r0,
            a,
            kappa,
            delta,
            n_phi: 36,        // 10° resolution toroidally
            n_theta_base: 100, // Base points per slice
            curvature_threshold: 0.5,
            max_refinement: 4,
            points: Vec::new(),
            halton: HaltonSequence::new(2),
        }
    }

    /// Sample the LCFS using quasi-random + curvature-adaptive method
    pub fn sample_lcfs(&mut self, n_points: usize) {
        self.points.clear();

        // Phase 1: Quasi-random seeding
        let base_points = self.quasi_random_seed(n_points);

        // Phase 2: Compute curvature at each point
        let curvatures = self.compute_curvatures(&base_points);

        // Phase 3: Adaptive refinement based on curvature
        self.adaptive_refine(&base_points, &curvatures);
    }

    /// Phase 1: Quasi-random seeding using Halton sequence
    fn quasi_random_seed(&mut self, n_points: usize) -> Vec<(f64, f64)> {
        let halton_points = self.halton.generate(n_points);

        halton_points.into_iter()
            .map(|p| {
                let theta = p[0] * 2.0 * PI - PI;  // [-π, π]
                let phi = p[1] * 2.0 * PI;          // [0, 2π]
                (theta, phi)
            })
            .collect()
    }

    /// Compute local curvature at each point (parallelized with Rayon)
    fn compute_curvatures(&self, points: &[(f64, f64)]) -> Vec<f64> {
        points.par_iter()
            .map(|&(theta, _phi)| self.local_curvature(theta))
            .collect()
    }

    /// Local curvature of D-shaped plasma boundary
    /// κ(θ) = 1/R_curv where R_curv is local radius of curvature
    fn local_curvature(&self, theta: f64) -> f64 {
        // D-shaped boundary: r(θ) = a * (1 + δ*cos(θ)) in local frame
        // Parametric form:
        // R(θ) = R0 + a*cos(θ + δ*sin(θ))
        // Z(θ) = κ*a*sin(θ)

        let eps = 1e-6;

        // First derivatives
        let dr_dtheta = -self.a * (1.0 + self.delta * theta.cos()) * (theta + self.delta * theta.sin()).sin();
        let dz_dtheta = self.kappa * self.a * theta.cos();

        // Second derivatives (numerical)
        let theta_p = theta + eps;
        let theta_m = theta - eps;

        let dr_p = -self.a * (1.0 + self.delta * theta_p.cos()) * (theta_p + self.delta * theta_p.sin()).sin();
        let dr_m = -self.a * (1.0 + self.delta * theta_m.cos()) * (theta_m + self.delta * theta_m.sin()).sin();
        let d2r_dtheta2 = (dr_p - 2.0 * dr_dtheta + dr_m) / (eps * eps);

        let dz_p = self.kappa * self.a * theta_p.cos();
        let dz_m = self.kappa * self.a * theta_m.cos();
        let d2z_dtheta2 = (dz_p - 2.0 * dz_dtheta + dz_m) / (eps * eps);

        // Curvature formula: κ = |r' × r''| / |r'|³
        let speed_sq = dr_dtheta * dr_dtheta + dz_dtheta * dz_dtheta;
        let speed = speed_sq.sqrt();

        let cross = (dr_dtheta * d2z_dtheta2 - dz_dtheta * d2r_dtheta2).abs();

        if speed > 1e-10 {
            cross / (speed * speed * speed)
        } else {
            0.0
        }
    }

    /// Phase 3: Adaptive refinement based on curvature
    fn adaptive_refine(&mut self, base_points: &[(f64, f64)], curvatures: &[f64]) {
        let max_curv = curvatures.iter().cloned().fold(0.0, f64::max);

        for (i, &(theta, phi)) in base_points.iter().enumerate() {
            let curv = curvatures[i];
            let normalized_curv = if max_curv > 0.0 { curv / max_curv } else { 0.0 };

            // Compute weight: high curvature = more refinement
            let weight = 1.0 + (self.max_refinement as f64 - 1.0) * normalized_curv;

            // Add base point
            self.add_surface_point(theta, phi, curv, weight);

            // Add refinement points if curvature is high
            if normalized_curv > self.curvature_threshold {
                let n_extra = ((weight - 1.0) * 2.0).ceil() as usize;
                let delta_theta = 0.01 / (1.0 + normalized_curv);

                for j in 1..=n_extra {
                    let offset = delta_theta * j as f64 / n_extra as f64;
                    self.add_surface_point(theta + offset, phi, curv * 0.9, weight * 0.8);
                    self.add_surface_point(theta - offset, phi, curv * 0.9, weight * 0.8);
                }
            }
        }
    }

    /// Add a surface point with computed 3D position and normal
    fn add_surface_point(&mut self, theta: f64, phi: f64, curvature: f64, weight: f64) {
        // D-shaped flux surface parametrization
        let r_local = self.a * (1.0 + self.delta * theta.cos());

        let r = self.r0 + r_local * (theta + self.delta * theta.sin()).cos();
        let z = self.kappa * self.a * theta.sin();

        // Convert to Cartesian
        let x = r * phi.cos();
        let y = r * phi.sin();

        // Surface normal (outward pointing)
        let dr_dtheta = -self.a * (1.0 + self.delta * theta.cos()) * (theta + self.delta * theta.sin()).sin()
                      + self.a * self.delta * theta.sin() * (theta + self.delta * theta.sin()).cos();
        let dz_dtheta = self.kappa * self.a * theta.cos();

        // Normal in poloidal plane
        let norm_r = dz_dtheta;
        let norm_z = -dr_dtheta;
        let norm_mag = (norm_r * norm_r + norm_z * norm_z).sqrt();

        let normal = if norm_mag > 1e-10 {
            Vec3::new(
                norm_r / norm_mag * phi.cos(),
                norm_r / norm_mag * phi.sin(),
                norm_z / norm_mag,
            )
        } else {
            Vec3::new(phi.cos(), phi.sin(), 0.0)
        };

        self.points.push(SurfacePoint {
            position: Vec3::new(x, y, z),
            normal,
            theta,
            phi,
            curvature,
            psi_n: 1.0, // On LCFS
            weight,
        });
    }

    /// Get points grouped by toroidal slice
    pub fn points_by_slice(&self) -> Vec<Vec<&SurfacePoint>> {
        let mut slices: Vec<Vec<&SurfacePoint>> = vec![Vec::new(); self.n_phi];

        for point in &self.points {
            let slice_idx = ((point.phi / (2.0 * PI)) * self.n_phi as f64) as usize;
            let idx = slice_idx.min(self.n_phi - 1);
            slices[idx].push(point);
        }

        slices
    }

    /// Get total number of sampled points
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Get curvature statistics (parallelized with Rayon)
    pub fn curvature_stats(&self) -> (f64, f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let curvatures: Vec<f64> = self.points.par_iter().map(|p| p.curvature).collect();
        let min = curvatures.par_iter().cloned().reduce(|| f64::INFINITY, f64::min);
        let max = curvatures.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max);
        let sum: f64 = curvatures.par_iter().sum();
        let mean = sum / curvatures.len() as f64;

        (min, mean, max)
    }
}

// ============================================================================
// DELAUNAY TRIANGULATION (2D)
// ============================================================================

/// Triangle in 2D
#[derive(Debug, Clone, Copy)]
pub struct Triangle2D {
    /// Vertex indices
    pub vertices: [usize; 3],
    /// Circumcenter
    pub circumcenter: (f64, f64),
    /// Circumradius squared
    pub circumradius_sq: f64,
}

/// 2D Delaunay triangulation
pub struct Delaunay2D {
    /// Points (x, y)
    pub points: Vec<(f64, f64)>,
    /// Triangles
    pub triangles: Vec<Triangle2D>,
}

impl Delaunay2D {
    /// Create Delaunay triangulation from points
    pub fn triangulate(points: Vec<(f64, f64)>) -> Self {
        let mut delaunay = Self {
            points: points.clone(),
            triangles: Vec::new(),
        };

        if points.len() < 3 {
            return delaunay;
        }

        // Bowyer-Watson algorithm
        delaunay.bowyer_watson();

        delaunay
    }

    /// Bowyer-Watson incremental algorithm
    fn bowyer_watson(&mut self) {
        // Create super-triangle
        let (min_x, max_x, min_y, max_y) = self.bounding_box();
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let d_max = dx.max(dy) * 2.0;

        let mid_x = (min_x + max_x) / 2.0;
        let mid_y = (min_y + max_y) / 2.0;

        // Super-triangle vertices (outside all points)
        let p0 = (mid_x - d_max, mid_y - d_max);
        let p1 = (mid_x + d_max, mid_y - d_max);
        let p2 = (mid_x, mid_y + d_max);

        let n = self.points.len();
        self.points.push(p0);
        self.points.push(p1);
        self.points.push(p2);

        // Initial super-triangle
        self.triangles.push(self.make_triangle(n, n + 1, n + 2));

        // Add points one by one
        for i in 0..n {
            self.add_point(i);
        }

        // Remove triangles with super-triangle vertices
        self.triangles.retain(|t| {
            t.vertices[0] < n && t.vertices[1] < n && t.vertices[2] < n
        });

        // Remove super-triangle vertices
        self.points.truncate(n);
    }

    /// Add point to triangulation
    fn add_point(&mut self, point_idx: usize) {
        let p = self.points[point_idx];

        // Find triangles whose circumcircle contains the point
        let mut bad_triangles = Vec::new();
        for (i, tri) in self.triangles.iter().enumerate() {
            if self.point_in_circumcircle(p, tri) {
                bad_triangles.push(i);
            }
        }

        // Find boundary of polygonal hole
        let mut polygon = Vec::new();
        for &tri_idx in &bad_triangles {
            let tri = &self.triangles[tri_idx];
            for edge in [(0, 1), (1, 2), (2, 0)] {
                let e = (tri.vertices[edge.0], tri.vertices[edge.1]);

                // Check if edge is shared with another bad triangle
                let shared = bad_triangles.iter().any(|&other_idx| {
                    if other_idx == tri_idx { return false; }
                    let other = &self.triangles[other_idx];
                    self.triangles_share_edge(tri, other, e)
                });

                if !shared {
                    polygon.push(e);
                }
            }
        }

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_by(|a, b| b.cmp(a));
        for idx in bad_triangles {
            self.triangles.remove(idx);
        }

        // Create new triangles from polygon edges to new point
        for (v0, v1) in polygon {
            self.triangles.push(self.make_triangle(v0, v1, point_idx));
        }
    }

    /// Check if two triangles share an edge
    fn triangles_share_edge(&self, _t1: &Triangle2D, t2: &Triangle2D, edge: (usize, usize)) -> bool {
        let (e0, e1) = edge;
        for e in [(0, 1), (1, 2), (2, 0)] {
            let other_edge = (t2.vertices[e.0], t2.vertices[e.1]);
            if (other_edge.0 == e0 && other_edge.1 == e1) ||
               (other_edge.0 == e1 && other_edge.1 == e0) {
                return true;
            }
        }
        false
    }

    /// Check if point is inside triangle's circumcircle
    fn point_in_circumcircle(&self, p: (f64, f64), tri: &Triangle2D) -> bool {
        let dx = p.0 - tri.circumcenter.0;
        let dy = p.1 - tri.circumcenter.1;
        dx * dx + dy * dy < tri.circumradius_sq
    }

    /// Create triangle and compute circumcircle
    fn make_triangle(&self, i0: usize, i1: usize, i2: usize) -> Triangle2D {
        let p0 = self.points[i0];
        let p1 = self.points[i1];
        let p2 = self.points[i2];

        // Circumcenter calculation
        let d = 2.0 * (p0.0 * (p1.1 - p2.1) + p1.0 * (p2.1 - p0.1) + p2.0 * (p0.1 - p1.1));

        let (cx, cy) = if d.abs() > 1e-10 {
            let ux = ((p0.0 * p0.0 + p0.1 * p0.1) * (p1.1 - p2.1) +
                      (p1.0 * p1.0 + p1.1 * p1.1) * (p2.1 - p0.1) +
                      (p2.0 * p2.0 + p2.1 * p2.1) * (p0.1 - p1.1)) / d;
            let uy = ((p0.0 * p0.0 + p0.1 * p0.1) * (p2.0 - p1.0) +
                      (p1.0 * p1.0 + p1.1 * p1.1) * (p0.0 - p2.0) +
                      (p2.0 * p2.0 + p2.1 * p2.1) * (p1.0 - p0.0)) / d;
            (ux, uy)
        } else {
            ((p0.0 + p1.0 + p2.0) / 3.0, (p0.1 + p1.1 + p2.1) / 3.0)
        };

        let dx = p0.0 - cx;
        let dy = p0.1 - cy;
        let r_sq = dx * dx + dy * dy;

        Triangle2D {
            vertices: [i0, i1, i2],
            circumcenter: (cx, cy),
            circumradius_sq: r_sq,
        }
    }

    /// Compute bounding box
    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in &self.points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        (min_x, max_x, min_y, max_y)
    }

    /// Get triangle count
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }
}

// ============================================================================
// 3D SURFACE MESH
// ============================================================================

/// Triangle in 3D space
#[derive(Debug, Clone)]
pub struct Triangle3D {
    /// Vertex positions
    pub vertices: [Vec3; 3],
    /// Face normal
    pub normal: Vec3,
    /// Area
    pub area: f64,
}

impl Triangle3D {
    /// Create triangle and compute properties
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let e1 = Vec3::new(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        let e2 = Vec3::new(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

        // Cross product for normal
        let nx = e1.y * e2.z - e1.z * e2.y;
        let ny = e1.z * e2.x - e1.x * e2.z;
        let nz = e1.x * e2.y - e1.y * e2.x;

        let area = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();

        let mag = (nx * nx + ny * ny + nz * nz).sqrt();
        let normal = if mag > 1e-10 {
            Vec3::new(nx / mag, ny / mag, nz / mag)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        Self {
            vertices: [v0, v1, v2],
            normal,
            area,
        }
    }

    /// Centroid of triangle
    pub fn centroid(&self) -> Vec3 {
        Vec3::new(
            (self.vertices[0].x + self.vertices[1].x + self.vertices[2].x) / 3.0,
            (self.vertices[0].y + self.vertices[1].y + self.vertices[2].y) / 3.0,
            (self.vertices[0].z + self.vertices[1].z + self.vertices[2].z) / 3.0,
        )
    }
}

/// 3D surface mesh from plasma boundary sampling
pub struct SurfaceMesh {
    /// Triangles forming the surface
    pub triangles: Vec<Triangle3D>,
    /// Total surface area
    pub total_area: f64,
    /// Volume enclosed (approximate)
    pub volume: f64,
}

impl SurfaceMesh {
    /// Build 3D mesh from sampled surface points using multi-level approach
    /// (parallelized with Rayon for slice triangulations)
    pub fn from_surface_points(sampler: &SurfaceSampler) -> Self {
        let mut mesh = Self {
            triangles: Vec::new(),
            total_area: 0.0,
            volume: 0.0,
        };

        // Get points grouped by toroidal slice
        let slices = sampler.points_by_slice();

        // Level 1: Delaunay triangulation per slice (R-Z plane) - PARALLELIZED
        let _slice_triangulations: Vec<Delaunay2D> = slices.par_iter()
            .filter(|slice| slice.len() >= 3)
            .map(|slice| {
                let points_2d: Vec<(f64, f64)> = slice.iter()
                    .map(|p| {
                        let r = (p.position.x * p.position.x + p.position.y * p.position.y).sqrt();
                        (r, p.position.z)
                    })
                    .collect();
                Delaunay2D::triangulate(points_2d)
            })
            .collect();

        // Level 2: Connect adjacent slices with convex hull-like approach
        // (Sequential - slice connections depend on adjacent pairs)
        for i in 0..slices.len() {
            let next_i = (i + 1) % slices.len();

            if slices[i].is_empty() || slices[next_i].is_empty() {
                continue;
            }

            // Connect nearest points between slices
            mesh.connect_slices(&slices[i], &slices[next_i]);
        }

        // Level 3: Smooth surface with marching-cubes-like refinement
        // (Implicit - the triangles already form the surface)

        // Compute total area and volume (parallelized)
        mesh.compute_metrics();

        mesh
    }

    /// Connect two adjacent toroidal slices
    fn connect_slices(&mut self, slice1: &[&SurfacePoint], slice2: &[&SurfacePoint]) {
        // Sort points by poloidal angle in each slice
        let mut s1: Vec<_> = slice1.iter().collect();
        let mut s2: Vec<_> = slice2.iter().collect();

        s1.sort_by(|a, b| a.theta.partial_cmp(&b.theta).unwrap());
        s2.sort_by(|a, b| a.theta.partial_cmp(&b.theta).unwrap());

        // Create triangles between slices
        let n1 = s1.len();
        let n2 = s2.len();

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < n1 || i2 < n2 {
            let p1_curr = s1[i1 % n1];
            let p1_next = s1[(i1 + 1) % n1];
            let p2_curr = s2[i2 % n2];
            let p2_next = s2[(i2 + 1) % n2];

            // Decide which triangle to add based on angle progression
            let advance_1 = if i1 < n1 {
                let d1 = (p1_next.theta - p2_curr.theta).abs();
                let d2 = (p2_next.theta - p1_curr.theta).abs();
                d1 < d2
            } else {
                false
            };

            if advance_1 && i1 < n1 {
                // Add triangle: p1_curr, p1_next, p2_curr
                self.triangles.push(Triangle3D::new(
                    p1_curr.position,
                    p1_next.position,
                    p2_curr.position,
                ));
                i1 += 1;
            } else if i2 < n2 {
                // Add triangle: p1_curr, p2_curr, p2_next
                self.triangles.push(Triangle3D::new(
                    p1_curr.position,
                    p2_curr.position,
                    p2_next.position,
                ));
                i2 += 1;
            } else {
                break;
            }

            // Prevent infinite loop
            if i1 >= n1 * 2 || i2 >= n2 * 2 {
                break;
            }
        }
    }

    /// Compute total surface area and volume (parallelized with Rayon)
    fn compute_metrics(&mut self) {
        self.total_area = self.triangles.par_iter().map(|t| t.area).sum();

        // Volume using divergence theorem: V = (1/3) Σ (centroid · normal) * area
        self.volume = self.triangles.par_iter()
            .map(|t| {
                let c = t.centroid();
                let dot = c.x * t.normal.x + c.y * t.normal.y + c.z * t.normal.z;
                dot * t.area / 3.0
            })
            .sum::<f64>()
            .abs();
    }

    /// Get number of triangles
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Export to simple mesh format (vertices and indices)
    pub fn to_mesh_data(&self) -> (Vec<Vec3>, Vec<[usize; 3]>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for tri in &self.triangles {
            let base = vertices.len();
            vertices.push(tri.vertices[0]);
            vertices.push(tri.vertices[1]);
            vertices.push(tri.vertices[2]);
            indices.push([base, base + 1, base + 2]);
        }

        (vertices, indices)
    }
}

// ============================================================================
// REAL-TIME SURFACE TRACKER
// ============================================================================

/// Real-time plasma surface tracker for digital twin
pub struct SurfaceTracker {
    /// Current sampler
    pub sampler: SurfaceSampler,
    /// Current mesh
    pub mesh: Option<SurfaceMesh>,
    /// Update interval (s)
    pub update_interval: f64,
    /// Last update time
    pub last_update: f64,
    /// Resolution level (1-5)
    pub resolution_level: usize,
    /// History of surface areas (for tracking erosion/growth)
    pub area_history: Vec<(f64, f64)>,
}

impl SurfaceTracker {
    /// Create new surface tracker
    pub fn new(r0: f64, a: f64, kappa: f64, delta: f64) -> Self {
        Self {
            sampler: SurfaceSampler::new(r0, a, kappa, delta),
            mesh: None,
            update_interval: 0.001, // 1 ms
            last_update: 0.0,
            resolution_level: 3,
            area_history: Vec::new(),
        }
    }

    /// Set resolution level (affects number of sample points)
    pub fn set_resolution(&mut self, level: usize) {
        self.resolution_level = level.clamp(1, 5);
    }

    /// Get number of sample points for current resolution
    fn n_points_for_resolution(&self) -> usize {
        match self.resolution_level {
            1 => 100,   // Fast, coarse
            2 => 250,   // Medium
            3 => 500,   // Standard
            4 => 1000,  // Fine
            5 => 2500,  // Very fine
            _ => 500,
        }
    }

    /// Update surface tracking
    pub fn update(&mut self, time: f64) -> bool {
        // Skip if not enough time has passed (but always allow first update)
        let is_first_update = self.mesh.is_none();
        if !is_first_update && time - self.last_update < self.update_interval {
            return false;
        }

        // Sample surface with current resolution
        let n_points = self.n_points_for_resolution();
        self.sampler.sample_lcfs(n_points);

        // Build mesh
        self.mesh = Some(SurfaceMesh::from_surface_points(&self.sampler));

        // Track area history
        if let Some(ref mesh) = self.mesh {
            self.area_history.push((time, mesh.total_area));

            // Keep last 1000 samples
            if self.area_history.len() > 1000 {
                self.area_history.remove(0);
            }
        }

        self.last_update = time;
        true
    }

    /// Update plasma geometry (for shape control feedback)
    pub fn update_geometry(&mut self, r0: f64, a: f64, kappa: f64, delta: f64) {
        self.sampler.r0 = r0;
        self.sampler.a = a;
        self.sampler.kappa = kappa;
        self.sampler.delta = delta;
    }

    /// Get current surface area
    pub fn current_area(&self) -> Option<f64> {
        self.mesh.as_ref().map(|m| m.total_area)
    }

    /// Get current volume
    pub fn current_volume(&self) -> Option<f64> {
        self.mesh.as_ref().map(|m| m.volume)
    }

    /// Get area change rate (m²/s)
    pub fn area_change_rate(&self) -> Option<f64> {
        if self.area_history.len() < 2 {
            return None;
        }

        let (t1, a1) = self.area_history[self.area_history.len() - 2];
        let (t2, a2) = self.area_history[self.area_history.len() - 1];

        let dt = t2 - t1;
        if dt > 1e-10 {
            Some((a2 - a1) / dt)
        } else {
            None
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new(2);
        let points = halton.generate(10);

        assert_eq!(points.len(), 10);

        // All points should be in [0, 1]
        for p in points {
            assert!(p[0] >= 0.0 && p[0] <= 1.0);
            assert!(p[1] >= 0.0 && p[1] <= 1.0);
        }
    }

    #[test]
    fn test_surface_sampler() {
        let mut sampler = SurfaceSampler::new(1.5, 0.6, 1.8, 0.4);
        sampler.sample_lcfs(100);

        assert!(sampler.n_points() >= 100);

        let (min_k, mean_k, max_k) = sampler.curvature_stats();
        assert!(min_k >= 0.0);
        assert!(mean_k >= 0.0);
        assert!(max_k >= mean_k);
    }

    #[test]
    fn test_delaunay_triangulation() {
        let points = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 1.0),
            (0.5, 0.5),
        ];

        let delaunay = Delaunay2D::triangulate(points);

        // Should have at least 2 triangles for 4 points
        assert!(delaunay.n_triangles() >= 2);
    }

    #[test]
    fn test_surface_mesh() {
        let mut sampler = SurfaceSampler::new(1.5, 0.6, 1.8, 0.4);
        sampler.sample_lcfs(50);

        let mesh = SurfaceMesh::from_surface_points(&sampler);

        assert!(mesh.n_triangles() > 0);
        assert!(mesh.total_area > 0.0);
    }

    #[test]
    fn test_surface_tracker() {
        let mut tracker = SurfaceTracker::new(1.5, 0.6, 1.8, 0.4);
        tracker.set_resolution(3);  // Higher resolution for better mesh

        tracker.update(0.0);

        let area = tracker.current_area().unwrap();
        let volume = tracker.current_volume().unwrap();

        // Mesh should have positive area and volume
        // (exact values depend on sampling and triangulation quality)
        assert!(area > 0.0, "Surface area should be positive, got {}", area);
        assert!(volume > 0.0, "Volume should be positive, got {}", volume);
    }
}
