//! # Monte Carlo Neutron Transport Module
//!
//! MCNP-style particle transport for fusion neutronics analysis.
//!
//! ## Theory
//!
//! The neutron transport equation (Boltzmann):
//!
//! ```text
//! Ω·∇ψ(r,E,Ω) + Σ_t(r,E)ψ(r,E,Ω) = ∫∫ Σ_s(r,E'→E,Ω'→Ω)ψ(r,E',Ω')dE'dΩ' + S(r,E,Ω)
//! ```
//!
//! Monte Carlo solves this stochastically by tracking individual particle histories.
//!
//! ## Key Features
//!
//! - Continuous-energy cross sections (ENDF/B-VIII.0 format)
//! - CSG geometry with toroidal primitives
//! - Variance reduction (weight windows, implicit capture, splitting/roulette)
//! - Criticality (k-eigenvalue) calculations
//! - Tallies: flux, current, heating, reaction rates
//!
//! ## References
//!
//! - X-5 Monte Carlo Team, "MCNP - A General Monte Carlo N-Particle Transport Code"
//! - ENDF/B-VIII.0: "Evaluated Nuclear Data File"
//! - Lux & Koblinger, "Monte Carlo Particle Transport Methods"
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//! January 2026

use crate::types::Vec3;
use crate::stochastic::RandomGenerator;

// ============================================================================
// PHYSICAL CONSTANTS FOR NEUTRONICS
// ============================================================================

/// Neutron mass (kg)
pub const NEUTRON_MASS: f64 = 1.674927471e-27;

/// Speed of light (m/s)
pub const C: f64 = 2.99792458e8;

/// Conversion: 1 MeV to Joules
pub const MEV_TO_JOULES: f64 = 1.602176634e-13;

/// Conversion: 1 barn to m²
pub const BARN_TO_M2: f64 = 1e-28;

/// D-T fusion neutron energy (MeV)
pub const DT_NEUTRON_ENERGY: f64 = 14.1;

/// Thermal neutron energy (eV)
pub const THERMAL_ENERGY: f64 = 0.0253;

/// Avogadro's number
pub const AVOGADRO: f64 = 6.02214076e23;

// ============================================================================
// NUCLEAR DATA STRUCTURES
// ============================================================================

/// Isotope identifier (Z*1000 + A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Isotope {
    /// Atomic number Z
    pub z: u32,
    /// Mass number A
    pub a: u32,
}

impl Isotope {
    pub fn new(z: u32, a: u32) -> Self {
        Self { z, a }
    }

    /// ZAID format (ZZAAA)
    pub fn zaid(&self) -> u32 {
        self.z * 1000 + self.a
    }

    /// Common isotopes
    pub fn h1() -> Self { Self::new(1, 1) }
    pub fn h2() -> Self { Self::new(1, 2) }   // Deuterium
    pub fn h3() -> Self { Self::new(1, 3) }   // Tritium
    pub fn he4() -> Self { Self::new(2, 4) }
    pub fn li6() -> Self { Self::new(3, 6) }
    pub fn li7() -> Self { Self::new(3, 7) }
    pub fn be9() -> Self { Self::new(4, 9) }
    pub fn c12() -> Self { Self::new(6, 12) }
    pub fn o16() -> Self { Self::new(8, 16) }
    pub fn fe56() -> Self { Self::new(26, 56) }
    pub fn ni58() -> Self { Self::new(28, 58) }
    pub fn w184() -> Self { Self::new(74, 184) }
    pub fn pb208() -> Self { Self::new(82, 208) }
    pub fn u235() -> Self { Self::new(92, 235) }
    pub fn u238() -> Self { Self::new(92, 238) }
}

/// Cross section data for an isotope at a specific energy
///
/// Cross sections in barns (1 barn = 10⁻²⁴ cm²)
#[derive(Debug, Clone)]
pub struct CrossSection {
    /// Energy (MeV)
    pub energy: f64,
    /// Total cross section (barns)
    pub sigma_t: f64,
    /// Elastic scattering (barns)
    pub sigma_el: f64,
    /// Inelastic scattering (barns)
    pub sigma_inel: f64,
    /// Radiative capture (n,γ) (barns)
    pub sigma_gamma: f64,
    /// (n,2n) reaction (barns)
    pub sigma_n2n: f64,
    /// (n,α) reaction (barns)
    pub sigma_nalpha: f64,
    /// (n,p) reaction (barns)
    pub sigma_np: f64,
    /// Fission (barns) - only for fissile isotopes
    pub sigma_f: f64,
    /// Average neutrons per fission (ν)
    pub nu_bar: f64,
}

impl CrossSection {
    /// Create cross section with only total and scatter
    pub fn simple(energy: f64, sigma_t: f64, sigma_el: f64, sigma_gamma: f64) -> Self {
        Self {
            energy,
            sigma_t,
            sigma_el,
            sigma_inel: 0.0,
            sigma_gamma,
            sigma_n2n: 0.0,
            sigma_nalpha: 0.0,
            sigma_np: 0.0,
            sigma_f: 0.0,
            nu_bar: 0.0,
        }
    }

    /// Absorption cross section (all non-scattering reactions)
    pub fn sigma_a(&self) -> f64 {
        self.sigma_gamma + self.sigma_nalpha + self.sigma_np + self.sigma_f
    }

    /// Total scattering
    pub fn sigma_s(&self) -> f64 {
        self.sigma_el + self.sigma_inel + self.sigma_n2n
    }
}

/// Cross section table for an isotope (energy-dependent)
#[derive(Debug, Clone)]
pub struct CrossSectionTable {
    pub isotope: Isotope,
    /// Atomic mass (amu)
    pub atomic_mass: f64,
    /// Cross sections at various energies (sorted by energy)
    pub data: Vec<CrossSection>,
}

impl CrossSectionTable {
    /// Interpolate cross section at given energy (log-log interpolation)
    pub fn at_energy(&self, e: f64) -> CrossSection {
        if self.data.is_empty() {
            return CrossSection::simple(e, 1.0, 0.5, 0.5);
        }

        // Find bracketing energies
        let mut i_low = 0;
        let mut i_high = self.data.len() - 1;

        if e <= self.data[0].energy {
            return self.data[0].clone();
        }
        if e >= self.data[i_high].energy {
            return self.data[i_high].clone();
        }

        // Binary search
        while i_high - i_low > 1 {
            let mid = (i_low + i_high) / 2;
            if e < self.data[mid].energy {
                i_high = mid;
            } else {
                i_low = mid;
            }
        }

        // Log-log interpolation
        let e1 = self.data[i_low].energy;
        let e2 = self.data[i_high].energy;
        let f = (e.ln() - e1.ln()) / (e2.ln() - e1.ln());

        let xs1 = &self.data[i_low];
        let xs2 = &self.data[i_high];

        let interp = |s1: f64, s2: f64| -> f64 {
            if s1 <= 0.0 || s2 <= 0.0 {
                s1 + f * (s2 - s1)
            } else {
                (s1.ln() + f * (s2.ln() - s1.ln())).exp()
            }
        };

        CrossSection {
            energy: e,
            sigma_t: interp(xs1.sigma_t, xs2.sigma_t),
            sigma_el: interp(xs1.sigma_el, xs2.sigma_el),
            sigma_inel: interp(xs1.sigma_inel, xs2.sigma_inel),
            sigma_gamma: interp(xs1.sigma_gamma, xs2.sigma_gamma),
            sigma_n2n: interp(xs1.sigma_n2n, xs2.sigma_n2n),
            sigma_nalpha: interp(xs1.sigma_nalpha, xs2.sigma_nalpha),
            sigma_np: interp(xs1.sigma_np, xs2.sigma_np),
            sigma_f: interp(xs1.sigma_f, xs2.sigma_f),
            nu_bar: xs1.nu_bar + f * (xs2.nu_bar - xs1.nu_bar),
        }
    }
}

// ============================================================================
// NUCLEAR DATA LIBRARY (Simplified ENDF/B-VIII.0)
// ============================================================================

/// Nuclear data library with cross sections for common fusion materials
pub struct NuclearDataLibrary {
    tables: std::collections::HashMap<u32, CrossSectionTable>,
}

impl NuclearDataLibrary {
    /// Create library with fusion-relevant isotopes
    pub fn fusion_library() -> Self {
        let mut tables = std::collections::HashMap::new();

        // Hydrogen-1 - water coolant
        tables.insert(Isotope::h1().zaid(), Self::hydrogen_xs());

        // Deuterium (H-2) - fuel
        tables.insert(Isotope::h2().zaid(), Self::deuterium_xs());

        // Tritium (H-3) - fuel
        tables.insert(Isotope::h3().zaid(), Self::tritium_xs());

        // Lithium-6 - breeding blanket
        tables.insert(Isotope::li6().zaid(), Self::li6_xs());

        // Lithium-7 - breeding blanket
        tables.insert(Isotope::li7().zaid(), Self::li7_xs());

        // Beryllium-9 - neutron multiplier
        tables.insert(Isotope::be9().zaid(), Self::be9_xs());

        // Oxygen-16 - water coolant
        tables.insert(Isotope::o16().zaid(), Self::o16_xs());

        // Iron-56 - structural
        tables.insert(Isotope::fe56().zaid(), Self::fe56_xs());

        // Tungsten-184 - plasma facing
        tables.insert(Isotope::w184().zaid(), Self::w184_xs());

        // Lead-208 - coolant/shielding
        tables.insert(Isotope::pb208().zaid(), Self::pb208_xs());

        Self { tables }
    }

    /// Get cross section table for isotope
    pub fn get(&self, isotope: &Isotope) -> Option<&CrossSectionTable> {
        self.tables.get(&isotope.zaid())
    }

    // Cross section data (simplified from ENDF/B-VIII.0)
    // Real implementation would read from ACE files

    fn hydrogen_xs() -> CrossSectionTable {
        CrossSectionTable {
            isotope: Isotope::h1(),
            atomic_mass: 1.008,
            data: vec![
                CrossSection::simple(1e-8, 20.5, 20.4, 0.33),
                CrossSection::simple(1e-5, 20.5, 20.4, 0.33),
                CrossSection::simple(0.001, 20.4, 20.4, 0.01),
                CrossSection::simple(0.1, 14.0, 14.0, 0.001),
                CrossSection::simple(1.0, 4.3, 4.3, 0.0001),
                CrossSection::simple(14.1, 0.7, 0.7, 0.00001),
            ],
        }
    }

    fn deuterium_xs() -> CrossSectionTable {
        CrossSectionTable {
            isotope: Isotope::h2(),
            atomic_mass: 2.014,
            data: vec![
                CrossSection::simple(1e-8, 3.4, 3.4, 0.0005),
                CrossSection::simple(1e-5, 3.4, 3.4, 0.0005),
                CrossSection::simple(0.001, 3.4, 3.4, 0.0004),
                CrossSection::simple(0.1, 3.2, 3.2, 0.0001),
                CrossSection::simple(1.0, 2.5, 2.5, 0.00001),
                CrossSection::simple(14.1, 0.8, 0.7, 0.00001),
            ],
        }
    }

    fn tritium_xs() -> CrossSectionTable {
        CrossSectionTable {
            isotope: Isotope::h3(),
            atomic_mass: 3.016,
            data: vec![
                CrossSection::simple(1e-8, 1.7, 1.5, 0.0),
                CrossSection::simple(0.001, 1.7, 1.5, 0.0),
                CrossSection::simple(0.1, 1.6, 1.5, 0.0),
                CrossSection::simple(1.0, 1.5, 1.4, 0.0),
                CrossSection::simple(14.1, 1.0, 0.9, 0.0),
            ],
        }
    }

    fn li6_xs() -> CrossSectionTable {
        // Li-6 + n → T + He-4 (tritium breeding)
        CrossSectionTable {
            isotope: Isotope::li6(),
            atomic_mass: 6.015,
            data: vec![
                CrossSection {
                    energy: 1e-8,
                    sigma_t: 950.0,
                    sigma_el: 0.7,
                    sigma_inel: 0.0,
                    sigma_gamma: 0.039,
                    sigma_n2n: 0.0,
                    sigma_nalpha: 940.0,  // (n,t) = (n,α) equivalent
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
                CrossSection {
                    energy: 0.001,
                    sigma_t: 30.0,
                    sigma_el: 0.7,
                    sigma_inel: 0.0,
                    sigma_gamma: 0.039,
                    sigma_n2n: 0.0,
                    sigma_nalpha: 29.0,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
                CrossSection {
                    energy: 0.24,  // Resonance
                    sigma_t: 5.0,
                    sigma_el: 2.0,
                    sigma_inel: 0.0,
                    sigma_gamma: 0.01,
                    sigma_n2n: 0.0,
                    sigma_nalpha: 2.9,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
                CrossSection {
                    energy: 14.1,
                    sigma_t: 1.5,
                    sigma_el: 0.8,
                    sigma_inel: 0.3,
                    sigma_gamma: 0.001,
                    sigma_n2n: 0.0,
                    sigma_nalpha: 0.02,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }

    fn li7_xs() -> CrossSectionTable {
        // Li-7 + n → T + He-4 + n' (threshold ~2.5 MeV)
        CrossSectionTable {
            isotope: Isotope::li7(),
            atomic_mass: 7.016,
            data: vec![
                CrossSection::simple(1e-8, 1.1, 1.0, 0.045),
                CrossSection::simple(0.001, 1.1, 1.0, 0.045),
                CrossSection::simple(1.0, 1.5, 1.3, 0.01),
                CrossSection {
                    energy: 14.1,
                    sigma_t: 1.8,
                    sigma_el: 0.6,
                    sigma_inel: 0.4,
                    sigma_gamma: 0.001,
                    sigma_n2n: 0.3,  // (n,n'α) produces tritium
                    sigma_nalpha: 0.0,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }

    fn be9_xs() -> CrossSectionTable {
        // Be-9 + n → 2n + 2He-4 (neutron multiplier)
        CrossSectionTable {
            isotope: Isotope::be9(),
            atomic_mass: 9.012,
            data: vec![
                CrossSection::simple(1e-8, 6.2, 6.1, 0.0076),
                CrossSection::simple(0.001, 6.2, 6.1, 0.0076),
                CrossSection::simple(1.0, 3.0, 2.5, 0.001),
                CrossSection {
                    energy: 3.0,  // (n,2n) threshold ~1.85 MeV
                    sigma_t: 2.5,
                    sigma_el: 1.5,
                    sigma_inel: 0.3,
                    sigma_gamma: 0.001,
                    sigma_n2n: 0.5,
                    sigma_nalpha: 0.0,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
                CrossSection {
                    energy: 14.1,
                    sigma_t: 1.8,
                    sigma_el: 0.8,
                    sigma_inel: 0.2,
                    sigma_gamma: 0.0001,
                    sigma_n2n: 0.55,  // Good multiplier at 14 MeV
                    sigma_nalpha: 0.1,
                    sigma_np: 0.0,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }

    fn o16_xs() -> CrossSectionTable {
        // Oxygen-16 - water coolant
        CrossSectionTable {
            isotope: Isotope::o16(),
            atomic_mass: 15.999,
            data: vec![
                CrossSection::simple(1e-8, 3.9, 3.9, 0.00019),
                CrossSection::simple(0.001, 3.9, 3.9, 0.00019),
                CrossSection::simple(0.1, 3.8, 3.8, 0.0001),
                CrossSection::simple(1.0, 2.8, 2.8, 0.00005),
                CrossSection::simple(14.1, 1.7, 1.2, 0.00001),
            ],
        }
    }

    fn fe56_xs() -> CrossSectionTable {
        CrossSectionTable {
            isotope: Isotope::fe56(),
            atomic_mass: 55.845,
            data: vec![
                CrossSection::simple(1e-8, 14.0, 11.6, 2.59),
                CrossSection::simple(0.001, 14.0, 11.6, 2.59),
                CrossSection::simple(0.1, 12.0, 10.0, 0.1),
                CrossSection::simple(1.0, 4.0, 3.0, 0.01),
                CrossSection {
                    energy: 14.1,
                    sigma_t: 2.8,
                    sigma_el: 1.2,
                    sigma_inel: 1.0,
                    sigma_gamma: 0.001,
                    sigma_n2n: 0.4,
                    sigma_nalpha: 0.01,
                    sigma_np: 0.05,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }

    fn w184_xs() -> CrossSectionTable {
        // Tungsten - plasma facing component
        CrossSectionTable {
            isotope: Isotope::w184(),
            atomic_mass: 183.84,
            data: vec![
                CrossSection::simple(1e-8, 23.0, 5.0, 18.0),
                CrossSection::simple(0.001, 23.0, 5.0, 10.0),
                CrossSection::simple(0.1, 15.0, 8.0, 2.0),
                CrossSection::simple(1.0, 8.0, 5.0, 0.5),
                CrossSection {
                    energy: 14.1,
                    sigma_t: 5.5,
                    sigma_el: 2.5,
                    sigma_inel: 2.0,
                    sigma_gamma: 0.01,
                    sigma_n2n: 0.8,
                    sigma_nalpha: 0.01,
                    sigma_np: 0.01,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }

    fn pb208_xs() -> CrossSectionTable {
        // Lead - coolant/shielding
        CrossSectionTable {
            isotope: Isotope::pb208(),
            atomic_mass: 207.2,
            data: vec![
                CrossSection::simple(1e-8, 11.1, 11.1, 0.00048),
                CrossSection::simple(0.001, 11.1, 11.1, 0.00048),
                CrossSection::simple(0.1, 11.0, 10.5, 0.01),
                CrossSection::simple(1.0, 7.0, 5.0, 0.005),
                CrossSection {
                    energy: 14.1,
                    sigma_t: 5.5,
                    sigma_el: 2.0,
                    sigma_inel: 2.5,
                    sigma_gamma: 0.001,
                    sigma_n2n: 0.9,  // Good multiplier
                    sigma_nalpha: 0.001,
                    sigma_np: 0.001,
                    sigma_f: 0.0,
                    nu_bar: 0.0,
                },
            ],
        }
    }
}

// ============================================================================
// MATERIAL COMPOSITION
// ============================================================================

/// Material composition for neutronics
#[derive(Debug, Clone)]
pub struct NeutronicMaterial {
    pub name: String,
    /// Atom density (atoms/barn-cm)
    pub atom_density: f64,
    /// Isotope fractions (isotope, atom fraction)
    pub composition: Vec<(Isotope, f64)>,
}

impl NeutronicMaterial {
    /// Calculate macroscopic cross section (1/cm)
    pub fn macro_xs(&self, library: &NuclearDataLibrary, energy: f64) -> MacroscopicXS {
        let mut sigma_t = 0.0;
        let mut sigma_s = 0.0;
        let mut sigma_a = 0.0;
        let mut sigma_f = 0.0;
        let mut nu_sigma_f = 0.0;

        for (isotope, frac) in &self.composition {
            if let Some(table) = library.get(isotope) {
                let xs = table.at_energy(energy);
                let n_i = self.atom_density * frac;  // atoms/barn-cm

                sigma_t += n_i * xs.sigma_t;
                sigma_s += n_i * xs.sigma_s();
                sigma_a += n_i * xs.sigma_a();
                sigma_f += n_i * xs.sigma_f;
                nu_sigma_f += n_i * xs.nu_bar * xs.sigma_f;
            }
        }

        MacroscopicXS {
            sigma_t,
            sigma_s,
            sigma_a,
            sigma_f,
            nu_sigma_f,
        }
    }

    /// Common fusion materials

    /// Li4SiO4 breeding ceramic (Li-6 enriched to 90%)
    pub fn li4sio4_enriched() -> Self {
        // Li4SiO4 density ~ 2.4 g/cm³
        // Molar mass ~ 120 g/mol
        // atom density ~ 0.048 atoms/barn-cm (total)
        Self {
            name: "Li4SiO4 (90% Li-6)".into(),
            atom_density: 0.048,
            composition: vec![
                (Isotope::li6(), 0.36),   // 4 Li atoms * 0.9 enrichment / 9 total
                (Isotope::li7(), 0.04),   // 4 Li atoms * 0.1 / 9 total
                (Isotope::new(14, 28), 0.11),  // Si-28
                (Isotope::o16(), 0.49),   // 4 O atoms / 9 total
            ],
        }
    }

    /// EUROFER-97 reduced activation steel
    pub fn eurofer97() -> Self {
        // Density ~ 7.75 g/cm³
        Self {
            name: "EUROFER-97".into(),
            atom_density: 0.0847,
            composition: vec![
                (Isotope::fe56(), 0.89),
                (Isotope::new(24, 52), 0.09),  // Cr-52
                (Isotope::new(23, 51), 0.01),  // V-51
                (Isotope::new(74, 184), 0.01), // W-184
            ],
        }
    }

    /// Tungsten plasma-facing
    pub fn tungsten() -> Self {
        Self {
            name: "Tungsten".into(),
            atom_density: 0.0632,
            composition: vec![
                (Isotope::new(74, 182), 0.265),
                (Isotope::new(74, 183), 0.143),
                (Isotope::w184(), 0.307),
                (Isotope::new(74, 186), 0.285),
            ],
        }
    }

    /// Lead-lithium eutectic (Pb-17Li)
    pub fn pbli() -> Self {
        Self {
            name: "Pb-17Li".into(),
            atom_density: 0.033,
            composition: vec![
                (Isotope::pb208(), 0.83),
                (Isotope::li6(), 0.153),  // 17% Li, 90% enriched
                (Isotope::li7(), 0.017),
            ],
        }
    }

    /// Water coolant
    pub fn water() -> Self {
        Self {
            name: "H2O".into(),
            atom_density: 0.1003,
            composition: vec![
                (Isotope::h1(), 0.667),
                (Isotope::o16(), 0.333),
            ],
        }
    }

    /// Void/vacuum
    pub fn void() -> Self {
        Self {
            name: "Void".into(),
            atom_density: 0.0,
            composition: vec![],
        }
    }
}

/// Macroscopic cross sections (1/cm)
#[derive(Debug, Clone, Copy)]
pub struct MacroscopicXS {
    pub sigma_t: f64,     // Total
    pub sigma_s: f64,     // Scattering
    pub sigma_a: f64,     // Absorption
    pub sigma_f: f64,     // Fission
    pub nu_sigma_f: f64,  // ν × σ_f (fission neutron production)
}

// ============================================================================
// GEOMETRY (CSG with toroidal primitives)
// ============================================================================

/// Geometry cell for neutron transport
#[derive(Debug, Clone)]
pub struct Cell {
    pub id: usize,
    pub name: String,
    pub material: NeutronicMaterial,
    pub region: Region,
    /// Importance for variance reduction (0 = kill particle)
    pub importance: f64,
}

/// Region definition (CSG)
#[derive(Debug, Clone)]
pub enum Region {
    /// Inside a surface (negative sense)
    Inside(Surface),
    /// Outside a surface (positive sense)
    Outside(Surface),
    /// Intersection (AND)
    Intersection(Vec<Region>),
    /// Union (OR)
    Union(Vec<Region>),
    /// Complement (NOT)
    Complement(Box<Region>),
}

impl Region {
    /// Check if point is inside region
    pub fn contains(&self, p: &Vec3) -> bool {
        match self {
            Region::Inside(s) => s.sense(p) < 0.0,
            Region::Outside(s) => s.sense(p) > 0.0,
            Region::Intersection(regions) => regions.iter().all(|r| r.contains(p)),
            Region::Union(regions) => regions.iter().any(|r| r.contains(p)),
            Region::Complement(r) => !r.contains(p),
        }
    }
}

/// Surface primitive
#[derive(Debug, Clone)]
pub enum Surface {
    /// Plane: ax + by + cz = d
    Plane { normal: Vec3, d: f64 },
    /// Sphere: (x-x0)² + (y-y0)² + (z-z0)² = R²
    Sphere { center: Vec3, radius: f64 },
    /// Cylinder along Z: (x-x0)² + (y-y0)² = R²
    CylinderZ { center: (f64, f64), radius: f64 },
    /// Torus along Z: (sqrt(x² + y²) - R)² + z² = r²
    TorusZ { major_radius: f64, minor_radius: f64 },
    /// General quadric: Ax² + By² + Cz² + Dxy + Eyz + Fxz + Gx + Hy + Iz + J = 0
    Quadric { coeffs: [f64; 10] },
}

impl Surface {
    /// Evaluate surface equation (negative = inside, positive = outside)
    pub fn sense(&self, p: &Vec3) -> f64 {
        match self {
            Surface::Plane { normal, d } => {
                normal.x * p.x + normal.y * p.y + normal.z * p.z - d
            }
            Surface::Sphere { center, radius } => {
                let dx = p.x - center.x;
                let dy = p.y - center.y;
                let dz = p.z - center.z;
                dx * dx + dy * dy + dz * dz - radius * radius
            }
            Surface::CylinderZ { center, radius } => {
                let dx = p.x - center.0;
                let dy = p.y - center.1;
                dx * dx + dy * dy - radius * radius
            }
            Surface::TorusZ { major_radius, minor_radius } => {
                let rho = (p.x * p.x + p.y * p.y).sqrt();
                let d = rho - major_radius;
                d * d + p.z * p.z - minor_radius * minor_radius
            }
            Surface::Quadric { coeffs } => {
                let [a, b, c, d, e, f, g, h, i, j] = coeffs;
                a * p.x * p.x + b * p.y * p.y + c * p.z * p.z +
                d * p.x * p.y + e * p.y * p.z + f * p.x * p.z +
                g * p.x + h * p.y + i * p.z + j
            }
        }
    }

    /// Distance to surface intersection along ray (None if no intersection)
    pub fn distance(&self, pos: &Vec3, dir: &Vec3) -> Option<f64> {
        match self {
            Surface::Plane { normal, d } => {
                let denom = normal.x * dir.x + normal.y * dir.y + normal.z * dir.z;
                if denom.abs() < 1e-12 {
                    return None;
                }
                let t = (d - (normal.x * pos.x + normal.y * pos.y + normal.z * pos.z)) / denom;
                if t > 1e-8 { Some(t) } else { None }
            }
            Surface::Sphere { center, radius } => {
                let oc = Vec3::new(pos.x - center.x, pos.y - center.y, pos.z - center.z);
                let a = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
                let b = 2.0 * (oc.x * dir.x + oc.y * dir.y + oc.z * dir.z);
                let c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - radius * radius;
                Self::solve_quadratic(a, b, c)
            }
            Surface::CylinderZ { center, radius } => {
                let ox = pos.x - center.0;
                let oy = pos.y - center.1;
                let a = dir.x * dir.x + dir.y * dir.y;
                let b = 2.0 * (ox * dir.x + oy * dir.y);
                let c = ox * ox + oy * oy - radius * radius;
                Self::solve_quadratic(a, b, c)
            }
            Surface::TorusZ { major_radius, minor_radius } => {
                // Torus intersection is quartic - use iterative method
                Self::torus_distance(pos, dir, *major_radius, *minor_radius)
            }
            Surface::Quadric { coeffs } => {
                let [aa, bb, cc, dd, ee, ff, gg, hh, ii, jj] = coeffs;

                // Quadratic in t: At² + Bt + C = 0
                let a = aa * dir.x * dir.x + bb * dir.y * dir.y + cc * dir.z * dir.z +
                        dd * dir.x * dir.y + ee * dir.y * dir.z + ff * dir.x * dir.z;

                let b = 2.0 * (aa * pos.x * dir.x + bb * pos.y * dir.y + cc * pos.z * dir.z) +
                        dd * (pos.x * dir.y + pos.y * dir.x) +
                        ee * (pos.y * dir.z + pos.z * dir.y) +
                        ff * (pos.x * dir.z + pos.z * dir.x) +
                        gg * dir.x + hh * dir.y + ii * dir.z;

                let c = aa * pos.x * pos.x + bb * pos.y * pos.y + cc * pos.z * pos.z +
                        dd * pos.x * pos.y + ee * pos.y * pos.z + ff * pos.x * pos.z +
                        gg * pos.x + hh * pos.y + ii * pos.z + jj;

                Self::solve_quadratic(a, b, c)
            }
        }
    }

    fn solve_quadratic(a: f64, b: f64, c: f64) -> Option<f64> {
        if a.abs() < 1e-12 {
            // Linear
            if b.abs() < 1e-12 { return None; }
            let t = -c / b;
            return if t > 1e-8 { Some(t) } else { None };
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);

        if t1 > 1e-8 {
            Some(t1)
        } else if t2 > 1e-8 {
            Some(t2)
        } else {
            None
        }
    }

    fn torus_distance(pos: &Vec3, dir: &Vec3, r_major: f64, r_minor: f64) -> Option<f64> {
        // Newton-Raphson iteration for torus intersection
        let mut t = 0.01;
        let max_dist = 1000.0;

        for _ in 0..50 {
            let p = Vec3::new(
                pos.x + t * dir.x,
                pos.y + t * dir.y,
                pos.z + t * dir.z,
            );

            let rho = (p.x * p.x + p.y * p.y).sqrt();
            let d = rho - r_major;
            let f = d * d + p.z * p.z - r_minor * r_minor;

            if f.abs() < 1e-10 && t > 1e-8 {
                return Some(t);
            }

            // Derivative df/dt
            let drho_dt = if rho > 1e-10 {
                (p.x * dir.x + p.y * dir.y) / rho
            } else {
                0.0
            };
            let df_dt = 2.0 * d * drho_dt + 2.0 * p.z * dir.z;

            if df_dt.abs() < 1e-12 {
                t += 0.1;
                continue;
            }

            let dt = -f / df_dt;
            t += dt.clamp(-1.0, 1.0);

            if t > max_dist || t < 0.0 {
                return None;
            }
        }

        None
    }
}

// ============================================================================
// PARTICLE AND TRANSPORT
// ============================================================================

/// Neutron particle for Monte Carlo transport
#[derive(Debug, Clone)]
pub struct Neutron {
    /// Position (cm)
    pub pos: Vec3,
    /// Direction (unit vector)
    pub dir: Vec3,
    /// Energy (MeV)
    pub energy: f64,
    /// Statistical weight
    pub weight: f64,
    /// Current cell ID
    pub cell: usize,
    /// History number
    pub history: usize,
    /// Is particle alive?
    pub alive: bool,
    /// Time (shakes, 1 shake = 10⁻⁸ s)
    pub time: f64,
}

impl Neutron {
    pub fn new(pos: Vec3, dir: Vec3, energy: f64) -> Self {
        Self {
            pos,
            dir: dir.normalize(),
            energy,
            weight: 1.0,
            cell: 0,
            history: 0,
            alive: true,
            time: 0.0,
        }
    }

    /// Velocity (cm/shake)
    pub fn velocity(&self) -> f64 {
        // v = sqrt(2E/m)
        // In useful units: v (cm/shake) = 1.3831e-3 * sqrt(E_MeV)
        1.3831e-3 * self.energy.sqrt()
    }

    /// Move particle by distance
    pub fn advance(&mut self, distance: f64) {
        self.pos.x += distance * self.dir.x;
        self.pos.y += distance * self.dir.y;
        self.pos.z += distance * self.dir.z;
        self.time += distance / self.velocity();
    }

    /// Isotropic scattering in lab frame
    pub fn scatter_isotropic(&mut self, rng: &mut RandomGenerator) {
        let (dx, dy, dz) = rng.isotropic_direction();
        self.dir = Vec3::new(dx, dy, dz);
    }

    /// Elastic scattering (center of mass → lab)
    /// Returns energy after collision
    pub fn scatter_elastic(&mut self, a: f64, rng: &mut RandomGenerator) -> f64 {
        // A = target mass / neutron mass
        // mu_cm = cosine of scattering angle in CM
        let mu_cm = 2.0 * rng.uniform() - 1.0;

        // Energy after collision (lab frame)
        let alpha = ((a - 1.0) / (a + 1.0)).powi(2);
        let e_ratio = 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm);
        let e_new = self.energy * e_ratio;

        // Lab scattering angle
        let mu_lab = (1.0 + a * mu_cm) / (1.0 + a * a + 2.0 * a * mu_cm).sqrt();

        // Rotate direction
        self.rotate_direction(mu_lab, rng);
        self.energy = e_new;

        e_new
    }

    /// Rotate direction by polar angle (mu = cos(theta))
    fn rotate_direction(&mut self, mu: f64, rng: &mut RandomGenerator) {
        let phi = 2.0 * std::f64::consts::PI * rng.uniform();
        let sin_theta = (1.0 - mu * mu).sqrt();

        // Local to global rotation
        let (u, v, w) = (self.dir.x, self.dir.y, self.dir.z);

        if w.abs() < 0.999 {
            let a = (1.0 - w * w).sqrt();
            self.dir = Vec3::new(
                mu * u + sin_theta * (u * w * phi.cos() - v * phi.sin()) / a,
                mu * v + sin_theta * (v * w * phi.cos() + u * phi.sin()) / a,
                mu * w - sin_theta * phi.cos() * a,
            );
        } else {
            // Nearly vertical
            self.dir = Vec3::new(
                sin_theta * phi.cos(),
                sin_theta * phi.sin(),
                mu * w.signum(),
            );
        }

        self.dir = self.dir.normalize();
    }
}

// ============================================================================
// TALLIES
// ============================================================================

/// Tally types for scoring
#[derive(Debug, Clone)]
pub enum TallyType {
    /// Track-length flux (1/cm²)
    Flux,
    /// Surface current (1/cm²)
    Current,
    /// Collision rate (1/cm³)
    Collision,
    /// Absorption rate (1/cm³)
    Absorption,
    /// Fission rate (1/cm³)
    Fission,
    /// Nuclear heating (MeV/cm³)
    Heating,
    /// Tritium production (reactions/cm³)
    TritiumProduction,
    /// Damage (DPA - displacements per atom)
    DPA,
}

/// Tally accumulator
#[derive(Debug, Clone)]
pub struct Tally {
    pub name: String,
    pub tally_type: TallyType,
    /// Cell or surface IDs to score
    pub locations: Vec<usize>,
    /// Energy bins (MeV), empty = total only
    pub energy_bins: Vec<f64>,
    /// Accumulated scores [location][energy_bin]
    scores: Vec<Vec<f64>>,
    /// Squared scores for variance
    scores_sq: Vec<Vec<f64>>,
    /// Number of histories
    n_histories: usize,
}

impl Tally {
    pub fn new(name: &str, tally_type: TallyType, locations: Vec<usize>, energy_bins: Vec<f64>) -> Self {
        let n_loc = locations.len().max(1);
        let n_bins = if energy_bins.is_empty() { 1 } else { energy_bins.len() };

        Self {
            name: name.into(),
            tally_type,
            locations,
            energy_bins,
            scores: vec![vec![0.0; n_bins]; n_loc],
            scores_sq: vec![vec![0.0; n_bins]; n_loc],
            n_histories: 0,
        }
    }

    /// Score a contribution
    pub fn score(&mut self, location_idx: usize, energy: f64, value: f64) {
        let bin = self.find_energy_bin(energy);
        if location_idx < self.scores.len() {
            self.scores[location_idx][bin] += value;
            self.scores_sq[location_idx][bin] += value * value;
        }
    }

    fn find_energy_bin(&self, energy: f64) -> usize {
        if self.energy_bins.is_empty() {
            return 0;
        }
        for (i, &e) in self.energy_bins.iter().enumerate() {
            if energy < e {
                return i;
            }
        }
        self.energy_bins.len() - 1
    }

    /// End history (for variance calculation)
    pub fn end_history(&mut self) {
        self.n_histories += 1;
    }

    /// Get mean and relative error
    pub fn result(&self, location_idx: usize, bin: usize) -> (f64, f64) {
        if self.n_histories == 0 {
            return (0.0, 1.0);
        }

        let n = self.n_histories as f64;
        let mean = self.scores[location_idx][bin] / n;
        let mean_sq = self.scores_sq[location_idx][bin] / n;

        // Relative error
        let variance = (mean_sq - mean * mean) / n;
        let rel_err = if mean.abs() > 1e-30 {
            variance.sqrt() / mean.abs()
        } else {
            1.0
        };

        (mean, rel_err)
    }

    /// Total over all bins
    pub fn total(&self, location_idx: usize) -> f64 {
        self.scores[location_idx].iter().sum::<f64>() / (self.n_histories.max(1) as f64)
    }
}

// ============================================================================
// MONTE CARLO TRANSPORT ENGINE
// ============================================================================

/// Monte Carlo neutron transport simulation
pub struct MonteCarloTransport {
    /// Geometry cells
    pub cells: Vec<Cell>,
    /// Nuclear data library
    pub library: NuclearDataLibrary,
    /// Tallies
    pub tallies: Vec<Tally>,
    /// Random number generator
    rng: RandomGenerator,
    /// Variance reduction: weight window lower bound
    pub weight_window_lower: f64,
    /// Variance reduction: weight window upper bound
    pub weight_window_upper: f64,
    /// Variance reduction: survival weight for Russian roulette
    pub survival_weight: f64,
    /// Energy cutoff (MeV)
    pub energy_cutoff: f64,
    /// Maximum history count
    pub max_collisions: usize,
    /// Statistics
    pub stats: TransportStats,
}

/// Transport statistics
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    pub histories_run: usize,
    pub total_collisions: usize,
    pub absorptions: usize,
    pub escapes: usize,
    pub weight_cutoffs: usize,
    pub energy_cutoffs: usize,
    pub russian_roulette_kills: usize,
    pub splits: usize,
    pub tritium_produced: f64,
    pub total_heating: f64,
}

impl MonteCarloTransport {
    pub fn new(cells: Vec<Cell>) -> Self {
        Self {
            cells,
            library: NuclearDataLibrary::fusion_library(),
            tallies: Vec::new(),
            rng: RandomGenerator::new(12345),
            weight_window_lower: 0.25,
            weight_window_upper: 2.0,
            survival_weight: 0.5,
            energy_cutoff: 1e-11,  // ~thermal
            max_collisions: 10000,
            stats: TransportStats::default(),
        }
    }

    /// Find cell containing point
    pub fn find_cell(&self, pos: &Vec3) -> Option<usize> {
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.region.contains(pos) {
                return Some(i);
            }
        }
        None
    }

    /// Add tally
    pub fn add_tally(&mut self, tally: Tally) {
        self.tallies.push(tally);
    }

    /// Run transport for N histories
    pub fn run(&mut self, source: &dyn NeutronSource, n_histories: usize) {
        for h in 0..n_histories {
            let mut neutron = source.sample(&mut self.rng);
            neutron.history = h;

            if let Some(cell_id) = self.find_cell(&neutron.pos) {
                neutron.cell = cell_id;
                self.transport_particle(&mut neutron);
            }

            // End history for tallies
            for tally in &mut self.tallies {
                tally.end_history();
            }

            self.stats.histories_run += 1;
        }
    }

    /// Transport single particle until termination
    fn transport_particle(&mut self, neutron: &mut Neutron) {
        let mut collisions = 0;

        while neutron.alive && collisions < self.max_collisions {
            // Get cell data (clone material to avoid borrow issues)
            let (importance, material, macro_xs) = {
                let cell = &self.cells[neutron.cell];
                let macro_xs = cell.material.macro_xs(&self.library, neutron.energy);
                (cell.importance, cell.material.clone(), macro_xs)
            };

            // Check importance (kill in void regions)
            if importance <= 0.0 {
                neutron.alive = false;
                self.stats.escapes += 1;
                break;
            }

            if macro_xs.sigma_t <= 0.0 {
                // Vacuum - stream to boundary
                if let Some(dist) = self.distance_to_boundary(neutron) {
                    neutron.advance(dist + 1e-8);
                    if let Some(new_cell) = self.find_cell(&neutron.pos) {
                        neutron.cell = new_cell;
                    } else {
                        neutron.alive = false;
                        self.stats.escapes += 1;
                    }
                } else {
                    neutron.alive = false;
                    self.stats.escapes += 1;
                }
                continue;
            }

            // Sample distance to collision
            let dist_collision = -self.rng.uniform().ln() / macro_xs.sigma_t;

            // Distance to cell boundary
            let dist_boundary = self.distance_to_boundary(neutron);

            let (distance, boundary_cross) = match dist_boundary {
                Some(db) if db < dist_collision => (db + 1e-8, true),
                _ => (dist_collision, false),
            };

            // Score track-length tallies
            self.score_track_length(neutron, distance, &macro_xs);

            // Advance particle
            neutron.advance(distance);

            if boundary_cross {
                // Crossed boundary - find new cell
                if let Some(new_cell) = self.find_cell(&neutron.pos) {
                    // Weight window check at boundary
                    self.apply_weight_window(neutron, new_cell);
                    neutron.cell = new_cell;
                } else {
                    neutron.alive = false;
                    self.stats.escapes += 1;
                }
            } else {
                // Collision
                self.process_collision(neutron, &macro_xs, &material);
                collisions += 1;
                self.stats.total_collisions += 1;
            }

            // Energy cutoff
            if neutron.energy < self.energy_cutoff {
                neutron.alive = false;
                self.stats.energy_cutoffs += 1;
            }

            // Weight cutoff with Russian roulette
            if neutron.weight < self.weight_window_lower * 0.1 {
                if self.rng.uniform() < neutron.weight / self.survival_weight {
                    neutron.weight = self.survival_weight;
                } else {
                    neutron.alive = false;
                    self.stats.russian_roulette_kills += 1;
                }
            }
        }
    }

    fn distance_to_boundary(&self, neutron: &Neutron) -> Option<f64> {
        let mut min_dist = f64::MAX;

        // Check all surfaces in current cell
        // (Simplified - real implementation tracks surface list per cell)
        for cell in &self.cells {
            if let Some(dist) = self.region_distance(&cell.region, &neutron.pos, &neutron.dir) {
                if dist < min_dist && dist > 1e-8 {
                    min_dist = dist;
                }
            }
        }

        if min_dist < f64::MAX {
            Some(min_dist)
        } else {
            None
        }
    }

    fn region_distance(&self, region: &Region, pos: &Vec3, dir: &Vec3) -> Option<f64> {
        match region {
            Region::Inside(s) | Region::Outside(s) => s.distance(pos, dir),
            Region::Intersection(regions) => {
                regions.iter().filter_map(|r| self.region_distance(r, pos, dir)).fold(None, |acc, d| {
                    match acc {
                        None => Some(d),
                        Some(a) => Some(a.min(d)),
                    }
                })
            }
            Region::Union(regions) => {
                regions.iter().filter_map(|r| self.region_distance(r, pos, dir)).fold(None, |acc, d| {
                    match acc {
                        None => Some(d),
                        Some(a) => Some(a.min(d)),
                    }
                })
            }
            Region::Complement(r) => self.region_distance(r, pos, dir),
        }
    }

    fn score_track_length(&mut self, neutron: &Neutron, distance: f64, macro_xs: &MacroscopicXS) {
        let flux_contrib = neutron.weight * distance;

        for tally in &mut self.tallies {
            if tally.locations.is_empty() || tally.locations.contains(&neutron.cell) {
                match tally.tally_type {
                    TallyType::Flux => {
                        tally.score(0, neutron.energy, flux_contrib);
                    }
                    TallyType::Collision => {
                        tally.score(0, neutron.energy, flux_contrib * macro_xs.sigma_t);
                    }
                    TallyType::Absorption => {
                        tally.score(0, neutron.energy, flux_contrib * macro_xs.sigma_a);
                    }
                    TallyType::Heating => {
                        // KERMA approximation
                        let heating = flux_contrib * macro_xs.sigma_a * neutron.energy;
                        tally.score(0, neutron.energy, heating);
                    }
                    _ => {}
                }
            }
        }
    }

    fn process_collision(&mut self, neutron: &mut Neutron, macro_xs: &MacroscopicXS, material: &NeutronicMaterial) {
        // Implicit capture (weight reduction)
        let p_absorb = macro_xs.sigma_a / macro_xs.sigma_t;
        neutron.weight *= 1.0 - p_absorb;

        // Check for tritium production (Li-6 n,α)
        for (isotope, frac) in &material.composition {
            if isotope.z == 3 && isotope.a == 6 {
                // Li-6
                if let Some(table) = self.library.get(isotope) {
                    let xs = table.at_energy(neutron.energy);
                    let n_i = material.atom_density * frac;
                    let p_nalpha = (n_i * xs.sigma_nalpha) / macro_xs.sigma_t;
                    self.stats.tritium_produced += neutron.weight * p_nalpha;
                }
            }
        }

        // Determine collision type
        let xi = self.rng.uniform();
        let p_scatter = macro_xs.sigma_s / macro_xs.sigma_t;

        if xi < p_scatter {
            // Scattering
            // Select target isotope
            let target = self.select_target_isotope(material, neutron.energy);
            if let Some(table) = self.library.get(&target) {
                neutron.scatter_elastic(table.atomic_mass, &mut self.rng);
            } else {
                neutron.scatter_isotropic(&mut self.rng);
            }
        } else {
            // Absorption (already accounted for by weight reduction)
            self.stats.absorptions += 1;
        }
    }

    fn select_target_isotope(&mut self, material: &NeutronicMaterial, energy: f64) -> Isotope {
        let mut total_xs = 0.0;
        let mut partial_xs = Vec::new();

        for (isotope, frac) in &material.composition {
            if let Some(table) = self.library.get(isotope) {
                let xs = table.at_energy(energy);
                let contrib = material.atom_density * frac * xs.sigma_t;
                total_xs += contrib;
                partial_xs.push((*isotope, total_xs));
            }
        }

        if partial_xs.is_empty() {
            return Isotope::h1();
        }

        let xi = self.rng.uniform() * total_xs;
        for (isotope, cumulative) in partial_xs {
            if xi < cumulative {
                return isotope;
            }
        }

        material.composition.last().map(|(i, _)| *i).unwrap_or(Isotope::h1())
    }

    fn apply_weight_window(&mut self, neutron: &mut Neutron, new_cell: usize) {
        let importance_ratio = self.cells[new_cell].importance /
                              self.cells[neutron.cell].importance.max(1e-30);

        if importance_ratio > 1.0 {
            // Entering more important region - splitting
            let n_split = (importance_ratio + 0.5) as usize;
            if n_split > 1 {
                neutron.weight /= n_split as f64;
                self.stats.splits += n_split - 1;
            }
        } else if importance_ratio < 1.0 {
            // Entering less important region - Russian roulette
            if self.rng.uniform() > importance_ratio {
                neutron.alive = false;
                self.stats.russian_roulette_kills += 1;
            } else {
                neutron.weight /= importance_ratio;
            }
        }
    }

    /// Calculate k-effective (criticality)
    pub fn calculate_keff(&mut self, source: &dyn NeutronSource, n_cycles: usize,
                          n_per_cycle: usize, skip_cycles: usize) -> KeffResult {
        let mut k_values = Vec::new();

        for cycle in 0..n_cycles {
            let mut fission_neutrons = 0.0;
            let mut source_neutrons = 0.0;

            for _ in 0..n_per_cycle {
                let mut neutron = source.sample(&mut self.rng);
                source_neutrons += neutron.weight;

                if let Some(cell_id) = self.find_cell(&neutron.pos) {
                    neutron.cell = cell_id;
                    fission_neutrons += self.transport_for_keff(&mut neutron);
                }
            }

            let current_k = fission_neutrons / source_neutrons.max(1e-30);

            if cycle >= skip_cycles {
                k_values.push(current_k);
            }
        }

        // Statistics
        let n = k_values.len() as f64;
        let mean_k = k_values.iter().sum::<f64>() / n;
        let variance = k_values.iter().map(|k| (k - mean_k).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = (variance / n).sqrt();

        KeffResult {
            k_eff: mean_k,
            std_dev,
            n_cycles: k_values.len(),
        }
    }

    fn transport_for_keff(&mut self, neutron: &mut Neutron) -> f64 {
        let mut fission_neutrons = 0.0;
        let mut collisions = 0;

        while neutron.alive && collisions < self.max_collisions {
            let cell = &self.cells[neutron.cell];
            let macro_xs = cell.material.macro_xs(&self.library, neutron.energy);

            if macro_xs.sigma_t <= 0.0 {
                break;
            }

            let dist = -self.rng.uniform().ln() / macro_xs.sigma_t;
            neutron.advance(dist);

            // Score fission neutrons
            fission_neutrons += neutron.weight * macro_xs.nu_sigma_f / macro_xs.sigma_t;

            // Implicit capture
            neutron.weight *= macro_xs.sigma_s / macro_xs.sigma_t;

            if neutron.weight < 1e-6 {
                break;
            }

            // Scatter
            neutron.scatter_isotropic(&mut self.rng);
            collisions += 1;
        }

        fission_neutrons
    }

    /// Print summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Monte Carlo Transport Summary ===\n");
        s.push_str(&format!("Histories run: {}\n", self.stats.histories_run));
        s.push_str(&format!("Total collisions: {}\n", self.stats.total_collisions));
        s.push_str(&format!("Absorptions: {}\n", self.stats.absorptions));
        s.push_str(&format!("Escapes: {}\n", self.stats.escapes));
        s.push_str(&format!("Russian roulette kills: {}\n", self.stats.russian_roulette_kills));
        s.push_str(&format!("Tritium produced: {:.4e}\n", self.stats.tritium_produced));

        s.push_str("\n--- Tallies ---\n");
        for tally in &self.tallies {
            let (mean, err) = tally.result(0, 0);
            s.push_str(&format!("{}: {:.4e} ± {:.2}%\n", tally.name, mean, err * 100.0));
        }

        s
    }
}

/// Result of k-effective calculation
#[derive(Debug, Clone)]
pub struct KeffResult {
    pub k_eff: f64,
    pub std_dev: f64,
    pub n_cycles: usize,
}

// ============================================================================
// NEUTRON SOURCES
// ============================================================================

/// Trait for neutron sources
pub trait NeutronSource {
    fn sample(&self, rng: &mut RandomGenerator) -> Neutron;
}

/// Point isotropic source
pub struct PointSource {
    pub position: Vec3,
    pub energy: f64,
}

impl NeutronSource for PointSource {
    fn sample(&self, rng: &mut RandomGenerator) -> Neutron {
        let (dx, dy, dz) = rng.isotropic_direction();
        Neutron::new(self.position, Vec3::new(dx, dy, dz), self.energy)
    }
}

/// Toroidal plasma source (D-T fusion)
pub struct ToroidalPlasmaSource {
    pub major_radius: f64,
    pub minor_radius: f64,
    /// Plasma profile exponent (higher = more peaked)
    pub profile_exp: f64,
}

impl NeutronSource for ToroidalPlasmaSource {
    fn sample(&self, rng: &mut RandomGenerator) -> Neutron {
        // Sample position in torus with peaked profile
        let theta = 2.0 * std::f64::consts::PI * rng.uniform();
        let phi = 2.0 * std::f64::consts::PI * rng.uniform();

        // Radial distribution (peaked at center)
        let r_norm = rng.uniform().powf(1.0 / (self.profile_exp + 1.0));
        let r = self.minor_radius * r_norm;

        // Convert to Cartesian
        let rho = self.major_radius + r * theta.cos();
        let x = rho * phi.cos();
        let y = rho * phi.sin();
        let z = r * theta.sin();

        // Isotropic direction
        let (dx, dy, dz) = rng.isotropic_direction();

        // D-T fusion neutron energy (14.1 MeV)
        Neutron::new(Vec3::new(x, y, z), Vec3::new(dx, dy, dz), DT_NEUTRON_ENERGY)
    }
}

/// Cylindrical shell source
pub struct CylindricalShellSource {
    pub r_inner: f64,
    pub r_outer: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub energy: f64,
}

impl NeutronSource for CylindricalShellSource {
    fn sample(&self, rng: &mut RandomGenerator) -> Neutron {
        // Sample in cylindrical shell
        let r = (self.r_inner.powi(2) + rng.uniform() * (self.r_outer.powi(2) - self.r_inner.powi(2))).sqrt();
        let theta = 2.0 * std::f64::consts::PI * rng.uniform();
        let z = self.z_min + rng.uniform() * (self.z_max - self.z_min);

        let pos = Vec3::new(r * theta.cos(), r * theta.sin(), z);
        let (dx, dy, dz) = rng.isotropic_direction();

        Neutron::new(pos, Vec3::new(dx, dy, dz), self.energy)
    }
}

// ============================================================================
// TRITIUM BREEDING RATIO CALCULATION
// ============================================================================

/// Calculate Tritium Breeding Ratio (TBR)
pub struct TBRCalculator {
    transport: MonteCarloTransport,
}

impl TBRCalculator {
    pub fn new(blanket_cells: Vec<Cell>) -> Self {
        let mut transport = MonteCarloTransport::new(blanket_cells);

        // Add tritium production tally
        transport.add_tally(Tally::new(
            "Tritium Production",
            TallyType::TritiumProduction,
            vec![],  // All cells
            vec![],  // All energies
        ));

        Self { transport }
    }

    /// Calculate TBR for given source
    pub fn calculate(&mut self, source: &dyn NeutronSource, n_histories: usize) -> TBRResult {
        self.transport.run(source, n_histories);

        let tbr = self.transport.stats.tritium_produced / n_histories as f64;

        TBRResult {
            tbr,
            n_histories,
            tritium_total: self.transport.stats.tritium_produced,
        }
    }
}

/// TBR calculation result
#[derive(Debug, Clone)]
pub struct TBRResult {
    /// Tritium Breeding Ratio (must be > 1.0 for self-sustaining)
    pub tbr: f64,
    pub n_histories: usize,
    pub tritium_total: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotope() {
        let li6 = Isotope::li6();
        assert_eq!(li6.z, 3);
        assert_eq!(li6.a, 6);
        assert_eq!(li6.zaid(), 3006);
    }

    #[test]
    fn test_cross_section_interpolation() {
        let table = CrossSectionTable {
            isotope: Isotope::h1(),
            atomic_mass: 1.008,
            data: vec![
                CrossSection::simple(0.001, 20.0, 20.0, 0.0),
                CrossSection::simple(1.0, 4.0, 4.0, 0.0),
            ],
        };

        let xs = table.at_energy(0.1);
        assert!(xs.sigma_t > 4.0 && xs.sigma_t < 20.0);
    }

    #[test]
    fn test_material_macro_xs() {
        let library = NuclearDataLibrary::fusion_library();
        let water = NeutronicMaterial::water();

        let xs = water.macro_xs(&library, 1.0);
        assert!(xs.sigma_t > 0.0);
    }

    #[test]
    fn test_surface_sphere() {
        let sphere = Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 10.0,
        };

        // Point inside
        assert!(sphere.sense(&Vec3::new(5.0, 0.0, 0.0)) < 0.0);
        // Point outside
        assert!(sphere.sense(&Vec3::new(15.0, 0.0, 0.0)) > 0.0);

        // Ray intersection
        let pos = Vec3::new(-20.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let dist = sphere.distance(&pos, &dir);
        assert!(dist.is_some());
        assert!((dist.unwrap() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_neutron_elastic_scatter() {
        let mut rng = RandomGenerator::new(42);
        let mut n = Neutron::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            14.1,
        );

        // Scatter off deuterium (A=2)
        let e_new = n.scatter_elastic(2.0, &mut rng);

        // Energy should decrease
        assert!(e_new < 14.1);
        assert!(e_new > 0.0);

        // Direction should change
        let dir_mag = (n.dir.x.powi(2) + n.dir.y.powi(2) + n.dir.z.powi(2)).sqrt();
        assert!((dir_mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_point_source() {
        let mut rng = RandomGenerator::new(42);
        let source = PointSource {
            position: Vec3::new(0.0, 0.0, 0.0),
            energy: 14.1,
        };

        let n = source.sample(&mut rng);
        assert_eq!(n.energy, 14.1);
        assert_eq!(n.pos.x, 0.0);
    }

    #[test]
    fn test_toroidal_source() {
        let mut rng = RandomGenerator::new(42);
        let source = ToroidalPlasmaSource {
            major_radius: 150.0,
            minor_radius: 50.0,
            profile_exp: 2.0,
        };

        // Sample multiple neutrons
        for _ in 0..100 {
            let n = source.sample(&mut rng);
            assert_eq!(n.energy, DT_NEUTRON_ENERGY);

            // Should be within torus bounds
            let rho = (n.pos.x.powi(2) + n.pos.y.powi(2)).sqrt();
            assert!(rho > 100.0 && rho < 200.0);
        }
    }

    #[test]
    fn test_tally() {
        let mut tally = Tally::new("Test", TallyType::Flux, vec![0], vec![]);

        tally.score(0, 1.0, 10.0);
        tally.end_history();
        tally.score(0, 1.0, 12.0);
        tally.end_history();

        let (mean, _err) = tally.result(0, 0);
        assert!((mean - 11.0).abs() < 0.1);
    }

    #[test]
    fn test_simple_transport() {
        // Simple sphere geometry
        let cells = vec![
            Cell {
                id: 0,
                name: "Inner".into(),
                material: NeutronicMaterial::water(),
                region: Region::Inside(Surface::Sphere {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 50.0,
                }),
                importance: 1.0,
            },
            Cell {
                id: 1,
                name: "Outer".into(),
                material: NeutronicMaterial::void(),
                region: Region::Outside(Surface::Sphere {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 50.0,
                }),
                importance: 0.0,  // Kill escaping particles
            },
        ];

        let mut mc = MonteCarloTransport::new(cells);
        mc.add_tally(Tally::new("Flux", TallyType::Flux, vec![0], vec![]));

        let source = PointSource {
            position: Vec3::new(0.0, 0.0, 0.0),
            energy: 2.0,
        };

        mc.run(&source, 100);

        assert!(mc.stats.histories_run == 100);
        assert!(mc.stats.total_collisions > 0);
    }
}
