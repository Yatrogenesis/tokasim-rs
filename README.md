# TOKASIM-RS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18301323.svg)](https://doi.org/10.5281/zenodo.18301323)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Tokamak Fusion Reactor Hyperrealistic Simulator**

A deterministic physics engine in Rust, competing with NVIDIA Omniverse + Commonwealth Fusion Systems' SPARC digital twin.

## Key Features

- **First-principles physics**: Particle-In-Cell (PIC) simulation with 10⁹-10¹⁴ particles
- **Deterministic control**: SYNTEX/PIRS rule-based system (no ML black boxes)
- **Auditable**: Every decision is explainable for nuclear regulators
- **Zero GPU dependencies**: Runs on commodity CPUs
- **YatroNet ready**: Designed for distributed computing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOKASIM-RS                                         │
│                  Tokamak Simulator in Rust                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  LEVEL 1: PARTICLES (Particle-In-Cell, Boris pusher)                        │
│  LEVEL 2: FIELDS (Maxwell Equations - FDTD solver)                          │
│  LEVEL 3: MHD (Grad-Shafranov equilibrium, stability analysis)              │
│  LEVEL 4: NUCLEAR (D-T fusion, Bosch-Hale rates, alpha heating)             │
│  LEVEL 5: CONTROL (PIRS deterministic rules)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Comparison vs NVIDIA Omniverse

| Feature           | NVIDIA Omniverse    | TOKASIM-RS + YatroNet           |
|-------------------|---------------------|----------------------------------|
| Physics           | Approximate (RT)    | First-principles (exact)         |
| Particles         | Visual only         | 10⁹-10¹⁴ simulated              |
| MHD               | Simplified          | Complete (instabilities)         |
| Nuclear reactions | Not simulated       | Monte Carlo + cross-sections     |
| Control           | ML/black box        | SYNTEX deterministic             |
| Hardware required | RTX GPUs ($$$)      | CPUs existing ($0)              |
| Determinism       | No                  | Yes (reproducible)               |
| Auditability      | Difficult           | Total (nuclear regulators)       |

## TS-1 Design (Our Optimized Tokamak)

| Parameter              | SPARC          | TS-1           | Improvement |
|------------------------|----------------|----------------|-------------|
| Toroidal field (B_t)   | 12.2 T         | 25 T           | +105%       |
| Major radius (R)       | 1.85 m         | 1.5 m          | -19%        |
| Fusion power           | 50-100 MW      | 300-500 MW     | +400%       |
| Q factor               | ~2-3           | 10             | +400%       |
| Weight                 | ~1,000 ton     | 400 ton        | -60%        |

## Usage

```rust
use tokasim_rs::*;
use tokasim_rs::simulator::{TokamakSimulator, SimulationParams};

fn main() {
    let params = SimulationParams {
        n_particles: 100_000,
        grid_resolution: 50,
        dt: 1e-10,
        ..Default::default()
    };

    let mut sim = TokamakSimulator::ts1(params);
    sim.run(1e-6);  // Run for 1 microsecond

    println!("{}", sim.summary());
    println!("{}", sim.controller.to_pirs());  // Export control rules
}
```

## Modules

- `constants` - Physical constants (SI units)
- `types` - Core data types (Vec3, Species, TokamakConfig)
- `particle` - Particle-In-Cell: Boris pusher, collisions, Maxwellian distributions
- `field` - FDTD Maxwell solver, Poisson solver
- `mhd` - Grad-Shafranov equilibrium, stability analysis, disruption prediction
- `nuclear` - Fusion rates (Bosch-Hale), alpha heating, Monte Carlo events
- `control` - PIRS-style deterministic control, PID controllers
- `geometry` - Tokamak geometry, coordinate systems
- `simulator` - Main simulation engine

## Integration with NL-SRE

The control module is designed for integration with the NL-SRE-Semantico motor:

```
Natural Language → NL-SRE → PIRS predicates → Control Actions
```

Example:
- Input: "Aumenta la potencia si la densidad cae por debajo de 2×10²⁰"
- PIRS: `control_rule(increase_power, [conditions([less_than(Density, 2e20)])], actions([adjust_heating(ICRF, 5e6)]))`

## Citation

If you use TOKASIM-RS in your research, please cite:

```bibtex
@software{molina_burgos_2026_tokasim,
  author       = {Molina-Burgos, Francisco},
  title        = {TOKASIM-RS: Tokamak Fusion Reactor Hyperrealistic Simulator},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18301323},
  url          = {https://doi.org/10.5281/zenodo.18301323}
}
```

## Author

Francisco Molina-Burgos
Avermex Research Division
Mérida, Yucatán, México

## License

MIT
