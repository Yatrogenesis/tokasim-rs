# TOKASIM-RS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18301323.svg)](https://doi.org/10.5281/zenodo.18301323)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Rapid Conceptual Design Engine for Tokamak Fusion Reactors**

A deterministic design exploration tool in Rust for rapid iteration through tokamak parameter space using empirical scaling laws.

## What This Tool IS

- **Pre-conceptual design tool**: Explore 10,000+ design configurations quickly
- **Scaling law calculator**: Uses validated empirical correlations (IPB98(y,2))
- **Deterministic control prototyping**: PIRS rule-based system for control logic design
- **Educational platform**: Understand tokamak physics relationships
- **Parameter sensitivity analyzer**: Quickly see how design choices affect performance

## What This Tool is NOT

- **NOT a digital twin**: Cannot predict actual plasma behavior in real-time
- **NOT first-principles simulation**: Uses empirical scaling laws, not FEM/MHD solvers
- **NOT engineering-grade**: Use COMSOL, ANSYS, or ITER IMAS for detailed engineering
- **NOT a neutronics code**: No ENDF/B cross-section database (use OpenMC/MCNP for that)
- **NOT comparable to Omniverse**: Different purpose entirely

## Physics Model (Honest Assessment)

### What We Calculate

| Module | Method | Accuracy |
|--------|--------|----------|
| Confinement time | IPB98(y,2) scaling law | ~20-30% for conventional tokamaks |
| Fusion power | Bosch-Hale σv parameterization | ~5% for D-T at relevant temperatures |
| Beta limits | Troyon scaling | Order of magnitude |
| Bootstrap current | Sauter fit | Approximate |
| Geometry | Analytic D-shape parametrization | Exact for idealized shape |

### What We DON'T Calculate

- Turbulent transport (use GENE, GYRO, or GS2)
- Edge physics and pedestal (use SOLPS, UEDGE)
- Disruption dynamics (use JOREK, NIMROD)
- Detailed neutronics (use OpenMC, MCNP, Serpent)
- Structural analysis (use ANSYS, COMSOL)
- Real MHD instabilities (use M3D-C1, NIMROD)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOKASIM-RS                                         │
│              Rapid Conceptual Design Engine                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  LEVEL 1: GEOMETRY - Analytic tokamak shape (D-shape parametrization)       │
│  LEVEL 2: SCALING LAWS - IPB98, Troyon, Greenwald limits                    │
│  LEVEL 3: POWER BALANCE - 0-D energy balance with heating/losses            │
│  LEVEL 4: FUSION - Bosch-Hale rates, alpha heating fraction                 │
│  LEVEL 5: CONTROL - PIRS deterministic rule prototyping                     │
│  LEVEL 6: OPTIMIZER - Genetic algorithm for design space exploration        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Appropriate Use Cases

1. **Scoping studies**: "What if we increase B_t to 25T?"
2. **Trade-off analysis**: "How does aspect ratio affect Q?"
3. **Control logic prototyping**: "Design SCRAM conditions before implementing in real PCS"
4. **Teaching**: "Show students how tokamak parameters relate"
5. **Quick feasibility checks**: "Is this design even in the ballpark?"

## TS-1 Reference Design

Our optimized conceptual design (NOT a validated engineering design):

| Parameter              | Value          | Source/Basis |
|------------------------|----------------|--------------|
| Major radius (R₀)      | 1.50 m         | Optimizer output |
| Minor radius (a)       | 0.60 m         | Optimizer output |
| Toroidal field (B_t)   | 25.0 T         | HTS REBCO limit |
| Plasma current (I_p)   | 12.0 MA        | q95 constraint |
| Elongation (κ)         | 1.97           | Stability limit |
| Triangularity (δ)      | 0.54           | Optimizer output |
| Estimated Q            | ~10            | IPB98 scaling (uncertain) |

**CAVEAT**: These are scaling-law extrapolations. Real performance requires validated simulations with ITER-grade codes.

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

## Benchmarks

Performance on commodity hardware (no GPU):

- **Throughput**: 1.72×10⁷ particle-steps/second
- **PIRS latency**: 2.11 μs average (deterministic control decisions)
- **Test coverage**: 142/142 tests passing

## The PIRS Control System

The deterministic control module is the most valuable part of this tool. Unlike ML-based control:

- **100% explainable**: Every decision has a traceable rule
- **100% deterministic**: Same input = same output, always
- **NRC-auditable**: Regulators can inspect every rule
- **Sub-millisecond latency**: 2.11 μs average response

```
Rule: emergency_shutdown_disruption (priority: 100)
  IF DisruptionRisk > 0.9 THEN emergency_shutdown()

Rule: vertical_stability (priority: 90)
  IF |VerticalPosition| > 0.05m THEN activate_VS_coils()
```

## Limitations (Read This)

1. **Scaling laws extrapolate poorly** beyond their training range (ITER-like devices)
2. **No turbulence**: We assume profiles, don't calculate them
3. **No edge physics**: Pedestal is parameterized, not predicted
4. **Simplified neutronics**: Blanket TBR is a fixed assumption
5. **No structural analysis**: We don't calculate stresses in magnets/vessel
6. **No transients**: Steady-state assumptions throughout
7. **Particle simulation is illustrative**: Not statistically significant for real physics

## When to Use Something Else

| Need | Use Instead |
|------|-------------|
| Detailed MHD stability | JOREK, NIMROD, M3D-C1 |
| Turbulent transport | GENE, GYRO, GS2, TGLF |
| Edge and SOL physics | SOLPS, UEDGE, EMC3-EIRENE |
| Neutronics | OpenMC, MCNP, Serpent |
| Structural analysis | ANSYS, COMSOL |
| Integrated modeling | IMAS/OMAS (ITER standard) |
| Production PCS | Real PCS vendors |

## Citation

```bibtex
@software{molina_burgos_2026_tokasim,
  author       = {Molina-Burgos, Francisco},
  title        = {TOKASIM-RS: Rapid Conceptual Design Engine for Tokamak Fusion Reactors},
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
