//! # Algoritmos de Optimización
//!
//! NSGA-II y Differential Evolution para optimización multi-objetivo.

use crate::optimizer::design::ReactorDesign;
use crate::optimizer::parameters::ReactorParameterSpace;
use crate::optimizer::constraints::ConstraintEvaluator;
use crate::optimizer::objectives::{ObjectiveFunctions, ObjectiveType};
use rand::Rng;

/// Configuración de optimización
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_prob: f64,
    pub mutation_prob: f64,
    pub seed: Option<u64>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            generations: 100,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            seed: None,
        }
    }
}

/// Optimizador NSGA-II
pub struct NSGA2Optimizer {
    config: OptimizationConfig,
    #[allow(dead_code)]
    objectives: Vec<ObjectiveType>,
    constraints: ConstraintEvaluator,
}

impl NSGA2Optimizer {
    pub fn new(population_size: usize, generations: usize, objectives: Vec<ObjectiveType>) -> Self {
        Self {
            config: OptimizationConfig {
                population_size,
                generations,
                ..Default::default()
            },
            objectives,
            constraints: ConstraintEvaluator::new(),
        }
    }

    pub fn with_config(config: OptimizationConfig, objectives: Vec<ObjectiveType>) -> Self {
        Self {
            config,
            objectives,
            constraints: ConstraintEvaluator::new(),
        }
    }

    /// Ejecuta la optimización
    pub fn optimize(&self, param_space: &ReactorParameterSpace) -> Vec<ReactorDesign> {
        let mut rng = rand::thread_rng();

        // 1. Inicializar población
        let mut population = self.initialize_population(param_space, &mut rng);

        // 2. Evaluar población inicial
        for design in &mut population {
            self.evaluate_design(design);
        }

        // 3. Loop evolutivo
        for gen in 0..self.config.generations {
            // 3.1 Crear offspring
            let mut offspring = self.create_offspring(&population, param_space, &mut rng);

            // 3.2 Evaluar offspring
            for design in &mut offspring {
                self.evaluate_design(design);
            }

            // 3.3 Combinar
            let mut combined: Vec<_> = population.into_iter().chain(offspring).collect();

            // 3.4 Non-dominated sorting
            let fronts = self.non_dominated_sort(&mut combined);

            // 3.5 Seleccionar siguiente generación
            population = self.select_next_generation(fronts);

            // 3.6 Reportar
            self.report_progress(gen, &population);
        }

        // Retornar frente de Pareto
        self.extract_pareto_front(&population)
    }

    fn initialize_population<R: Rng>(
        &self,
        space: &ReactorParameterSpace,
        rng: &mut R,
    ) -> Vec<ReactorDesign> {
        (0..self.config.population_size)
            .map(|i| {
                let mut design = ReactorDesign::new(&format!("gen0_ind{}", i));
                design.generation = 0;

                // Valores aleatorios dentro de rangos
                design.density = rng.gen_range(space.density.min..=space.density.max);
                design.ion_temperature_kev = rng.gen_range(space.ion_temperature.min..=space.ion_temperature.max);
                design.electron_temperature_kev = rng.gen_range(space.electron_temperature.min..=space.electron_temperature.max);
                design.major_radius = rng.gen_range(space.major_radius.min..=space.major_radius.max);
                design.minor_radius = rng.gen_range(space.minor_radius.min..=space.minor_radius.max);
                design.elongation = rng.gen_range(space.elongation.min..=space.elongation.max);
                design.triangularity = rng.gen_range(space.triangularity.min..=space.triangularity.max);
                design.toroidal_field = rng.gen_range(space.toroidal_field.min..=space.toroidal_field.max);
                design.plasma_current_ma = rng.gen_range(space.plasma_current.min..=space.plasma_current.max);
                design.icrf_power_mw = rng.gen_range(space.icrf_power.min..=space.icrf_power.max);
                design.ecrh_power_mw = rng.gen_range(space.ecrh_power.min..=space.ecrh_power.max);
                design.nbi_power_mw = rng.gen_range(space.nbi_power.min..=space.nbi_power.max);
                design.shield_thickness = rng.gen_range(space.shield_thickness.min..=space.shield_thickness.max);
                design.magnet_technology = space.magnet_technology;
                design.first_wall_material = space.first_wall_material;
                design.blanket_type = space.blanket_type;

                design
            })
            .collect()
    }

    fn evaluate_design(&self, design: &mut ReactorDesign) {
        // Evaluar objetivos
        ObjectiveFunctions::evaluate_all(design);

        // Evaluar restricciones
        let result = self.constraints.evaluate(design);
        design.feasible = result.feasible;
        design.constraint_violations = result.violations
            .iter()
            .map(|v| format!("{:?}", v))
            .collect();
    }

    fn create_offspring<R: Rng>(
        &self,
        population: &[ReactorDesign],
        space: &ReactorParameterSpace,
        rng: &mut R,
    ) -> Vec<ReactorDesign> {
        let mut offspring = Vec::with_capacity(self.config.population_size);

        while offspring.len() < self.config.population_size {
            // Selección por torneo
            let parent1 = self.tournament_select(population, rng);
            let parent2 = self.tournament_select(population, rng);

            // Cruce
            let mut child = if rng.gen::<f64>() < self.config.crossover_prob {
                self.crossover(parent1, parent2, rng)
            } else {
                parent1.clone()
            };

            // Mutación
            if rng.gen::<f64>() < self.config.mutation_prob {
                self.mutate(&mut child, space, rng);
            }

            child.id = ReactorDesign::generate_id();
            offspring.push(child);
        }

        offspring
    }

    fn tournament_select<'a, R: Rng>(&self, population: &'a [ReactorDesign], rng: &mut R) -> &'a ReactorDesign {
        let i1 = rng.gen_range(0..population.len());
        let i2 = rng.gen_range(0..population.len());

        let p1 = &population[i1];
        let p2 = &population[i2];

        if p1.pareto_rank < p2.pareto_rank {
            p1
        } else if p2.pareto_rank < p1.pareto_rank {
            p2
        } else if p1.crowding_distance > p2.crowding_distance {
            p1
        } else {
            p2
        }
    }

    fn crossover<R: Rng>(&self, p1: &ReactorDesign, p2: &ReactorDesign, rng: &mut R) -> ReactorDesign {
        let mut child = p1.clone();
        let eta = 20.0; // Distribution index for SBX

        // SBX crossover para cada parámetro
        if rng.gen::<bool>() { child.density = Self::sbx_value(p1.density, p2.density, eta, rng); }
        if rng.gen::<bool>() { child.ion_temperature_kev = Self::sbx_value(p1.ion_temperature_kev, p2.ion_temperature_kev, eta, rng); }
        if rng.gen::<bool>() { child.major_radius = Self::sbx_value(p1.major_radius, p2.major_radius, eta, rng); }
        if rng.gen::<bool>() { child.minor_radius = Self::sbx_value(p1.minor_radius, p2.minor_radius, eta, rng); }
        if rng.gen::<bool>() { child.toroidal_field = Self::sbx_value(p1.toroidal_field, p2.toroidal_field, eta, rng); }
        if rng.gen::<bool>() { child.plasma_current_ma = Self::sbx_value(p1.plasma_current_ma, p2.plasma_current_ma, eta, rng); }

        child
    }

    fn sbx_value<R: Rng>(p1: f64, p2: f64, eta: f64, rng: &mut R) -> f64 {
        let u = rng.gen::<f64>();
        let beta = if u < 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };

        0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    }

    fn mutate<R: Rng>(&self, design: &mut ReactorDesign, space: &ReactorParameterSpace, rng: &mut R) {
        let eta = 20.0;

        design.density = Self::polynomial_mutation(design.density, space.density.min, space.density.max, eta, rng);
        design.ion_temperature_kev = Self::polynomial_mutation(design.ion_temperature_kev, space.ion_temperature.min, space.ion_temperature.max, eta, rng);
        design.major_radius = Self::polynomial_mutation(design.major_radius, space.major_radius.min, space.major_radius.max, eta, rng);
        design.minor_radius = Self::polynomial_mutation(design.minor_radius, space.minor_radius.min, space.minor_radius.max, eta, rng);
        design.toroidal_field = Self::polynomial_mutation(design.toroidal_field, space.toroidal_field.min, space.toroidal_field.max, eta, rng);
    }

    fn polynomial_mutation<R: Rng>(value: f64, min: f64, max: f64, eta: f64, rng: &mut R) -> f64 {
        let u = rng.gen::<f64>();
        let delta = if u < 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0)) - 1.0
        } else {
            1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0))
        };

        (value + delta * (max - min)).max(min).min(max)
    }

    fn non_dominated_sort(&self, population: &mut [ReactorDesign]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
        let mut domination_count = vec![0usize; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if population[i].dominates(&population[j]) {
                        dominated_by[i].push(j);
                    } else if population[j].dominates(&population[i]) {
                        domination_count[i] += 1;
                    }
                }
            }

            if domination_count[i] == 0 {
                population[i].pareto_rank = 0;
                fronts[0].push(i);
            }
        }

        let mut current_front = 0;
        while !fronts[current_front].is_empty() {
            let mut next_front = Vec::new();
            for &i in &fronts[current_front] {
                for &j in &dominated_by[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].pareto_rank = current_front + 1;
                        next_front.push(j);
                    }
                }
            }
            current_front += 1;
            fronts.push(next_front);
        }

        fronts.pop(); // Remove empty last front
        fronts
    }

    fn select_next_generation(&self, fronts: Vec<Vec<usize>>) -> Vec<ReactorDesign> {
        // Simplified: just take first N individuals by rank
        let mut selected = Vec::new();
        for front in fronts {
            if selected.len() + front.len() <= self.config.population_size {
                selected.extend(front);
            } else {
                let remaining = self.config.population_size - selected.len();
                selected.extend(front.into_iter().take(remaining));
                break;
            }
        }
        // This is a placeholder - actual implementation needs population reference
        Vec::new()
    }

    fn report_progress(&self, gen: usize, population: &[ReactorDesign]) {
        let feasible_count = population.iter().filter(|d| d.feasible).count();
        let best_q = population.iter()
            .filter(|d| d.feasible)
            .filter_map(|d| d.objectives.get("Q"))
            .fold(0.0f64, |a, &b| a.max(b));

        println!("Gen {}: Feasible={}/{}, Best Q={:.2}",
            gen, feasible_count, population.len(), best_q);
    }

    fn extract_pareto_front(&self, population: &[ReactorDesign]) -> Vec<ReactorDesign> {
        population.iter()
            .filter(|d| d.pareto_rank == 0 && d.feasible)
            .cloned()
            .collect()
    }
}

/// Differential Evolution optimizer
pub struct DifferentialEvolution {
    pub population_size: usize,
    pub f: f64,  // Scale factor
    pub cr: f64, // Crossover probability
    pub generations: usize,
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self {
            population_size: 100,
            f: 0.8,
            cr: 0.9,
            generations: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::parameters::ReactorParameterSpace;

    #[test]
    fn test_nsga2_initialization() {
        let objectives = vec![
            ObjectiveType::Maximize("Q".to_string()),
            ObjectiveType::Minimize("CAPEX".to_string()),
        ];
        let _optimizer = NSGA2Optimizer::new(10, 5, objectives);
        let _space = ReactorParameterSpace::medium_reactor();

        // Test would run optimization
        // let results = optimizer.optimize(&space);
    }
}
