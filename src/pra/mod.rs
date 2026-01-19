//! # Probabilistic Risk Assessment (PRA) Module
//!
//! Monte Carlo-based safety analysis for tokamak systems.
//!
//! ## Methodology
//!
//! The PRA methodology follows:
//! 1. **Event Tree Analysis (ETA)**: Model accident sequences from initiating events
//! 2. **Fault Tree Analysis (FTA)**: Calculate failure probabilities of safety systems
//! 3. **Monte Carlo Simulation**: Sample from uncertainty distributions
//!
//! ## Key Metrics
//!
//! - **PFD**: Probability of Failure on Demand
//! - **PFH**: Probability of Failure per Hour
//! - **CDF**: Core (Plasma) Damage Frequency
//! - **LERF**: Large Early Release Frequency (for tritium)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] NUREG-1150: Severe Accident Risks
//! [2] IEC 61508: Functional Safety
//! [3] ITER Safety Analysis Report

use crate::stochastic::RandomGenerator;

/// Failure probability distribution
#[derive(Debug, Clone)]
pub enum FailureDistribution {
    /// Constant failure rate (exponential lifetime)
    Exponential { lambda: f64 },
    /// Log-normal distribution (common for mechanical components)
    LogNormal { mu: f64, sigma: f64 },
    /// Weibull distribution (for wear-out failures)
    Weibull { k: f64, lambda: f64 },
    /// Point estimate (fixed probability)
    Point(f64),
    /// Beta distribution (for Bayesian estimates)
    Beta { alpha: f64, beta: f64 },
}

impl FailureDistribution {
    /// Sample failure time from distribution
    pub fn sample_failure_time(&self, rng: &mut RandomGenerator) -> f64 {
        match self {
            Self::Exponential { lambda } => rng.exponential(*lambda),
            Self::LogNormal { mu, sigma } => rng.log_normal(*mu, *sigma),
            Self::Weibull { k, lambda } => {
                // F(t) = 1 - exp(-(t/λ)^k)
                // t = λ * (-ln(1-U))^(1/k)
                let u = rng.uniform();
                lambda * (-( 1.0 - u).ln()).powf(1.0 / k)
            }
            Self::Point(p) => {
                // Interpret as failure in unit time interval
                if rng.uniform() < *p { 0.5 } else { f64::INFINITY }
            }
            Self::Beta { alpha, beta } => {
                // Use ratio of gammas (simplified - real impl would use proper gamma sampling)
                // For now, approximate with normal for large alpha, beta
                let mean = alpha / (alpha + beta);
                let var = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
                rng.normal_params(mean, var.sqrt()).clamp(0.0, 1.0)
            }
        }
    }

    /// Calculate probability of failure before time t
    pub fn cdf(&self, t: f64) -> f64 {
        match self {
            Self::Exponential { lambda } => 1.0 - (-lambda * t).exp(),
            Self::LogNormal { mu, sigma } => {
                // Approximate CDF using error function
                let z = (t.ln() - mu) / (sigma * std::f64::consts::SQRT_2);
                0.5 * (1.0 + erf(z))
            }
            Self::Weibull { k, lambda } => 1.0 - (-(t / lambda).powf(*k)).exp(),
            Self::Point(p) => if t >= 1.0 { *p } else { p * t },
            Self::Beta { .. } => {
                // For time interpretation, assume constant hazard
                t.min(1.0)
            }
        }
    }

    /// Calculate failure rate at time t (hazard function)
    pub fn hazard(&self, t: f64) -> f64 {
        match self {
            Self::Exponential { lambda } => *lambda,
            Self::Weibull { k, lambda } => (k / lambda) * (t / lambda).powf(k - 1.0),
            Self::LogNormal { mu, sigma } => {
                // h(t) = f(t) / (1 - F(t))
                let z = (t.ln() - mu) / sigma;
                let pdf = (-0.5 * z * z).exp() / (t * sigma * (2.0 * std::f64::consts::PI).sqrt());
                let cdf = self.cdf(t);
                if cdf < 0.9999 { pdf / (1.0 - cdf) } else { f64::INFINITY }
            }
            Self::Point(p) => *p,
            Self::Beta { .. } => 1.0,  // Simplified
        }
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation 7.1.26
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Basic event in fault tree
#[derive(Debug, Clone)]
pub struct BasicEvent {
    /// Unique identifier
    pub id: String,
    /// Description
    pub description: String,
    /// Failure distribution
    pub distribution: FailureDistribution,
    /// Test/inspection interval (hours)
    pub test_interval: f64,
    /// Mean time to repair (hours)
    pub mttr: f64,
    /// Common cause failure group (if any)
    pub ccf_group: Option<String>,
}

impl BasicEvent {
    /// Create new basic event with exponential failure distribution
    pub fn exponential(id: &str, description: &str, failure_rate_per_hour: f64) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            distribution: FailureDistribution::Exponential { lambda: failure_rate_per_hour },
            test_interval: 8760.0,  // Annual test default
            mttr: 24.0,  // 24 hour repair default
            ccf_group: None,
        }
    }

    /// Calculate average unavailability (PFD)
    ///
    /// For continuously monitored: λ * MTTR
    /// For periodically tested: λ * T_test / 2
    pub fn unavailability(&self) -> f64 {
        match &self.distribution {
            FailureDistribution::Exponential { lambda } => {
                // Average unavailability for periodically tested component
                // Q = λ * (T/2 + MTTR) for T >> MTTR
                lambda * (self.test_interval / 2.0 + self.mttr)
            }
            FailureDistribution::Point(p) => *p,
            _ => self.distribution.cdf(self.test_interval / 2.0),
        }
    }
}

/// Fault tree gate type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    /// AND gate - all inputs must fail
    And,
    /// OR gate - any input failure causes output
    Or,
    /// K-out-of-N gate
    KOutOfN { k: usize, n: usize },
    /// NOT gate (for success paths)
    Not,
}

/// Fault tree node
#[derive(Debug, Clone)]
pub struct FaultTreeNode {
    /// Node identifier
    pub id: String,
    /// Description
    pub description: String,
    /// Gate type (None for basic events)
    pub gate: Option<GateType>,
    /// Child node IDs (empty for basic events)
    pub children: Vec<String>,
    /// Basic event data (None for gates)
    pub basic_event: Option<BasicEvent>,
}

/// Complete fault tree
#[derive(Debug, Clone)]
pub struct FaultTree {
    /// Root node (top event)
    pub top_event: String,
    /// All nodes indexed by ID
    nodes: std::collections::HashMap<String, FaultTreeNode>,
}

impl FaultTree {
    /// Create new fault tree
    pub fn new(top_event: &str) -> Self {
        Self {
            top_event: top_event.to_string(),
            nodes: std::collections::HashMap::new(),
        }
    }

    /// Add gate node
    pub fn add_gate(&mut self, id: &str, description: &str, gate: GateType, children: Vec<&str>) {
        self.nodes.insert(id.to_string(), FaultTreeNode {
            id: id.to_string(),
            description: description.to_string(),
            gate: Some(gate),
            children: children.iter().map(|s| s.to_string()).collect(),
            basic_event: None,
        });
    }

    /// Add basic event
    pub fn add_basic_event(&mut self, event: BasicEvent) {
        let id = event.id.clone();
        self.nodes.insert(id.clone(), FaultTreeNode {
            id,
            description: event.description.clone(),
            gate: None,
            children: Vec::new(),
            basic_event: Some(event),
        });
    }

    /// Calculate probability of top event (analytical, for simple trees)
    pub fn calculate_probability(&self) -> f64 {
        self.node_probability(&self.top_event)
    }

    fn node_probability(&self, node_id: &str) -> f64 {
        let node = match self.nodes.get(node_id) {
            Some(n) => n,
            None => return 0.0,
        };

        match &node.gate {
            None => {
                // Basic event
                node.basic_event.as_ref().map(|e| e.unavailability()).unwrap_or(0.0)
            }
            Some(GateType::And) => {
                // Product of child probabilities
                node.children.iter()
                    .map(|c| self.node_probability(c))
                    .product()
            }
            Some(GateType::Or) => {
                // 1 - product of (1 - p_i)
                let product: f64 = node.children.iter()
                    .map(|c| 1.0 - self.node_probability(c))
                    .product();
                1.0 - product
            }
            Some(GateType::KOutOfN { k, n }) => {
                // Sum of combinations where k or more fail
                let probs: Vec<f64> = node.children.iter()
                    .map(|c| self.node_probability(c))
                    .collect();

                self.k_out_of_n_probability(*k, *n, &probs)
            }
            Some(GateType::Not) => {
                1.0 - self.node_probability(&node.children[0])
            }
        }
    }

    fn k_out_of_n_probability(&self, k: usize, n: usize, probs: &[f64]) -> f64 {
        // Recursive calculation of P(at least k of n fail)
        if k == 0 {
            return 1.0;
        }
        if k > n || probs.len() < n {
            return 0.0;
        }
        if k == n {
            return probs.iter().take(n).product();
        }

        // P(k of n) = p_n * P(k-1 of n-1) + (1-p_n) * P(k of n-1)
        let p_last = probs[n - 1];
        let sub_probs = &probs[..n-1];

        p_last * self.k_out_of_n_probability(k - 1, n - 1, sub_probs)
            + (1.0 - p_last) * self.k_out_of_n_probability(k, n - 1, sub_probs)
    }

    /// Monte Carlo simulation for probability
    pub fn monte_carlo(&self, n_trials: usize, mission_time: f64, rng: &mut RandomGenerator) -> MonteCarloResult {
        let mut failures = 0;
        let mut failure_times = Vec::new();

        for _ in 0..n_trials {
            if let Some(t) = self.simulate_failure(mission_time, rng) {
                failures += 1;
                failure_times.push(t);
            }
        }

        let probability = failures as f64 / n_trials as f64;
        let variance = probability * (1.0 - probability) / n_trials as f64;

        MonteCarloResult {
            probability,
            std_error: variance.sqrt(),
            confidence_95: (
                probability - 1.96 * variance.sqrt(),
                probability + 1.96 * variance.sqrt(),
            ),
            n_trials,
            mean_failure_time: if failure_times.is_empty() {
                f64::INFINITY
            } else {
                failure_times.iter().sum::<f64>() / failure_times.len() as f64
            },
        }
    }

    fn simulate_failure(&self, mission_time: f64, rng: &mut RandomGenerator) -> Option<f64> {
        self.simulate_node(&self.top_event, mission_time, rng)
    }

    fn simulate_node(&self, node_id: &str, mission_time: f64, rng: &mut RandomGenerator) -> Option<f64> {
        let node = self.nodes.get(node_id)?;

        match &node.gate {
            None => {
                // Basic event - sample failure time
                let failure_time = node.basic_event.as_ref()?
                    .distribution.sample_failure_time(rng);
                if failure_time <= mission_time {
                    Some(failure_time)
                } else {
                    None
                }
            }
            Some(GateType::And) => {
                // All must fail - return latest failure time
                let times: Vec<f64> = node.children.iter()
                    .filter_map(|c| self.simulate_node(c, mission_time, rng))
                    .collect();

                if times.len() == node.children.len() {
                    times.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap())
                } else {
                    None
                }
            }
            Some(GateType::Or) => {
                // Any failure - return earliest
                node.children.iter()
                    .filter_map(|c| self.simulate_node(c, mission_time, rng))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
            }
            Some(GateType::KOutOfN { k, .. }) => {
                let mut times: Vec<f64> = node.children.iter()
                    .filter_map(|c| self.simulate_node(c, mission_time, rng))
                    .collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());

                if times.len() >= *k {
                    Some(times[k - 1])  // Time when k-th failure occurs
                } else {
                    None
                }
            }
            Some(GateType::Not) => {
                // Invert - failure of NOT gate means child did NOT fail
                if self.simulate_node(&node.children[0], mission_time, rng).is_none() {
                    Some(0.0)  // "Failed" immediately
                } else {
                    None
                }
            }
        }
    }
}

/// Result of Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Estimated probability
    pub probability: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% confidence interval
    pub confidence_95: (f64, f64),
    /// Number of trials
    pub n_trials: usize,
    /// Mean time to failure (for failed cases)
    pub mean_failure_time: f64,
}

/// Event tree for accident sequence analysis
#[derive(Debug, Clone)]
pub struct EventTree {
    /// Initiating event
    pub initiating_event: InitiatingEvent,
    /// Sequence of safety systems (branch points)
    pub branches: Vec<EventTreeBranch>,
    /// End states
    pub end_states: Vec<EndState>,
}

/// Initiating event
#[derive(Debug, Clone)]
pub struct InitiatingEvent {
    pub id: String,
    pub description: String,
    /// Frequency per hour
    pub frequency: f64,
}

/// Branch point in event tree
#[derive(Debug, Clone)]
pub struct EventTreeBranch {
    pub id: String,
    pub description: String,
    /// Probability of success
    pub success_probability: f64,
    /// Reference to fault tree (if detailed analysis available)
    pub fault_tree: Option<FaultTree>,
}

/// End state of accident sequence
#[derive(Debug, Clone)]
pub struct EndState {
    pub id: String,
    pub description: String,
    /// Sequence of branch outcomes (true = success, false = failure)
    pub sequence: Vec<bool>,
    /// Consequence severity (1 = minor, 5 = catastrophic)
    pub severity: u8,
    /// Frequency (calculated)
    pub frequency: f64,
}

impl EventTree {
    /// Calculate frequencies for all end states
    pub fn calculate_frequencies(&mut self) {
        let ie_freq = self.initiating_event.frequency;

        for end_state in &mut self.end_states {
            let mut freq = ie_freq;

            for (i, &success) in end_state.sequence.iter().enumerate() {
                let p_success = self.branches[i].success_probability;
                freq *= if success { p_success } else { 1.0 - p_success };
            }

            end_state.frequency = freq;
        }
    }

    /// Get Core Damage Frequency (sum of severe end states)
    pub fn core_damage_frequency(&self) -> f64 {
        self.end_states.iter()
            .filter(|e| e.severity >= 4)
            .map(|e| e.frequency)
            .sum()
    }
}

/// Common Cause Failure (CCF) analysis using beta-factor method
pub struct CCFAnalysis {
    /// Beta factor (fraction of failures that are common cause)
    pub beta: f64,
    /// Components in group
    pub components: Vec<String>,
}

impl CCFAnalysis {
    /// Create CCF group with beta factor
    ///
    /// Typical values: β = 0.05-0.1 for diverse redundancy
    pub fn new(beta: f64, components: Vec<String>) -> Self {
        Self { beta, components }
    }

    /// Calculate CCF contribution to system failure
    ///
    /// For 2oo3 system: P_CCF = β * Q
    /// where Q is single component unavailability
    pub fn ccf_probability(&self, single_unavailability: f64) -> f64 {
        self.beta * single_unavailability
    }
}

/// Complete PRA analysis
pub struct PRAAnalysis {
    /// Fault trees for each safety system
    pub fault_trees: Vec<FaultTree>,
    /// Event trees for each initiating event
    pub event_trees: Vec<EventTree>,
    /// Random number generator
    rng: RandomGenerator,
}

impl PRAAnalysis {
    /// Create new PRA analysis
    pub fn new(seed: u64) -> Self {
        Self {
            fault_trees: Vec::new(),
            event_trees: Vec::new(),
            rng: RandomGenerator::new(seed),
        }
    }

    /// Add fault tree
    pub fn add_fault_tree(&mut self, ft: FaultTree) {
        self.fault_trees.push(ft);
    }

    /// Add event tree
    pub fn add_event_tree(&mut self, et: EventTree) {
        self.event_trees.push(et);
    }

    /// Run full Monte Carlo analysis
    pub fn run_monte_carlo(&mut self, n_trials: usize, mission_time: f64) -> PRAResults {
        let mut results = PRAResults::default();

        // Analyze each fault tree
        for ft in &self.fault_trees {
            let mc_result = ft.monte_carlo(n_trials, mission_time, &mut self.rng);
            results.fault_tree_results.push((ft.top_event.clone(), mc_result));
        }

        // Calculate event tree frequencies
        for et in &mut self.event_trees {
            et.calculate_frequencies();
            results.cdf += et.core_damage_frequency();
        }

        // Calculate total risk metrics
        results.total_pfd = results.fault_tree_results.iter()
            .map(|(_, r)| r.probability)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        results
    }

    /// Create example PRA for tokamak plasma control system
    pub fn tokamak_plasma_control() -> Self {
        let mut pra = Self::new(42);

        // Fault tree for plasma emergency shutdown
        let mut ft = FaultTree::new("TOP");
        ft.add_gate("TOP", "Failure to shutdown plasma", GateType::And, vec!["MECH", "CTRL"]);

        // Mechanical system - 2oo3 redundancy
        ft.add_gate("MECH", "Mechanical systems failure", GateType::KOutOfN { k: 2, n: 3 },
            vec!["MGI1", "MGI2", "MGI3"]);

        ft.add_basic_event(BasicEvent::exponential(
            "MGI1", "MGI Valve 1 fails to open", 1e-6));
        ft.add_basic_event(BasicEvent::exponential(
            "MGI2", "MGI Valve 2 fails to open", 1e-6));
        ft.add_basic_event(BasicEvent::exponential(
            "MGI3", "MGI Valve 3 fails to open", 1e-6));

        // Control system
        ft.add_gate("CTRL", "Control system failure", GateType::Or,
            vec!["PLC", "SENSOR", "POWER"]);

        ft.add_basic_event(BasicEvent::exponential(
            "PLC", "PLC fails", 1e-7));
        ft.add_basic_event(BasicEvent::exponential(
            "SENSOR", "Disruption sensor fails", 1e-5));
        ft.add_basic_event(BasicEvent::exponential(
            "POWER", "Control power loss", 1e-6));

        pra.add_fault_tree(ft);

        // Event tree for disruption
        let ie = InitiatingEvent {
            id: "DISRUPTION".to_string(),
            description: "Uncontrolled plasma disruption".to_string(),
            frequency: 0.1,  // per pulse
        };

        let branches = vec![
            EventTreeBranch {
                id: "MGI".to_string(),
                description: "Massive Gas Injection".to_string(),
                success_probability: 0.999,
                fault_tree: None,
            },
            EventTreeBranch {
                id: "VDE_CTRL".to_string(),
                description: "VDE Control".to_string(),
                success_probability: 0.99,
                fault_tree: None,
            },
            EventTreeBranch {
                id: "STRUCT".to_string(),
                description: "Structural integrity".to_string(),
                success_probability: 0.9999,
                fault_tree: None,
            },
        ];

        let end_states = vec![
            EndState {
                id: "OK".to_string(),
                description: "Controlled shutdown".to_string(),
                sequence: vec![true, true, true],
                severity: 1,
                frequency: 0.0,
            },
            EndState {
                id: "MINOR".to_string(),
                description: "Minor first wall damage".to_string(),
                sequence: vec![false, true, true],
                severity: 2,
                frequency: 0.0,
            },
            EndState {
                id: "DAMAGE".to_string(),
                description: "Significant component damage".to_string(),
                sequence: vec![false, false, true],
                severity: 4,
                frequency: 0.0,
            },
            EndState {
                id: "BREACH".to_string(),
                description: "Vessel breach".to_string(),
                sequence: vec![false, false, false],
                severity: 5,
                frequency: 0.0,
            },
        ];

        let et = EventTree {
            initiating_event: ie,
            branches,
            end_states,
        };

        pra.add_event_tree(et);

        pra
    }
}

/// Results of PRA analysis
#[derive(Debug, Clone, Default)]
pub struct PRAResults {
    /// Results for each fault tree
    pub fault_tree_results: Vec<(String, MonteCarloResult)>,
    /// Core Damage Frequency (per year or per demand)
    pub cdf: f64,
    /// Total system PFD
    pub total_pfd: f64,
}

impl PRAResults {
    /// Generate report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== PROBABILISTIC RISK ASSESSMENT RESULTS ===\n\n");

        report.push_str("FAULT TREE ANALYSIS:\n");
        report.push_str("-------------------\n");
        for (name, result) in &self.fault_tree_results {
            report.push_str(&format!(
                "{}: P = {:.2e} ± {:.2e} (95% CI: [{:.2e}, {:.2e}])\n",
                name, result.probability, result.std_error,
                result.confidence_95.0, result.confidence_95.1
            ));
        }

        report.push_str(&format!("\nCORE DAMAGE FREQUENCY: {:.2e} per demand\n", self.cdf));
        report.push_str(&format!("TOTAL SYSTEM PFD: {:.2e}\n", self.total_pfd));

        // SIL assessment
        let sil = if self.total_pfd < 1e-5 {
            "SIL 4"
        } else if self.total_pfd < 1e-4 {
            "SIL 3"
        } else if self.total_pfd < 1e-3 {
            "SIL 2"
        } else if self.total_pfd < 1e-2 {
            "SIL 1"
        } else {
            "Below SIL 1"
        };
        report.push_str(&format!("\nACHIEVED SAFETY INTEGRITY LEVEL: {}\n", sil));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_distribution() {
        let dist = FailureDistribution::Exponential { lambda: 0.001 };

        // CDF at t=0 should be 0
        assert!((dist.cdf(0.0) - 0.0).abs() < 1e-10);

        // CDF at t=∞ should be 1 (approximately at large t)
        assert!((dist.cdf(10000.0) - 1.0).abs() < 0.01);

        // CDF at t=1/λ should be 1 - 1/e ≈ 0.632
        assert!((dist.cdf(1000.0) - 0.632).abs() < 0.01);
    }

    #[test]
    fn test_basic_event() {
        let event = BasicEvent::exponential("TEST", "Test component", 1e-5);

        // Unavailability should be ~ λ * T/2 for periodically tested
        let expected = 1e-5 * 8760.0 / 2.0;  // ~0.044
        assert!((event.unavailability() - expected).abs() / expected < 0.1);
    }

    #[test]
    fn test_simple_fault_tree() {
        let mut ft = FaultTree::new("TOP");

        // Simple OR gate
        ft.add_gate("TOP", "System failure", GateType::Or, vec!["A", "B"]);
        ft.add_basic_event(BasicEvent {
            id: "A".to_string(),
            description: "Component A".to_string(),
            distribution: FailureDistribution::Point(0.01),
            test_interval: 8760.0,
            mttr: 24.0,
            ccf_group: None,
        });
        ft.add_basic_event(BasicEvent {
            id: "B".to_string(),
            description: "Component B".to_string(),
            distribution: FailureDistribution::Point(0.01),
            test_interval: 8760.0,
            mttr: 24.0,
            ccf_group: None,
        });

        let p = ft.calculate_probability();
        // P(A OR B) = 1 - (1-0.01)(1-0.01) = 0.0199
        assert!((p - 0.0199).abs() < 0.001);
    }

    #[test]
    fn test_and_gate() {
        let mut ft = FaultTree::new("TOP");

        ft.add_gate("TOP", "System failure", GateType::And, vec!["A", "B"]);
        ft.add_basic_event(BasicEvent {
            id: "A".to_string(),
            description: "Component A".to_string(),
            distribution: FailureDistribution::Point(0.1),
            test_interval: 8760.0,
            mttr: 24.0,
            ccf_group: None,
        });
        ft.add_basic_event(BasicEvent {
            id: "B".to_string(),
            description: "Component B".to_string(),
            distribution: FailureDistribution::Point(0.1),
            test_interval: 8760.0,
            mttr: 24.0,
            ccf_group: None,
        });

        let p = ft.calculate_probability();
        // P(A AND B) = 0.1 * 0.1 = 0.01
        assert!((p - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_monte_carlo() {
        let mut ft = FaultTree::new("TOP");
        ft.add_basic_event(BasicEvent::exponential("TOP", "Single component", 1e-4));

        let mut rng = RandomGenerator::new(42);
        let result = ft.monte_carlo(10000, 1000.0, &mut rng);

        // Probability should be approximately 1 - exp(-0.1) ≈ 0.095
        let expected = 1.0 - (-0.1_f64).exp();
        assert!((result.probability - expected).abs() < 0.02);
    }

    #[test]
    fn test_tokamak_pra() {
        let mut pra = PRAAnalysis::tokamak_plasma_control();
        let results = pra.run_monte_carlo(1000, 8760.0);

        // Should have results
        assert!(!results.fault_tree_results.is_empty());

        // CDF should be very low for redundant systems
        assert!(results.cdf < 0.01);
    }
}
