//! # Stochastic Module
//!
//! Stochastic noise models for sensors, actuators, and control systems.
//!
//! ## Philosophy
//!
//! Real-world sensors and actuators have inherent noise and uncertainties:
//! - **Measurement noise**: Gaussian noise, quantization errors
//! - **Drift**: Slow systematic errors that accumulate
//! - **Latency**: Time delays between measurement and availability
//! - **Actuator noise**: Imprecise response to commands
//!
//! ## Mathematical Models
//!
//! ### Sensor Noise
//! y(t) = x(t) + η(t) + d(t)
//! where:
//! - x(t) = true value
//! - η(t) ~ N(0, σ²) is white Gaussian noise
//! - d(t) is slow drift (random walk or Ornstein-Uhlenbeck)
//!
//! ### Actuator Response
//! u_actual(t) = G(s) · u_commanded(t) + ε(t)
//! where G(s) is transfer function (delay + dynamics)
//!
//! ## Author
//! Francisco Molina-Burgos, Avermex Research Division
//!
//! ## References
//!
//! [1] Gelb, A. "Applied Optimal Estimation", MIT Press, 1974
//! [2] Simon, D. "Optimal State Estimation", Wiley, 2006
//! [3] ITER Instrumentation & Control Design Description Document

use std::f64::consts::PI;

/// Pseudo-random number generator (xoshiro256**)
///
/// Fast, high-quality PRNG suitable for Monte Carlo simulations.
/// Period: 2^256 - 1
#[derive(Clone)]
pub struct RandomGenerator {
    state: [u64; 4],
}

impl RandomGenerator {
    /// Create new RNG with seed
    pub fn new(seed: u64) -> Self {
        // Initialize state using SplitMix64
        let mut s = seed;
        let mut state = [0u64; 4];
        for i in 0..4 {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            state[i] = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Create with time-based seed (for non-reproducible simulations)
    pub fn from_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self::new(seed)
    }

    /// Generate next u64
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Generate uniform [0, 1)
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate uniform in range [a, b)
    pub fn uniform_range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }

    /// Generate standard normal N(0, 1) using Box-Muller transform
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);  // Avoid log(0)
        let u2 = self.uniform();

        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Generate normal N(mean, std)
    pub fn normal_params(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.normal()
    }

    /// Generate exponential distribution with rate λ
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.uniform().max(1e-300).ln() / lambda
    }

    /// Generate Poisson distribution with rate λ
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            // Direct method for small λ
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= self.uniform();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Normal approximation for large λ
            let n = self.normal_params(lambda, lambda.sqrt());
            n.max(0.0).round() as u64
        }
    }

    /// Generate log-normal distribution
    ///
    /// X ~ LogNormal(μ, σ) means ln(X) ~ N(μ, σ)
    pub fn log_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.normal()).exp()
    }
}

impl Default for RandomGenerator {
    fn default() -> Self {
        Self::new(42)  // Reproducible default
    }
}

/// Sensor noise model
#[derive(Debug, Clone)]
pub struct SensorNoise {
    /// Measurement standard deviation
    pub std_dev: f64,
    /// Drift rate (random walk std per sqrt(s))
    pub drift_rate: f64,
    /// Current drift value
    drift: f64,
    /// Quantization step (0 = no quantization)
    pub quantization: f64,
    /// Measurement latency (s)
    pub latency: f64,
    /// Saturation limits (min, max)
    pub saturation: Option<(f64, f64)>,
    /// Bias (systematic error)
    pub bias: f64,
    /// Dead band (minimum detectable change)
    pub dead_band: f64,
    /// History buffer for latency simulation
    history: Vec<(f64, f64)>,  // (time, value)
    /// Last output value (for dead band)
    last_output: f64,
}

impl SensorNoise {
    /// Create new sensor noise model
    ///
    /// ## Arguments
    /// * `std_dev` - Measurement noise standard deviation
    /// * `drift_rate` - Drift random walk rate (units per sqrt(second))
    pub fn new(std_dev: f64, drift_rate: f64) -> Self {
        Self {
            std_dev,
            drift_rate,
            drift: 0.0,
            quantization: 0.0,
            latency: 0.0,
            saturation: None,
            bias: 0.0,
            dead_band: 0.0,
            history: Vec::with_capacity(100),
            last_output: 0.0,
        }
    }

    /// Builder: set quantization step
    pub fn with_quantization(mut self, step: f64) -> Self {
        self.quantization = step;
        self
    }

    /// Builder: set latency
    pub fn with_latency(mut self, latency_s: f64) -> Self {
        self.latency = latency_s;
        self
    }

    /// Builder: set saturation limits
    pub fn with_saturation(mut self, min: f64, max: f64) -> Self {
        self.saturation = Some((min, max));
        self
    }

    /// Builder: set bias
    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }

    /// Builder: set dead band
    pub fn with_dead_band(mut self, dead_band: f64) -> Self {
        self.dead_band = dead_band;
        self
    }

    /// Apply noise to a measurement
    ///
    /// ## Arguments
    /// * `true_value` - Actual physical value
    /// * `time` - Current simulation time
    /// * `dt` - Timestep
    /// * `rng` - Random number generator
    ///
    /// ## Returns
    /// Noisy measurement as would be seen by control system
    pub fn apply(&mut self, true_value: f64, time: f64, dt: f64, rng: &mut RandomGenerator) -> f64 {
        // 1. Update drift (random walk)
        self.drift += rng.normal() * self.drift_rate * dt.sqrt();

        // 2. Add Gaussian noise + drift + bias
        let mut measured = true_value + rng.normal() * self.std_dev + self.drift + self.bias;

        // 3. Apply quantization
        if self.quantization > 0.0 {
            measured = (measured / self.quantization).round() * self.quantization;
        }

        // 4. Apply saturation
        if let Some((min, max)) = self.saturation {
            measured = measured.clamp(min, max);
        }

        // 5. Handle latency (store in history buffer)
        if self.latency > 0.0 {
            self.history.push((time, measured));
            // Clean old entries
            self.history.retain(|(t, _)| time - t < self.latency * 2.0);
            // Find value from latency ago
            let target_time = time - self.latency;
            measured = self.history.iter()
                .filter(|(t, _)| *t <= target_time)
                .last()
                .map(|(_, v)| *v)
                .unwrap_or(measured);
        }

        // 6. Apply dead band
        if self.dead_band > 0.0 {
            if (measured - self.last_output).abs() < self.dead_band {
                measured = self.last_output;
            } else {
                self.last_output = measured;
            }
        }

        measured
    }

    /// Reset sensor state (drift, history)
    pub fn reset(&mut self) {
        self.drift = 0.0;
        self.history.clear();
        self.last_output = 0.0;
    }
}

/// Actuator noise model
#[derive(Debug, Clone)]
pub struct ActuatorNoise {
    /// Response noise standard deviation (fraction of command)
    pub noise_fraction: f64,
    /// Absolute noise floor
    pub noise_floor: f64,
    /// Slew rate limit (units per second)
    pub slew_rate: f64,
    /// Transport delay (s)
    pub delay: f64,
    /// First-order time constant (s)
    pub time_constant: f64,
    /// Current output value
    current_output: f64,
    /// Command buffer for delay
    command_buffer: Vec<(f64, f64)>,
}

impl ActuatorNoise {
    /// Create new actuator noise model
    pub fn new(noise_fraction: f64, slew_rate: f64) -> Self {
        Self {
            noise_fraction,
            noise_floor: 0.0,
            slew_rate,
            delay: 0.0,
            time_constant: 0.0,
            current_output: 0.0,
            command_buffer: Vec::with_capacity(100),
        }
    }

    /// Builder: set transport delay
    pub fn with_delay(mut self, delay: f64) -> Self {
        self.delay = delay;
        self
    }

    /// Builder: set time constant
    pub fn with_time_constant(mut self, tau: f64) -> Self {
        self.time_constant = tau;
        self
    }

    /// Builder: set noise floor
    pub fn with_noise_floor(mut self, floor: f64) -> Self {
        self.noise_floor = floor;
        self
    }

    /// Apply actuator dynamics to command
    ///
    /// ## Arguments
    /// * `command` - Commanded value
    /// * `time` - Current time
    /// * `dt` - Timestep
    /// * `rng` - Random generator
    ///
    /// ## Returns
    /// Actual actuator output
    pub fn apply(&mut self, command: f64, time: f64, dt: f64, rng: &mut RandomGenerator) -> f64 {
        // 1. Store command with timestamp
        self.command_buffer.push((time, command));
        self.command_buffer.retain(|(t, _)| time - t < self.delay * 2.0 + 1.0);

        // 2. Get delayed command
        let target_time = time - self.delay;
        let delayed_command = self.command_buffer.iter()
            .filter(|(t, _)| *t <= target_time)
            .last()
            .map(|(_, c)| *c)
            .unwrap_or(command);

        // 3. Apply slew rate limit
        let max_change = self.slew_rate * dt;
        let delta = (delayed_command - self.current_output).clamp(-max_change, max_change);

        // 4. Apply first-order dynamics
        let target = self.current_output + delta;
        if self.time_constant > 0.0 {
            let alpha = dt / (self.time_constant + dt);
            self.current_output += alpha * (target - self.current_output);
        } else {
            self.current_output = target;
        }

        // 5. Add noise
        let noise_std = (self.noise_fraction * self.current_output.abs()).max(self.noise_floor);
        self.current_output + rng.normal() * noise_std
    }

    /// Reset actuator state
    pub fn reset(&mut self) {
        self.current_output = 0.0;
        self.command_buffer.clear();
    }

    /// Set initial output
    pub fn set_output(&mut self, value: f64) {
        self.current_output = value;
    }
}

/// Stochastic process models
pub mod processes {
    use super::RandomGenerator;

    /// Ornstein-Uhlenbeck process
    ///
    /// dX = θ(μ - X)dt + σdW
    ///
    /// Mean-reverting stochastic process, useful for:
    /// - Temperature fluctuations
    /// - Market prices
    /// - Sensor drift with auto-correction
    #[derive(Debug, Clone)]
    pub struct OrnsteinUhlenbeck {
        /// Current value
        pub x: f64,
        /// Mean level
        pub mu: f64,
        /// Mean reversion rate
        pub theta: f64,
        /// Volatility
        pub sigma: f64,
    }

    impl OrnsteinUhlenbeck {
        pub fn new(x0: f64, mu: f64, theta: f64, sigma: f64) -> Self {
            Self { x: x0, mu, theta, sigma }
        }

        /// Step forward in time
        pub fn step(&mut self, dt: f64, rng: &mut RandomGenerator) {
            // Exact discretization
            let exp_theta = (-self.theta * dt).exp();
            let var = self.sigma * self.sigma * (1.0 - exp_theta * exp_theta) / (2.0 * self.theta);

            self.x = self.mu + (self.x - self.mu) * exp_theta + var.sqrt() * rng.normal();
        }

        /// Get current value
        pub fn value(&self) -> f64 {
            self.x
        }

        /// Stationary variance
        pub fn stationary_variance(&self) -> f64 {
            self.sigma * self.sigma / (2.0 * self.theta)
        }
    }

    /// Geometric Brownian Motion
    ///
    /// dS = μSdt + σSdW
    ///
    /// Useful for:
    /// - Component failure rates
    /// - Cost escalation
    #[derive(Debug, Clone)]
    pub struct GeometricBrownian {
        pub x: f64,
        pub mu: f64,
        pub sigma: f64,
    }

    impl GeometricBrownian {
        pub fn new(x0: f64, mu: f64, sigma: f64) -> Self {
            Self { x: x0, mu, sigma }
        }

        pub fn step(&mut self, dt: f64, rng: &mut RandomGenerator) {
            // Exact solution: X(t+dt) = X(t) * exp((μ - σ²/2)dt + σ√dt * Z)
            let drift = (self.mu - 0.5 * self.sigma * self.sigma) * dt;
            let diffusion = self.sigma * dt.sqrt() * rng.normal();
            self.x *= (drift + diffusion).exp();
        }

        pub fn value(&self) -> f64 {
            self.x
        }
    }

    /// Jump-diffusion process (Merton model)
    ///
    /// For modeling rare events like disruptions
    #[derive(Debug, Clone)]
    pub struct JumpDiffusion {
        pub x: f64,
        pub mu: f64,
        pub sigma: f64,
        /// Jump intensity (expected jumps per unit time)
        pub lambda: f64,
        /// Jump size mean
        pub jump_mean: f64,
        /// Jump size std
        pub jump_std: f64,
    }

    impl JumpDiffusion {
        pub fn new(x0: f64, mu: f64, sigma: f64, lambda: f64, jump_mean: f64, jump_std: f64) -> Self {
            Self { x: x0, mu, sigma, lambda, jump_mean, jump_std }
        }

        pub fn step(&mut self, dt: f64, rng: &mut RandomGenerator) {
            // Diffusion part
            let drift = self.mu * dt;
            let diffusion = self.sigma * dt.sqrt() * rng.normal();

            // Jump part (Poisson number of jumps)
            let n_jumps = rng.poisson(self.lambda * dt);
            let mut jump_sum = 0.0;
            for _ in 0..n_jumps {
                jump_sum += rng.normal_params(self.jump_mean, self.jump_std);
            }

            self.x += drift + diffusion + jump_sum;
        }

        pub fn value(&self) -> f64 {
            self.x
        }
    }
}

/// Collection of sensor noise models for a tokamak diagnostic suite
pub struct DiagnosticSuite {
    /// Magnetic probes (B field measurement)
    pub magnetic_probe: SensorNoise,
    /// Rogowski coil (plasma current)
    pub rogowski_coil: SensorNoise,
    /// Interferometer (density)
    pub interferometer: SensorNoise,
    /// ECE radiometer (temperature)
    pub ece_radiometer: SensorNoise,
    /// Bolometer (radiated power)
    pub bolometer: SensorNoise,
    /// Position measurement
    pub position_sensor: SensorNoise,
}

impl DiagnosticSuite {
    /// Create realistic diagnostic suite for tokamak
    ///
    /// Values based on ITER diagnostic specifications
    pub fn iter_like() -> Self {
        Self {
            // Magnetic probes: ~1% accuracy, low drift
            magnetic_probe: SensorNoise::new(0.001, 1e-6)
                .with_quantization(1e-4)
                .with_latency(1e-4),

            // Rogowski coil: ~0.1% accuracy for plasma current
            rogowski_coil: SensorNoise::new(0.001, 1e-5)
                .with_quantization(1e-3)
                .with_latency(1e-5),

            // Interferometer: ~2% accuracy for density
            interferometer: SensorNoise::new(0.02, 1e-4)
                .with_latency(1e-3),

            // ECE: ~5% accuracy for temperature
            ece_radiometer: SensorNoise::new(0.05, 1e-4)
                .with_latency(1e-4),

            // Bolometer: ~10% for radiated power
            bolometer: SensorNoise::new(0.1, 1e-3)
                .with_latency(1e-3),

            // Position: ~1 mm accuracy
            position_sensor: SensorNoise::new(0.001, 1e-5)
                .with_quantization(1e-4)
                .with_latency(1e-5)
                .with_saturation(-1.0, 1.0),
        }
    }

    /// Create high-fidelity diagnostic suite (R&D quality)
    pub fn high_fidelity() -> Self {
        Self {
            magnetic_probe: SensorNoise::new(1e-4, 1e-7),
            rogowski_coil: SensorNoise::new(1e-4, 1e-6),
            interferometer: SensorNoise::new(0.005, 1e-5),
            ece_radiometer: SensorNoise::new(0.01, 1e-5),
            bolometer: SensorNoise::new(0.02, 1e-4),
            position_sensor: SensorNoise::new(1e-4, 1e-6),
        }
    }
}

/// Collection of actuator noise models
pub struct ActuatorSuite {
    /// Poloidal field coil power supply
    pub pf_coil_supply: ActuatorNoise,
    /// Toroidal field coil power supply
    pub tf_coil_supply: ActuatorNoise,
    /// Gas puff valve
    pub gas_valve: ActuatorNoise,
    /// ICRF power system
    pub icrf_power: ActuatorNoise,
    /// ECRH power system
    pub ecrh_power: ActuatorNoise,
    /// NBI power system
    pub nbi_power: ActuatorNoise,
}

impl ActuatorSuite {
    /// Create realistic actuator suite for tokamak
    pub fn iter_like() -> Self {
        Self {
            // PF coils: ~0.1% accuracy, 10 kA/s slew rate
            pf_coil_supply: ActuatorNoise::new(0.001, 1e4)
                .with_delay(1e-3)
                .with_time_constant(10e-3),

            // TF coils: very stable, slow response
            tf_coil_supply: ActuatorNoise::new(0.0001, 100.0)
                .with_delay(0.1)
                .with_time_constant(1.0),

            // Gas valve: ~5% accuracy, fast response
            gas_valve: ActuatorNoise::new(0.05, 1e22)
                .with_delay(5e-3)
                .with_time_constant(10e-3),

            // ICRF: ~1% power control
            icrf_power: ActuatorNoise::new(0.01, 1e7)
                .with_delay(1e-3)
                .with_time_constant(1e-3),

            // ECRH: ~1% power control
            ecrh_power: ActuatorNoise::new(0.01, 1e7)
                .with_delay(1e-4)
                .with_time_constant(1e-4),

            // NBI: ~2% power control, slow modulation
            nbi_power: ActuatorNoise::new(0.02, 1e6)
                .with_delay(0.1)
                .with_time_constant(0.1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_uniform() {
        let mut rng = RandomGenerator::new(12345);

        // Generate many samples
        let samples: Vec<f64> = (0..10000).map(|_| rng.uniform()).collect();

        // Check range [0, 1)
        assert!(samples.iter().all(|&x| x >= 0.0 && x < 1.0));

        // Check mean is approximately 0.5
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.02, "Mean {} should be ~0.5", mean);
    }

    #[test]
    fn test_rng_normal() {
        let mut rng = RandomGenerator::new(12345);

        let samples: Vec<f64> = (0..10000).map(|_| rng.normal()).collect();

        // Check mean is approximately 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.05, "Mean {} should be ~0", mean);

        // Check std is approximately 1
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.05, "Std {} should be ~1", std);
    }

    #[test]
    fn test_sensor_noise() {
        let mut sensor = SensorNoise::new(0.01, 1e-5)
            .with_quantization(0.001)
            .with_saturation(-1.0, 1.0);

        let mut rng = RandomGenerator::new(42);

        // Apply noise to constant value
        let true_value = 0.5;
        let measurements: Vec<f64> = (0..1000)
            .map(|i| sensor.apply(true_value, i as f64 * 0.001, 0.001, &mut rng))
            .collect();

        // Mean should be close to true value
        let mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
        assert!((mean - true_value).abs() < 0.05);

        // All values should be within saturation
        assert!(measurements.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_actuator_slew_rate() {
        let mut actuator = ActuatorNoise::new(0.0, 100.0);  // 100 units/s slew rate
        let mut rng = RandomGenerator::new(42);

        // Command step change
        actuator.set_output(0.0);
        let dt = 0.01;  // 10 ms timestep
        let command = 10.0;  // Step to 10

        // Should take ~100 ms to reach 10 with 100 units/s slew rate
        let mut outputs = Vec::new();
        for i in 0..20 {
            let out = actuator.apply(command, i as f64 * dt, dt, &mut rng);
            outputs.push(out);
        }

        // Check slew rate is respected (max 1 unit per 10 ms)
        for i in 1..outputs.len() {
            let delta = (outputs[i] - outputs[i-1]).abs();
            assert!(delta <= 100.0 * dt * 1.5, "Slew rate violated: delta = {}", delta);
        }

        // Should reach target eventually
        assert!((outputs.last().unwrap() - command).abs() < 1.0);
    }

    #[test]
    fn test_ornstein_uhlenbeck() {
        use processes::OrnsteinUhlenbeck;

        let mut ou = OrnsteinUhlenbeck::new(0.0, 1.0, 1.0, 0.1);
        let mut rng = RandomGenerator::new(42);

        // Run for long time
        for _ in 0..10000 {
            ou.step(0.01, &mut rng);
        }

        // Should be near mean
        assert!((ou.value() - 1.0).abs() < 1.0, "OU should converge to mean");
    }

    #[test]
    fn test_diagnostic_suite() {
        let suite = DiagnosticSuite::iter_like();

        // Check reasonable parameters
        assert!(suite.magnetic_probe.std_dev > 0.0);
        assert!(suite.rogowski_coil.std_dev > 0.0);
    }
}
