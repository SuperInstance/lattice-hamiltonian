//! Metropolis-Hastings Monte Carlo sampler.

use crate::ising::IsingModel;

/// Metropolis sampler for the Ising model.
pub struct MetropolisSampler {
    pub temperature: f64,
    pub steps_per_sweep: usize,
}

impl MetropolisSampler {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature: 1.0,
            steps_per_sweep: 0,
        }
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Perform one Metropolis sweep over the lattice.
    pub fn sweep(&self, model: &mut IsingModel) {
        let n = model.size;
        for _ in 0..n * n {
            let r = ((rand_simple() * n as f64) as usize) % n;
            let c = ((rand_simple() * n as f64) as usize) % n;
            let de = model.delta_energy(r, c);
            if de < 0.0 || rand_simple() < (-de / self.temperature).exp() {
                model.flip(r, c);
            }
        }
    }

    /// Run multiple sweeps and collect observables.
    pub fn sample(&self, model: &mut IsingModel, n_sweeps: usize) -> SamplingResult {
        let mut energies = Vec::with_capacity(n_sweeps);
        let mut magnetizations = Vec::with_capacity(n_sweeps);

        for _ in 0..n_sweeps {
            self.sweep(model);
            energies.push(model.energy());
            magnetizations.push(model.magnetization());
        }

        let avg_e = energies.iter().sum::<f64>() / n_sweeps as f64;
        let avg_m = magnetizations.iter().sum::<f64>() / n_sweeps as f64;
        let var_e = energies.iter().map(|e| (e - avg_e).powi(2)).sum::<f64>() / n_sweeps as f64;

        SamplingResult {
            energies,
            magnetizations,
            avg_energy: avg_e,
            avg_magnetization: avg_m,
            energy_variance: var_e,
        }
    }

    /// Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / (N T²).
    pub fn specific_heat(result: &SamplingResult, n_spins: usize, temperature: f64) -> f64 {
        result.energy_variance / (n_spins as f64 * temperature * temperature)
    }
}

/// Simple pseudo-random number generator (xoshiro-like).
fn rand_simple() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(123456789) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x & 0x7FFFFFFF) as f64 / 0x7FFFFFFF as f64
    })
}

/// Results from a Monte Carlo sampling run.
pub struct SamplingResult {
    pub energies: Vec<f64>,
    pub magnetizations: Vec<f64>,
    pub avg_energy: f64,
    pub avg_magnetization: f64,
    pub energy_variance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_creation() {
        let sampler = MetropolisSampler::new(2.0).with_temperature(2.269);
        assert!((sampler.temperature - 2.269).abs() < 1e-10);
    }

    #[test]
    fn test_sweep() {
        let mut model = IsingModel::new(4, 1.0, 0.0);
        let sampler = MetropolisSampler::new(2.0).with_temperature(2.0);
        sampler.sweep(&mut model);
        // Model should still have valid spins
        for &s in &model.spins {
            assert!(s == 1 || s == -1);
        }
    }

    #[test]
    fn test_sample() {
        let mut model = IsingModel::new(4, 1.0, 0.0);
        let sampler = MetropolisSampler::new(2.0).with_temperature(2.0);
        let result = sampler.sample(&mut model, 100);
        assert_eq!(result.energies.len(), 100);
        assert!(result.avg_energy.is_finite());
    }

    #[test]
    fn test_specific_heat() {
        let mut model = IsingModel::new(4, 1.0, 0.0);
        let sampler = MetropolisSampler::new(2.0).with_temperature(2.0);
        let result = sampler.sample(&mut model, 50);
        let cv = MetropolisSampler::specific_heat(&result, model.spins.len(), 2.0);
        assert!(cv >= 0.0);
    }
}
