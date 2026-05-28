//! Potts model: generalization of Ising to q > 2 states.

/// q-state Potts model.
///
/// H = -J Σ_{<i,j>} δ(s_i, s_j)
/// where δ is the Kronecker delta.
pub struct PottsModel {
    pub size: usize,
    pub q: usize,
    pub j: f64,
    pub states: Vec<usize>,
}

impl PottsModel {
    pub fn new(size: usize, q: usize, j: f64) -> Self {
        let n = size * size;
        let states = (0..n).map(|i| i % q).collect();
        Self { size, q, j, states }
    }

    fn idx(&self, r: usize, c: usize) -> usize {
        (r % self.size) * self.size + (c % self.size)
    }

    /// Energy: -J * Σ δ(s_i, s_j) for nearest neighbors.
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for r in 0..self.size {
            for c in 0..self.size {
                let i = self.idx(r, c);
                let right = self.idx(r, c + 1);
                let down = self.idx(r + 1, c);
                if self.states[i] == self.states[right] {
                    e -= self.j;
                }
                if self.states[i] == self.states[down] {
                    e -= self.j;
                }
            }
        }
        e
    }

    /// Order parameter (generalized magnetization).
    pub fn order_parameter(&self) -> f64 {
        let mut counts = vec![0usize; self.q];
        for &s in &self.states {
            counts[s] += 1;
        }
        let n = self.states.len() as f64;
        let max_count = counts.iter().max().copied().unwrap_or(0) as f64;
        (self.q as f64 * max_count / n - 1.0) / (self.q as f64 - 1.0)
    }

    /// Set all states to the same value (ordered state).
    pub fn set_ordered(&mut self, state: usize) {
        for s in self.states.iter_mut() {
            *s = state % self.q;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potts_creation() {
        let model = PottsModel::new(10, 3, 1.0);
        assert_eq!(model.states.len(), 100);
    }

    #[test]
    fn test_energy() {
        let model = PottsModel::new(4, 3, 1.0);
        let e = model.energy();
        assert!(e.is_finite());
    }

    #[test]
    fn test_order_parameter_ordered() {
        let mut model = PottsModel::new(4, 3, 1.0);
        model.set_ordered(0);
        let op = model.order_parameter();
        assert!((op - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_parameter_disordered() {
        let model = PottsModel::new(100, 10, 1.0);
        let op = model.order_parameter();
        assert!(op < 1.0);
    }
}
