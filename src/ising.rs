//! Ising model on a 2D lattice.

/// 2D Ising model with periodic boundary conditions.
///
/// H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i
/// where s_i ∈ {-1, +1}.
pub struct IsingModel {
    pub size: usize,
    pub j: f64, // Coupling constant
    pub h: f64, // External field
    pub spins: Vec<i8>,
}

impl IsingModel {
    pub fn new(size: usize, j: f64, h: f64) -> Self {
        let n = size * size;
        let mut spins = vec![1i8; n];
        // Random initialization
        for i in 0..n {
            if i % 3 == 0 {
                spins[i] = -1;
            }
        }
        Self { size, j, h, spins }
    }

    /// Index from (row, col).
    fn idx(&self, r: usize, c: usize) -> usize {
        (r % self.size) * self.size + (c % self.size)
    }

    /// Compute the Hamiltonian (total energy).
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for r in 0..self.size {
            for c in 0..self.size {
                let i = self.idx(r, c);
                let right = self.idx(r, c + 1);
                let down = self.idx(r + 1, c);
                e -= self.j * (self.spins[i] * self.spins[right]) as f64;
                e -= self.j * (self.spins[i] * self.spins[down]) as f64;
                e -= self.h * self.spins[i] as f64;
            }
        }
        e
    }

    /// Magnetization: average spin.
    pub fn magnetization(&self) -> f64 {
        let sum: i64 = self.spins.iter().map(|&s| s as i64).sum();
        sum as f64 / self.spins.len() as f64
    }

    /// Energy change from flipping spin at position (r, c).
    pub fn delta_energy(&self, r: usize, c: usize) -> f64 {
        let i = self.idx(r, c);
        let s = self.spins[i] as f64;
        let neighbors = [
            self.spins[self.idx(r.wrapping_sub(1), c)],
            self.spins[self.idx(r + 1, c)],
            self.spins[self.idx(r, c.wrapping_sub(1))],
            self.spins[self.idx(r, c + 1)],
        ];
        let neighbor_sum: f64 = neighbors.iter().map(|&n| n as f64).sum();
        2.0 * s * (self.j * neighbor_sum + self.h)
    }

    /// Flip the spin at (r, c).
    pub fn flip(&mut self, r: usize, c: usize) {
        let i = self.idx(r, c);
        self.spins[i] *= -1;
    }

    /// Total spin.
    pub fn total_spin(&self) -> i64 {
        self.spins.iter().map(|&s| s as i64).sum()
    }

    /// Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / (N k_B T²).
    /// At critical temperature T_c = 2J / (k_B ln(1 + √2)) ≈ 2.269J.
    pub fn critical_temperature(&self) -> f64 {
        2.0 * self.j / (1.0 + 2.0_f64.sqrt()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_creation() {
        let model = IsingModel::new(10, 1.0, 0.0);
        assert_eq!(model.spins.len(), 100);
    }

    #[test]
    fn test_energy() {
        let model = IsingModel::new(4, 1.0, 0.0);
        let _e = model.energy();
        assert!(e.is_finite());
    }

    #[test]
    fn test_magnetization() {
        let model = IsingModel::new(4, 1.0, 0.0);
        let m = model.magnetization();
        assert!(m >= -1.0 && m <= 1.0);
    }

    #[test]
    fn test_flip() {
        let mut model = IsingModel::new(4, 1.0, 0.0);
        let before = model.spins[0];
        model.flip(0, 0);
        assert_eq!(model.spins[0], -before);
    }

    #[test]
    fn test_delta_energy() {
        let model = IsingModel::new(4, 1.0, 0.0);
        let de = model.delta_energy(0, 0);
        assert!(de.is_finite());
    }

    #[test]
    fn test_critical_temperature() {
        let model = IsingModel::new(10, 1.0, 0.0);
        let tc = model.critical_temperature();
        assert!((tc - 2.269).abs() < 0.01);
    }
}
