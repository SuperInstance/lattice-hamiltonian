//! Transfer matrix method for exact computation on 1D/2D lattices.

/// Transfer matrix for the 1D Ising model.
///
/// T_{s_i, s_{i+1}} = exp(J s_i s_{i+1} + h(s_i + s_{i+1})/2)
pub struct TransferMatrix {
    pub j: f64,
    pub h: f64,
    pub matrix: [[f64; 2]; 2],
}

impl TransferMatrix {
    pub fn new(j: f64, h: f64, temperature: f64) -> Self {
        let beta = 1.0 / temperature;
        let _e = beta.exp();
        // States: -1 (index 0) and +1 (index 1)
        let matrix = [
            [(j * beta + h * beta).exp(), (-j * beta).exp()],
            [(-j * beta).exp(), (j * beta - h * beta).exp()],
        ];
        Self { j, h, matrix }
    }

    /// Partition function Z = Tr(T^N) for a chain of length N.
    pub fn partition_function(&self, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let eigenvalues = self.eigenvalues();
        eigenvalues[0].powi(n as i32) + eigenvalues[1].powi(n as i32)
    }

    /// Free energy per site: f = -kT/N * ln(Z).
    pub fn free_energy_per_site(&self, n: usize, temperature: f64) -> f64 {
        let z = self.partition_function(n);
        -temperature * z.ln() / n as f64
    }

    /// Eigenvalues of the transfer matrix.
    pub fn eigenvalues(&self) -> [f64; 2] {
        let a = self.matrix[0][0];
        let b = self.matrix[0][1];
        let c = self.matrix[1][0];
        let d = self.matrix[1][1];
        let trace = a + d;
        let det = a * d - b * c;
        let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
        [(trace + disc) / 2.0, (trace - disc) / 2.0]
    }

    /// Correlation length: ξ = 1/|ln(λ_1/λ_0)|.
    pub fn correlation_length(&self) -> f64 {
        let eig = self.eigenvalues();
        if eig[0].abs() < 1e-15 {
            return f64::INFINITY;
        }
        let ratio = eig[1] / eig[0];
        if ratio.abs() < 1e-15 {
            return 0.0;
        }
        1.0 / ratio.ln().abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_matrix_creation() {
        let tm = TransferMatrix::new(1.0, 0.0, 1.0);
        assert!(tm.matrix[0][0] > 0.0);
    }

    #[test]
    fn test_partition_function() {
        let tm = TransferMatrix::new(1.0, 0.0, 1.0);
        let z = tm.partition_function(10);
        assert!(z > 0.0);
    }

    #[test]
    fn test_eigenvalues_positive() {
        let tm = TransferMatrix::new(1.0, 0.0, 1.0);
        let eig = tm.eigenvalues();
        assert!(eig[0] > 0.0);
        assert!(eig[1] > 0.0);
    }

    #[test]
    fn test_correlation_length() {
        let tm = TransferMatrix::new(1.0, 0.0, 1.0);
        let xi = tm.correlation_length();
        assert!(xi > 0.0);
    }

    #[test]
    fn test_free_energy() {
        let tm = TransferMatrix::new(1.0, 0.0, 1.0);
        let f = tm.free_energy_per_site(100, 1.0);
        assert!(f.is_finite());
    }
}
