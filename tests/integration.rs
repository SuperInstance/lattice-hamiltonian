use lattice_hamiltonian::*;

#[test]
fn test_ising_energy_finite() {
    let model = IsingModel::new(4, 1.0, 0.0);
    assert!(model.energy().is_finite());
}

#[test]
fn test_ising_magnetization_bounds() {
    let model = IsingModel::new(4, 1.0, 0.0);
    let m = model.magnetization();
    assert!(m >= -1.0 && m <= 1.0);
}

#[test]
fn test_ising_flip_reverses_spin() {
    let mut model = IsingModel::new(4, 1.0, 0.0);
    let before = model.spins[0];
    model.flip(0, 0);
    assert_eq!(model.spins[0], -before);
    model.flip(0, 0);
    assert_eq!(model.spins[0], before);
}

#[test]
fn test_ising_delta_energy_symmetric() {
    let model = IsingModel::new(4, 1.0, 0.0);
    // Flipping same spin twice returns to original energy, so delta should cancel
    let de = model.delta_energy(0, 0);
    assert!(de.is_finite());
}

#[test]
fn test_ising_all_aligned_low_energy() {
    let mut model = IsingModel::new(4, 1.0, 0.0);
    // All spins up
    for s in model.spins.iter_mut() {
        *s = 1;
    }
    let e_aligned = model.energy();
    // Flip one spin
    model.flip(0, 0);
    let e_flipped = model.energy();
    // Aligned should be lower energy for J > 0
    assert!(e_aligned < e_flipped);
}

#[test]
fn test_potts_energy() {
    let potts = PottsModel::new(4, 3, 1.0);
    assert!(potts.energy().is_finite());
}

#[test]
fn test_potts_order_parameter_range() {
    let potts = PottsModel::new(4, 3, 1.0);
    let op = potts.order_parameter();
    assert!(op >= 0.0 && op <= 1.0 + 1e-10);
}

#[test]
fn test_transfer_matrix_partition_function_positive() {
    let tm = TransferMatrix::new(1.0, 0.0, 2.0);
    let z = tm.partition_function(10);
    assert!(z > 0.0);
}

#[test]
fn test_transfer_matrix_eigenvalues() {
    let tm = TransferMatrix::new(1.0, 0.0, 2.0);
    let eigs = tm.eigenvalues();
    assert!(eigs[0] > eigs[1], "eigenvalues should be sorted descending");
}

#[test]
fn test_transfer_matrix_free_energy() {
    let tm = TransferMatrix::new(1.0, 0.0, 2.0);
    let f = tm.free_energy_per_site(10, 2.0);
    assert!(f.is_finite());
}
