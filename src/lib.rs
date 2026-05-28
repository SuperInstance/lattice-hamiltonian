#![allow(
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code
)]
//! # Lattice Hamiltonian
//!
//! Ising/Potts models, transfer matrices, phase transitions, and Monte Carlo on discrete manifolds.

mod ising;
mod monte_carlo;
mod potts;
mod transfer;

pub use ising::IsingModel;
pub use monte_carlo::MetropolisSampler;
pub use potts::PottsModel;
pub use transfer::TransferMatrix;
