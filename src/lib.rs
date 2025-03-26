use crate::algorithm::{random_sample_and_recombine, run_pca};
use crate::utils::combine_two_arrays;
/// This is essentially a re-implementation of the scrublet algorithm implemented in python.
/// We improved over the original algorithm in several small ways and expanded the capabilities a little,
/// but the general approach presented in the paper stayed the same. (This is also the reason why most of the names stayed the same for convenience.
use anndata_memory::IMAnnData;
use nalgebra_sparse::CsrMatrix;
use ndarray::{Array1, Array2};
use num_traits::{Bounded, Float, FromPrimitive, One, Zero};
use single_algebra::sparse::MatrixSum;
use single_algebra::{Direction, Log1P, Normalize};
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::dimred::pca::PCAResult;
use single_svdlib::SMat;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, Deref, MulAssign, SubAssign};

pub(crate) mod algorithm;
pub(crate) mod utils;

#[derive(Clone, Debug)]
pub struct StrangerCellsParams {
    sim_doublet_ratio: f32,
    n_neighbors: usize,
    expected_doublet_rate: f32,
    stddev_doublet_rate: f32,
    random_sate: u32,
    use_approx_neighbors: bool,
    log1p: bool,
    norm_target: u32,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    max_iter: Option<usize>,
}

pub struct StrangerCellsWorkspace<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum,
{
    pca: Option<PCAResult<T>>,
    combined_pca: Option<Array2<T>>,
    combined_pca_sim_vec: Option<Vec<bool>>,
    obs_scores: Option<Array1<T>>,
    sim_scores: Option<Array1<T>>,
    obs_errors: Option<Array1<T>>,
    sim_errors: Option<Array1<T>>,
    z_score: Option<Array1<T>>,
}
impl<T> StrangerCellsWorkspace<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum,
{
    pub fn new_empty() -> Self {
        Self {
            pca: None,
            combined_pca: None,
            combined_pca_sim_vec: None,
            obs_scores: None,
            sim_scores: None,
            obs_errors: None,
            sim_errors: None,
            z_score: None,
        }
    }
}

pub struct StrangerCellsResult<T>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum, {}

pub fn run_doublet_removal<T, const K: usize>(adata: &IMAnnData, params: StrangerCellsParams) -> anyhow::Result<()>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum,
{

    let x = adata.x();

    let read_guard = x.0.read_inner();
    let data = read_guard.deref();

    Ok(())
}

fn doublet_remove<T, const K: usize>(csr_matrix: &CsrMatrix<T>, params: StrangerCellsParams) -> anyhow::Result<()>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + Zero
        + One
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum
        + Bounded
        + single_svdlib::SvdFloat
        + 'static,
{
    let mut workspace = StrangerCellsWorkspace::<T>::new_empty();
    let pca = run_pca(
        csr_matrix,
        params.feature_selection_method,
        params.center,
        Some(false),
        Some(K),
        None,
        Some(params.random_sate),
        params.max_iter,
    )?;

    let num_simulated_cells =
        ((csr_matrix.nrows() as f32) * params.sim_doublet_ratio).round() as usize;
    let mut simulated =
        random_sample_and_recombine(csr_matrix, num_simulated_cells, params.random_sate as u64)?;
    let sums: Vec<f64> = simulated.sum_row()?;
    simulated.normalize(&sums, params.norm_target as f64, &Direction::ROW)?;
    if (params.log1p) {
        simulated.log1p_normalize()?;
    }
    let transformed_original = pca.transform(csr_matrix)?;
    let transformed_simulated = pca.transform(&simulated)?;

    let combined_transformed = combine_two_arrays(transformed_original, transformed_simulated)?;
    workspace.combined_pca = Some(combined_transformed);
    let mut v_obs = vec![false; csr_matrix.nrows()];
    let mut v_sim = vec![true; num_simulated_cells];
    v_obs.append(&mut v_sim);
    workspace.combined_pca_sim_vec = Some(v_obs);

    Ok(())
}
