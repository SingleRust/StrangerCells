/// This is essentially a re-implementation of the scrublet algorithm implemented in python.
/// We improved over the original algorithm in several small ways and expanded the capabilities a little,
/// but the general approach presented in the paper stayed the same. (This is also the reason why most of the names stayed the same for convenience.
use crate::algorithm::{call_doublets, knn_classifier, random_sample_and_recombine, run_pca};
use crate::utils::{arr1_conversion, combine_two_arrays};
use anndata::ArrayData;
use anndata::data::DynCsrMatrix;
use anndata_memory::IMAnnData;
use anyhow::anyhow;
use kiddo::traits::DistanceMetric;
use kiddo::{Manhattan, SquaredEuclidean};
use nalgebra_sparse::CsrMatrix;
use ndarray::Array1;
use num_traits::float::FloatCore;
use num_traits::{Bounded, Float, FromPrimitive, One, Zero};
use single_algebra::sparse::MatrixSum;
use single_algebra::{Direction, Log1P, Normalize};
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_svdlib::SMat;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{AddAssign, Deref, MulAssign, SubAssign};

pub(crate) mod algorithm;
pub(crate) mod utils;

#[derive(Clone, Debug)]
pub enum DistanceMeasure {
    SquaredEuclidean,
    Manhattan,
}

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
    verbose: bool,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    max_iter: Option<usize>,
    doublet_threshold: Option<f32>,
}

pub fn run_doublet_removal<T, const K: usize>(
    adata: &IMAnnData,
    params: StrangerCellsParams,
    distance_measure: DistanceMeasure,
) -> anyhow::Result<(Vec<bool>, Array1<T>, T, T, T, T)>
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
    match data {
        ArrayData::CsrMatrix(dyncsr) => {
            match dyncsr {
                DynCsrMatrix::F32(csr) => {
                    let (
                        predicted_doublets,
                        z_scores,
                        threshold,
                        detected_doublet_rate,
                        detectable_doublet_fraction,
                        overall_doublet_rate,
                    ) = match distance_measure {
                        DistanceMeasure::SquaredEuclidean => {
                            doublet_remove::<f32, K, SquaredEuclidean>(csr, params)
                        }
                        DistanceMeasure::Manhattan => {
                            doublet_remove::<f32, K, Manhattan>(csr, params)
                        }
                    }?;
                    let z_scores: Array1<T> = arr1_conversion(z_scores)?;
                    let threshold = T::from(threshold).unwrap();
                    let detected_doublet_rate = T::from(detected_doublet_rate).unwrap();
                    let detectable_doublet_fraction = T::from(detectable_doublet_fraction).unwrap();
                    let overall_doublet_rate = T::from(overall_doublet_rate).unwrap();
                    Ok((
                        predicted_doublets,
                        z_scores,
                        threshold,
                        detected_doublet_rate,
                        detectable_doublet_fraction,
                        overall_doublet_rate,
                    ))
                }
                DynCsrMatrix::F64(csr) => {
                    let (
                        predicted_doublets,
                        z_scores,
                        threshold,
                        detected_doublet_rate,
                        detectable_doublet_fraction,
                        overall_doublet_rate,
                    ) = match distance_measure {
                        DistanceMeasure::SquaredEuclidean => {
                            doublet_remove::<f64, K, SquaredEuclidean>(csr, params)
                        }
                        DistanceMeasure::Manhattan => {
                            doublet_remove::<f64, K, Manhattan>(csr, params)
                        }
                    }?;
                    let z_scores: Array1<T> = arr1_conversion(z_scores)?;
                    let threshold = T::from(threshold).unwrap();
                    let detected_doublet_rate = T::from(detected_doublet_rate).unwrap();
                    let detectable_doublet_fraction = T::from(detectable_doublet_fraction).unwrap();
                    let overall_doublet_rate = T::from(overall_doublet_rate).unwrap();
                    Ok((
                        predicted_doublets,
                        z_scores,
                        threshold,
                        detected_doublet_rate,
                        detectable_doublet_fraction,
                        overall_doublet_rate,
                    ))
                }
                other => Err(anyhow!(
                    "This CSRMatrix datatype is currently not supported for StrangerCells detection. Please convert the matrix with the appropriate SingleRust function int f32 or f64."
                )), // TODO: add error code
            }
        }
        other => Err(anyhow!(
            "This matrix type is currently not supported for StrangerCells detection. Please use a CSRMatrix in the f32 or f64 format!"
        )), // TODO: add error code
    }
}

fn doublet_remove<T, const K: usize, D>(
    csr_matrix: &CsrMatrix<T>,
    mut params: StrangerCellsParams,
) -> anyhow::Result<(Vec<bool>, Array1<T>, T, T, T, T)>
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
        + FloatCore
        + single_svdlib::SvdFloat
        + Display
        + 'static
        + std::default::Default,
    D: DistanceMetric<T, K>,
{
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

    let doublet_threshold = match params.doublet_threshold {
        None => None,
        Some(s) => Some(T::from(s).unwrap()),
    };

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

    let combined_transformed = combine_two_arrays(transformed_original, transformed_simulated)?; // individuals get dropped here!
    let mut combined_pca_sim_vec = vec![false; csr_matrix.nrows()];
    let mut v_sim = vec![true; num_simulated_cells];
    combined_pca_sim_vec.append(&mut v_sim);

    let (obs_scores, sim_scores, obs_errors, sim_errors) = knn_classifier::<T, K, D>(
        &combined_transformed,
        combined_pca_sim_vec.as_slice(),
        params.n_neighbors,
        params.expected_doublet_rate,
        params.stddev_doublet_rate,
    )?;

    let call_doublet_res = call_doublets(
        &obs_scores,
        &sim_scores,
        &obs_errors,
        doublet_threshold,
        T::from(params.expected_doublet_rate).unwrap(),
        params.verbose,
    )?;

    Ok(call_doublet_res)
}
