use crate::utils::{find_min_threshold, generate_random_mask};
use anyhow::anyhow;
use kiddo::KdTree;
use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::{Array1, Array2};
use num_traits::real::Real;
use num_traits::{Float, FromPrimitive, One, Zero};
use rand::SeedableRng;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use single_algebra::dimred::pca::{MaskedSparsePCA, MaskedSparsePCABuilder};
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

pub(crate) fn run_pca<T>(
    matrix: &CsrMatrix<T>,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    verbose: Option<bool>,
    n_components: Option<usize>,
    alpha: Option<T>,
    random_seed: Option<u32>,
    max_iter: Option<usize>,
) -> anyhow::Result<MaskedSparsePCA<T>>
where
    T: single_svdlib::SvdFloat + num_traits::Bounded + 'static,
{
    let feature_selection_method =
        feature_selection_method.unwrap_or(FeatureSelectionMethod::RandomSelection(1000));
    let ncols = matrix.ncols();
    let center = center.unwrap_or(false);
    let verbose = verbose.unwrap_or(false);
    let n_components = n_components.unwrap_or(50);
    let random_seed = random_seed.unwrap_or(42);
    let selected = match feature_selection_method {
        FeatureSelectionMethod::FullFeatures => {
            vec![true; ncols]
        }
        FeatureSelectionMethod::HighlyVariableSelection(vec) => vec,
        FeatureSelectionMethod::RandomSelection(num_genes) => {
            generate_random_mask(ncols, num_genes)
        }
    };

    let mut masked_pca = MaskedSparsePCABuilder::new()
        .mask(selected)
        .center(center)
        .verbose(verbose)
        .alpha(alpha.unwrap_or(T::one()))
        .n_components(n_components)
        .random_seed(random_seed)
        .build();
    masked_pca.fit(matrix, max_iter)?;
    Ok(masked_pca)
}

pub(crate) fn random_sample_and_recombine<T>(
    base_matrix: &CsrMatrix<T>,
    target_num_cells: usize,
    random_seed: u64,
) -> anyhow::Result<CsrMatrix<T>>
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
        + 'static,
{
    let mut rng = StdRng::seed_from_u64(random_seed);
    let uniform = Uniform::new(0, base_matrix.nrows());
    let mut random_sample: Vec<(usize, usize)> = Vec::with_capacity(target_num_cells);
    for _ in 0..target_num_cells {
        random_sample.push((uniform.sample(&mut rng), uniform.sample(&mut rng)));
    }

    for (primary_rand_ind, secondary_rand_ind) in &random_sample {
        if *primary_rand_ind >= base_matrix.nrows() || *secondary_rand_ind >= base_matrix.nrows() {
            return Err(anyhow!(
                "Generated random indices exceeded the number of rows of the base matrix.({}, {}), max: {}",
                *primary_rand_ind,
                *secondary_rand_ind,
                base_matrix.nrows()
            ));
        }
    }

    let triplets: Vec<(usize, usize, T)> = random_sample
        .par_iter()
        .enumerate()
        .flat_map(|(i, (primary_rand_ind, secondary_rand_ind))| {
            let prim_rand_row = base_matrix.row(*primary_rand_ind);
            let sec_rand_row = base_matrix.row(*secondary_rand_ind);

            let mut combined_values =
                std::collections::HashMap::with_capacity(prim_rand_row.nnz() + sec_rand_row.nnz());

            for (&col, &val) in prim_rand_row
                .col_indices()
                .iter()
                .zip(prim_rand_row.values())
            {
                combined_values.insert(col, val);
            }

            for (&col, &val) in sec_rand_row.col_indices().iter().zip(sec_rand_row.values()) {
                combined_values
                    .entry(col)
                    .and_modify(|e| *e = *e + val)
                    .or_insert(val);
            }

            combined_values
                .into_iter()
                .map(|(col, val)| (i, col, val))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut new_data = CooMatrix::new(target_num_cells, base_matrix.ncols());

    for (i, j, val) in triplets {
        new_data.push(i, j, val);
    }

    Ok(CsrMatrix::from(&new_data))
}

fn build_knn_tree<T, const K: usize>(data: &Array2<T>) -> anyhow::Result<KdTree<T, K>>
where
    T: Default + AddAssign + Debug + num_traits::float::FloatCore + Send + Sync,
{
    let nrows = data.nrows() as u64;
    let mut kdtree: KdTree<T, K> = KdTree::new();

    for i in 0..nrows {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = data[(i as usize, j)]
        }
        kdtree.add(&point_array, i);
    }
    Ok(kdtree)
}

pub(crate) fn knn_classifier<T, const K: usize, D>(
    data: &Array2<T>,
    vec_sim: &[bool],
    n_neighbors: usize,
    exp_doublet_rate: f32,
    stddev_doublet_rate: f32,
) -> anyhow::Result<(Array1<T>, Array1<T>, Array1<T>, Array1<T>)>
where
    T: Default
        + AddAssign
        + Debug
        + num_traits::float::FloatCore
        + Send
        + Sync
        + FromPrimitive
        + Float,
    D: DistanceMetric<T, K>,
{
    let n_obs = vec_sim.iter().filter(|&&x| !x).count();
    let n_sim = vec_sim.iter().filter(|&&x| x).count();
    let n_total = n_obs + n_sim;

    let n_frac = (n_sim as f32) / (n_obs as f32);
    let k_adj = ((n_neighbors as f32) * (1.0 + n_frac)).round() as usize;

    let kdtree = build_knn_tree::<T, K>(data)?;

    let mut all_scores = Array1::zeros(n_total);
    let mut all_errors = Array1::zeros(n_total);

    let rho = T::from_f32(exp_doublet_rate).unwrap();
    let r = T::from_f32((n_sim as f32) / (n_obs as f32)).unwrap();
    let se_rho = T::from_f32(stddev_doublet_rate).unwrap();
    let n_float = T::from_usize(k_adj).unwrap();
    let one = T::one();
    let two = one + one;
    let three = two + one;

    for i in 0..n_total {
        // Create query array for this cell
        let mut query_array = [T::zero(); K];
        for j in 0..K.min(data.ncols()) {
            query_array[j] = data[(i, j)];
        }

        let neighbors = kdtree.nearest_n::<D>(&query_array, k_adj);

        let n_sim_neigh = neighbors
            .iter()
            .filter(|&n| vec_sim[n.item as usize])
            .count();
        let n_obs_neigh = neighbors.len() - n_sim_neigh;

        let nd = T::from_usize(n_sim_neigh).unwrap();
        let ns = T::from_usize(n_obs_neigh).unwrap();
        let n = T::from_usize(neighbors.len()).unwrap();

        let q = (nd + one) / (n + two);
        let one_minus_q = one - q;
        let one_minus_rho = one - rho;
        let rho_over_r = rho / r;
        let denominator = one_minus_rho - q * (one_minus_rho - rho_over_r);
        let ld = q * rho_over_r / denominator;

        let se_q = (q * one_minus_q / (n + three)).sqrt();
        let term1 = <T as Float>::powi((se_q / q * one_minus_rho), 2);
        let term2 = <T as Float>::powi(se_rho / rho * one_minus_q, 2);
        let se_ld = q * rho_over_r / <T as Float>::powi(denominator, 2) * (term1 + term2).sqrt();

        // Store results
        all_scores[i] = ld;
        all_errors[i] = se_ld;
    }

    // Split results into observed and simulated
    let mut obs_scores = Array1::zeros(n_obs);
    let mut obs_errors = Array1::zeros(n_obs);
    let mut sim_scores = Array1::zeros(n_sim);
    let mut sim_errors = Array1::zeros(n_sim);

    let mut obs_idx = 0;
    let mut sim_idx = 0;

    for i in 0..n_total {
        if vec_sim[i] {
            // This is a simulated cell
            sim_scores[sim_idx] = all_scores[i];
            sim_errors[sim_idx] = all_errors[i];
            sim_idx += 1;
        } else {
            // This is an observed cell
            obs_scores[obs_idx] = all_scores[i];
            obs_errors[obs_idx] = all_errors[i];
            obs_idx += 1;
        }
    }

    Ok((obs_scores, sim_scores, obs_errors, sim_errors))
}

fn array_to_fixed_size_array<A, const N: usize>(vec: &[A]) -> [A; N]
where
    A: Copy + Default,
{
    let mut arr = [A::default(); N];
    for (i, &val) in vec.iter().enumerate().take(N) {
        arr[i] = val;
    }
    arr
}

pub(crate) fn call_doublets<T>(
    obs_scores: &Array1<T>,
    sim_scores: &Array1<T>,
    obs_errors: &Array1<T>,
    threshold: Option<T>,
    expected_rate: T,
    verbose: bool,
) -> anyhow::Result<(Vec<bool>, Array1<T>, T, T, T, T)>
where
    T: Float + FromPrimitive + Debug + Display + PartialOrd,
{
    let threshold = match threshold {
        Some(t) => t,
        None => {
            // Automatic threshold detection
            match find_min_threshold(sim_scores, 50) {
                Ok(t) => {
                    if verbose {
                        println!("Automatically set threshold at doublet score = {:.2}", t);
                    }
                    t
                }
                Err(_) => {
                    if verbose {
                        println!(
                            "Warning: failed to automatically identify doublet score threshold. Please specify a threshold manually."
                        );
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to automatically identify threshold"
                    ));
                }
            }
        }
    };

    let mut z_scores = Array1::zeros(obs_scores.len());
    for (i, (&score, &error)) in obs_scores.iter().zip(obs_errors.iter()).enumerate() {
        z_scores[i] = (score - threshold) / error;
    }

    let predicted_doublets: Vec<bool> = obs_scores.iter().map(|&score| score > threshold).collect();

    let detected_doublet_rate = T::from_usize(predicted_doublets.iter().filter(|&&x| x).count())
        .unwrap()
        / T::from_usize(predicted_doublets.len()).unwrap();

    let detectable_doublet_fraction = T::from_usize(
        sim_scores
            .iter()
            .filter(|&&score| score > threshold)
            .count(),
    )
    .unwrap()
        / T::from_usize(sim_scores.len()).unwrap();

    let overall_doublet_rate = if detectable_doublet_fraction > T::zero() {
        detected_doublet_rate / detectable_doublet_fraction
    } else {
        // Avoid division by zero
        T::zero()
    };

    if verbose {
        println!(
            "Detected doublet rate = {:.1}%",
            T::from_f32(100.0).unwrap() * detected_doublet_rate
        );
        println!(
            "Estimated detectable doublet fraction = {:.1}%",
            T::from_f32(100.0).unwrap() * detectable_doublet_fraction
        );
        println!("Overall doublet rate:");
        println!(
            "\tExpected   = {:.1}%",
            T::from_f32(100.0).unwrap() * expected_rate
        );
        println!(
            "\tEstimated  = {:.1}%",
            T::from_f32(100.0).unwrap() * overall_doublet_rate
        );
    }

    Ok((
        predicted_doublets,
        z_scores,
        threshold,
        detected_doublet_rate,
        detectable_doublet_fraction,
        overall_doublet_rate,
    ))
}
