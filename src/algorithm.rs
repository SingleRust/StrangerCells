use crate::utils::generate_random_mask;
use anyhow::anyhow;
use kiddo::KdTree;
use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::Array2;
use num_traits::{Float, FromPrimitive, One, Zero};
use rand::SeedableRng;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use single_algebra::dimred::pca::{MaskedSparsePCA, MaskedSparsePCABuilder};
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use std::fmt::Debug;
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
    T: Float + Default + AddAssign + Debug + num_traits::float::FloatCore + Send + Sync,
{
    let nrows = data.nrows() as u64;
    let ncols = data.ncols();
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
    vec_sim: &Vec<bool>,
    n_neighbors: usize,
    exp_doublet_rate: f32,
    stddev_doublet_rate: f32
) -> anyhow::Result<()>
where
    T: Float + Default + AddAssign + Debug + num_traits::float::FloatCore + Send + Sync,
    D: DistanceMetric<T, K>,
{
    let mut ind_obs = Vec::new();
    let mut ind_sim = Vec::new();
    for (ind, &v) in vec_sim.iter().enumerate() {
        if v {
            ind_obs.push(ind as u64);
        } else {
            ind_sim.push(ind as u64);
        }
    }
    let n_sim = ind_sim.len() as f32;
    let n_obs = ind_obs.len() as f32;
    let n_frac = n_sim / n_obs;
    let k_adj = ((n_neighbors as f32) * (1.0 + n_frac)).round() as usize;
    let kdtree = build_knn_tree::<T, K>(data)?;

    for i in 0..data.nrows() {
        // look at all cells, generated and generous
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = data[(i as usize, j)];
        }
        let mut n_sim_neigh = 0;
        let mut n_obs_neigh = 0;
        let neighbors = kdtree.nearest_n::<D>(&query_array, k_adj);
        for neighbor in neighbors {
            let neighbor_id = neighbor.item;
            //let neighbor_distance = neighbor.distance; // maybe add some weights later on??? TODO!
            if ind_obs.contains(&neighbor_id) {
                n_obs_neigh += 1;
            } else {
                n_sim_neigh += 1;
            }
        }
        let rho = exp_doublet_rate;
        let r = n_sim / n_obs;
        let n_sim_neigh = n_sim_neigh as f32;
        let n_obs_neigh = n_obs_neigh as f32;
        let N = k_adj as f32;

        // Bayesian calcualtion
        let q = (n_sim_neigh+1.0)/(N+2.0);
        let ld = q * (rho/r/(1.0-rho-q*(1.0-rho-rho/r))); // formula from the paper and the script
        let se_q = (q*(1.0-q)/(N+3.0)).sqrt();
        let sq_rho = stddev_doublet_rate;
        

    }
    Ok(())
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
