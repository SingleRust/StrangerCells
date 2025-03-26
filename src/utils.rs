use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};
use ndarray::{Array1, Array2, Axis};
use num_traits::{Bounded, Float, FromPrimitive, One, Zero};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::thread_rng;

pub(crate) fn generate_random_mask(n_genes: usize, num_random_selection: usize) -> Vec<bool> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0, n_genes);
    let mut vec = vec![false; num_random_selection];
    for _ in 0..num_random_selection {
        let v = uniform.sample(&mut rng);
        vec[v] = true;
    }
    vec
}

pub(crate) fn combine_two_arrays<T>(arr1: Array2<T>, arr2: Array2<T>) -> anyhow::Result<Array2<T>>
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
    + single_svdlib::SvdFloat {
    Ok(ndarray::concatenate(Axis(0), &[arr1.view(), arr2.view()])?)
}