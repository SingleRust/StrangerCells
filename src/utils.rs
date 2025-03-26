use anyhow::anyhow;
use ndarray::{Array1, Array2, Axis};
use num_traits::{Bounded, Float, FromPrimitive, Num, NumCast, One, Zero};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

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
        + single_svdlib::SvdFloat,
{
    Ok(ndarray::concatenate(Axis(0), &[arr1.view(), arr2.view()])?)
}

pub(crate) fn find_min_threshold<T>(scores: &Array1<T>, n_bins: usize) -> anyhow::Result<T>
where
    T: Float + FromPrimitive + Debug,
{
    let scores_f64: Vec<f64> = scores.iter().map(|&x| x.to_f64().unwrap()).collect();

    if scores_f64.is_empty() {
        return Err(anyhow!("Empty scores array!")); // TODO add error code!
    }

    let min_score = *scores_f64
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_score = *scores_f64
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let bin_width = (max_score - min_score) / (n_bins as f64);
    let mut histogram = vec![0; n_bins];

    for &score in &scores_f64 {
        let bin_idx = ((score - min_score) / (n_bins as f64)).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        histogram[bin_idx] += 1;
    }

    let mut smoothed = vec![0; n_bins];
    for i in 1..n_bins - 1 {
        smoothed[i] = (histogram[i - 1] + histogram[i] + histogram[i + 1]) / 3;
    }

    smoothed[0] = (histogram[0] + histogram[1]) / 2;
    smoothed[n_bins - 1] = (histogram[n_bins - 2] + histogram[n_bins - 1]) / 2;

    let mut peak1_idx = 0;
    for i in 1..n_bins - 1 {
        if smoothed[i] > smoothed[peak1_idx] {
            peak1_idx = i;
        }
    }

    let mut peak2_idx = peak1_idx;
    for i in peak1_idx + 1..n_bins {
        if smoothed[i] > smoothed[peak2_idx] {
            peak2_idx = i;
        }
    }

    if peak1_idx == peak2_idx {
        for i in 0..peak1_idx {
            if smoothed[i] > smoothed[peak2_idx] {
                peak2_idx = i;
            }
        }
    }

    if peak1_idx > peak2_idx {
        std::mem::swap(&mut peak1_idx, &mut peak2_idx);
    }

    let start_idx = peak1_idx;
    let end_idx = peak2_idx;

    if start_idx == end_idx || start_idx + 1 >= n_bins || end_idx >= n_bins {
        // Fallback to a simple threshold at the median
        let median_idx = n_bins / 2;
        return Ok(T::from_f64(min_score + (median_idx as f64) * bin_width).unwrap());
    }

    let mut min_idx = start_idx + 1;
    for i in (start_idx + 1)..end_idx {
        if smoothed[i] < smoothed[min_idx] {
            min_idx = i;
        }
    }

    // Convert back to score value
    let threshold = min_score + ((min_idx as f64) + 0.5) * bin_width;
    Ok(T::from_f64(threshold).unwrap())
}

pub(crate) fn arr2_conversion<M, T>(array2: Array2<M>) -> anyhow::Result<Array2<T>>
where
    M: Num + Copy + num_traits::ToPrimitive,
    T: Num + NumCast + Clone,
{
    let mut result = Array2::zeros(array2.dim());

    for (target, &source) in result.iter_mut().zip(array2.iter()) {
        *target = T::from(source).ok_or_else(|| anyhow!("Failed to convert value"))?;
    }

    Ok(result)
}

pub(crate) fn arr1_conversion<M, T>(array1: Array1<M>) -> anyhow::Result<Array1<T>>
where
    M: Num + Copy + num_traits::ToPrimitive,
    T: Num + NumCast + Clone,
{
    let mut result = Array1::zeros(array1.dim());

    for (target, &source) in result.iter_mut().zip(array1.iter()) {
        *target = T::from(source).ok_or_else(|| anyhow!("Failed to convert value"))?;
    }

    Ok(result)
}