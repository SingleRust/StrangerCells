# StrangerCells 🧫

A Rust implementation of the Scrublet algorithm for doublet detection in single-cell RNA-seq data.

## About 📝

StrangerCells is a direct port of the Python-based Scrublet algorithm to Rust, with a focus on performance and memory efficiency. It detects doublets (two cells captured as one) in single-cell RNA sequencing data using a nearest-neighbor classifier approach.

## Features ✨

- Core Scrublet algorithm implemented in Rust
- Sparse matrix operations for memory efficiency
- Parallel processing using Rayon
- Seamless integration with AnnData objects through SingleRust
- Support for Squared Euclidean and Manhattan distance metrics

## Installation 💾

Add StrangerCells to your Cargo.toml:

```toml
[dependencies]
stranger_cells = "0.1.0"
```

## Usage 🚀

```rust
use stranger_cells::{DistanceMeasure, StrangerCellsParams, run_doublet_removal};
use anndata_memory::IMAnnData;
use single_rust::memory::processing::dimred::FeatureSelectionMethod;

fn main() -> anyhow::Result<()> {
    // Load AnnData object
    let adata = IMAnnData::read_h5ad("my_data.h5ad")?;
    
    // Configure parameters
    let params = StrangerCellsParams {
        sim_doublet_ratio: 2.0,
        n_neighbors: 30,
        expected_doublet_rate: 0.1,
        stddev_doublet_rate: 0.02,
        random_sate: 42,
        use_approx_neighbors: true,
        log1p: true,
        norm_target: 1_000_000,
        verbose: true,
        feature_selection_method: Some(FeatureSelectionMethod::RandomSelection(1000)),
        center: Some(true),
        max_iter: None,
        doublet_threshold: None,
    };
    
    // Run doublet detection
    let (predicted_doublets, z_scores, threshold, 
         detected_rate, detectable_fraction, overall_rate) = 
        run_doublet_removal::<f32, 50>(&adata, params, DistanceMeasure::SquaredEuclidean)?;
    
    println!("Detected {} doublets", 
        predicted_doublets.iter().filter(|&&x| x).count());
    
    Ok(())
}
```

## How It Works 🔍

The algorithm follows the same approach as the original Scrublet:

1. Generates synthetic doublets by combining random cell pairs
2. Uses PCA for dimensionality reduction
3. Builds a KNN classifier to identify real cells similar to synthetic doublets
4. Calculates doublet scores and applies thresholding

## Future Improvements 🔮

Planned enhancements for future releases:

- VAE implementation for improved doublet detection
- Additional dimensionality reduction options
- Improved doublet simulation strategies
- Smart thresholding algorithms

## License 📄

BSD 3-Clause License

Copyright (c) 2025 Ian F. Diks

All rights reserved.

## Acknowledgments 👏

- Original Scrublet algorithm by Samuel L. Wolock, et al.