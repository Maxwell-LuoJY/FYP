# Research Report: A Comparative Analysis of Stable Diffusion VAE Embeddings using the SPEC Framework

**Author:** GitHub Copilot
**Date:** October 31, 2025

## 1. Abstract

This report details an experiment designed to compare the latent representations of two distinct Variational Autoencoders (VAEs) used within the Stable Diffusion ecosystem: the standard VAE from version 2.0 ('v2') and the VAE from the x4 upscaling model ('x4'). Using a small, curated dataset of 15 images spanning three categories (cats, dogs, and humans), we extract deterministic embeddings from each VAE. The core of our analysis employs the SPEC (Spectral Principal-Component-like Embedding Comparison) framework to quantify and visualize the differences between these two embedding spaces. By applying a kernel-based method (approximated with Random Fourier Features) and performing a differential analysis on the covariance structures, we identify the principal modes of variation between the models. This methodology allows us to interpret not just *if* the models differ, but *how* they differ in their representation of the same visual data.

## 2. Introduction

Variational Autoencoders are a cornerstone of modern generative models like Stable Diffusion, responsible for compressing high-dimensional image data into a compact, structured latent space and reconstructing it. The quality and characteristics of this latent space are critical for the performance of the overall diffusion model. Different VAEs, trained for different purposes (e.g., general-purpose generation vs. upscaling), are expected to develop distinct representational biases.

Understanding these differences is crucial for model selection and fine-tuning. However, a direct comparison of high-dimensional, non-linear embedding spaces is non-trivial. The SPEC framework offers a principled approach to this problem by identifying a "difference subspace" that highlights the most significant variations between two embedding sets. This report documents the application of SPEC to compare the 'v2' and 'x4' VAEs, providing a reproducible pipeline for analysis.

## 3. Methodology

The experiment is structured into three main stages: data preparation, embedding extraction, and comparative analysis.

### 3.1. Dataset

The dataset is a small, local collection of 15 images, sourced from three folders: `images/cats/`, `images/dogs/`, and `images/human/`. This multi-category dataset, while small, provides a basis for observing how the VAEs might differ in representing distinct semantic concepts.

### 32. VAE Models and Configurations

Two VAE models are compared:
1.  **`v2` VAE**: The standard VAE from Stable Diffusion v2, configured via `stable_diffusion/v2-inference-v.yaml`. Its weights are loaded from `checkpoints/stable_diffusion_v2_vae.safetensors`.
2.  **`x4` VAE**: The VAE from the Stable Diffusion x4 upscaling model, configured via `stable_diffusion/x4-upscaling.yaml`. Its weights are loaded from `checkpoints/stable_diffusion_x4_vae.safetensors`.

### 3.3. Embedding Extraction

For each image in the dataset, the following pipeline is executed for both VAEs:
1.  **Preprocessing**: Images are loaded using PIL and converted to RGB. A `torchvision.transforms` pipeline resizes each image to the resolution specified in the model's configuration file (e.g., 256x256) and normalizes pixel values to the range `[-1, 1]`.
2.  **Encoding**: The preprocessed image tensor is passed to the `model.encode()` method of the `AutoencoderKL` instance.
3.  **Deterministic Embedding**: The encoder outputs a `DiagonalGaussianDistribution` representing the posterior distribution. We use the mean (`posterior.mean`) of this distribution as a deterministic, point-estimate embedding.
4.  **Flattening**: The resulting latent tensor, with shape `[1, C, H, W]`, is flattened into a 1D vector.

This process yields two embedding matrices, **X** (from `v2`) and **Y** (from `x4`), both of shape `[15, D]`, where `D` is the flattened latent dimension.

### 3.4. SPEC Comparative Analysis

The core comparison leverages the `DiffEmbed_by_covariance_matrix` function from the SPEC framework.
1.  **Kernel Approximation**: To capture non-linear relationships, a Gaussian kernel is used. To avoid computing the full `N x N` kernel matrix, we use **Random Fourier Features (RFF)** as an approximation. The `gaussian_covariance` function projects the embeddings **X** and **Y** into an RFF space, producing feature matrices `phi_x` and `phi_y`.
    $$ k(u, v) = \exp(-\frac{\|u-v\|^2}{2\sigma^2}) \approx \phi(u)^T \phi(v) $$
2.  **Differential Eigendecomposition**: The SPEC algorithm constructs a difference operator based on the covariance structures of `phi_x` and `phi_y`. The eigendecomposition of this operator yields eigenvalues (`λ`) and eigenvectors (`v`) that represent the principal modes of difference. A large eigenvalue indicates a direction in the feature space where the two embedding sets differ significantly.
3.  **Mode Visualization**: The `visualize_modes_covariance` function uses these results to provide interpretability. For each top mode (a high-eigenvalue eigenvector), it calculates a score for every sample in the dataset. The samples with the highest scores for a given mode are the ones that most exemplify that particular type of difference. These images are then saved as a grid, providing a visual summary of the mode.

## 4. Experimental Setup

The experiment is implemented in `test.py` with the following key parameters:
- **Device**: `cuda` if available, otherwise `cpu`.
- **RFF Dimension (`rff_dim`)**: 512
- **Gaussian Kernel Sigma (`sigma`)**: 10.0
- **Batch Size (`batchsize`)**: 1 (for covariance calculation)
- **SPEC `eta` parameter**: 1.0
- **Number of Modes to Visualize**: 10 (or fewer if not enough eigenvalues)
- **Samples per Mode**: 20 (or fewer if the dataset is smaller)

**Dependencies:** `torch`, `pytorch-lightning`, `numpy`, `pyyaml`, `pillow`, `matplotlib`, `safetensors`.

## 5. Expected Results and Interpretation

The execution of `test.py` will produce the following outputs:
1.  **Console Output**:
    *   The shape of the embedding matrices for `v2` and `x4`.
    *   A list of the top-k eigenvalues from the SPEC analysis. The magnitude of these values indicates the strength of the corresponding difference mode.
2.  **File Outputs** (in `outputs/v2_vs_x4/`):
    *   `compare_cholesky.png`: A 1D scatter plot of all computed eigenvalues.
    *   `mode={i}_summary.jpg`: For each of the top modes, a grid of images that best represent that mode of difference. For example, if Mode 0 primarily shows images of cats with fine fur texture, it suggests the two VAEs differ significantly in how they represent such textures.
    *   `eigenvalues.npy` and `eigenvectors.npy`: The raw numerical results for further analysis.

**Interpretation**: By examining the images associated with each top mode, we can form hypotheses about the representational biases of the two VAEs. For instance, the 'x4' upscaling VAE might preserve finer texture details, which would surface as a mode distinguishing images with complex textures (e.g., fur, fabric) from those with smooth surfaces. Conversely, the 'v2' VAE might focus more on global structure.

## 6. Limitations and Future Work

The primary limitation of this study is the extremely small dataset size (N=15). This limits the statistical significance of the findings. Future work should expand the dataset to hundreds or thousands of images across more diverse categories.

Additionally, the choice of hyperparameters (`rff_dim`, `sigma`) can influence the results. A sensitivity analysis by running the experiment with a range of these values would provide more robust conclusions.

Finally, this analysis could be extended to compare other VAEs, or to study the effect of fine-tuning on a VAE's latent space.

## 7. Conclusion

This report outlines a clear and reproducible methodology for the comparative analysis of VAE latent spaces using the SPEC framework. The `test.py` script provides a complete pipeline from data loading to visualization, enabling a qualitative and quantitative assessment of the differences between the Stable Diffusion 'v2' and 'x4' VAEs. While the current experiment is a small-scale proof-of-concept, it establishes a strong foundation for more extensive future studies into the representational properties of generative models.
