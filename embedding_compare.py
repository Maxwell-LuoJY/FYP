import numpy as np
import torch
from spec.utils import gaussian_covariance
from spec.diff_embeddigs import EmbeddingEvaluation

# Each .npz holds an array of shape [N, D]
clip = np.load('features/clip_imagenet.npz')['features']
dino = np.load('features/dino_imagenet.npz')['features']

sigma_clip = 3.5
sigma_dino = 25.0

cov_clip, _, phi_clip = gaussian_covariance(
    torch.from_numpy(clip).float(),
    rff_dim=2000,
    batchsize=128,
    sigma=sigma_clip,
    return_features=True)

cov_dino, _, phi_dino = gaussian_covariance(
    torch.from_numpy(dino).float(),
    rff_dim=2000,
    batchsize=128,
    sigma=sigma_dino,
    return_features=True)

spec = EmbeddingEvaluation(sigma=0)
eigenvalues, eigenvectors = spec.DiffEmbed_by_covariance_matrix(
    x=clip,
    y=dino,
    cov_function=None,
    phi_x=phi_clip,
    phi_y=phi_dino,
    eta=1)

from spec.utils import visualize_modes_covariance

# Try to import ImageFilesDataset, but make it optional
try:
    from spec_core.dataset import ImageFilesDataset
    has_dataset_class = True
except ImportError:
    print("Warning: spec_core.dataset not available, skipping image visualization")
    has_dataset_class = False

# Load image paths if available
try:
    image_paths = np.load('paths/imagenet_paths.npy')
    if has_dataset_class:
        dataset = ImageFilesDataset(path='', name='imagenet-val', path_files=image_paths)
    else:
        dataset = None
except FileNotFoundError:
    print("Warning: Image paths not found, skipping image visualization")
    dataset = None

visualize_modes_covariance(
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    x_feature=phi_clip,
    y_feature=phi_dino,
    num_visual_mode=10,
    num_samples_per_mode=20,
    save_dir='outputs/clip_vs_dino/',
    dataset=dataset,
    save_file=(dataset is not None),  # Only save files if dataset is available
    x=clip,
    y=dino,
    model_names=('CLIP', 'DINOv2'),
    plot_tsne=False  # Disable t-SNE to avoid pacmap dependency issues
)