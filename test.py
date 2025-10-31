import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from vae import AutoencoderKL
import yaml
import numpy as np

# spec-align comparison utilities
from spec.utils import gaussian_covariance
from spec.diff_embeddigs import EmbeddingEvaluation

"""
目标：
1) 将 images/cats 与 images/dogs 中的 10 张图片组成一个数据集
2) 分别用 v2 与 x4 两套配置/权重将该数据集编码为 VAE embeddings
3) 使用 SPEC 框架对比两套 embeddings 的差异
"""

# device 选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

root = Path(__file__).resolve().parent

def load_config(config_path: Path):
    with open(str(config_path), 'r') as f:
        cfg = yaml.safe_load(f)
    fs = cfg['model']['params']['first_stage_config']['params']
    return fs['embed_dim'], fs['ddconfig'], fs['lossconfig']

def build_transform(resolution: int):
    if resolution % 64 != 0:
        print(f"[Warn] ddconfig.resolution={resolution} 不是 64 的倍数，可能与 VAE 下采样不匹配。")
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def list_images(folders):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = []
    for folder in folders:
        p = root / 'images' / folder
        if not p.is_dir():
            print(f"[Warn] 图像文件夹不存在: {p}")
            continue
        for fp in sorted(p.iterdir()):
            if fp.suffix.lower() in exts:
                files.append(fp)
    return files

def load_pil_images(paths):
    ims = []
    for p in paths:
        try:
            ims.append(Image.open(str(p)).convert('RGB'))
        except Exception as e:
            print(f"[Skip] 无法读取图像 {p}: {e}")
    return ims

def encode_dataset(model: AutoencoderKL, transform, pil_images):
    model.eval().to(device)
    latents = []
    with torch.no_grad():
        for idx, img in enumerate(pil_images, 1):
            inp = transform(img).unsqueeze(0).to(device)
            posterior = model.encode(inp)
            z = posterior.mean  # 确定性 embedding
            latents.append(z.detach().cpu().view(1, -1))
            if idx <= 2:
                # 打印前两张的形状供参考
                print(f"  [debug] img{idx}: input={tuple(inp.shape)}, mean={tuple(posterior.mean.shape)}, logvar={tuple(posterior.logvar.shape)}")
    if not latents:
        return np.zeros((0, 0), dtype=np.float32)
    X = torch.cat(latents, dim=0).numpy().astype(np.float32)  # [N, D]
    return X

# 两套设置：名称、配置文件、checkpoint
model_specs = [
    {
        'name': 'v2',
        'config': root / 'stable_diffusion' / 'v2-inference-v.yaml',
        'ckpt': root / 'checkpoints' / 'stable_diffusion_v2_vae.safetensors',
    },
    {
        'name': 'x4',
        'config': root / 'stable_diffusion' / 'x4-upscaling.yaml',
        'ckpt': root / 'checkpoints' / 'stable_diffusion_x4_vae.safetensors',
    }
]

# 1) 组装数据集（cats+human+dogs 共 15 张）
img_paths = list_images(['cats', 'human', 'dogs'])
pil_images = load_pil_images(img_paths)
print(f"Loaded {len(pil_images)} images from cats+human+dogs.")

embeds_dict = {}

for spec in model_specs:
    name = spec['name']
    cfg_path = spec['config']
    ckpt_path = spec['ckpt']
    if not ckpt_path.is_file():
        print(f"[Warn] {name} 的 checkpoint 不存在，跳过：{ckpt_path}")
        continue
    print(f"\n[{name}] config: {cfg_path.name}, ckpt: {ckpt_path.name}")

    embed_dim, ddconfig, lossconfig = load_config(cfg_path)
    transform = build_transform(ddconfig.get('resolution', 256))

    model = AutoencoderKL(ddconfig, lossconfig, embed_dim, ckpt_path=str(ckpt_path))
    X = encode_dataset(model, transform, pil_images)
    print(f"[{name}] embeddings shape: {tuple(X.shape)}")
    embeds_dict[name] = X

# 3) SPEC 对比
if len(embeds_dict) >= 2 and ('v2' in embeds_dict and 'x4' in embeds_dict):
    x = embeds_dict['v2']
    y = embeds_dict['x4']
    try:
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        rff_dim = 512
        sigma = 10.0
        batchsize = 1

        cov_x, _, phi_x = gaussian_covariance(
            x_t.float(), rff_dim=rff_dim, batchsize=batchsize, sigma=sigma, return_features=True
        )
        cov_y, _, phi_y = gaussian_covariance(
            y_t.float(), rff_dim=rff_dim, batchsize=batchsize, sigma=sigma, return_features=True
        )

        spec = EmbeddingEvaluation(sigma=0)
        eigenvalues, eigenvectors = spec.DiffEmbed_by_covariance_matrix(
            x=x, y=y, cov_function=None, phi_x=phi_x, phi_y=phi_y, eta=1
        )

        topk = min(5, eigenvalues.numel())
        top_vals = eigenvalues[:topk].detach().cpu().numpy()
        print(f"\n[SPEC] Top-{topk} eigenvalues:", np.round(top_vals, 6))
    except Exception as e:
        print("[SPEC] 对比失败：", repr(e))
        print("可以稍后调小 rff_dim 或调整 sigma，或确保 spec 依赖完整。")
else:
    print("[Info] 需要同时得到 v2 与 x4 的 embeddings 才能进行 SPEC 对比。请检查 checkpoints 是否齐全。")

# 可视化与模式分析（在成功得到 eigenvalues/eigenvectors 时执行）
try:
    from spec.utils import visualize_modes_covariance
    import matplotlib.pyplot as plt
    save_dir = root / 'outputs' / 'v2_vs_x4'
    os.makedirs(save_dir, exist_ok=True)
    # 简单的特征值散点
    if 'eigenvalues' in locals():
        plt.figure()
        plt.scatter(eigenvalues.real.cpu(), [0]*eigenvalues.shape[0], s=5, c='blue')
        plt.savefig(str(save_dir / 'compare_cholesky.png'))
        # 模式可视化（使用我们刚刚编码的 PIL 图片列表）
        visualize_modes_covariance(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            x_feature=phi_x,
            y_feature=phi_y,
            num_visual_mode=min(10, eigenvalues.numel()),
            num_samples_per_mode=min(20, x.shape[0]),
            save_dir=str(save_dir),
            dataset=pil_images,  # 直接使用 PIL 图像列表
            data_type='image',
            save_file=True,
            x=x,
            y=y,
            model_names=('v2', 'x4'),
            plot_tsne=False  # 避免额外依赖
        )
except Exception as e:
    print('[Vis] 可视化跳过：', repr(e))