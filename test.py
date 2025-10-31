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

with open('stable_diffusion/v2-inference-v.yaml', 'r') as f:
    config = yaml.safe_load(f)

embed_dim = config['model']['params']['first_stage_config']['params']['embed_dim']
ddconfig = config['model']['params']['first_stage_config']['params']['ddconfig']
lossconfig = config['model']['params']['first_stage_config']['params']['lossconfig']

"""
本脚本：
1) 读取 SD v2 配置，构建 AutoencoderKL
2) （可选）加载预训练 VAE 权重：
    - 通过环境变量 VAE_CKPT 指定 ckpt 路径，或项目根目录存在 'vae.ckpt'（包含 'state_dict' 的权重）
3) 将输入图像归一化到 [-1, 1]
4) 选择合适的 device（CUDA 优先）
5) 对两张图片进行编码，打印 posterior 的 mean/logvar 形状与一次采样的形状
"""

# device 选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载图片（支持跨平台路径）
root = Path(__file__).resolve().parent
image_path1 = root / 'images' / '1.png'
image_path2 = root / 'images' / '1.png'  # 如需对比不同图片，修改此处
image1 = Image.open(str(image_path1)).convert('RGB')
image2 = Image.open(str(image_path2)).convert('RGB')

# 2. 定义图像的预处理
target_res = ddconfig.get('resolution', 256)
if target_res % 64 != 0:
    print(f"[Warn] ddconfig.resolution={target_res} 不是 64 的倍数，可能与 VAE 下采样不匹配。")

transform = transforms.Compose([
    transforms.Resize((target_res, target_res)),  # SD VAE 常用 64 的倍数
    transforms.ToTensor(),
    # 将 [0,1] 归一化为 [-1,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 3. 预处理图像
input_image = transform(image1).unsqueeze(0)  # 增加批次维度

# 4. 加载模型 + （可选）预训练权重
# 优先读取环境变量 VAE_CKPT；若不存在，则尝试项目根目录下的 'vae.ckpt'
ckpt_candidates = []
env_ckpt = os.environ.get('VAE_CKPT')
if env_ckpt:
    ckpt_candidates.append(Path(env_ckpt))
ckpt_candidates.append(root / 'vae.ckpt')

ckpt_path = None
for p in ckpt_candidates:
    if p.is_file():
        ckpt_path = str(p)
        break

if ckpt_path:
    print(f"Loading VAE weights from: {ckpt_path}")
else:
    print("[Info] 未找到 VAE 预训练权重（VAE_CKPT 或 ./vae.ckpt）。将使用随机初始化权重运行，仅验证形状是否正确。")

model = AutoencoderKL(ddconfig, lossconfig, embed_dim, ckpt_path=ckpt_path)
model.eval()
model.to(device)

# 5. 使用编码器进行编码
with torch.no_grad():
    latents = []
    for idx, image in enumerate([image1, image2], 1):
        input_image = transform(image).unsqueeze(0).to(device)
        posterior = model.encode(input_image)
        z = posterior.mean  # 使用均值作为确定性 embedding
        latents.append(z.detach().cpu())

        # 基本形状信息
        print(f"Image {idx}:")
        print("  input shape:", tuple(input_image.shape))
        print("  mean shape:", tuple(posterior.mean.shape))
        print("  logvar shape:", tuple(posterior.logvar.shape))

    # 将两张图的潜变量展平成 [1, D] 的向量
    x = latents[0].view(1, -1).numpy().astype(np.float32)
    y = latents[1].view(1, -1).numpy().astype(np.float32)

    # 直接给出一个简单的相似度作为快速参考
    def cosine_sim(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return float((a @ b.T)[0, 0])

    cos = cosine_sim(x, y)
    print(f"\n[Quick metric] Cosine similarity of flattened VAE latents: {cos:.6f}")

    # 使用 spec 的 DiffEmbed_by_covariance_matrix 做对比（与 embedding_compare.py 保持一致风格）
    try:
        # 将 numpy 转回 torch 以复用 gaussian_covariance 的实现
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        # 这些超参可以按需调整；rff_dim 越大越稳定，但更耗时
        rff_dim = 512
        sigma = 10.0
        batchsize = 128

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

        # 输出前若干个特征值，作为差异性刻画
        topk = min(5, eigenvalues.numel())
        top_vals = eigenvalues[:topk].detach().cpu().numpy()
        print(f"[SPEC] Top-{topk} eigenvalues:", np.round(top_vals, 6))
    except Exception as e:
        print("[SPEC] 对比失败：", repr(e))
        print("可以稍后调小 rff_dim 或调整 sigma，或确保 spec 依赖完整。")