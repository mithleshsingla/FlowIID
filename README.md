# FlowIID: Single-Step Intrinsic Image Decomposition via Latent Flow Matching

**FlowIID** is a novel approach for **Intrinsic Image Decomposition (IID)** that separates an input image into **Albedo** (reflectance) and **Shading** components using **Latent Flow Matching**. Unlike existing methods that require multiple inference steps or large parameter counts, FlowIID achieves competitive results in a **single forward pass** with only **52M parameters**.

<p align="center">
  <img src="docs/model_architecture.png" width="1200"/>
</p>  

---

## 🔹 What is Flow Matching?

**Flow Matching** is a generative modeling technique that learns to **transform a simple prior distribution (e.g., Gaussian) into a complex data distribution** by solving an Ordinary Differential Equation (ODE).

Instead of iteratively denoising (like diffusion models), flow matching directly learns a **velocity field** that tells us how to move particles in latent space toward realistic samples.  

<p align="center">
  <img src="docs/flow_matching.png" width="600"/>
</p>  

In this project:  
- We first compress shading images using a **VAE** into latent space.  
- Then, we train a **UNet-based flow matching network** on these latents.  
- We pass our input image to **Encoder** that passes features from last **3** layers to the **Unet** network.
- Then we apply **Euler's** method on our model output **(velocity)** to get the latent representation.
- We pass our latent representation through the decoder to get the output **Shading** image.
- Then we divide our input image with the **Shading** image to get the **Albedo** component: **A = I / S**.

---

## 📖 Method Overview

### 🔹 Variational Autoencoder (VAE)
- Trained on **shading** images resized to **256×256×3**.  
- Latent space: **8 × 32 × 32**.  
- Loss function combines pixel-wise, perceptual, KL divergence, and adversarial terms.  

#### Loss Functions

**Reconstruction (L2) loss:**

```math
\mathcal{L}_{\text{rec}} = \|s_0 - \hat{s}_0\|_2^2
```

**Perceptual (feature) loss** (using a fixed feature extractor $\phi$, e.g., VGG):

```math
\mathcal{L}_{\text{perc}} = \|\phi(s_0) - \phi(\hat{s}_0)\|_2^2
```

**Kullback–Leibler divergence** (with prior $p(z) = \mathcal{N}(0, I)$ and posterior $`q_\phi(z \mid s_0)`$):

```math
\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left(q_\phi(z \mid s_0)\,\|\,p(z)\right)
```

**Adversarial (GAN) loss:**

```math
\mathcal{L}_A = \mathbb{E}_{s_0} [\log (D (s_0))] + \mathbb{E}_z [\log (1 - D (G (z)))]
```

**Total VAE objective:**

For first 90 epochs:
```math
\mathcal{L}(E, D) = \mathcal{L}_{\text{rec}} + 0.005 \cdot \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{perc}}
```

For subsequent 200 epochs:
```math
\mathcal{L}(E, D) = \mathcal{L}_{\text{rec}} + 0.005 \cdot \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{perc}} + 0.1 \cdot \mathcal{L}_A
```

### 🔹 Flow Matching Network
- Based on **UNet + encoder features**
- Trained on latent representation of size **8 × 32 × 32**.
- ODE solved using **Euler method** with just **1** timestep

#### Flow Matching Loss

```math
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_t}\big[\,\|\,u_\theta(x_t, t) - v_t\,\|_2^2\big]
```

---

## 📊 Results

### Qualitative Results

| Input | Albedo | Shading |
|-------|--------|---------|
| <img src="docs/input_images/_DSC4383.png" width="200"> | <img src="docs/albedo/_DSC4383.png" width="200"> | <img src="docs/shading/_DSC4383.png" width="200"> |
| <img src="docs/input_images/frame_0000.png" width="200"> | <img src="docs/albedo/frame_0000.png" width="200"> | <img src="docs/shading/frame_0000.png" width="200"> |
| <img src="docs/input_images/frame_0007.png" width="200"> | <img src="docs/albedo/frame_0007.png" width="200"> | <img src="docs/shading/frame_0007.png" width="200"> |
| <img src="docs/input_images/frame_0008.png" width="200"> | <img src="docs/albedo/frame_0008.png" width="200"> | <img src="docs/shading/frame_0008.png" width="200"> |
| <img src="docs/input_images/frame_0009.png" width="200"> | <img src="docs/albedo/frame_0009.png" width="200"> | <img src="docs/shading/frame_0009.png" width="200"> |

### Comparison with Previous Methods

<p align="center">
  <img src="docs/collage.pdf" width="800"/>
</p>

The collage shows comparison of our FlowIID with existing state-of-the-art methods, demonstrating superior albedo and shading decomposition quality.

---

### Quantitative Results
#### ARAP Dataset - Albedo Results
*Note: * implies model is finetuned on ARAP dataset*

| Method | LMSE↓ | RMSE↓ | SSIM↑ |
|--------|-------|-------|-------|
| Niid-net* | 0.023 | 0.129 | **0.788** |
| Lettry et al. | 0.042 | 0.163 | 0.670 |
| Kocsis et al. | 0.030 | 0.160 | 0.738 |
| Zhu et al. | 0.029 | 0.184 | 0.729 |
| IntrinsicAnything | 0.038 | 0.171 | 0.692 |
| Careaga and Aksoy | 0.025 | 0.140 | 0.671 |
| PIENet | 0.031 | 0.139 | 0.718 |
| Careaga and Aksoy (2024) | 0.023 | 0.145 | 0.700 |
| **FlowIID (Ours)** | **0.021** | **0.108** | 0.760 |

#### ARAP Dataset - Shading Results
*Note: * implies model is finetuned on ARAP dataset*

| Method | LMSE↓ | RMSE↓ | SSIM↑ |
|--------|-------|-------|-------|
| Niid-net* | 0.022 | 0.206 | **0.781** |
| Lettry et al. | 0.042 | 0.193 | 0.610 |
| Careaga and Aksoy | 0.026 | 0.168 | 0.680 |
| PIENet | 0.037 | 0.170 | 0.718 |
| **FlowIID (Ours)** | **0.022** | **0.132** | 0.744 |

#### MIT Intrinsic Dataset

| Method | Albedo ||| Shading |||
|--------|--------|--------|--------|--------|--------|--------|
|| MSE↓ | LMSE↓ | DSSIM↓ | MSE↓ | LMSE↓ | DSSIM↓ |
| CasQNet | 0.0091 | 0.0212 | 0.0730 | 0.0081 | 0.0192 | 0.0659 |
| PAIDNet | 0.0038 | 0.0239 | 0.0368 | **0.0032** | 0.0267 | **0.0475** |
| USI3D | 0.0156 | 0.0640 | 0.1158 | 0.0102 | 0.0474 | 0.1310 |
| CGIntrinsics | 0.0167 | 0.0319 | 0.1287 | 0.0127 | 0.0211 | 0.1376 |
| PIENet | **0.0028** | 0.0126 | **0.0340** | 0.0035 | 0.0203 | 0.0485 |
| **FlowIID (Ours)** | 0.0040 | **0.0043** | 0.0435 | 0.0109 | **0.0119** | 0.0823 |
---

## 🔹 Dataset Preprocessing

1. **Download datasets** (Hypersim, InteriorVerse, MID)  
   - Extract **albedo** and **HDR** images.  
2. **Tonemap HDR → LDR** (without gamma correction).  
3. **Normalize** to range [0, 1].  
4. **Compute shading ground truth**:  
   
   ```math
   \text{Shading} = \frac{\text{HDR}}{\text{Albedo}}
   ```

5. Final ground truth images:  
   - Albedo  
   - Shading  
   - LDR input image  

---

## 📂 Repository Structure
```
├── README.md                    # Project overview
├── requirements.txt            # Python dependencies
├── .gitattributes
├── checkpoints/               # Saved model weights
│   ├── epoch_290_best_autoencoder_model_checkpoint.pth
│   └── result.pth
├── config/                    # YAML configuration files
│   ├── autoen_alb.yaml
│   ├── fine.yaml
│   └── unet_hyperism.yaml
├── data_preprocessing/        # Data preprocessing utilities
│   ├── calculate_shading.py
│   ├── download.py
│   ├── hdr_to_ldr_converter.py
│   └── resize_image.py
├── docs/                      # Documentation & figures
│   ├── FlowIID.pdf
│   ├── collage.pdf
│   ├── flow_matching.png
│   ├── model_architecture.png
│   ├── input_images/         # Sample input images
│   ├── albedo/               # Sample albedo results
│   └── shading/              # Sample shading results
├── eval/                      # Evaluation scripts
│   ├── eval_arap.py
│   ├── eval_mit.py
│   └── mit_finetune.py
├── inference.py              # Main inference script
├── models/                   # Model architectures
│   ├── discriminator.py
│   ├── unet.py
│   └── vae.py
└── src/                      # Training scripts
    ├── dataloader_image_hyperism.py
    ├── dataloader_latent_hyperism.py
    ├── train_unet.py
    └── train_vae.py
```

---

## 🚀 Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/FlowIID.git
cd FlowIID
pip install -r requirements.txt
```

---

## 🔹 Usage

### Train VAE
```bash
python src/train_vae.py --config config/autoen_alb.yaml
```

### Train Flow Matching Network
```bash
python src/train_unet.py --config config/unet_hyperism.yaml
```

### Inference
```bash
python inference.py
```

### Evaluation
```bash
python eval/eval_arap.py
python eval/eval_mit.py
```

---

## 🔧 Model Architecture Details

### VAE Components
- **Encoder**: 6 downsampling blocks processing 3×H×W input
- **Decoder**: Generates shading from 8×H/8×W/8 latent representation
- **Latent space**: 8 × 32 × 32 for 256×256 input images

### Flow Matching Network
- **UNet**: 2 downsampling + 2 upsampling blocks with skip connections
- **Modified Residual Blocks (MRB)**: Integrated in both UNet and encoder
- **Feature fusion**: Encoder features from last 3 blocks added to UNet
- **Attention**: Selective attention layers for efficiency-accuracy balance

### Training Strategy
1. **Stage 1**: VAE + Discriminator training (290 epochs total)
   - First 90 epochs: Reconstruction + KL + Perceptual loss
   - Next 200 epochs: Add adversarial loss
2. **Stage 2**: Flow Matching training (250 epochs)
   - Batch size: 32
   - Learning rate: 1×10⁻⁴

---

## 📌 Key Features

- **Single-step inference**: Results in just one forward pass
- **Parameter efficient**: Only **52M parameters** vs. hundreds of millions in competing methods
- **Fast training**: Deterministic training compared to stochastic diffusion methods
- **Strong performance**: Competitive or superior results on MIT Intrinsic and ARAP benchmarks
- **Practical deployment**: Suitable for real-time and resource-constrained applications

---
