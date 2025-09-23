import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import yaml
from collections import OrderedDict

# Import your models
from models.unet import Unet, Encoder
from models.vae import VAE
# Import from the flow matching library
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def strip_prefix_if_present(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def strip_orig_mod(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict

class WrappedModel(ModelWrapper):
    def __init__(self, model, encoder=None):
        super().__init__(model)
        self.encoder = encoder
        self.condition = None
    
    def set_condition(self, condition):
        self.condition = condition
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if self.condition is None:
            raise ValueError("Condition not set. Call set_condition() first.")
        return self.model(x, t, self.condition)

# Load models globally
print("Loading models...")

# Load config
with open('config/unet_hyperism.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset_config = config['dataset_params_input']
autoencoder_model_config = config['autoencoder_params']
train_config = config['train_params']

# Initialize models
encoder = Encoder(im_channels=dataset_config['im_channels']).to(device)
model = Unet(im_channels=autoencoder_model_config['z_channels']).to(device)
vae = VAE(latent_dim=8).to(device)

# Set models to evaluation mode
encoder.eval()
model.eval()
vae.eval()

# Load model checkpoint
checkpoint_path = "checkpoints/result.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
        encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
    else:
        model.load_state_dict(strip_orig_mod(checkpoint))
    print("Model loaded successfully")

# Load VAE
vae_path = os.path.join("checkpoints", train_config['vae_autoencoder_ckpt_name'])
if os.path.exists(vae_path):
    print(f'Loading VAE checkpoint from {vae_path}')
    checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
    model_state_dict = strip_prefix_if_present(checkpoint_vae['model_state_dict'], '_orig_mod.')
    vae.load_state_dict(model_state_dict)
    print('VAE loaded successfully')

# Create wrapped model for sampling
wrapped_model = WrappedModel(model, encoder)
solver = ODESolver(velocity_model=wrapped_model)

print("Models loaded successfully!")

def process_image(input_image):
    """Process a single image and return albedo and shading components"""
    try:
        if input_image is None:
            return None, None, "Please upload an image"
        
        # Resize to model input size
        img_size = dataset_config['im_size']
        input_image = input_image.resize((img_size, img_size), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(input_image, dtype=np.float32) / 255.0
        img_array = 2 * img_array - 1  # Convert to [-1, 1] range
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get latent dimensions
            latent_channels = autoencoder_model_config['z_channels']
            latent_size = train_config['im_size_lt']
            
            # Generate initial noise
            x_init = torch.randn(1, latent_channels, latent_size, latent_size).to(device)
            
            # Encode LDR image
            ldr_encoded = encoder(img_tensor)
            wrapped_model.set_condition(ldr_encoded)
            
            # Sample shading latents
            shading_latents = solver.sample(
                x_init=x_init,
                method='euler',
                step_size=1,
                return_intermediates=False
            )
            
            # Decode shading latents to image space
            shading_images = vae.decoder(shading_latents)
            
            # Convert tensors to numpy
            ldr_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
            shading_np = shading_images[0].cpu().numpy().squeeze()
            
            # Convert from [-1,1] to [0,1]
            ldr_final = (ldr_np + 1) / 2
            shading_final = (shading_np + 1) / 2
            
            # Calculate albedo: albedo = ldr / shading 
            shading_3d = np.stack([shading_final] * 3, axis=2)
            albedo_final = ldr_final / (shading_3d + 1e-8)
            albedo_final = np.clip(albedo_final, 0, 1)
            
            # Convert back to PIL Images (both as RGB for better display)
            albedo_pil = Image.fromarray((albedo_final * 255).astype(np.uint8))
            shading_display = np.stack([shading_final] * 3, axis=2)
            shading_pil = Image.fromarray((shading_display * 255).astype(np.uint8))
            
            return albedo_pil, shading_pil, "Success!"
            
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return None, None, error_msg

# Create Gradio interface
title = "FlowIID: Single-Step Intrinsic Image Decomposition"
description = """
Upload an image to decompose it into **albedo** (reflectance) and **shading** components using FlowIID.

- **Albedo**: Material properties and colors
- **Shading**: Illumination and shadows

📄 [Read our paper](./docs/FlowIID.pdf)
"""

# Use your sample images as examples
examples = [
    ["docs/input_images/_DSC4383.png"],
    ["docs/input_images/frame_0000.png"],
    ["docs/input_images/frame_0007.png"],
]

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image")
    ],
    outputs=[
        gr.Image(type="pil", label="Albedo (Reflectance)"),
        gr.Image(type="pil", label="Shading (Illumination)"),
        gr.Textbox(label="Status")
    ],
    title=title,
    description=description,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()