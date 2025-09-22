import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# Import your models
from models.unet import Unet, Encoder
from models.vae import VAE
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

class InferenceDataset(Dataset):
    def __init__(self, im_path, im_size):
        """
        Dataset for inference on PNG images
        """
        self.im_path = im_path
        self.im_size = im_size
        
        # Find all PNG image files
        self.image_files = []
        
        # Support both single directory and nested directory structures
        if os.path.isfile(im_path) and im_path.lower().endswith('.png'):
            # Single file
            self.image_files = [im_path]
        else:
            # Directory or nested directories
            for root, dirs, files in os.walk(im_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        self.image_files.append(os.path.join(root, file))
        
        if not self.image_files:
            raise ValueError(f"No PNG files found in {im_path}")
        
        print(f"Found {len(self.image_files)} PNG images")
    
    def __len__(self):
        return len(self.image_files)
    
    def get_image_path(self, idx):
        return self.image_files[idx]
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load PNG image using PIL
            img = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            if img.size != (self.im_size, self.im_size):
                img = img.resize((self.im_size, self.im_size), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert to [-1, 1] range
            img_array = 2 * img_array - 1
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            
            return img_tensor
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros(3, self.im_size, self.im_size)

def save_png_image(image_array, output_path):
    """
    Save numpy array as PNG file
    image_array: numpy array of shape (H, W, C) or (H, W) for grayscale
    Values should be in [0, 1] range
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clip values to [0, 1] and convert to [0, 255]
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    
    if len(image_array.shape) == 2:  # Grayscale
        img = Image.fromarray(image_array, mode='L')
    else:  # RGB
        img = Image.fromarray(image_array, mode='RGB')
    
    img.save(output_path)

def get_output_filename(input_path, output_dir, suffix):
    """
    Generate output filename based on input path
    """
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_{suffix}.png"
    return os.path.join(output_dir, output_filename)


def inference(args):
    # Load config
    with open(args.config_path, 'r') as file:
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
    checkpoint_path = args.model_checkpoint 
    print(checkpoint_path)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
        else:
            model.load_state_dict(strip_orig_mod(checkpoint))
        print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load VAE
    vae_path = os.path.join("checkpoints", train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        model_state_dict = strip_prefix_if_present(checkpoint_vae['model_state_dict'], '_orig_mod.')
        vae.load_state_dict(model_state_dict)
        print('VAE loaded successfully')
    else:
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    
    # Create dataset and dataloader
    inference_dataset = InferenceDataset(args.input_path, dataset_config['im_size'])
    dataloader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create output directories
    albedo_dir = os.path.join(args.output_path, 'albedo')
    shading_dir = os.path.join(args.output_path, 'shading')
    os.makedirs(albedo_dir, exist_ok=True)
    os.makedirs(shading_dir, exist_ok=True)
    
    # Create wrapped model for sampling
    wrapped_model = WrappedModel(model, encoder)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Process images
    with torch.no_grad():
        for batch_idx, ldr_batch in enumerate(tqdm(dataloader, desc="Processing images")):
            ldr_batch = ldr_batch.to(device)
            batch_size = ldr_batch.size(0)
            
            # Get latent dimensions
            latent_channels = autoencoder_model_config['z_channels']
            latent_size = train_config['im_size_lt']
            
            # Generate initial noise
            x_init = torch.randn(batch_size, latent_channels, latent_size, latent_size).to(device)
            
            # Encode LDR image
            ldr_encoded = encoder(ldr_batch)
            wrapped_model.set_condition(ldr_encoded)
            
            # Sample shading latents
            shading_latents = solver.sample(
                x_init=x_init,
                method='euler',
                step_size=1,  # Adjust based on your needs
                return_intermediates=False
            )
            
            # Decode shading latents to image space
            shading_images = vae.decoder(shading_latents)
            
            # Process each image in the batch
            for i in range(batch_size):
                # Get input image path
                img_idx = batch_idx * args.batch_size + i
                if img_idx >= len(inference_dataset):
                    break
                    
                input_path = inference_dataset.get_image_path(img_idx)
                
                # Convert tensors to numpy
                ldr_np = ldr_batch[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                shading_np = shading_images[i].cpu().numpy().squeeze()  # Remove channel dim for grayscale
                
                # Convert from [-1,1] to [0,1]
                ldr_final = (ldr_np + 1) / 2
                shading_final = (shading_np + 1) / 2
                
                # Calculate albedo: albedo = ldr / shading 
                shading_3d = np.stack([shading_final] * 3, axis=2)  # Make shading 3-channel
                albedo_final = ldr_final / (shading_3d)
                albedo_final = np.clip(albedo_final, 0, 1)  # Ensure valid range
                
                # Generate output paths
                albedo_path = get_output_filename(input_path, albedo_dir, "albedo")
                shading_path = get_output_filename(input_path, shading_dir, "shading")
                
                # Save images as PNG
                save_png_image(albedo_final, albedo_path)
                save_png_image(shading_final, shading_path)
                
                if (img_idx + 1) % 10 == 0:
                    print(f"Processed {img_idx + 1} images")
    
    print(f"Inference completed! Results saved to:")
    print(f"  Albedo: {albedo_dir}")
    print(f"  Shading: {shading_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for LDR to Albedo/Shading separation')
    parser.add_argument('--config', dest='config_path', 
                        default='config/unet_hyperism.yaml', type=str,
                        help='Path to config file')
    parser.add_argument('--model_checkpoint', type=str,default="checkpoints/result.pth",
                        help='Path to trained model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input PNG images (file or directory)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory (will create albedo/ and shading/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    inference(args)