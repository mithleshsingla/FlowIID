import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from unet_img_my import Unet
from vae_new import VAE
from unet_img_my import Encoder
import gc
from dataloader_image_hyperism import HDRGrayscaleEXRDataset_new,ImageDataset_d
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import logging
import re
import random
import torchvision.transforms as transforms
import torchvision.utils as vutils
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device index: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from torch.amp import autocast, GradScaler
#scaler = GradScaler()
torch.set_float32_matmul_precision('high')
import wandb
wandb.init(project="ldr_to_al_training_latent_flow_matching")  
from collections import OrderedDict


def strip_prefix_if_present(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def to_rgb(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    return image

def reparameterize( mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class CombinedDataset(Dataset):
    def __init__(self, sh_dataset, ldr_dataset, augment=True, augmentation_prob=0.5):
        """
        A dataset that matches corresponding images across the three datasets based on scene metadata.
        
        Args:
            sh_dataset: The HDRGrayscaleEXRDataset for spherical harmonics shading
            ldr_dataset: The ImageDataset for LDR input (dequantize.exr)
            augment: Whether to apply augmentations
            augmentation_prob: Probability of applying augmentations to each sample
        """
        self.sh_dataset = sh_dataset
        self.ldr_dataset = ldr_dataset
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        
        # Create a mapping from scene info to indices for each dataset
        self.matching_indices = self._find_matching_indices()
        
        # Define augmentation transforms (modify as needed for your data)
        self.augmentation_transforms = self._get_augmentation_transforms()
        
        print(f"Found {len(self.matching_indices)} matching image triplets out of:")
        print(f"  {len(sh_dataset)} shading images")
        print(f"  {len(ldr_dataset)} LDR images")
        print(f"Augmentation: {'Enabled' if augment else 'Disabled'}")
    
    def _get_augmentation_transforms(self):
        """Define augmentation transforms that can be applied to both images consistently"""
        # Note: Adjust these transforms based on your specific image types and requirements
        return {
            'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
            'vertical_flip': transforms.RandomVerticalFlip(p=1.0),
            'rotation': transforms.RandomRotation(degrees=(-10, 10), fill=0),
            'crop_and_resize': transforms.RandomResizedCrop(
                size=(256, 256),  # Adjust size as needed
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            # Add more transforms as needed
        }
    
    def _apply_synchronized_augmentation(self, sh_image, ldr_image):
        """Apply the same augmentation to both images"""
        if not self.augment or random.random() > self.augmentation_prob:
            return sh_image, ldr_image
        
        # Convert to PIL Images or tensors as needed by your transforms
        # This assumes your images are already in the correct format
        
        # Choose which augmentations to apply randomly
        available_augs = list(self.augmentation_transforms.keys())
        num_augs = random.randint(0, min(2, len(available_augs)))  # Apply 0-2 augmentations
        selected_augs = random.sample(available_augs, num_augs)
        
        # Set the same random seed for both images to ensure identical transforms
        for aug_name in selected_augs:
            transform = self.augmentation_transforms[aug_name]
            
            # Set random seed to ensure same transform is applied to both images
            seed = random.randint(0, 2**32 - 1)
            
            # Apply to sh_image
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            sh_image = transform(sh_image)
            
            # Apply to ldr_image with same seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            ldr_image = transform(ldr_image)
        
        return sh_image, ldr_image
    
    def _find_matching_indices(self):
        """Find matching indices across all three datasets based on scene info"""
        # Create dictionaries to map scene info to indices for each dataset
        sh_indices = {}
        ldr_indices = {}
        
        # Create key-to-index mappings for each dataset
        for idx in range(len(self.sh_dataset)):
            info = self.sh_dataset.get_scene_info(idx)
            key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
            sh_indices[key] = idx
        
        for idx in range(len(self.ldr_dataset)):
            info = self.ldr_dataset.get_scene_info(idx)
            key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
            ldr_indices[key] = idx
        
        # Find common keys across all datasets
        sh_keys = set(sh_indices.keys())
        ldr_keys = set(ldr_indices.keys())

        common_keys = sh_keys.intersection(ldr_keys)

        # Create a list of matching indices
        matching_indices = [
            (sh_indices[key], ldr_indices[key]) 
            for key in common_keys
        ]
        
        return matching_indices
    
    def __len__(self):
        return len(self.matching_indices)
    
    def __getitem__(self, idx):
        # Get the matching indices for all three datasets
        sh_idx, ldr_idx = self.matching_indices[idx]
        
        # Get the items from each dataset
        sh_image = self.sh_dataset[sh_idx]
        ldr_image = self.ldr_dataset[ldr_idx]
        
        # Apply synchronized augmentations
        sh_image, ldr_image = self._apply_synchronized_augmentation(sh_image, ldr_image)
        
        # Also store the scene info for saving output images
        #info = self.sh_dataset.get_scene_info(sh_idx)
        
        return sh_image, ldr_image

def convert_state_dict(old_state_dict):
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("decoder.model"):
            parts = k.split(".")
            layer_idx = int(parts[2])
            subkey = ".".join(parts[3:])
            # Map old indices to new deconv blocks
            if layer_idx in [0, 1, 2]:
                new_key = f"decoder.deconv1.{layer_idx}.{subkey}"
            elif layer_idx in [3, 4, 5]:
                new_key = f"decoder.deconv2.{layer_idx-3}.{subkey}"
            elif layer_idx in [6, 7, 8]:
                new_key = f"decoder.deconv3.{layer_idx-6}.{subkey}"
            elif layer_idx in [9, 10, 11]:
                new_key = f"decoder.deconv4.{layer_idx-9}.{subkey}"
            elif layer_idx in [12, 13]:
                new_key = f"decoder.deconv5.{layer_idx-12}.{subkey}"
            else:
                continue  # Skip unknown
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def strip_orig_mod(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


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

# Add this after creating all datasets, before starting the training loop
def check_dataset_sync(im_dataset, im_dataset_encoder, im_shading):
    """Check if the two datasets are synchronized"""
    # Skip if using latents directly
    if not hasattr(im_dataset, 'images') or not isinstance(im_dataset.images, list) or not im_dataset.images:
        print("Skipping sync check - dataset doesn't have image list")
        return
    
    if not hasattr(im_dataset_encoder, 'image_files') or not im_dataset_encoder.image_files:
        print("Skipping sync check - encoder dataset doesn't have image_files list")
        return

    if not hasattr(im_shading, 'image_files') or not im_shading.image_files:
        print("Skipping sync check - shading dataset doesn't have image_files list")
        return
    
    # Get basenames for comparison
    im_basenames = [os.path.basename(str(path)) for path in im_dataset.images[:5]]
    encoder_basenames = [os.path.basename(str(path)) for path in im_dataset_encoder.image_files[:5]]
    shading_basenames = [os.path.basename(str(path)) for path in im_shading.image_files[:5]]

    print("First 5 images in latent dataset:", im_basenames)
    print("First 5 images in encoder dataset:", encoder_basenames)
    print("First 5 images in shading dataset:", shading_basenames)
    # Check if they follow the same pattern (might not be exactly the same files)
    im_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_dataset.images[0]) if im_dataset.images else "")
    encoder_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_dataset_encoder.image_files[0]) if im_dataset_encoder.image_files else "")
    shading_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_shading.image_files[0]) if im_shading.image_files else "")

    if im_pattern and encoder_pattern and shading_pattern and im_pattern.group(1) != encoder_pattern.group(1) and im_pattern.group(1) != shading_pattern.group(1):
        print(f"WARNING: Datasets might be using different file patterns: {im_pattern.group(1)} vs {encoder_pattern.group(1)}")

def save_training_samples(output, image, gt_image, train_config, step_count, img_save_count, guidance_scale=None, suffix=""):
    """
    Save training samples with support for different guidance scales and directories
    
    Args:
        output: Generated samples
        image: Original/shading images
        gt_image: Ground truth images
        train_config: Training configuration
        step_count: Current training step
        img_save_count: Image save counter
        guidance_scale: CFG guidance scale (for directory naming)
        suffix: Additional suffix for filename
    """
    
    sample_size = min(8, output.shape[0])
    gt_image = gt_image[:sample_size].detach().cpu()
    save_output = output[:sample_size].detach().cpu()
    shading_image = image[:sample_size].detach().cpu()

    # Base save path
    #base_save_path = os.path.join('/home/project/mithlesh/Hyperism', train_config['task_name'], 'Flow_samples')
    base_save_path = os.path.join('/home/project/dataset/Hyperism', train_config['task_name'], 'Flow_samples')

    # Create guidance-scale specific directory if guidance_scale is provided
    if guidance_scale is not None:
        guidance_dir = f"guidance_{guidance_scale}"
        save_path = os.path.join(base_save_path, guidance_dir)
    else:
        save_path = base_save_path
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create collage
    collage = torch.cat([save_output, shading_image], dim=0)
    collage_g=torch.cat([gt_image], dim=0)
    # Construct filename with suffix
    filename = f"{step_count}{suffix}.png"
    filename_2 = f"{step_count}{suffix}_ldr.png"
    output_path = os.path.join(save_path, filename)
    output_path_2 = os.path.join(save_path, filename_2)

    vutils.save_image(collage, output_path, nrow=4, normalize=True)
    vutils.save_image(collage_g, output_path_2, nrow=4, normalize=True)
    # Also save numbered samples in subdirectory
    # numbered_save_path = os.path.join(save_path, 'numbered_samples')
    # os.makedirs(numbered_save_path, exist_ok=True)
    
    # numbered_filename = f"sample_{img_save_count:06d}{suffix}.png"
    # numbered_output_path = os.path.join(numbered_save_path, numbered_filename)
    # vutils.save_image(collage, numbered_output_path, nrow=4, normalize=True)
    
    return img_save_count + 1

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
    dataset_shading = config['dataset_params_shading']
    dataset_config = config['dataset_params_input']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Add image saving configuration
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    step_count = 0
    
    # Check if latent path is an H5 file
    latent_path = train_config['vae_latent_dir_name']
    use_h5_latents = latent_path.endswith('.h5')
    
    print(f"Using latent path: {latent_path}")
    print(f"Use H5 latents: {use_h5_latents}")
    print(f"train_config: {train_config['im_size_lt']}")
    print(f"Image save steps: {image_save_steps}")
    
    # im_dataset_cls = {
    #     'ldr_to_sh_flow': ldr_to_sh_Dataset,
    # }.get(dataset_config['name'])
    
    # Initialize models
    encoder = Encoder(im_channels=dataset_config['im_channels']).to(device)
    encoder.train()

    # Instantiate the Unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels']).to(device)
    model.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters
    encoder_params = count_parameters(encoder)
    model_params = count_parameters(model)

    # Print the results
    print(f"Encoder Parameters: {encoder_params:,}")
    print(f"Unet Model Parameters: {model_params:,}")
    print(f"Total Parameters: {encoder_params + model_params:,}")

    # Load VAE - Now we always load it for decoder functionality
    print('Loading VAE model for decoder functionality')
    vae = VAE(latent_dim=8).to(device)
    vae.eval()
  
    # Load vae if found
    vae_path = os.path.join("ldr_to_sh", train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        model_state_dict = strip_prefix_if_present(checkpoint_vae['model_state_dict'], '_orig_mod.')
        #checkpoint_vae['model_state_dict'] = convert_state_dict(checkpoint_vae['model_state_dict'])
        #vae.load_state_dict(checkpoint_vae['model_state_dict'])
        vae.load_state_dict(model_state_dict)
        print('VAE loaded successfully')
    else:
        print(f'Warning: VAE checkpoint not found at {vae_path}. Decoder visualization will not work properly.')

    # Freeze VAE parameters since we're only using it for visualization
    for param in vae.parameters():
        param.requires_grad = False
        
    
    shading_dataset = HDRGrayscaleEXRDataset_new(im_path=dataset_shading['im_path'],
                               im_size=dataset_config['im_size'])
    
    # Load the encoder dataset (always uses images, not latents)
    ldr_dataset = ImageDataset_d(im_path=dataset_config['im_path'],
                               im_size=dataset_config['im_size'], file_suffix='reconstructed_ldr.exr')
    
    combined_dataset = CombinedDataset(shading_dataset, ldr_dataset)
    
    # Split dataset into train and validation (90:10 ratio)
    dataset_size = len(combined_dataset)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    
    #train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    
    
    indices = np.arange(len(combined_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    data_loader = DataLoader(train_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True , num_workers=4 , pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    val_loader = DataLoader(val_dataset,
                           batch_size=train_config['ldm_batch_size'],
                           shuffle=False , num_workers=4 , pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer_model = Adam(model.parameters(), lr=train_config['ldm_lr'])
    optimizer_encoder = Adam(encoder.parameters(), lr=train_config['ldm_lr'])
    
    # Add learning rate schedulers
    scheduler_model = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.1, patience=5,min_lr=0.00001)
    scheduler_encoder = ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.1, patience=5,min_lr=0.00001)
    scaler = GradScaler()

    # Initialize variables for tracking best model
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(train_config['task_name'], exist_ok=True)

    # Try to load existing checkpoints if available
    checkpoint_path="/home/project/ldr_image_to_ldr_shading/LDR_image_to_LDR_shading_hyperism/train_vae_mithlesh/ldr_to_sh_flow/flow_model_ckpt.pth"
    #checkpoint_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            step_count = checkpoint.get('step_count', 0)
            img_save_count = checkpoint.get('img_save_count', 0)
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        else:
            model.load_state_dict(strip_orig_mod(checkpoint))
            print("Loaded model weights only")
            start_epoch = 0
    else:
        start_epoch = 0

    
    # Check if the model is already compiled
    if hasattr(torch, 'compile'):
       model = torch.compile(model)
       encoder = torch.compile(encoder)
    
    # Training loop
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        flow_losses = []
        lpm_losses = []
        
        # Create a zip iterator but ensure it has the correct length (from the shorter of the two loaders)
        # data_iter = zip(data_loader, data_loader_encoder,data_loader_albedo)
        # total_batches = min(len(data_loader), len(data_loader_encoder), len(data_loader_albedo))

        # Training phase
        model.train()
        encoder.train()

        for batch_idx, (shading_im,cond_img) in enumerate(tqdm(data_loader)):
            step_count += 1
            
            optimizer_encoder.zero_grad()
            optimizer_model.zero_grad()
            
            cond_img = cond_img.float().to(device)
            shading_im = shading_im.float().to(device)
            
            # Sample random noise
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                #_,_,im=vae(shading_im)
                _, mu, logvar = vae.encoder(shading_im)
                im = reparameterize(mu, logvar)
                noise = torch.randn_like(im).to(device)
                #t = torch.rand(im.shape[0]).to(device)
                if step_count % 2 == 0:
                    t = torch.zeros(im.shape[0], device=device)  # t=0
                else:
                    t= torch.rand(im.shape[0], device=device)  # uniform
                
                path_sample = path.sample(t=t, x_0=noise, x_1=im)
            
                # Process conditional image through encoder
                encoder_out = encoder(cond_img)
                
            
                # Calculate flow matching loss
                model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                #loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)
                # Base flow matching loss
                flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)

                # Reconstruct image from predicted and ground truth latents
                # Total loss: flow + perceptual
                loss = flow_loss 

            losses.append(loss.item())
            flow_losses.append(flow_loss.item())
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_model)
            scaler.unscale_(optimizer_encoder)

            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

            scaler.step(optimizer_model)
            scaler.step(optimizer_encoder)
            scaler.update()
            
            # Log batch loss to W&B
            wandb.log({"batch_loss": loss.item(), "step": step_count})
            
            # Image Saving Logic - Generate proper samples using velocity integration
            
            # Image Saving Logic - Generate proper samples using velocity integration
            if step_count % image_save_steps == 0 or step_count == 1:
                model.eval()
                encoder.eval()
                
                with torch.no_grad():
                    try:
                        
                        # Create wrapped model for sampling (similar to inference code)
                        wrapped_model = WrappedModel(model, encoder)
                        solver = ODESolver(velocity_model=wrapped_model)
                        
                        # Use fewer time steps for training visualization (faster)
                        #T = get_time_discretization(10, rho=5).to(device)  # Use 10 steps instead of 20 for faster training
                        
                        # Generate initial noise with same shape as latent
                        current_batch_size = cond_img.size(0)
                        #latent_channels = train_config.get('latent_channels', 4)  # Adjust based on your config
                        latent_channels = config['autoencoder_params']['z_channels']
                        latent_size = train_config['im_size_lt']
                        
                        x_init = torch.randn(current_batch_size, latent_channels, latent_size, latent_size).to(device)
                        
                        # Process condition images through encoder
                        cond_encoded = encoder(cond_img)
                        
                        # Set condition for the wrapped model
                        wrapped_model.set_condition(cond_encoded)
                        
                        # Sample from the model (integrate velocity to get actual sample)
                        samples = solver.sample(
                            #time_grid=T,
                            x_init=x_init,
                            method='euler',
                            step_size=1,  # Larger step size for faster training visualization
                            return_intermediates=False
                        )
                        
                        # Now decode the proper samples using VAE decoder
                        samples = samples.float()
                        decoded_output = vae.decoder(samples)
                        #image=vae.decoder(im)
                        
                        # Save the decoded samples
                        img_save_count = save_training_samples(
                            decoded_output,shading_im, cond_img, train_config, step_count, img_save_count
                        )
                        #print(f"Saved decoded training samples at step {step_count}")
                        
                    except Exception as e:
                        print(f"Error saving training samples: {e}")
                        # Fallback: save a simple visualization of the velocity field
                        try:
                            # Just save the raw model output for debugging
                            velocity_viz = model_out.float()
                            # You could also save this for debugging purposes
                            print(f"Saved velocity visualization at step {step_count}")
                        except:
                            print("Failed to save any visualization")
                
        # End of training epoch
        train_loss_avg = np.mean(losses)
        train_lpm_avg = np.mean(lpm_losses)
        print('Finished epoch:{} | Average Training Loss: {:.4f}'.format(
            epoch_idx + 1, train_loss_avg))
        
        wandb.log({"epoch": epoch_idx + 1, "train_loss": train_loss_avg, "step": step_count})
        wandb.log({"epoch": epoch_idx + 1, "train_lpm": train_lpm_avg, "step": step_count})

        # Validation phase
        if len(val_dataset) > 0:
            model.eval()
            encoder.eval()
            val_losses = []
            val_flow_losses = []
            
            
            # Create validation iterator
            # val_iter = zip(val_loader, val_loader_encoder, val_loader_albedo)
            # val_total_batches = min(len(val_loader), len(val_loader_encoder), len(val_loader_albedo))

            with torch.no_grad():
                for   val_shading_im,val_cond_img in tqdm(val_loader, desc="Validation"):

                    val_cond_img = val_cond_img.float().to(device)
                    val_shading_im = val_shading_im.float().to(device)

                    # Sample random noise
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        #_,_,val_im=vae(val_shading_im)
                        _, mu, logvar = vae.encoder(val_shading_im)
                        val_im = reparameterize(mu, logvar)
                
                        noise = torch.randn_like(val_im).to(device)
                        t = torch.rand(val_im.shape[0]).to(device)
                    
                        path_sample = path.sample(t=t, x_0=noise, x_1=val_im)
                    
                        # Process conditional image through encoder
                        encoder_out = encoder(val_cond_img)
                        
                        # Calculate flow matching loss
                        model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                        #val_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)
                        val_flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)

                        # Reconstruct image from predicted and ground truth latents
                        
                        # Total loss: flow + perceptual
                        val_loss = val_flow_loss 

                        # for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        #     t = torch.full((val_im.shape[0],), t_val, device=device)
                        #     path_sample = path.sample(noise, val_im, t)
                        #     pred = model(path_sample.x_t, t, encoder_out)
                        #     error = torch.nn.functional.mse_loss(pred, path_sample.dx_t)
                        #     print(f"t={t_val}: error={error.item()}")

                    val_losses.append(val_loss.item())
                    val_flow_losses.append(val_flow_loss.item())
                    
            
            val_loss_avg = np.mean(val_losses)
            val_flow_loss = np.mean(val_flow_losses)
            
            print(f'Validation Loss: {val_loss_avg:.4f}')
            
            # Log validation metrics to W&B
            wandb.log({"val_loss": val_loss_avg, "step": step_count})
            wandb.log({"val_flow_loss": val_flow_loss, "step": step_count})
            
            
            # Update learning rate schedulers
            scheduler_model.step(val_loss_avg)
            scheduler_encoder.step(val_loss_avg)
            
            # Check if model improved
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                
                # Save best model
                best_checkpoint_dict = {
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_model_state_dict': optimizer_model.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss_avg,
                    'step_count': step_count,
                    'img_save_count': img_save_count
                }
                best_checkpoint_path = os.path.join(train_config['task_name'], 'best_' + train_config['ldm_ckpt_name'])
                torch.save(best_checkpoint_dict, best_checkpoint_path)
                print(f"Saved best model with validation loss: {val_loss_avg:.4f}")
            else:
                patience_counter += 1
                print(f"Validation did not improve. Patience: {patience_counter}/{patience}")
            
            if epoch_idx % 5 == 0:
                best_val_loss = val_loss_avg
                patience_counter = 0
                
                # Save best model
                best_checkpoint_dict = {
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_model_state_dict': optimizer_model.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss_avg,
                    'step_count': step_count,
                    'img_save_count': img_save_count
                }
                best_checkpoint_path = os.path.join(train_config['task_name'], 'epoch_' + str(epoch_idx + 1) + '_' + train_config['ldm_ckpt_name'])
                torch.save(best_checkpoint_dict, best_checkpoint_path)
                print(f"Saved best model with validation loss: {val_loss_avg:.4f}")
            
            # Early stopping check
            # if patience_counter >= patience:
            #     print(f"Early stopping triggered after {epoch_idx+1} epochs")
            #     break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Always save latest checkpoint
        checkpoint_dict = {
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss_avg,
            'step_count': step_count,
            'img_save_count': img_save_count
        }
        checkpoint_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
        torch.save(checkpoint_dict, checkpoint_path)
        
        # Save encoder separately for convenience
        encoder_checkpoint_path = os.path.join(train_config['task_name'], 'encoder_' + train_config['ldm_ckpt_name'])
        torch.save(encoder.state_dict(), encoder_checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        if epoch_idx % 1 == 0:  # Every 5 epochs
            gc.collect()
            torch.cuda.empty_cache()    
        
    
    print('Done Training...')
    
    # Load best model for final save
    best_checkpoint_path = os.path.join(train_config['task_name'], 'best_' + train_config['ldm_ckpt_name'])
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        print(f"Loaded best model with validation loss: {best_checkpoint['best_val_loss']:.4f}")
    
    # Final model save (state dict only, for inference)
    final_model_path = os.path.join(train_config['task_name'], 'final_model_for_inf.pth')
    final_encoder_path = os.path.join(train_config['task_name'], 'final_encoder_for_inf.pth')
    torch.save(model.state_dict(), final_model_path)
    torch.save(encoder.state_dict(), final_encoder_path)
    print(f"Saved final model to {final_model_path}")
    print(f"Saved final encoder to {final_encoder_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/unet_hyperism_1.yaml', type=str)
    
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    train(args)