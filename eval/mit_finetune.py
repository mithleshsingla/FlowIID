import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from unet_img_my_4 import Unet
from vae_new import VAE
from unet_img_my_4 import Encoder
import gc
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import logging
import re
import random
import torchvision.transforms as transforms
import torchvision.utils as vutils
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from torch.utils.data import Dataset
from PIL import Image
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device index: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")
loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from torch.amp import autocast, GradScaler
torch.set_float32_matmul_precision('high')
import wandb
wandb.init(project="mit_intrinsic_finetuning")  
from collections import OrderedDict

class MITIntrinsicDataset(Dataset):
    def __init__(self, root_dir, im_size=256, augment=True, augmentation_prob=0.5, num_lights=10):
        """
        MIT-Intrinsic Dataset loader
        
        Args:
            root_dir: Path to MIT-intrinsic train directory (/mnt/zone/B/mithlesh/dataset/mit/MIT-intrinsic/train)
            im_size: Target image size (256)
            augment: Whether to apply augmentations
            augmentation_prob: Probability of applying augmentations
            num_lights: Number of light images per scene (10)
        """
        self.root_dir = root_dir
        self.im_size = im_size
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.num_lights = num_lights
        
        # Find all scenes (directories in root_dir)
        self.scenes = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
        self.scenes.sort()
        
        # Create data pairs (scene, light_idx)
        self.data_pairs = []
        valid_scenes = []
        
        for scene in self.scenes:
            scene_path = os.path.join(root_dir, scene)
            
            # Check for light and corresponding shading images
            light_shading_pairs = []
            for i in range(1, num_lights + 1):
                light_path = os.path.join(scene_path, f'light{i:02d}.png')
                shading_path = os.path.join(scene_path, f'shading{i:02d}.png')
                
                if os.path.exists(light_path) and os.path.exists(shading_path):
                    light_shading_pairs.append(i)
                else:
                    if not os.path.exists(light_path):
                        print(f"Warning: Missing {light_path}")
                    if not os.path.exists(shading_path):
                        print(f"Warning: Missing {shading_path}")
            
            if len(light_shading_pairs) > 0:  # At least some pairs exist
                valid_scenes.append(scene)
                for light_idx in light_shading_pairs:
                    self.data_pairs.append((scene, light_idx))  # (scene_name, light_number)
        
        self.scenes = valid_scenes
        
        # Image transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
        ])
        
        # Augmentation transforms
        self.augmentation_transforms = self._get_augmentation_transforms()
        
        print(f"MIT-Intrinsic Dataset initialized:")
        print(f"  Found {len(self.scenes)} valid scenes")
        print(f"  Total data pairs: {len(self.data_pairs)}")
        print(f"  Augmentation: {'Enabled' if augment else 'Disabled'}")
        if len(self.scenes) > 0:
            print(f"  Sample scenes: {self.scenes[:5]}")
    
    def _get_augmentation_transforms(self):
        """Define augmentation transforms that can be applied to both images consistently"""
        return {
            'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
            'vertical_flip': transforms.RandomVerticalFlip(p=1.0),
            'rotation': transforms.RandomRotation(degrees=(-10, 10), fill=0),
            'crop_and_resize': transforms.RandomResizedCrop(
                size=(self.im_size, self.im_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
        }
    
    def _apply_synchronized_augmentation(self, light_image, shading_image):
        """Apply the same augmentation to both images"""
        if not self.augment or random.random() > self.augmentation_prob:
            return light_image, shading_image
        
        # Choose which augmentations to apply randomly
        available_augs = list(self.augmentation_transforms.keys())
        num_augs = random.randint(0, min(2, len(available_augs)))  # Apply 0-2 augmentations
        selected_augs = random.sample(available_augs, num_augs)
        
        # Set the same random seed for both images to ensure identical transforms
        for aug_name in selected_augs:
            transform = self.augmentation_transforms[aug_name]
            
            # Set random seed to ensure same transform is applied to both images
            seed = random.randint(0, 2**32 - 1)
            
            # Apply to light_image
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            light_image = transform(light_image)
            
            # Apply to shading_image with same seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            shading_image = transform(shading_image)
        
        return light_image, shading_image
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        scene_name, light_idx = self.data_pairs[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        # Load light image
        light_path = os.path.join(scene_path, f'light{light_idx:02d}.png')
        light_image = Image.open(light_path).convert('RGB')
        
        # Load corresponding shading image (now specific to this light)
        shading_path = os.path.join(scene_path, f'shading{light_idx:02d}.png')
        shading_image = Image.open(shading_path).convert('L')
        
        # Apply base transforms
        light_tensor = self.base_transform(light_image)
        shading_tensor = self.base_transform(shading_image)
        #print(f"shading_tensor shape: {shading_tensor.shape}, light_tensor shape: {light_tensor.shape}")
        
        # Apply synchronized augmentations
        light_tensor, shading_tensor = self._apply_synchronized_augmentation(light_tensor, shading_tensor)
        
        return shading_tensor, light_tensor  # (ground_truth_shading, input_light)

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


def to_rgb(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    return image


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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


def save_training_samples(output, image, gt_image, train_config, step_count, img_save_count, guidance_scale=None, suffix=""):
    """
    Save training samples with support for different guidance scales and directories
    """
    sample_size = min(8, output.shape[0])
    gt_image = gt_image[:sample_size].detach().cpu()
    save_output = output[:sample_size].detach().cpu()
    light_image = image[:sample_size].detach().cpu()

    # Base save path for MIT intrinsic
    base_save_path = os.path.join(train_config['task_name'], 'MIT_Flow_samples')

    # Create guidance-scale specific directory if guidance_scale is provided
    if guidance_scale is not None:
        guidance_dir = f"guidance_{guidance_scale}"
        save_path = os.path.join(base_save_path, guidance_dir)
    else:
        save_path = base_save_path
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create collage: [generated_shading, input_light]
    collage = torch.cat([save_output, light_image], dim=0)
    collage_gt = torch.cat([gt_image], dim=0)
    
    # Construct filename with suffix
    filename = f"generated_{step_count}{suffix}.png"
    filename_gt = f"ground_truth_{step_count}{suffix}.png"
    output_path = os.path.join(save_path, filename)
    output_path_gt = os.path.join(save_path, filename_gt)

    vutils.save_image(collage, output_path, nrow=4, normalize=True)
    vutils.save_image(collage_gt, output_path_gt, nrow=4, normalize=True)
    
    return img_save_count + 1


def finetune_mit_intrinsic(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Update config for MIT dataset
    train_config = config['train_params']
    autoencoder_model_config = config['autoencoder_params']
    
    # MIT-specific parameters
    mit_dataset_path = "/mnt/zone/B/mithlesh/dataset/mit/MIT-intrinsic/train"
    
    # Add image saving configuration
    image_save_steps = train_config.get('autoencoder_img_save_steps', 500)
    img_save_count = 0
    step_count = 0
    
    print(f"MIT Dataset path: {mit_dataset_path}")
    print(f"Image save steps: {image_save_steps}")
    
    # Initialize models
    encoder = Encoder(im_channels=3).to(device)  # RGB input from PNG
    encoder.train()

    # Instantiate the Unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels']).to(device)
    model.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters
    encoder_params = count_parameters(encoder)
    model_params = count_parameters(model)

    print(f"Encoder Parameters: {encoder_params:,}")
    print(f"Unet Model Parameters: {model_params:,}")
    print(f"Total Parameters: {encoder_params + model_params:,}")

    # Load VAE for encoder/decoder functionality
    print('Loading VAE model for decoder functionality')
    vae = VAE(latent_dim=8).to(device)
    vae.eval()
  
    # Load VAE checkpoint
    vae_path = os.path.join("ldr_to_sh", train_config['vae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        model_state_dict = strip_prefix_if_present(checkpoint_vae['model_state_dict'], '_orig_mod.')
        vae.load_state_dict(model_state_dict)
        print('VAE loaded successfully')
    else:
        print(f'Warning: VAE checkpoint not found at {vae_path}. Decoder visualization will not work properly.')

    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create MIT intrinsic dataset
    mit_dataset = MITIntrinsicDataset(
        root_dir=mit_dataset_path,
        im_size=256,
        augment=args.augment,  # Can be controlled via command line
        augmentation_prob=0.5,
        num_lights=10
    )
    
    # Split dataset into train and validation (90:10 ratio)
    dataset_size = len(mit_dataset)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = np.arange(len(mit_dataset))
    np.random.shuffle(indices)  # Shuffle for random split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(mit_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(mit_dataset, val_indices)

    print(f"MIT Intrinsic Dataset:")
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    data_loader = DataLoader(train_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True, prefetch_factor=2)
    
    val_loader = DataLoader(val_dataset,
                           batch_size=train_config['ldm_batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True, 
                           persistent_workers=True, prefetch_factor=2)
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    
    # Use lower learning rate for fine-tuning
    finetune_lr = train_config.get('finetune_lr', train_config['ldm_lr'] * 0.1)
    optimizer_model = Adam(model.parameters(), lr=finetune_lr)
    optimizer_encoder = Adam(encoder.parameters(), lr=finetune_lr)
    
    # Add learning rate schedulers
    scheduler_model = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.1, patience=5, min_lr=0.00001)
    scheduler_encoder = ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.1, patience=5, min_lr=0.00001)
    scaler = GradScaler()

    # Initialize variables for tracking best model
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())
    
    # Create checkpoint directory for MIT finetuning
    mit_task_name = train_config['task_name'] + '_MIT_finetune'
    os.makedirs(mit_task_name, exist_ok=True)

    # Load pre-trained checkpoint for fine-tuning
    pretrained_checkpoint_path = "/home/project/ldr_image_to_ldr_shading/LDR_image_to_LDR_shading_hyperism/train_vae_mithlesh/result"
    
    if os.path.exists(pretrained_checkpoint_path):
        print(f"Loading pre-trained checkpoint from {pretrained_checkpoint_path}")
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
            print(f"Loaded pre-trained model for fine-tuning")
        else:
            model.load_state_dict(strip_orig_mod(checkpoint))
            print("Loaded model weights only for fine-tuning")
        start_epoch = 0
    else:
        print(f"Warning: Pre-trained checkpoint not found at {pretrained_checkpoint_path}")
        start_epoch = 0

    # Check if resuming from previous MIT fine-tuning
    mit_checkpoint_path = os.path.join(mit_task_name, 'mit_finetune_ckpt.pth')
    if os.path.exists(mit_checkpoint_path):
        print(f"Resuming MIT fine-tuning from {mit_checkpoint_path}")
        mit_checkpoint = torch.load(mit_checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(strip_orig_mod(mit_checkpoint['model_state_dict']))
        encoder.load_state_dict(strip_orig_mod(mit_checkpoint['encoder_state_dict']))
        optimizer_model.load_state_dict(mit_checkpoint['optimizer_model_state_dict'])
        optimizer_encoder.load_state_dict(mit_checkpoint['optimizer_encoder_state_dict'])
        start_epoch = mit_checkpoint['epoch']
        best_val_loss = mit_checkpoint.get('best_val_loss', float('inf'))
        step_count = mit_checkpoint.get('step_count', 0)
        img_save_count = mit_checkpoint.get('img_save_count', 0)
        print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    
    # Compile models if available
    if hasattr(torch, 'compile'):
       model = torch.compile(model)
       encoder = torch.compile(encoder)
    
    print("Starting MIT Intrinsic fine-tuning...")
    
    # Training loop
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        flow_losses = []
        
        # Training phase
        model.train()
        encoder.train()

        for batch_idx, (shading_gt, light_input) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch_idx+1}/{num_epochs}")):
            step_count += 1
            
            optimizer_encoder.zero_grad()
            optimizer_model.zero_grad()
            
            light_input = light_input.float().to(device)  # Input light images
            shading_gt = shading_gt.float().to(device)    # Ground truth shading
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Encode ground truth shading to latent space
                _, mu, logvar = vae.encoder(shading_gt)
                shading_latent = reparameterize(mu, logvar)
                
                # Sample noise and time
                noise = torch.randn_like(shading_latent).to(device)
                
                t = torch.zeros(shading_latent.shape[0], device=device)

                # Sample path
                path_sample = path.sample(t=t, x_0=noise, x_1=shading_latent)
            
                # Process light input through encoder
                encoder_out = encoder(light_input)
                
                # Calculate flow matching loss
                model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)
                print(f"Batch {batch_idx+1}, Flow Loss: {flow_loss.item():.4f}")
                # recon_pred_z = path_sample.x_t + (1.0 - t).view(-1,1,1,1) * model_out  # Predicted latent
            
                # recon_pred_z = recon_pred_z.float()
                # output = vae.decoder(recon_pred_z)
                # output_rgb= to_rgb(output)
                # shading_rgb = to_rgb(shading_gt)
                # lpm = (loss_fn_alex(output_rgb, shading_rgb).mean())  # Ensure lpips_loss is a scalar
                # print(f"Batch {batch_idx+1}, LPIPS Loss: {lpm.item():.4f}")
                # Total loss
                loss =  flow_loss #+ 10*lpm

            losses.append(loss.item())
            flow_losses.append(flow_loss.item())

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_model)
            scaler.unscale_(optimizer_encoder)

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

            scaler.step(optimizer_model)
            scaler.step(optimizer_encoder)
            scaler.update()
            
            # Log batch loss to W&B
            wandb.log({"batch_loss": loss.item(), "step": step_count})
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                model.eval()
                encoder.eval()
                
                with torch.no_grad():
                    try:
                        # Create wrapped model for sampling
                        wrapped_model = WrappedModel(model, encoder)
                        solver = ODESolver(velocity_model=wrapped_model)
                        
                        # Generate initial noise with same shape as latent
                        current_batch_size = light_input.size(0)
                        latent_channels = config['autoencoder_params']['z_channels']
                        latent_size = train_config['im_size_lt']
                        
                        x_init = torch.randn(current_batch_size, latent_channels, latent_size, latent_size).to(device)
                        
                        # Process condition images through encoder
                        cond_encoded = encoder(light_input)
                        
                        # Set condition for the wrapped model
                        wrapped_model.set_condition(cond_encoded)
                        
                        # Sample from the model
                        samples = solver.sample(
                            x_init=x_init,
                            method='euler',
                            step_size=1,
                            return_intermediates=False
                        )
                        
                        # Decode the samples using VAE decoder
                        samples = samples.float()
                        decoded_output = vae.decoder(samples)
                        
                        # Save the decoded samples
                        img_save_count = save_training_samples(
                            decoded_output, light_input, shading_gt, 
                            {'task_name': mit_task_name}, step_count, img_save_count
                        )
                        
                    except Exception as e:
                        print(f"Error saving training samples: {e}")
                
                model.train()
                encoder.train()
        
        # End of training epoch
        train_loss_avg = np.mean(losses)
        train_flow_avg = np.mean(flow_losses)
        print('Finished epoch:{} | Average Training Loss: {:.4f}'.format(
            epoch_idx + 1, train_loss_avg))
        
        wandb.log({"epoch": epoch_idx + 1, "train_loss": train_loss_avg, "train_flow_loss": train_flow_avg, "step": step_count})

        # Validation phase
        if len(val_dataset) > 0:
            model.eval()
            encoder.eval()
            val_losses = []
            val_flow_losses = []

            with torch.no_grad():
                for val_shading_gt, val_light_input in tqdm(val_loader, desc="Validation"):

                    val_light_input = val_light_input.float().to(device)
                    val_shading_gt = val_shading_gt.float().to(device)

                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        # Encode validation shading to latent space
                        _, mu, logvar = vae.encoder(val_shading_gt)
                        val_shading_latent = reparameterize(mu, logvar)
                
                        noise = torch.randn_like(val_shading_latent).to(device)
                        #t = torch.rand(val_shading_latent.shape[0]).to(device)
                        t = torch.zeros(val_shading_latent.shape[0], device=device)
                        path_sample = path.sample(t=t, x_0=noise, x_1=val_shading_latent)
                    
                        # Process conditional image through encoder
                        encoder_out = encoder(val_light_input)
                        
                        # Calculate flow matching loss
                        model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                        val_flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)

                        # Total loss
                        val_loss = val_flow_loss 

                    val_losses.append(val_loss.item())
                    val_flow_losses.append(val_flow_loss.item())
            
            val_loss_avg = np.mean(val_losses)
            val_flow_loss_avg = np.mean(val_flow_losses)
            
            print(f'Validation Loss: {val_loss_avg:.4f}')
            
            # Log validation metrics to W&B
            wandb.log({"val_loss": val_loss_avg, "val_flow_loss": val_flow_loss_avg, "step": step_count})
            
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
                best_checkpoint_path = os.path.join(mit_task_name, 'best_mit_finetune_ckpt.pth')
                torch.save(best_checkpoint_dict, best_checkpoint_path)
                print(f"Saved best MIT model with validation loss: {val_loss_avg:.4f}")
            else:
                patience_counter += 1
                print(f"Validation did not improve. Patience: {patience_counter}/{patience}")
            
            # Save model every 5 epochs
            if epoch_idx % 50 == 0:
                epoch_checkpoint_dict = {
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
                epoch_checkpoint_path = os.path.join(mit_task_name, f'epoch_{epoch_idx + 1}_mit_finetune_ckpt.pth')
                torch.save(epoch_checkpoint_dict, epoch_checkpoint_path)
                print(f"Saved epoch {epoch_idx + 1} checkpoint")
        
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
        checkpoint_path = os.path.join(mit_task_name, 'mit_finetune_ckpt.pth')
        torch.save(checkpoint_dict, checkpoint_path)
        
        # Save encoder separately for convenience
        encoder_checkpoint_path = os.path.join(mit_task_name, 'encoder_mit_finetune_ckpt.pth')
        torch.save(encoder.state_dict(), encoder_checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if epoch_idx % 1 == 0:  # Every epoch
            gc.collect()
            torch.cuda.empty_cache()    
    
    print('Done MIT Intrinsic Fine-tuning...')
    
    # Load best model for final save
    best_checkpoint_path = os.path.join(mit_task_name, 'best_mit_finetune_ckpt.pth')
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        print(f"Loaded best MIT model with validation loss: {best_checkpoint['best_val_loss']:.4f}")
    
    # Final model save (state dict only, for inference)
    final_model_path = os.path.join(mit_task_name, 'final_mit_model_for_inf.pth')
    final_encoder_path = os.path.join(mit_task_name, 'final_mit_encoder_for_inf.pth')
    torch.save(model.state_dict(), final_model_path)
    torch.save(encoder.state_dict(), final_encoder_path)
    print(f"Saved final MIT model to {final_model_path}")
    print(f"Saved final MIT encoder to {final_encoder_path}")


def verify_mit_dataset(dataset_path):
    """
    Verify the MIT-intrinsic dataset structure and print statistics
    """
    print(f"Verifying MIT dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    scenes = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found {len(scenes)} scenes")
    
    valid_scenes = 0
    total_light_images = 0
    
    for scene in scenes[:5]:  # Check first 5 scenes
        scene_path = os.path.join(dataset_path, scene)
        print(f"\nChecking scene: {scene}")
        
        # Check shading (should be grayscale/1-channel)
        shading_path = os.path.join(scene_path, 'shading.png')
        has_shading = os.path.exists(shading_path)
        if has_shading:
            # Check if shading is actually 1-channel
            try:
                shading_img = Image.open(shading_path)
                shading_mode = shading_img.mode
                print(f"  - shading.png: ✓ (mode: {shading_mode})")
            except:
                print(f"  - shading.png: ✓ (could not read mode)")
        else:
            print(f"  - shading.png: ✗")
        
        # Check light images
        light_count = 0
        for i in range(1, 11):
            light_path = os.path.join(scene_path, f'light{i:02d}.png')
            if os.path.exists(light_path):
                light_count += 1
        
        print(f"  - light images: {light_count}/10")
        total_light_images += light_count
        
        if has_shading and light_count > 0:
            valid_scenes += 1
    
    print(f"\nSummary:")
    print(f"Valid scenes (first 5 checked): {valid_scenes}/5")
    print(f"Average light images per scene: {total_light_images/5:.1f}")
    
    return valid_scenes > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for MIT-intrinsic fine-tuning')
    parser.add_argument('--config', dest='config_path',
                        default='config/fine.yaml', type=str)
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='Disable data augmentation')
    parser.add_argument('--verify-dataset', action='store_true',
                        help='Verify dataset structure before training')
    
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    # Verify dataset if requested
    if args.verify_dataset:
        mit_dataset_path = "/mnt/zone/B/mithlesh/dataset/mit/MIT-intrinsic/train"
        verify_mit_dataset(mit_dataset_path)
        sys.exit(0)
    
    print(f"Data augmentation: {'Enabled' if args.augment else 'Disabled'}")
    finetune_mit_intrinsic(args)