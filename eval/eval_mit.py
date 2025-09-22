import numpy as np
import os
import sys
import torch
import yaml
from pathlib import Path
import argparse
from skimage.metrics import structural_similarity as ssim
import cv2
import glob
from PIL import Image

# Import your model components
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.vae import VAE
from models.unet import Unet, Encoder

def compute_psnr(true, pred, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = compute_mse(true, pred)
    if mse == 0 or mse == float('inf'):
        return float('inf') if mse == 0 else 0.0
    return 20 * np.log10(max_val / np.sqrt(mse))

def compute_mse(true, pred):
    """Mean Squared Error"""
    return np.mean((true - pred) ** 2)

def compute_lmse(true, pred, window_size=20, window_stride=10, epsilon=1e-6):
    """
    Compute Local Mean Squared Error (LMSE) between true and pred.
    """
    H, W = true.shape
    total_mse = 0.0
    count = 0
    
    for i in range(0, H - window_size + 1, window_stride):
        for j in range(0, W - window_size + 1, window_stride):
            true_win = true[i:i+window_size, j:j+window_size]
            pred_win = pred[i:i+window_size, j:j+window_size]
            
            mse = np.mean((true_win - pred_win) ** 2)
            total_mse += mse
            count += 1
    
    if count == 0:
        return float('inf')
    
    return total_mse / count

def compute_dssim(true, pred):
    """Dissimilarity version of SSIM (1 - SSIM)"""
    ssim_score = ssim(true, pred, data_range=1.0)
    return (1 - ssim_score) / 2

class WrappedModel(ModelWrapper):
    def __init__(self, model, encoder=None):
        super().__init__(model)
        self.encoder = encoder
        self.condition = None
    
    def set_condition(self, condition):
        self.condition = condition
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.forward(x, t, **extras)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if self.condition is None:
            raise ValueError("Condition not set. Call set_condition() first.")
        return self.model(x, t, self.condition)

class YourModelEstimator:
    """Your Deep Learning Model Estimator for MIT Dataset Evaluation"""
    
    def __init__(self, vae_checkpoint, flow_checkpoint, config_path, device='cuda:1'):
        self.device = 'cuda:1'  # As specified in original code
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        autoencoder_config = config['autoencoder_params']
        
        # Initialize models
        self.vae = VAE(latent_dim=8).to(self.device)
        self.encoder = Encoder(im_channels=3).to(self.device)
        self.model = Unet(im_channels=autoencoder_config['z_channels']).to(self.device)
        
        # Load checkpoints
        self._load_vae_checkpoint(vae_checkpoint)
        self._load_flow_checkpoint(flow_checkpoint)
        
        # Set to evaluation mode
        self.vae.eval()
        self.encoder.eval()
        self.model.eval()
        
        # Freeze parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Setup wrapped model and solver
        self.wrapped_model = WrappedModel(self.model, self.encoder)
        self.solver = ODESolver(velocity_model=self.wrapped_model)
    
    def _strip_prefix_if_present(self, state_dict, prefix='_orig_mod.'):
        """Remove prefix from state dict keys if present"""
        keys = list(state_dict.keys())
        if not any(key.startswith(prefix) for key in keys):
            return state_dict
        
        stripped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                stripped_key = key[len(prefix):]
                stripped_state_dict[stripped_key] = value
            else:
                stripped_state_dict[key] = value
        return stripped_state_dict
    
    def _load_vae_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint_vae = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
            model_state_dict = self._strip_prefix_if_present(checkpoint_vae['model_state_dict'], '_orig_mod.')
            self.vae.load_state_dict(model_state_dict)
            print('VAE loaded successfully')
        else:
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    
    def _load_flow_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(self._strip_prefix_if_present(checkpoint['model_state_dict']))
            self.encoder.load_state_dict(self._strip_prefix_if_present(checkpoint['encoder_state_dict']))
            print("Model and encoder loaded successfully")
        else:
            raise FileNotFoundError(f"Flow checkpoint not found: {checkpoint_path}")
    
    def _load_image(self, image_path, target_size=(256, 256)):
        """Load and preprocess image from file"""
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0  # Convert to [0,1]
        
        # Resize if needed
        if image.shape[:2] != target_size:
            image_uint8 = (image * 255).astype(np.uint8)
            image_resized = cv2.resize(image_uint8, target_size, interpolation=cv2.INTER_LINEAR)
            image = image_resized.astype(np.float32) / 255.0
        
        return image
    
    def _preprocess_image(self, image, target_size=(256, 256)):
        """Preprocess image for your model"""
        # Resize if needed
        if image.shape[:2] != target_size:
            image_uint8 = (image * 255).astype(np.uint8)
            if len(image.shape) == 3:
                image_resized = cv2.resize(image_uint8, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                image_resized = cv2.resize(image_uint8, target_size, interpolation=cv2.INTER_LINEAR)
            image = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and normalize to [-1, 1]
        if len(image.shape) == 2:  # Grayscale
            image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            image_tensor = image_tensor.repeat(3, 1, 1)  # Convert grayscale to RGB
        else:  # RGB
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] to [-1, 1]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        return image_tensor
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor from [-1, 1] to [0, 1] numpy array"""
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        # Remove batch dimension and convert to numpy
        tensor = tensor.squeeze(0)
        if tensor.shape[0] == 1:  # Grayscale
            return tensor.squeeze(0).cpu().numpy()
        else:  # RGB - convert to grayscale by taking mean
            tensor_rgb = tensor.permute(1, 2, 0).cpu().numpy()
            return np.mean(tensor_rgb, axis=2)
    
    def estimate_shading_from_light(self, light_image):
        """
        Estimate shading from a single light image
        
        Args:
            light_image: RGB light image (H, W, 3) numpy array in [0,1]
        
        Returns:
            shading: grayscale shading estimate (H, W) numpy array
        """
        with torch.no_grad():
            # Preprocess image for your model
            input_tensor = self._preprocess_image(light_image)
            
            # Encode condition
            cond_encoded = self.encoder(input_tensor)
            self.wrapped_model.set_condition(cond_encoded)
            
            # Generate initial noise
            batch_size = input_tensor.size(0)
            latent_size_h = int(input_tensor.size(2) / 8)
            latent_size_w = int(input_tensor.size(3) / 8)
            latent_channels = 8
            
            x_init = torch.randn(batch_size, latent_channels, latent_size_h, latent_size_w, device=self.device)
            
            # Sample from the model
            samples = self.solver.sample(
                x_init=x_init,
                method="euler",
                step_size=1,
                return_intermediates=False
            )
            
            # Decode to get predicted shading
            predicted_shading = self.vae.decoder(samples.float())
            
            # Convert shading back to numpy
            shading_np = self._tensor_to_numpy(predicted_shading)
            
            # Resize shading to match input image size if needed
            if shading_np.shape != light_image.shape[:2]:
                shading_uint8 = (shading_np * 255).astype(np.uint8)
                shading_resized = cv2.resize(shading_uint8, (light_image.shape[1], light_image.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
                shading_np = shading_resized.astype(np.float32) / 255.0
            
            return shading_np

def load_scene_data(dataset_path, scene_name):
    """
    Load all data for a scene
    
    Args:
        dataset_path: Path to MIT-intrinsic dataset
        scene_name: Name of the scene (e.g., 'cup2')
    
    Returns:
        light_images: List of light images
        shading_images: List of corresponding shading ground truth images
        reflectance_image: Single reflectance ground truth image
    """
    scene_path = os.path.join(dataset_path, 'test', scene_name)
    
    # Load light images (light01.png to light10.png)
    light_images = []
    light_files = sorted(glob.glob(os.path.join(scene_path, 'light*.png')))
    for light_file in light_files:
        light_img = Image.open(light_file).convert('RGB')
        light_images.append(np.array(light_img).astype(np.float32) / 255.0)
    
    # Load shading ground truth images (shading01.png to shading10.png)
    shading_images = []
    shading_files = sorted(glob.glob(os.path.join(scene_path, 'shading*.png')))
    for shading_file in shading_files:
        shading_img = Image.open(shading_file).convert('L')  # Convert to grayscale
        shading_images.append(np.array(shading_img).astype(np.float32) / 255.0)
    
    # Load reflectance ground truth
    reflectance_path = os.path.join(scene_path, 'reflectance.png')
    if os.path.exists(reflectance_path):
        reflectance_img = Image.open(reflectance_path).convert('RGB')
        reflectance_image = np.array(reflectance_img).astype(np.float32) / 255.0
    else:
        # Try diffuse.png as fallback
        diffuse_path = os.path.join(scene_path, 'diffuse.png')
        if os.path.exists(diffuse_path):
            reflectance_img = Image.open(diffuse_path).convert('RGB')
            reflectance_image = np.array(reflectance_img).astype(np.float32) / 255.0
        else:
            raise FileNotFoundError(f"Neither reflectance.png nor diffuse.png found in {scene_path}")
    
    print(f"Loaded scene {scene_name}:")
    print(f"  Light images: {len(light_images)}")
    print(f"  Shading images: {len(shading_images)}")
    print(f"  Reflectance shape: {reflectance_image.shape}")
    
    return light_images, shading_images, reflectance_image

def evaluate_your_model_new(vae_checkpoint, flow_checkpoint, config_path, dataset_path, results_dir='results_your_model_no_mask'):
    """
    Evaluate your model on the new MIT Intrinsic Images dataset structure without using masks
    
    Args:
        vae_checkpoint: path to VAE checkpoint
        flow_checkpoint: path to flow model checkpoint  
        config_path: path to config file
        dataset_path: path to MIT-intrinsic dataset
        results_dir: directory to save results
    """
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize your model
    print("Loading your model...")
    your_estimator = YourModelEstimator(vae_checkpoint, flow_checkpoint, config_path)
    
    # Get all scene directories
    test_path = os.path.join(dataset_path, 'test')
    scene_names = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    scene_names.sort()
    
    print(f"Found {len(scene_names)} scenes: {scene_names}")
    print("Evaluation without mask - using entire image")
    
    results = []
    all_results = []
    
    print(f"Evaluating on {len(scene_names)} scenes...")
    
    for i, scene_name in enumerate(scene_names):
        print(f"\nProcessing scene {scene_name} ({i+1}/{len(scene_names)})...")
        
        try:
            # Load scene data
            light_images, shading_images, reflectance_image = load_scene_data(dataset_path, scene_name)
            
            # Ensure we have the same number of light and shading images
            if len(light_images) != len(shading_images):
                print(f"Warning: Mismatch in number of light ({len(light_images)}) and shading ({len(shading_images)}) images")
                min_count = min(len(light_images), len(shading_images))
                light_images = light_images[:min_count]
                shading_images = shading_images[:min_count]
            
            scene_results = []
            
            # Process each light-shading pair
            for j, (light_img, true_shading) in enumerate(zip(light_images, shading_images)):
                print(f"  Processing light {j+1:02d}...")
                
                # Get prediction from your model
                est_shading = your_estimator.estimate_shading_from_light(light_img)
                
                # Compute metrics for this light-shading pair (no masking)
                shading_mse = compute_mse(true_shading, est_shading)
                shading_lmse = compute_lmse(true_shading, est_shading)
                shading_dssim = compute_dssim(true_shading, est_shading)
                shading_psnr = compute_psnr(true_shading, est_shading)
                
                light_result = {
                    'light_id': j+1,
                    'shading_mse': shading_mse,
                    'shading_lmse': shading_lmse,
                    'shading_dssim': shading_dssim,
                    'shading_psnr': shading_psnr
                }
                scene_results.append(light_result)
                
                print(f"    Light {j+1:02d} - S_MSE: {shading_mse:.4f}, S_LMSE: {shading_lmse:.4f}, S_DSSIM: {shading_dssim:.4f}, S_PSNR: {shading_psnr:.2f}")
            
            # Calculate average metrics for this scene
            avg_shading_mse = np.mean([r['shading_mse'] for r in scene_results])
            avg_shading_lmse = np.mean([r['shading_lmse'] for r in scene_results])
            avg_shading_dssim = np.mean([r['shading_dssim'] for r in scene_results])
            avg_shading_psnr = np.mean([r['shading_psnr'] for r in scene_results])
            
            # For reflectance evaluation, use average of all light images
            avg_light_image = np.mean(light_images, axis=0)
            
            # Compute reflectance: R = I / S (using average light and average predicted shading)
            avg_est_shading = np.mean([your_estimator.estimate_shading_from_light(light_img) for light_img in light_images], axis=0)
            epsilon = 1e-6
            
            # Convert reflectance ground truth to grayscale for comparison
            true_refl_gray = np.mean(reflectance_image, axis=2)
            
            # Compute estimated reflectance
            est_reflectance = avg_light_image.mean(axis=2) / (avg_est_shading + epsilon)
            est_reflectance = np.clip(est_reflectance, 0, 1)
            
            # Compute reflectance metrics (no masking)
            albedo_mse = compute_mse(true_refl_gray, est_reflectance)
            albedo_lmse = compute_lmse(true_refl_gray, est_reflectance)
            albedo_dssim = compute_dssim(true_refl_gray, est_reflectance)
            albedo_psnr = compute_psnr(true_refl_gray, est_reflectance)
            
            # Store scene results
            scene_summary = {
                'scene': scene_name,
                'num_lights': len(light_images),
                'avg_shading_mse': avg_shading_mse,
                'avg_shading_lmse': avg_shading_lmse,
                'avg_shading_dssim': avg_shading_dssim,
                'avg_shading_psnr': avg_shading_psnr,
                'albedo_mse': albedo_mse,
                'albedo_lmse': albedo_lmse,
                'albedo_dssim': albedo_dssim,
                'albedo_psnr': albedo_psnr,
                'individual_results': scene_results
            }
            
            all_results.append(scene_summary)
            results.append(scene_summary)
            
            print(f"  Scene {scene_name} Summary:")
            print(f"    Avg Shading - MSE: {avg_shading_mse:.4f}, LMSE: {avg_shading_lmse:.4f}, DSSIM: {avg_shading_dssim:.4f}, PSNR: {avg_shading_psnr:.2f}")
            print(f"    Albedo - MSE: {albedo_mse:.4f}, LMSE: {albedo_lmse:.4f}, DSSIM: {albedo_dssim:.4f}, PSNR: {albedo_psnr:.2f}")
            
        except Exception as e:
            print(f"  Error processing scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate overall averages
    if results:
        overall_avg_s_mse = np.mean([r['avg_shading_mse'] for r in results])
        overall_avg_s_lmse = np.mean([r['avg_shading_lmse'] for r in results])
        overall_avg_s_dssim = np.mean([r['avg_shading_dssim'] for r in results])
        overall_avg_s_psnr = np.mean([r['avg_shading_psnr'] for r in results])
        overall_avg_a_mse = np.mean([r['albedo_mse'] for r in results])
        overall_avg_a_lmse = np.mean([r['albedo_lmse'] for r in results])
        overall_avg_a_dssim = np.mean([r['albedo_dssim'] for r in results])
        overall_avg_a_psnr = np.mean([r['albedo_psnr'] for r in results])
        
        print(f"\n" + "="*60)
        print(f"OVERALL AVERAGE RESULTS (No Mask - Full Image):")
        print(f"Shading - MSE: {overall_avg_s_mse:.4f}, LMSE: {overall_avg_s_lmse:.4f}, DSSIM: {overall_avg_s_dssim:.4f}, PSNR: {overall_avg_s_psnr:.2f}")
        print(f"Albedo  - MSE: {overall_avg_a_mse:.4f}, LMSE: {overall_avg_a_lmse:.4f}, DSSIM: {overall_avg_a_dssim:.4f}, PSNR: {overall_avg_a_psnr:.2f}")
        print(f"Successful evaluations: {len(results)}/{len(scene_names)}")
        
        # Save detailed results
        results_file = os.path.join(results_dir, 'detailed_results.txt')
        with open(results_file, 'w') as f:
            f.write("Detailed Results for Your Model on MIT Intrinsic Dataset (No Mask - Full Image)\n")
            f.write("="*80 + "\n\n")
            
            for scene_result in results:
                f.write(f"Scene: {scene_result['scene']} ({scene_result['num_lights']} lights)\n")
                f.write(f"  Average Shading Metrics:\n")
                f.write(f"    MSE: {scene_result['avg_shading_mse']:.4f}\n")
                f.write(f"    LMSE: {scene_result['avg_shading_lmse']:.4f}\n")
                f.write(f"    DSSIM: {scene_result['avg_shading_dssim']:.4f}\n")
                f.write(f"    PSNR: {scene_result['avg_shading_psnr']:.2f}\n")
                f.write(f"  Albedo Metrics:\n")
                f.write(f"    MSE: {scene_result['albedo_mse']:.4f}\n")
                f.write(f"    LMSE: {scene_result['albedo_lmse']:.4f}\n")
                f.write(f"    DSSIM: {scene_result['albedo_dssim']:.4f}\n")
                f.write(f"    PSNR: {scene_result['albedo_psnr']:.2f}\n")
                f.write(f"  Individual Light Results:\n")
                for light_result in scene_result['individual_results']:
                    f.write(f"    Light {light_result['light_id']:02d}: MSE={light_result['shading_mse']:.4f}, "
                           f"LMSE={light_result['shading_lmse']:.4f}, DSSIM={light_result['shading_dssim']:.4f}, "
                           f"PSNR={light_result['shading_psnr']:.2f}\n")
                f.write("\n")
            
            f.write(f"OVERALL AVERAGES (No Mask - Full Image):\n")
            f.write(f"Shading - MSE: {overall_avg_s_mse:.4f}, LMSE: {overall_avg_s_lmse:.4f}, DSSIM: {overall_avg_s_dssim:.4f}, PSNR: {overall_avg_s_psnr:.2f}\n")
            f.write(f"Albedo  - MSE: {overall_avg_a_mse:.4f}, LMSE: {overall_avg_a_lmse:.4f}, DSSIM: {overall_avg_a_dssim:.4f}, PSNR: {overall_avg_a_psnr:.2f}\n")
            f.write(f"Successful evaluations: {len(results)}/{len(scene_names)}\n")
        
        print(f"\nResults saved to {results_dir}/")
        print(f"Detailed results saved to {results_file}")
    else:
        print("No successful evaluations!")

def main():
    parser = argparse.ArgumentParser(description='Evaluate your model on MIT Intrinsic Images dataset (no mask)')
    parser.add_argument('--vae_checkpoint', type=str, 
                       default="/home/project/ldr_image_to_ldr_shading/LDR_image_to_LDR_shading_hyperism/train_vae_mithlesh/ldr_to_sh/epoch_290_best_autoencoder_model_checkpoint.pth",
                       help='Path to VAE checkpoint')
    parser.add_argument('--flow_checkpoint', type=str, 
                       default="/home/project/ldr_image_to_ldr_shading/LDR_image_to_LDR_shading_hyperism/train_vae_mithlesh/ldr_to_sh_fine_MIT_finetune/mit_finetune_ckpt.pth",
                       help='Path to flow model checkpoint')
    parser.add_argument('--config_path', type=str, 
                       default='/home/project/ldr_image_to_ldr_shading/LDR_image_to_LDR_shading_hyperism/train_vae_mithlesh/config/fine.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset_path', type=str, 
                       default='/mnt/zone/B/mithlesh/dataset/mit/MIT-intrinsic',
                       help='Path to MIT-intrinsic dataset')
    parser.add_argument('--results_dir', type=str, default='results_your_model_no_mask',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    test_path = os.path.join(args.dataset_path, 'test')
    if not os.path.exists(test_path):
        print(f"Error: Dataset test folder not found at {test_path}")
        print("Please make sure the dataset path is correct")
        return
    
    evaluate_your_model_new(
        vae_checkpoint=args.vae_checkpoint,
        flow_checkpoint=args.flow_checkpoint,
        config_path=args.config_path,
        dataset_path=args.dataset_path,
        results_dir=args.results_dir
    )

if __name__ == '__main__':
    main()