"""
GradCAM Implementation for Model Interpretability
Generates heatmaps showing which regions the model focuses on
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability
    """
    
    def __init__(self, model, target_layer, device='cpu'):
        """
        Args:
            model: PyTorch model
            target_layer: Name of the target layer for visualization
            device: cuda or cpu
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        self.gradient = None
        self.activation = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activation = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        # Get the target layer
        target_module = self._get_target_module()
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def _get_target_module(self):
        """Get target module by name"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
        raise ValueError(f"Target layer '{self.target_layer}' not found in model")
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index. If None, uses predicted class
        
        Returns:
            cam: Generated CAM
            pred_class: Predicted class index
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()
        
        # Generate CAM
        gradients = self.gradient[0]  # (C, H, W)
        activations = self.activation[0]  # (C, H, W)
        
        # Weight the activations by the gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        weighted_activations = weights * activations
        
        # Sum across channels
        cam = weighted_activations.sum(dim=0)  # (H, W)
        
        # ReLU to keep only positive activations
        cam = F.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy(), target_class
    
    def generate_heatmap(self, input_tensor, target_class=None, overlay_alpha=0.4):
        """
        Generate colored heatmap overlay on input image
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W) normalized to [0, 1]
            target_class: Target class for CAM
            overlay_alpha: Transparency of overlay (0-1)
        
        Returns:
            heatmap: Heatmap image
            input_image: Original input image
            cam: CAM array
            pred_class: Predicted class
        """
        # Generate CAM
        cam, pred_class = self.generate_cam(input_tensor, target_class)
        
        # Get original image
        input_image = input_tensor[0].cpu().numpy()
        # Denormalize if needed (assuming ImageNet normalization)
        input_image = (input_image * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + 
                      np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1))
        input_image = np.clip(input_image, 0, 1)
        input_image = np.transpose(input_image, (1, 2, 0))
        
        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))
        
        # Create heatmap
        heatmap_normalized = np.uint8(255 * cam_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(float) / 255.0
        
        # Overlay on original image
        overlay = overlay_alpha * heatmap_colored + (1 - overlay_alpha) * input_image
        overlay = np.clip(overlay, 0, 1)
        
        return overlay, input_image, cam_resized, pred_class


def generate_gradcam_visualization(model, image_tensor, class_names, 
                                   target_layer='backbone.layer4', 
                                   device='cpu', save_path=None):
    """
    Convenience function to generate and optionally save GradCAM visualization
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, 3, H, W)
        class_names: List of class names
        target_layer: Name of target layer
        device: Device to use
        save_path: Optional path to save visualization
    
    Returns:
        Dict with visualization data
    """
    gradcam = GradCAM(model, target_layer, device)
    overlay, original, cam, pred_class = gradcam.generate_heatmap(image_tensor)
    
    result = {
        'overlay': overlay,
        'original': original,
        'cam': cam,
        'predicted_class': pred_class,
        'predicted_class_name': class_names[pred_class]
    }
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save overlay
        overlay_uint8 = np.uint8(overlay * 255)
        overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path / f'overlay_{class_names[pred_class]}.png'), overlay_bgr)
        
        # Save CAM
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cv2.imwrite(str(save_path / f'cam_{class_names[pred_class]}.png'), cam_colored)
        
        # Save original
        original_uint8 = np.uint8(original * 255)
        original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path / f'original.png'), original_bgr)
        
        print(f"Saved visualizations to {save_path}")
    
    return result


if __name__ == "__main__":
    print("GradCAM module for heatmap visualization")
    print("Import this module to use GradCAM functionality")
