import torch
from torchvision import transforms
from overcomplete.models import BaseModel

class DinoV2(BaseModel):
    """
    Concrete class for DiNoV2 model with multi-layer extraction capabilities.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
    extract_layers : list of int, optional
        List of block indices (0-indexed) to extract features from.
        e.g., [2, 5, 8, 11].

    Methods
    -------
    forward_features(x):
        Returns a dictionary containing features from requested layers.
    """

    def __init__(self, use_half=False, device='cpu', extract_layers=None):
        super().__init__(use_half, device)
        
        # Load the model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval().to(self.device)
        
        if self.use_half:
            self.model = self.model.half()

        # Preprocessing (Standard DINOv2)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- Hook Setup ---
        self.extract_layers = extract_layers if extract_layers is not None else []
        self._activations = {} 

        # Verify blocks exist and register hooks
        if self.extract_layers:
            # DINOv2 ViT stores transformer blocks in .blocks
            for layer_idx in self.extract_layers:
                if 0 <= layer_idx < len(self.model.blocks):
                    self.model.blocks[layer_idx].register_forward_hook(self._get_hook(layer_idx))
                else:
                    print(f"Warning: Layer {layer_idx} requested but model only has {len(self.model.blocks)} blocks.")

    def _get_hook(self, layer_idx):
        """Creates a closure to save the output of a specific layer."""
        def hook(module, input, output):
            # output of a ViT block is usually (Batch, Tokens, Dim)
            self._activations[layer_idx] = output
        return hook

    def forward_features(self, x):
        """
        Perform a forward pass and extract intermediate features.

        Returns
        -------
        dict
            Keys are layer indices (int) or 'final'. 
            Values are torch.Tensor of shape (batch, tokens, dim).
        """
        self._activations = {} # Clear previous info
        
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            
            # Run the standard forward pass. 
            # The hooks we registered in __init__ will automatically populate self._activations
            final_output = self.model.forward_features(x)
            
            # You can store the final standard output as well if needed
            results = self._activations.copy()
            results['final'] = final_output['x_patchtokens']
            
            return results