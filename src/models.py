import torch
from torchvision import transforms

from overcomplete.models import BaseModel

class DinoV2(BaseModel):
    """
    Concrete class for DiNoV2 model.

    Specifically, we use the DiNoV2 model from Oquab, Darcet & al (2021),
    see https://github.com/facebookresearch/dinov2

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Perform a forward pass on the input tensor.

    preprocess(img):
        Preprocess input images for the DiNoV2 model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval().to(self.device)
        if self.use_half:
            self.model = self.model.half()

        self.preprocess = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Assume input is in the same device as the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output features.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)['x_norm_patchtokens']