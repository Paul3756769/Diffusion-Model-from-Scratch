
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------
# Config
# -----------------------------
@dataclass
class DiffusionConfig:
    """Configuration for diffusion training and sampling."""
    image_size: int = 28
    channels: int = 1
    batch_size: int = 128
    epochs: int = 5
    lr: float = 2e-4
    beta_min: float = 0.1
    beta_max: float = 20.0
    num_steps: int = 1000
    num_sample_grid: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Helpers
# -----------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """
    Build sinusoidal time embeddings.

    Mathematical form:
    $$\text{emb}(t) = [\sin(\omega_1 t), \dots, \sin(\omega_k t), \cos(\omega_1 t), \dots, \cos(\omega_k t)]$$

    Args:
        t: Normalized time tensor of shape [B].
        dim: Embedding dimension.

    Returns:
        Tensor of shape [B, dim].
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------------
# Model
# -----------------------------
class Block(nn.Module):
    """Residual convolution block with time conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class SimpleScoreNet(nn.Module):
    """
    Small score network for image diffusion.

    Input:
        x: Noisy image tensor of shape [B, C, H, W].
        t: Normalized time tensor of shape [B].

    Output:
        Tensor with the same shape as x.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.block1 = Block(base_channels, base_channels, time_dim)
        self.down = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.block2 = Block(base_channels * 2, base_channels * 2, time_dim)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.block3 = Block(base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x = self.block1(x, t_emb)
        x = self.down(x)
        x = self.block2(x, t_emb)
        x = self.up(x)
        x = self.block3(x, t_emb)
        return self.out(x)

class Diffusion:
    """
    Diffusion process utilities.

    Stored state is limited to init parameters only:
    - beta schedule bounds
    - number of reverse steps
    - target device

    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, num_steps: int = 1000, device: str = "cpu"):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_steps = num_steps
        self.device = device

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear beta schedule.

        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            Beta values for each time entry.
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative signal retention.
        
        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            alpha_bar(t).
        """
        return torch.exp(
            -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        )

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward noise scale.

        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            sigma(t).
        """
        return torch.sqrt(1.0 - self.alpha_bar(t).clamp(max=0.999999))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the forward diffusion process.

        Args:
            x0: Clean image batch.
            t: Normalized time tensor.
            noise: Optional Gaussian noise tensor.

        Returns:
            Tuple of (noisy images, noise used, sigma values).
        """
        
        # If noise is not provided, sample standard Gaussian noise.
        if noise is None:
            noise = torch.randn_like(x0)

        # Compute the alpha_bar and sigma values for the given time tensor.
        alpha_bar = self.alpha_bar(t)[:, None, None, None]
        sigma = self.sigma(t)[:, None, None, None]
        
        # sampling step with gaussian distribution
        xt = torch.sqrt(alpha_bar) * x0 + sigma * noise
        
        # return parameters needed for loss computation
        return xt, noise, sigma

    @torch.no_grad()
    def reverse_sample(self, score_model: nn.Module, n: int = 16) -> torch.Tensor:
        """
        Generate images by running the reverse diffusion process.

        Args:
            score_model: Trained score model.
            n: Number of images to generate.

        Returns:
            Generated image batch of shape [n, C, H, W].
        """
        # setup
        score_model.eval()
        device = torch.device(self.device)
        
        # sample gaussian noise as initial state
        x = torch.randn(n, self.channels, self.image_size, self.image_size, device=device)
        
        # create time steps for reverse process
        steps = torch.linspace(1.0, 1e-3, self.num_steps, device=device) # avoiding zero
        dt = 1.0 / self.num_steps
        
        # sample the reverse process iteratively
        for t in steps:
            # precomputation
            t_batch = torch.full((n,), t, device=device)
            beta_t = self.beta(t_batch)[:, None, None, None]
            score = score_model(x, t_batch)

            # update step according to the reverse SDE
            drift = 0.5 * beta_t * x + beta_t * score
            noise = torch.randn_like(x)
            x = x + drift * dt + torch.sqrt(beta_t * dt) * noise

        return x.clamp(-1, 1)

    @property
    def image_size(self) -> int:
        return getattr(self, "_image_size", 28)

    @property
    def channels(self) -> int:
        return getattr(self, "_channels", 1)

    @channels.setter
    def channels(self, value: int) -> None:
        self._channels = value

    @image_size.setter
    def image_size(self, value: int) -> None:
        self._image_size = value

# -----------------------------
# Loss
# -----------------------------
def diffusion_loss(
    diffusion: Diffusion,
    model: nn.Module,
    x0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the denoising score matching loss.
    
    Args:
        cfg: Diffusion configuration.
        diffusion: Diffusion process helper.
        model: Score model taking (x, t).
        x0: Clean image batch.

    Returns:
        Scalar loss tensor.
    """
    t = torch.rand(x0.shape[0], device=x0.device).clamp(1e-5, 1.0)
    xt, noise, sigma = diffusion.q_sample(x0, t)
    pred = model(xt, t)
    target = -noise / sigma
    return ((sigma ** 2) * (pred - target).pow(2)).mean()

# -----------------------------
# Training
# -----------------------------
def train_diffusion_model(
    cfg: DiffusionConfig,
    diffusion: Diffusion,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """
    Train the diffusion model on a dataset.

    Args:
        cfg: Diffusion configuration.
        diffusion: Diffusion helper object.
        model: Score model to train.
        loader: DataLoader yielding image batches.
        optimizer: Optimizer used for parameter updates.

    Returns:
        Dictionary with the trained model and loss history.
    """
    history = []

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0

        for x, _ in loader:
            x = x.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss( diffusion, model, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        history.append(avg_loss)
        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {avg_loss:.4f}")

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------
# Config
# -----------------------------
@dataclass
class DiffusionConfig:
    """Configuration for diffusion training and sampling."""
    image_size: int = 28
    channels: int = 1
    batch_size: int = 128
    epochs: int = 5
    lr: float = 2e-4
    beta_min: float = 0.1
    beta_max: float = 20.0
    num_steps: int = 1000
    num_sample_grid: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Helpers
# -----------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """
    Build sinusoidal time embeddings.

    Mathematical form:
    $$\text{emb}(t) = [\sin(\omega_1 t), \dots, \sin(\omega_k t), \cos(\omega_1 t), \dots, \cos(\omega_k t)]$$

    Args:
        t: Normalized time tensor of shape [B].
        dim: Embedding dimension.

    Returns:
        Tensor of shape [B, dim].
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------------
# Model
# -----------------------------
class Block(nn.Module):
    """Residual convolution block with time conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class SimpleScoreNet(nn.Module):
    """
    Small score network for image diffusion.

    Input:
        x: Noisy image tensor of shape [B, C, H, W].
        t: Normalized time tensor of shape [B].

    Output:
        Tensor with the same shape as x.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.block1 = Block(base_channels, base_channels, time_dim)
        self.down = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.block2 = Block(base_channels * 2, base_channels * 2, time_dim)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.block3 = Block(base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x = self.block1(x, t_emb)
        x = self.down(x)
        x = self.block2(x, t_emb)
        x = self.up(x)
        x = self.block3(x, t_emb)
        return self.out(x)

class Diffusion:
    """
    Diffusion process utilities.

    Stored state is limited to init parameters only:
    - beta schedule bounds
    - number of reverse steps
    - target device

    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, num_steps: int = 1000, device: str = "cpu"):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_steps = num_steps
        self.device = device

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear beta schedule.

        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            Beta values for each time entry.
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative signal retention.
        
        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            alpha_bar(t).
        """
        return torch.exp(
            -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        )

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward noise scale.

        Args:
            t: Normalized time tensor in [0, 1].

        Returns:
            sigma(t).
        """
        return torch.sqrt(1.0 - self.alpha_bar(t).clamp(max=0.999999))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the forward diffusion process.

        Args:
            x0: Clean image batch.
            t: Normalized time tensor.
            noise: Optional Gaussian noise tensor.

        Returns:
            Tuple of (noisy images, noise used, sigma values).
        """
        
        # If noise is not provided, sample standard Gaussian noise.
        if noise is None:
            noise = torch.randn_like(x0)

        # Compute the alpha_bar and sigma values for the given time tensor.
        alpha_bar = self.alpha_bar(t)[:, None, None, None]
        sigma = self.sigma(t)[:, None, None, None]
        
        # sampling step with gaussian distribution
        xt = torch.sqrt(alpha_bar) * x0 + sigma * noise
        
        # return parameters needed for loss computation
        return xt, noise, sigma

    @torch.no_grad()
    def reverse_sample(self, score_model: nn.Module, n: int = 16) -> torch.Tensor:
        """
        Generate images by running the reverse diffusion process.

        Args:
            score_model: Trained score model.
            n: Number of images to generate.

        Returns:
            Generated image batch of shape [n, C, H, W].
        """
        # setup
        score_model.eval()
        device = torch.device(self.device)
        
        # sample gaussian noise as initial state
        x = torch.randn(n, self.channels, self.image_size, self.image_size, device=device)
        
        # create time steps for reverse process
        steps = torch.linspace(1.0, 1e-3, self.num_steps, device=device) # avoiding zero
        dt = 1.0 / self.num_steps
        
        # sample the reverse process iteratively
        for t in steps:
            # precomputation
            t_batch = torch.full((n,), t, device=device)
            beta_t = self.beta(t_batch)[:, None, None, None]
            score = score_model(x, t_batch)

            # update step according to the reverse SDE
            drift = 0.5 * beta_t * x + beta_t * score
            noise = torch.randn_like(x)
            x = x + drift * dt + torch.sqrt(beta_t * dt) * noise

        return x.clamp(-1, 1)

    @property
    def image_size(self) -> int:
        return getattr(self, "_image_size", 28)

    @property
    def channels(self) -> int:
        return getattr(self, "_channels", 1)

    @channels.setter
    def channels(self, value: int) -> None:
        self._channels = value

    @image_size.setter
    def image_size(self, value: int) -> None:
        self._image_size = value

# -----------------------------
# Loss
# -----------------------------
def diffusion_loss(
    diffusion: Diffusion,
    model: nn.Module,
    x0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the denoising score matching loss.
    
    Args:
        cfg: Diffusion configuration.
        diffusion: Diffusion process helper.
        model: Score model taking (x, t).
        x0: Clean image batch.

    Returns:
        Scalar loss tensor.
    """
    t = torch.rand(x0.shape[0], device=x0.device).clamp(1e-5, 1.0)
    xt, noise, sigma = diffusion.q_sample(x0, t)
    pred = model(xt, t)
    target = -noise / sigma
    return ((sigma ** 2) * (pred - target).pow(2)).mean()

# -----------------------------
# Training
# -----------------------------
def train_diffusion_model(
    cfg: DiffusionConfig,
    diffusion: Diffusion,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """
    Train the diffusion model on a dataset.

    Args:
        cfg: Diffusion configuration.
        diffusion: Diffusion helper object.
        model: Score model to train.
        loader: DataLoader yielding image batches.
        optimizer: Optimizer used for parameter updates.

    Returns:
        Dictionary with the trained model and loss history.
    """
    history = []

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0

        for x, _ in loader:
            x = x.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss( diffusion, model, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        history.append(avg_loss)
        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {avg_loss:.4f}")

    return {"model": model, "history": history}


@torch.no_grad()
def generate_images(
    cfg: DiffusionConfig,
    diffusion: Diffusion,
    model: nn.Module,
    num_images: int = 16,
) -> torch.Tensor:
    """
    Generate images from a trained diffusion model.

    Args:
        cfg: Diffusion configuration.
        diffusion: Diffusion helper object.
        model: Trained score model.
        num_images: Number of images to generate.

    Returns:
        Generated image batch.
    """
    diffusion.channels = cfg.channels
    diffusion.image_size = cfg.image_size
    return diffusion.reverse_sample(model, n=num_images)
 
