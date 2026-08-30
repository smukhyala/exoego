"""Frozen visual backbones.

The backbone is never trained. Per-frame features are extracted once and cached,
which makes every downstream experiment (3 configs x 7 label budgets x 5 seeds)
a matter of seconds rather than hours.
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# `kind` selects how a clip is encoded:
#   image -- one forward per frame; the temporal head sees T frame embeddings.
#   video -- one forward per clip; motion is modelled inside the backbone.
# Per-frame image features turn out to be a poor fit for manipulation verbs:
# "pick up" and "put down" share every frame and differ only in order, and
# frozen appearance features latch onto the recording rather than the action.
BACKBONES = {
    "dinov2s": {"hf_id": "facebook/dinov2-small", "dim": 384, "kind": "image"},
    "dinov2b": {"hf_id": "facebook/dinov2-base", "dim": 768, "kind": "image"},
    "clip": {"hf_id": "openai/clip-vit-base-patch16", "dim": 768, "kind": "image"},
    # Kinetics-finetuned: pretraining is explicitly about verbs/motion.
    "videomae": {
        "hf_id": "MCG-NJU/videomae-base-finetuned-kinetics",
        "dim": 768, "kind": "video", "tokens": 8,
    },
}


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def restore_videomae_attention_bias(model, hf_id: str) -> int:
    """Copy VideoMAE's q_bias/v_bias into query.bias/value.bias.

    VideoMAE checkpoints store the attention biases as separate `q_bias` and
    `v_bias` tensors (key bias is structurally zero). transformers expects
    `query.bias` / `key.bias` / `value.bias`, so a plain from_pretrained drops
    them and silently re-initialises: the load report lists them as MISSING
    while the checkpoint's own entries show up as UNEXPECTED. The weights load
    fine, so the model looks healthy while running without its learned biases.
    """
    try:
        checkpoint = hf_hub_download(hf_id, "model.safetensors")
        raw = load_file(checkpoint)
    except Exception:
        return 0

    restored = 0
    for index, layer in enumerate(model.encoder.layer):
        attention = layer.attention.attention
        prefix = f"videomae.encoder.layer.{index}.attention.attention"
        for suffix, target in [("q_bias", attention.query), ("v_bias", attention.value)]:
            tensor = raw.get(f"{prefix}.{suffix}")
            if tensor is not None and target.bias is not None:
                target.bias.data.copy_(tensor)
                restored += 1
    return restored


class FrozenEncoder:
    """Wraps a pretrained image backbone; maps uint8 RGB frames to embeddings."""

    def __init__(self, name: str = "dinov2s", device: str = "auto"):
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(BACKBONES)}")
        spec = BACKBONES[name]
        self.name = name
        self.dim = spec["dim"]
        self.kind = spec.get("kind", "image")
        self.tokens = spec.get("tokens", 1)
        self.device = pick_device(device)

        model = AutoModel.from_pretrained(spec["hf_id"])
        if name == "clip":
            model = model.vision_model
        if self.kind == "video":
            restore_videomae_attention_bias(model, spec["hf_id"])
        model.eval()
        model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    def _to_tensor(self, frames_uint8: np.ndarray) -> torch.Tensor:
        batch = frames_uint8.astype(np.float32) / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(batch)
        return tensor.to(self.device)

    @torch.no_grad()
    def encode(self, frames_uint8: np.ndarray) -> np.ndarray:
        """(B, 224, 224, 3) uint8 RGB -> (B, dim) float32. Image backbones only."""
        tensor = self._to_tensor(frames_uint8).permute(0, 3, 1, 2).contiguous()
        outputs = self.model(pixel_values=tensor)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token.float().cpu().numpy()

    @torch.no_grad()
    def encode_clips(self, clips_uint8: np.ndarray) -> np.ndarray:
        """(B, T, 224, 224, 3) uint8 RGB -> (B, tokens, dim) float32.

        VideoMAE emits 8 temporal x 196 spatial tokens for 16 input frames. We
        pool over space only, keeping the temporal axis so the same TemporalHead
        can sit on top and the ordering cue that separates "pick up" from
        "put down" survives.
        """
        tensor = self._to_tensor(clips_uint8).permute(0, 1, 4, 2, 3).contiguous()
        outputs = self.model(pixel_values=tensor)
        hidden = outputs.last_hidden_state
        batch_size = hidden.shape[0]
        spatial = hidden.shape[1] // self.tokens
        pooled = hidden.reshape(batch_size, self.tokens, spatial, -1).mean(dim=2)
        return pooled.float().cpu().numpy()
