"""Backbone wiring, including a regression guard for a silent weight-loading bug."""

from pathlib import Path

import pytest

from exoego.encoders import BACKBONES, restore_videomae_attention_bias

HF_ID = BACKBONES["videomae"]["hf_id"]
CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CACHED = (CACHE / f"models--{HF_ID.replace('/', '--')}").exists()


def test_backbone_specs_are_complete():
    for name, spec in BACKBONES.items():
        assert "hf_id" in spec and "dim" in spec, name
        assert spec.get("kind") in {"image", "video"}, name
        if spec["kind"] == "video":
            assert spec.get("tokens", 0) > 0, name


@pytest.mark.skipif(not CACHED, reason="VideoMAE weights not cached locally")
def test_videomae_attention_biases_are_restored():
    """VideoMAE stores attention bias as `q_bias`/`v_bias`; transformers expects
    `query.bias`/`value.bias`.

    A plain from_pretrained therefore drops them and silently re-initialises to
    zero -- the model loads "successfully" and runs with no learned attention
    bias. This asserts the remapping actually fires, because nothing else would
    reveal it: the failure is silent and only shows up as degraded features.
    """
    from transformers import VideoMAEModel

    model = VideoMAEModel.from_pretrained(HF_ID)
    attention = model.encoder.layer[0].attention.attention

    assert attention.query.bias.abs().sum().item() == 0.0, (
        "expected transformers to leave query.bias zeroed before the fix"
    )

    restored = restore_videomae_attention_bias(model, HF_ID)
    assert restored == 2 * len(model.encoder.layer)
    assert attention.query.bias.abs().sum().item() > 0.0
    assert attention.value.bias.abs().sum().item() > 0.0
