"""Memory profiles for the Klein pipeline. Each trades VRAM for speed or precision; every
profile here was measured before it is offered.

  offload        model CPU offload: one whole component on the GPU at a time
  group          block-level group offloading with CUDA streams: one transformer block at a
                 time, prefetched; lossless
  offload_fp8    model offload, weights stored in fp8 (e4m3) and computed in bf16
  group_fp8      group offloading plus fp8 storage
  fast           transformer and VAE stay on the GPU, the text encoder is offloaded by layer;
                 lossless; about 9 GB
  resident       everything on the GPU; lossless; about 15 GB

`auto` picks the fastest lossless profile that fits the free video memory (see choose());
`auto/group_fp8` falls back to group_fp8 instead of group.
"""

from __future__ import annotations

import torch

PROFILES = ("offload", "group", "offload_fp8", "group_fp8", "fast", "resident")
KLEIN_TEXT_LAYERS = (9, 18, 27)  # the Qwen3 hidden states Klein reads (Flux2KleinPipeline default)

# Free video memory each lossless profile needs at the 1 MP working size, measured with the
# driver (weights, CUDA context, activations and headroom for the selection models' work).
NEEDS_GB = {"resident": 17.5, "fast": 10.5}


def choose(free_gb: float | None, fallback: str = "group") -> str:
    """The fastest lossless profile for this much free video memory; `fallback` when neither
    fits (group, or group_fp8 on machines with less system memory)."""
    if free_gb is not None:
        for name in ("resident", "fast"):
            if free_gb >= NEEDS_GB[name]:
                return name
    return fallback


def shorten_text_encoder(pipe) -> int:
    """Drop the Qwen3 layers after the last hidden state Klein reads. Call before offloading.

    hidden_states[k] is the input of layer k, so reading index 27 needs layers 0 to 27; the last
    entry is passed through the final norm, so one layer past the last index is kept. Returns
    how many layers were dropped. The embeddings are bit-identical.
    """
    model = pipe.text_encoder.model
    keep = max(KLEIN_TEXT_LAYERS) + 1
    n = len(model.layers)
    if n <= keep:
        return 0
    model.layers = model.layers[:keep]
    return n - keep


def encode_short(pipe, prompt: str):
    """Klein's prompt embeddings without the language-model head (same tensors as encode_prompt)."""
    tok = pipe.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    dev = pipe._execution_device
    te = pipe.text_encoder
    # Model CPU offload hooks only the top-level module, so call through it there (its language
    # head then runs too); block and leaf offload, fast and resident can skip the head.
    hooked = hasattr(te, "_hf_hook")
    call = te if hooked else te.model
    out = call(input_ids=inputs["input_ids"].to(dev), attention_mask=inputs["attention_mask"].to(dev),
               output_hidden_states=True, use_cache=False, **({"logits_to_keep": 1} if hooked else {}))
    x = torch.stack([out.hidden_states[k] for k in KLEIN_TEXT_LAYERS], dim=1).to(dtype=pipe.text_encoder.dtype, device=dev)
    b, c, n, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, c * d)


def apply_profile(pipe, profile: str, device: str = "cuda") -> None:
    from diffusers.hooks import apply_group_offloading, apply_layerwise_casting

    if profile not in PROFILES:
        raise ValueError(f"unknown memory profile {profile!r}; one of {PROFILES}")
    if profile.endswith("fp8"):
        for name in ("transformer", "text_encoder"):
            apply_layerwise_casting(getattr(pipe, name), storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
                                    skip_modules_pattern=("pos_embed", "patch_embed", "norm", "proj_in", "proj_out", "embed_tokens", "lm_head", "rotary"))
    if profile == "resident":
        pipe.to(device)
        return
    if profile.startswith("offload"):
        pipe.enable_model_cpu_offload()
        return
    pipe.vae.to(device)
    on, off = torch.device(device), torch.device("cpu")
    if profile == "fast":
        pipe.transformer.to(device)
        apply_group_offloading(pipe.text_encoder, onload_device=on, offload_device=off, offload_type="leaf_level",
                               use_stream=True, low_cpu_mem_usage=True)
        return
    # The transformer exposes its blocks as top-level ModuleLists, so block-level grouping works.
    apply_group_offloading(pipe.transformer, onload_device=on, offload_device=off, offload_type="block_level",
                           num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True)
    # Qwen3ForCausalLM nests its layers under .model, which block-level grouping does not find
    # (the whole 8 GB encoder was onloaded at once in the first measurement); leaf level works.
    apply_group_offloading(pipe.text_encoder, onload_device=on, offload_device=off, offload_type="leaf_level",
                           use_stream=True, low_cpu_mem_usage=True)
