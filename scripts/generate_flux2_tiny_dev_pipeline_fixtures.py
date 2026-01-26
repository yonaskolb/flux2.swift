#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file, save_file
from transformers import Mistral3Config, Mistral3ForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2-dev (embedded guidance) pipeline fixtures."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny_dev_pipeline",
        help="Output directory for pipeline snapshot and fixtures.",
    )
    parser.add_argument(
        "--source-snapshot",
        default="fixtures/flux2_tiny_pipeline_guidance",
        help="Source snapshot for transformer/scheduler/VAE.",
    )
    parser.add_argument(
        "--source-text-encoder",
        default="fixtures/flux2_tiny_mistral3_text_encoder/text_encoder",
        help="Source text_encoder directory (config.json + model.safetensors).",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for the reference implementation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model weights and fixtures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=8,
        help="Pixel height passed to the pipeline (must match VAE scale constraints).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=8,
        help="Pixel width passed to the pipeline (must match VAE scale constraints).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=3,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale passed to the transformer.",
    )
    parser.add_argument(
        "--layers",
        default="2",
        help="Comma-separated hidden state layer indices to stack for prompt embeds.",
    )
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=3,
        help="Text sequence length for input_ids/attention_mask.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def prepare_text_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(1, dtype=torch.int64, device=device)
    h = torch.arange(1, dtype=torch.int64, device=device)
    w = torch.arange(1, dtype=torch.int64, device=device)
    l = torch.arange(seq_len, dtype=torch.int64, device=device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def prepare_latent_ids(latents_nchw: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = latents_nchw.shape
    t = torch.arange(1, dtype=torch.int64, device=latents_nchw.device)
    h = torch.arange(height, dtype=torch.int64, device=latents_nchw.device)
    w = torch.arange(width, dtype=torch.int64, device=latents_nchw.device)
    l = torch.arange(1, dtype=torch.int64, device=latents_nchw.device)
    coords = torch.cartesian_prod(t, h, w, l)
    return coords.unsqueeze(0).expand(batch, -1, -1)


def pack_latents(latents_nchw: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents_nchw.shape
    return latents_nchw.reshape(b, c, h * w).permute(0, 2, 1).contiguous()


def main() -> None:
    args = parse_args()

    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    out_dir = Path(args.out_dir)
    transformer_dir = out_dir / "transformer"
    scheduler_dir = out_dir / "scheduler"
    vae_dir = out_dir / "vae"
    text_encoder_dir = out_dir / "text_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir.mkdir(parents=True, exist_ok=True)
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    source_snapshot = Path(args.source_snapshot)
    shutil.copy2(source_snapshot / "transformer" / "config.json", transformer_dir / "config.json")
    shutil.copy2(
        source_snapshot / "transformer" / "model.safetensors",
        transformer_dir / "model.safetensors",
    )
    shutil.copy2(source_snapshot / "scheduler" / "config.json", scheduler_dir / "config.json")
    shutil.copy2(source_snapshot / "vae" / "config.json", vae_dir / "config.json")
    shutil.copy2(source_snapshot / "vae" / "model.safetensors", vae_dir / "model.safetensors")

    source_text_encoder = Path(args.source_text_encoder)
    shutil.copy2(source_text_encoder / "config.json", text_encoder_dir / "config.json")
    shutil.copy2(source_text_encoder / "model.safetensors", text_encoder_dir / "model.safetensors")

    with (text_encoder_dir / "config.json").open("r", encoding="utf-8") as f:
        text_encoder_config = json.load(f)
    mistral_config = Mistral3Config.from_dict(text_encoder_config)
    text_encoder = Mistral3ForConditionalGeneration(mistral_config)
    text_encoder_weights = load_file(str(text_encoder_dir / "model.safetensors"))
    text_encoder.load_state_dict(text_encoder_weights, strict=False)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    with (transformer_dir / "config.json").open("r", encoding="utf-8") as f:
        transformer_config = json.load(f)
    transformer = Flux2Transformer2DModel(**transformer_config)
    transformer_weights = load_file(str(transformer_dir / "model.safetensors"))
    transformer.load_state_dict(transformer_weights)
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()

    with (scheduler_dir / "config.json").open("r", encoding="utf-8") as f:
        scheduler_config = json.load(f)
    scheduler_config = {
        key: value for key, value in scheduler_config.items() if not key.startswith("_")
    }
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)
    sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
    scheduler.set_timesteps(args.num_inference_steps, device=device, sigmas=sigmas)
    timesteps = scheduler.timesteps.to(device=device, dtype=dtype)

    with (vae_dir / "config.json").open("r", encoding="utf-8") as f:
        vae_config = json.load(f)
    vae = AutoencoderKLFlux2(**vae_config).to(device=device, dtype=dtype)
    vae_weights = load_file(str(vae_dir / "model.safetensors"))
    vae.load_state_dict(vae_weights)
    vae.eval()

    layer_indices = [int(item) for item in args.layers.split(",") if item.strip()]

    batch = 1
    text_seq_len = args.text_seq_len
    input_ids = torch.randint(
        low=0,
        high=mistral_config.text_config.vocab_size,
        size=(batch, text_seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((batch, text_seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        enc_out = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    stacked = torch.stack([enc_out.hidden_states[i] for i in layer_indices], dim=1)
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch, text_seq_len, -1)
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    text_ids = prepare_text_ids(batch, text_seq_len, device=device)

    # Prepare latents in NCHW form, matching pipeline_flux2.py + Flux2LatentPreparation.
    num_latent_channels = transformer_config["in_channels"] // 4
    vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)
    height = 2 * (int(args.height) // (vae_scale_factor * 2))
    width = 2 * (int(args.width) // (vae_scale_factor * 2))
    latents_nchw = torch.randn(
        (batch, num_latent_channels * 4, height // 2, width // 2),
        device=device,
        dtype=dtype,
    )
    latents_packed = pack_latents(latents_nchw)
    latent_ids = prepare_latent_ids(latents_nchw)

    guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32).expand(
        batch
    )

    with torch.no_grad():
        latents = latents_packed
        for t in timesteps:
            timestep = (t / 1000).expand(batch)
            noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                guidance=guidance,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    unpacked = unpacked * latents_bn_std + latents_bn_mean
    unpatchified = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded = vae.decode(unpatchified, return_dict=False)[0]

    inputs_out = out_dir / "dev_inputs.safetensors"
    expected_out = out_dir / "dev_expected.safetensors"

    save_file(
        {
            "input_ids": input_ids.contiguous().cpu(),
            "attention_mask": attention_mask.contiguous().cpu(),
            "latents": latents_nchw.contiguous().cpu(),
        },
        str(inputs_out),
    )

    save_file(
        {
            "packed_latents": latents.contiguous().cpu(),
            "decoded": decoded.contiguous().cpu(),
            "prompt_embeds": prompt_embeds.contiguous().cpu(),
            "text_ids": text_ids.contiguous().cpu(),
            "latent_ids": latent_ids.contiguous().cpu(),
        },
        str(expected_out),
    )

    print(f"Wrote dev inputs to {inputs_out}")
    print(f"Wrote dev expected outputs to {expected_out}")


if __name__ == "__main__":
    main()
