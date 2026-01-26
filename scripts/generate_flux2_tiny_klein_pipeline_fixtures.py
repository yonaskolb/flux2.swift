#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 Klein end-to-end fixtures (Qwen3 -> transformer -> VAE)."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny_klein_pipeline",
        help="Output directory for the klein pipeline snapshot and fixtures.",
    )
    parser.add_argument(
        "--pipeline-snapshot",
        default="fixtures/flux2_tiny_pipeline",
        help="Snapshot containing transformer/scheduler/vae.",
    )
    parser.add_argument(
        "--text-encoder-snapshot",
        default="fixtures/flux2_tiny_text_encoder",
        help="Snapshot containing text_encoder weights/config.",
    )
    parser.add_argument(
        "--text-fixture",
        default="fixtures/flux2_tiny_text_encoder/prompt_embeds.safetensors",
        help="Fixture with input_ids/attention_mask for the text encoder.",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for reference helpers.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights and fixtures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=3,
        help="Number of inference steps for the denoise loop.",
    )
    parser.add_argument("--height", type=int, default=8, help="Image height.")
    parser.add_argument("--width", type=int, default=8, help="Image width.")
    parser.add_argument(
        "--layers",
        default="0",
        help="Comma-separated hidden state layer indices to stack.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (end - start) / float(count - 1)
    return [start + step * i for i in range(count)]


def ensure_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_snapshot(src: Path, dst: Path, subdir: str) -> None:
    src_dir = src / subdir
    dst_dir = dst / subdir
    if not src_dir.exists():
        raise SystemExit(f"Missing {subdir} in {src}")
    shutil.copytree(src_dir, dst_dir)


def load_qwen3_model(text_encoder_dir: Path, dtype: torch.dtype, device: torch.device) -> Qwen3Model:
    config_path = text_encoder_dir / "config.json"
    weights_path = text_encoder_dir / "model.safetensors"
    with config_path.open("r", encoding="utf-8") as f:
        config = Qwen3Config(**json.load(f))
    model = Qwen3Model(config).to(device=device, dtype=dtype)
    weights = load_file(str(weights_path))
    trimmed = {k.replace("model.", ""): v for k, v in weights.items()}
    model.load_state_dict(trimmed)
    model.eval()
    return model


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
    ensure_dir(out_dir)

    pipeline_snapshot = Path(args.pipeline_snapshot)
    text_snapshot = Path(args.text_encoder_snapshot)

    copy_snapshot(pipeline_snapshot, out_dir, "transformer")
    copy_snapshot(pipeline_snapshot, out_dir, "scheduler")
    copy_snapshot(pipeline_snapshot, out_dir, "vae")
    copy_snapshot(text_snapshot, out_dir, "text_encoder")

    transformer_dir = out_dir / "transformer"
    scheduler_dir = out_dir / "scheduler"
    vae_dir = out_dir / "vae"
    text_encoder_dir = out_dir / "text_encoder"

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

    with (vae_dir / "config.json").open("r", encoding="utf-8") as f:
        vae_config = json.load(f)
    vae = AutoencoderKLFlux2(**vae_config)
    vae_weights = load_file(str(vae_dir / "model.safetensors"))
    vae.load_state_dict(vae_weights)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    layer_indices = [int(item) for item in args.layers.split(",") if item.strip()]

    text_fixtures = load_file(args.text_fixture)
    input_ids = text_fixtures["input_ids"].to(device=device)
    attention_mask = text_fixtures["attention_mask"].to(device=device)

    text_model = load_qwen3_model(text_encoder_dir, dtype=dtype, device=device)
    with torch.no_grad():
        output = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    stacked = torch.stack([output.hidden_states[i] for i in layer_indices], dim=1)
    stacked = stacked.to(dtype=dtype)
    batch_size, num_layers, seq_len, hidden_dim = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_layers * hidden_dim
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(device)

    vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)
    patch_h, patch_w = vae_config["patch_size"]
    height = patch_h * (args.height // (vae_scale_factor * patch_h))
    width = patch_w * (args.width // (vae_scale_factor * patch_w))
    patch_area = patch_h * patch_w
    num_latents_channels = transformer_config["in_channels"] // patch_area
    latents_shape = (
        batch_size,
        num_latents_channels * patch_area,
        height // patch_h,
        width // patch_w,
    )
    latents = torch.randn(latents_shape, device=device, dtype=dtype)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents).to(device)
    packed_latents = Flux2KleinPipeline._pack_latents(latents)

    sigmas = None
    if not scheduler_config.get("use_flow_sigmas", False):
        sigmas = linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
    mu = None
    if scheduler_config.get("use_dynamic_shifting", False):
        mu = compute_empirical_mu(
            image_seq_len=packed_latents.shape[1], num_steps=args.num_inference_steps
        )
    scheduler.set_timesteps(args.num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = scheduler.timesteps.to(device=device, dtype=dtype)

    with torch.no_grad():
        for t in timesteps:
            timestep = (t / 1000).expand(batch_size)
            noise_pred = transformer(
                hidden_states=packed_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : packed_latents.shape[1], :]
            packed_latents = scheduler.step(noise_pred, t, packed_latents).prev_sample

    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(packed_latents, latent_ids)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    unpacked = unpacked * latents_bn_std + latents_bn_mean
    unpatchified = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded = vae.decode(unpatchified, return_dict=False)[0]

    inputs_out = out_dir / "klein_inputs.safetensors"
    expected_out = out_dir / "klein_expected.safetensors"

    save_file(
        {
            "input_ids": input_ids.contiguous().cpu(),
            "attention_mask": attention_mask.contiguous().cpu(),
            "prompt_embeds": prompt_embeds.contiguous().cpu(),
            "text_ids": text_ids.contiguous().cpu(),
            "latents": latents.contiguous().cpu(),
            "latent_ids": latent_ids.contiguous().cpu(),
            "height": torch.tensor([args.height], dtype=torch.int32),
            "width": torch.tensor([args.width], dtype=torch.int32),
            "num_inference_steps": torch.tensor([args.num_inference_steps], dtype=torch.int32),
        },
        str(inputs_out),
    )

    save_file(
        {
            "packed_latents": packed_latents.contiguous().cpu(),
            "decoded": decoded.contiguous().cpu(),
        },
        str(expected_out),
    )

    print(f"Wrote klein inputs to {inputs_out}")
    print(f"Wrote klein expected outputs to {expected_out}")


if __name__ == "__main__":
    main()
