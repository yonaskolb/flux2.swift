#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 VAE fixtures for numerical parity checks."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny",
        help="Output directory for the tiny VAE snapshot and fixtures.",
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
        help="Data type for weights and fixtures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    parser.add_argument("--height", type=int, default=8, help="Input image height.")
    parser.add_argument("--width", type=int, default=8, help="Input image width.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [4, 8],
        "layers_per_block": 1,
        "act_fn": "silu",
        "latent_channels": 4,
        "norm_num_groups": 2,
        "sample_size": max(args.height, args.width),
        "force_upcast": True,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "mid_block_add_attention": True,
        "batch_norm_eps": 1e-4,
        "batch_norm_momentum": 0.1,
        "patch_size": [2, 2],
    }

    model = AutoencoderKLFlux2(**config).to(device=device, dtype=dtype)
    model.eval()

    batch = 1
    image = torch.randn(
        (batch, config["in_channels"], args.height, args.width),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        posterior = model.encode(image).latent_dist
        moments = posterior.parameters
        latents = posterior.mode().clone()
        decoded = model.decode(latents, return_dict=False)[0]

    out_dir = Path(args.out_dir)
    vae_dir = out_dir / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)

    with (vae_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(weights, str(vae_dir / "model.safetensors"))

    inputs = {"image": image.cpu()}
    save_file(inputs, str(out_dir / "vae_inputs.safetensors"))

    expected = {
        "moments": moments.cpu(),
        "latents": latents.cpu(),
        "decoded": decoded.cpu(),
    }
    save_file(expected, str(out_dir / "vae_expected.safetensors"))

    print(f"Wrote tiny VAE snapshot to {vae_dir}")
    print(f"Wrote inputs to {out_dir / 'vae_inputs.safetensors'}")
    print(f"Wrote expected output to {out_dir / 'vae_expected.safetensors'}")


if __name__ == "__main__":
    main()
