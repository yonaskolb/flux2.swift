#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 latent patchify/BN fixtures."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="fixtures/flux2_tiny",
        help="Snapshot directory containing the tiny VAE fixtures.",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for the reference implementation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for running the reference model.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for fixtures.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def patchify(latents: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    ph, pw = patch_size
    latents = latents.view(batch, channels, height // ph, ph, width // pw, pw)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch, channels * ph * pw, height // ph, width // pw)
    return latents


def unpatchify(latents: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    ph, pw = patch_size
    latents = latents.reshape(batch, channels // (ph * pw), ph, pw, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch, channels // (ph * pw), height * ph, width * pw)
    return latents


def main() -> None:
    args = parse_args()
    diffusers_src = Path(args.diffusers_path)
    if diffusers_src.exists():
        sys.path.insert(0, str(diffusers_src))
    else:
        raise SystemExit(f"diffusers path not found: {diffusers_src}")

    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    snapshot_dir = Path(args.snapshot_dir)
    config_path = snapshot_dir / "vae" / "config.json"
    weights_path = snapshot_dir / "vae" / "model.safetensors"
    inputs_path = snapshot_dir / "vae_inputs.safetensors"
    expected_path = snapshot_dir / "latent_expected.safetensors"

    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise SystemExit(f"Missing weights: {weights_path}")
    if not inputs_path.exists():
        raise SystemExit(f"Missing inputs: {inputs_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    model = AutoencoderKLFlux2(**config).to(device=device, dtype=dtype)
    model.load_state_dict(load_file(str(weights_path)))
    model.eval()

    inputs = load_file(str(inputs_path))
    image = inputs["image"].to(device=device, dtype=dtype)

    with torch.no_grad():
        posterior = model.encode(image).latent_dist
        latents = posterior.mode()
        patchified = patchify(latents, tuple(config["patch_size"]))

        bn_mean = model.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
        bn_std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
        bn_std = bn_std.to(device=device, dtype=dtype)

        normalized = (patchified - bn_mean) / bn_std
        denormalized = normalized * bn_std + bn_mean
        unpatchified = unpatchify(denormalized, tuple(config["patch_size"]))

    expected = {
        "patchified": patchified.cpu().clone(),
        "normalized": normalized.cpu().clone(),
        "denormalized": denormalized.cpu().clone(),
        "unpatchified": unpatchified.cpu().clone(),
    }
    save_file(expected, str(expected_path))

    print(f"Wrote expected latents to {expected_path}")


if __name__ == "__main__":
    main()
