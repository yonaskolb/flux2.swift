#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 prepare_latents and prepare_image_latents fixtures."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny",
        help="Output directory for fixtures.",
    )
    parser.add_argument(
        "--vae-dir",
        default="fixtures/flux2_tiny/vae",
        help="Path to tiny VAE snapshot.",
    )
    parser.add_argument(
        "--transformer-config",
        default="fixtures/flux2_tiny/transformer/config.json",
        help="Path to tiny transformer config (for in_channels).",
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
    parser.add_argument("--num-images", type=int, default=2, help="Number of reference images.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for repeats.")
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
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.transformer_config).open("r", encoding="utf-8") as f:
        transformer_config = json.load(f)
    in_channels = transformer_config["in_channels"]
    num_latents_channels = in_channels // 4

    with Path(args.vae_dir, "config.json").open("r", encoding="utf-8") as f:
        vae_config = json.load(f)
    block_out_channels = vae_config["block_out_channels"]
    patch_size = vae_config["patch_size"]
    patch_h = patch_size[0]
    patch_w = patch_size[1]
    vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    adjusted_height = patch_h * (args.height // (vae_scale_factor * patch_h))
    adjusted_width = patch_w * (args.width // (vae_scale_factor * patch_w))

    latent_shape = (
        args.batch_size,
        num_latents_channels * patch_h * patch_w,
        adjusted_height // patch_h,
        adjusted_width // patch_w,
    )
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents)
    packed_latents = Flux2KleinPipeline._pack_latents(latents)

    save_file(
        {
            "latents": latents.contiguous().cpu(),
            "height": torch.tensor([args.height], dtype=torch.int32),
            "width": torch.tensor([args.width], dtype=torch.int32),
        },
        str(out_dir / "prepare_latents_inputs.safetensors"),
    )
    save_file(
        {
            "latents": packed_latents.contiguous().cpu(),
            "latent_ids": latent_ids.contiguous().cpu(),
        },
        str(out_dir / "prepare_latents_expected.safetensors"),
    )

    vae = AutoencoderKLFlux2(**vae_config)
    weights = load_file(str(Path(args.vae_dir) / "model.safetensors"))
    vae.load_state_dict(weights)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    images = torch.randn(
        (args.num_images, vae_config["in_channels"], args.height, args.width),
        device=device,
        dtype=dtype,
    )

    image_latents = []
    for idx in range(args.num_images):
        image = images[idx : idx + 1]
        posterior = vae.encode(image).latent_dist
        latents = posterior.mode()
        latents = Flux2KleinPipeline._patchify_latents(latents)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        )
        latents = (latents - latents_bn_mean) / latents_bn_std
        image_latents.append(latents)

    image_latent_ids = Flux2KleinPipeline._prepare_image_ids(image_latents)
    packed_list = []
    for latent in image_latents:
        packed = Flux2KleinPipeline._pack_latents(latent).squeeze(0)
        packed_list.append(packed)

    image_latents = torch.cat(packed_list, dim=0).unsqueeze(0)
    image_latents = image_latents.repeat(args.batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(args.batch_size, 1, 1)

    save_file(
        {
            "images": images.contiguous().cpu(),
            "batch_size": torch.tensor([args.batch_size], dtype=torch.int32),
        },
        str(out_dir / "prepare_image_latents_inputs.safetensors"),
    )
    save_file(
        {
            "image_latents": image_latents.contiguous().cpu(),
            "image_latent_ids": image_latent_ids.contiguous().cpu(),
        },
        str(out_dir / "prepare_image_latents_expected.safetensors"),
    )

    print("Wrote prepare_latents fixtures.")
    print("Wrote prepare_image_latents fixtures.")


if __name__ == "__main__":
    main()
