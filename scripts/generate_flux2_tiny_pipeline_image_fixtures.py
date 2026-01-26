#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Flux2 image-conditioned pipeline fixtures."
    )
    parser.add_argument(
        "--snapshot",
        default="fixtures/flux2_tiny_pipeline",
        help="Snapshot root containing transformer/scheduler/vae.",
    )
    parser.add_argument(
        "--inputs",
        default="fixtures/flux2_tiny/transformer_inputs.safetensors",
        help="Source transformer inputs for text/image ids and encoder states.",
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
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=3,
        help="Number of inference steps for the denoise loop.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2,
        help="Number of conditioning images.",
    )
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
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    snapshot = Path(args.snapshot)
    transformer_dir = snapshot / "transformer"
    scheduler_dir = snapshot / "scheduler"
    vae_dir = snapshot / "vae"

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
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    with (vae_dir / "config.json").open("r", encoding="utf-8") as f:
        vae_config = json.load(f)
    vae = AutoencoderKLFlux2(**vae_config)
    vae_weights = load_file(str(vae_dir / "model.safetensors"))
    vae.load_state_dict(vae_weights)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    input_reader = load_file(args.inputs)
    encoder_hidden_states = input_reader["encoder_hidden_states"].to(device=device, dtype=dtype)
    img_ids = input_reader["img_ids"].to(device=device)
    txt_ids = input_reader["txt_ids"].to(device=device)

    batch = encoder_hidden_states.shape[0]
    seq_len = img_ids.shape[1]
    in_channels = transformer_config["in_channels"]
    latents = torch.randn((batch, seq_len, in_channels), device=device, dtype=dtype)
    latents_init = latents.clone()

    images = torch.randn(
        (args.num_images, vae_config["in_channels"], vae_config["sample_size"], vae_config["sample_size"]),
        device=device,
        dtype=dtype,
    )

    image_latents = []
    for idx in range(args.num_images):
        image = images[idx : idx + 1]
        posterior = vae.encode(image).latent_dist
        latents_img = posterior.mode()
        latents_img = Flux2KleinPipeline._patchify_latents(latents_img)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_img.device, latents_img.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        )
        latents_img = (latents_img - latents_bn_mean) / latents_bn_std
        image_latents.append(latents_img)

    image_latent_ids = Flux2KleinPipeline._prepare_image_ids(image_latents)
    packed_list = []
    for latent in image_latents:
        packed = Flux2KleinPipeline._pack_latents(latent).squeeze(0)
        packed_list.append(packed)
    image_latents = torch.cat(packed_list, dim=0).unsqueeze(0)
    image_latents = image_latents.repeat(batch, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch, 1, 1)

    timesteps = scheduler.timesteps.to(device=device, dtype=dtype)
    for t in timesteps:
        timestep = (t / 1000).expand(latents.shape[0])
        latent_model_input = torch.cat([latents, image_latents], dim=1).to(dtype)
        latent_ids = torch.cat([img_ids, image_latent_ids], dim=1)
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=latent_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.shape[1], :]
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latent_ids = img_ids.clone()
    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    unpacked = unpacked * latents_bn_std + latents_bn_mean
    unpatchified = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded = vae.decode(unpatchified, return_dict=False)[0]

    inputs_out = snapshot / "pipeline_image_inputs.safetensors"
    expected_out = snapshot / "pipeline_image_expected.safetensors"

    save_file(
        {
            "latents": latents_init.contiguous().cpu(),
            "encoder_hidden_states": encoder_hidden_states.contiguous().cpu(),
            "timesteps": timesteps.contiguous().cpu(),
            "img_ids": img_ids.contiguous().cpu(),
            "txt_ids": txt_ids.contiguous().cpu(),
            "images": images.contiguous().cpu(),
            "batch_size": torch.tensor([batch], dtype=torch.int32),
        },
        str(inputs_out),
    )

    save_file(
        {
            "packed_latents": latents.contiguous().cpu(),
            "decoded": decoded.contiguous().cpu(),
            "image_latents": image_latents.contiguous().cpu(),
            "image_latent_ids": image_latent_ids.contiguous().cpu(),
        },
        str(expected_out),
    )

    print(f"Wrote pipeline image inputs to {inputs_out}")
    print(f"Wrote pipeline image expected outputs to {expected_out}")


if __name__ == "__main__":
    main()
