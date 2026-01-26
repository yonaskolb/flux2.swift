#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional, List

import torch
from safetensors.torch import save_file
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FLUX.2 klein-4B prompt embedding fixtures via diffusers-compatible Qwen3 encoder."
    )
    parser.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--token", default=None, help="HF token if required.")
    parser.add_argument(
        "--text-encoder-subfolder",
        default="text_encoder",
        help="Subfolder for the text encoder weights.",
    )
    parser.add_argument(
        "--tokenizer-subfolder",
        default="tokenizer",
        help="Subfolder for tokenizer assets.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading from HF.",
    )
    parser.add_argument(
        "--prompt",
        default="A fluffy orange cat sitting on a windowsill",
        help="Prompt string to embed.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length.")
    parser.add_argument(
        "--layers",
        default="9,18,27",
        help="Comma-separated hidden state layer indices.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Output dtype for embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Torch device; default picks mps if available, else cpu.",
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_klein4b",
        help="Output directory for fixtures.",
    )
    parser.add_argument(
        "--diffusers-path",
        default="temp/diffusers/src",
        help="Path to diffusers src/ for ground-truth implementation.",
    )
    return parser.parse_args()


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    layers = [int(item) for item in args.layers.split(",") if item.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "revision": args.revision,
        "dtype": dtype,
        "subfolder": args.text_encoder_subfolder,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.token:
        model_kwargs["token"] = args.token
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.model_id, **model_kwargs).to(device)
    text_encoder.eval()

    tokenizer_kwargs = {
        "revision": args.revision,
        "subfolder": args.tokenizer_subfolder,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.token:
        tokenizer_kwargs["token"] = args.token
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_id, **tokenizer_kwargs)
    tokenizer_out = out_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_out)

    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
        dtype=dtype,
        device=device,
        max_sequence_length=args.max_seq_len,
        hidden_states_layers=layers,
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tensors = {
        "prompt_embeds": prompt_embeds.cpu(),
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
    }

    out_file = out_dir / "prompt_embeds.safetensors"
    save_file(tensors, str(out_file))

    print(f"Wrote {out_file}")
    print(f"prompt_embeds shape: {tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")


if __name__ == "__main__":
    main()
