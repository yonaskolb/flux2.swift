#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    MistralConfig,
    PixtralVisionConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny Mistral3 text encoder fixtures for Flux2-dev."
    )
    parser.add_argument(
        "--out-dir",
        default="fixtures/flux2_tiny_mistral3_text_encoder",
        help="Output directory for fixtures and text_encoder weights.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights and fixtures.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length.")
    parser.add_argument("--vocab-size", type=int, default=32, help="Vocabulary size.")
    parser.add_argument("--hidden-size", type=int, default=8, help="Hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--intermediate-size", type=int, default=16, help="MLP hidden size.")
    parser.add_argument("--attention-heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--kv-heads", type=int, default=1, help="Number of key/value heads.")
    parser.add_argument("--max-pos", type=int, default=128, help="Max position embeddings.")
    parser.add_argument(
        "--layers",
        default="0,1,2",
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


def main() -> None:
    args = parse_args()
    if args.hidden_size % args.attention_heads != 0:
        raise SystemExit("hidden-size must be divisible by attention-heads.")
    if args.kv_heads > args.attention_heads:
        raise SystemExit("kv-heads must be <= attention-heads.")

    torch.manual_seed(args.seed)
    dtype = resolve_dtype(args.dtype)

    head_dim = args.hidden_size // args.attention_heads
    text_config = MistralConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        vocab_size=args.vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=args.max_pos,
        head_dim=head_dim,
        rope_theta=10_000.0,
        sliding_window=None,
        tie_word_embeddings=True,
    )

    # Vision tower is unused for these fixtures but required by the config/model.
    vision_config = PixtralVisionConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=args.attention_heads,
        image_size=16,
        patch_size=8,
    )

    layer_types = ["full_attention"] * args.num_layers
    config = Mistral3Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        rope_parameters={
            "rope_theta": 10_000.0,
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": args.max_pos,
        },
        layer_types=layer_types,
    )

    model = Mistral3ForConditionalGeneration(config)
    model.eval()

    layer_indices = [int(item) for item in args.layers.split(",") if item.strip()]

    batch = args.batch_size
    seq_len = args.seq_len
    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(batch, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    pad_len = max(1, seq_len // 4)
    if pad_len < seq_len:
        input_ids[:, -pad_len:] = 0
        attention_mask[:, -pad_len:] = 0

    with torch.no_grad():
        output = model(
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

    t = torch.arange(1, dtype=torch.int64)
    h = torch.arange(1, dtype=torch.int64)
    w = torch.arange(1, dtype=torch.int64)
    l = torch.arange(seq_len, dtype=torch.int64)
    coords = torch.cartesian_prod(t, h, w, l)
    text_ids = coords.unsqueeze(0).expand(batch_size, -1, -1)

    out_dir = Path(args.out_dir)
    text_encoder_dir = out_dir / "text_encoder"
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    config_path = text_encoder_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)

    # Mistral3ForConditionalGeneration ties lm_head.weight to embed_tokens.weight by default,
    # which safetensors refuses to serialize. We do not need lm_head for Flux2 prompt embeds.
    weights = {
        key: value.to(dtype=dtype).cpu()
        for key, value in model.state_dict().items()
        if key != "lm_head.weight"
    }
    save_file(weights, str(text_encoder_dir / "model.safetensors"))

    save_file(
        {
            "prompt_embeds": prompt_embeds.cpu(),
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "text_ids": text_ids.contiguous().cpu(),
        },
        str(out_dir / "prompt_embeds.safetensors"),
    )

    print(f"Wrote {config_path}")
    print(f"Wrote {text_encoder_dir / 'model.safetensors'}")
    print(f"Wrote {out_dir / 'prompt_embeds.safetensors'}")


if __name__ == "__main__":
    main()
