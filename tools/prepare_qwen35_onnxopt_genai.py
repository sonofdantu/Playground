#!/usr/bin/env python3
"""Prepare an experimental ORT GenAI package from Qwen3.5 ONNX-OPT graphs.

The public ONNX-OPT export is shaped for Transformers.js/Optimum. It has the
right decoder, embedding, and vision graphs, but not the ORT GenAI package
contract. This script downloads one graph variant, patches graph I/O names that
ORT GenAI expects, adds image-feature injection to the embedding graph, and
writes a prototype genai_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper

sys.path.append(str(Path(__file__).resolve().parent))
from model_provenance import write_provenance  # noqa: E402


DEFAULT_REPO_ID = "onnx-community/Qwen3.5-2B-ONNX-OPT"
DEFAULT_UPSTREAM_REPO_ID = "Qwen/Qwen3.5-2B"

VARIANTS = {
    "q4f16": {
        "decoder": "decoder_model_merged_q4f16",
        "embedding": "embed_tokens_q4f16",
        "vision": "vision_encoder_q4f16",
    },
    "q4": {
        "decoder": "decoder_model_merged_q4",
        "embedding": "embed_tokens_q4",
        "vision": "vision_encoder_q4",
    },
    "quantized": {
        "decoder": "decoder_model_merged_quantized",
        "embedding": "embed_tokens_quantized",
        "vision": "vision_encoder_quantized",
    },
    "fp16": {
        "decoder": "decoder_model_merged_fp16",
        "embedding": "embed_tokens_fp16",
        "vision": "vision_encoder_fp16",
    },
}

LINEAR_LAYER_INDICES = [
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
]

QWEN35_IMAGE_TOKEN_ID = 248056
ORT_GENAI_QWEN_PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
QWEN35_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--upstream-repo-id", default=DEFAULT_UPSTREAM_REPO_ID)
    parser.add_argument("--output-dir", default="models/qwen3.5-2b-onnxopt-q4f16")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="q4f16")
    parser.add_argument("--force", action="store_true", help="Remove an existing output directory first.")
    parser.add_argument("--metadata-only", action="store_true", help="Download only small configs and ONNX graph headers.")
    parser.add_argument("--skip-provenance", action="store_true")
    return parser.parse_args()


def download(repo_id: str, filename: str, cache_dir: Path, revision: str | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def copy_downloaded(repo_id: str, filename: str, target: Path, cache_dir: Path, revision: str | None) -> None:
    cached = download(repo_id, filename, cache_dir, revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target)
    print(f"{filename} -> {target}")


def rename_graph_values(model: onnx.ModelProto, mapping: dict[str, str]) -> None:
    def rename_name(name: str) -> str:
        return mapping.get(name, name)

    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        value.name = rename_name(value.name)

    for initializer in model.graph.initializer:
        initializer.name = rename_name(initializer.name)

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = rename_name(name)
        for index, name in enumerate(node.output):
            node.output[index] = rename_name(name)


def patch_decoder_graph(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    mapping: dict[str, str] = {}
    for index in LINEAR_LAYER_INDICES:
        mapping[f"past_conv.{index}"] = f"past_key_values.{index}.conv_state"
        mapping[f"past_recurrent.{index}"] = f"past_key_values.{index}.recurrent_state"
        mapping[f"present_conv.{index}"] = f"present.{index}.conv_state"
        mapping[f"present_recurrent.{index}"] = f"present.{index}.recurrent_state"

    rename_graph_values(model, mapping)
    for index, graph_input in enumerate(model.graph.input):
        if graph_input.name == "num_logits_to_keep":
            del model.graph.input[index]
            if not any(initializer.name == "num_logits_to_keep" for initializer in model.graph.initializer):
                model.graph.initializer.append(helper.make_tensor("num_logits_to_keep", TensorProto.INT64, [], [0]))
            print("inlined decoder num_logits_to_keep=0 initializer")
            break
    onnx.save(model, path)
    print(f"patched decoder recurrent I/O names: {path}")


def make_int64_constant(name: str, value: int | Iterable[int], dims: list[int] | None = None) -> onnx.NodeProto:
    if isinstance(value, int):
        values = [value]
        tensor_dims = [] if dims is None else dims
    else:
        values = list(value)
        tensor_dims = [len(values)] if dims is None else dims
    tensor = helper.make_tensor(name + "_tensor", TensorProto.INT64, tensor_dims, values)
    return helper.make_node("Constant", [], [name], name=name + "_const", value=tensor)


def patch_embedding_graph(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph

    if any(input_value.name == "image_features" for input_value in graph.input):
        print(f"embedding graph already has image_features: {path}")
        return

    original_output_name = graph.output[0].name
    base_output_name = "/qwen35/base_inputs_embeds"
    rename_graph_values(model, {original_output_name: base_output_name})

    image_features = helper.make_tensor_value_info(
        "image_features",
        TensorProto.FLOAT,
        ["num_logical_patches", 2048],
    )
    graph.input.append(image_features)
    graph.output[0].name = "inputs_embeds"

    nodes = [
        make_int64_constant("/qwen35/image_token_id", QWEN35_IMAGE_TOKEN_ID),
        helper.make_node("Equal", ["input_ids", "/qwen35/image_token_id"], ["/qwen35/image_token_mask"], name="/qwen35/EqualImageToken"),
        make_int64_constant("/qwen35/axes_neg1", [-1]),
        helper.make_node("Unsqueeze", ["/qwen35/image_token_mask", "/qwen35/axes_neg1"], ["/qwen35/image_token_mask_unsqueezed"], name="/qwen35/UnsqueezeMask"),
        helper.make_node("Shape", [base_output_name], ["/qwen35/base_shape"], name="/qwen35/BaseShape"),
        helper.make_node("Expand", ["/qwen35/image_token_mask_unsqueezed", "/qwen35/base_shape"], ["/qwen35/expanded_mask"], name="/qwen35/ExpandMask"),
        helper.make_node("Cast", ["/qwen35/expanded_mask"], ["/qwen35/expanded_mask_bool"], name="/qwen35/CastMaskBool", to=TensorProto.BOOL),
        helper.make_node("NonZero", ["/qwen35/expanded_mask_bool"], ["/qwen35/nonzero_indices_raw"], name="/qwen35/NonZeroImagePositions"),
        helper.make_node("Transpose", ["/qwen35/nonzero_indices_raw"], ["/qwen35/nonzero_indices"], name="/qwen35/TransposeIndices", perm=[1, 0]),
        helper.make_node("Cast", ["image_features"], ["/qwen35/image_features_float"], name="/qwen35/CastImageFeatures", to=TensorProto.FLOAT),
        make_int64_constant("/qwen35/reshape_flat", [-1]),
        helper.make_node("Reshape", ["/qwen35/image_features_float", "/qwen35/reshape_flat"], ["/qwen35/image_features_flat"], name="/qwen35/FlattenImageFeatures", allowzero=0),
        helper.make_node("Shape", ["/qwen35/nonzero_indices"], ["/qwen35/index_shape"], name="/qwen35/IndexShape"),
        make_int64_constant("/qwen35/axis0_scalar", 0),
        helper.make_node("Gather", ["/qwen35/index_shape", "/qwen35/axis0_scalar"], ["/qwen35/update_count"], name="/qwen35/GatherUpdateCount", axis=0),
        make_int64_constant("/qwen35/axes0", [0]),
        helper.make_node("Unsqueeze", ["/qwen35/update_count", "/qwen35/axes0"], ["/qwen35/slice_end"], name="/qwen35/UnsqueezeSliceEnd"),
        make_int64_constant("/qwen35/slice_start", [0]),
        make_int64_constant("/qwen35/slice_axis", [0]),
        helper.make_node("Slice", ["/qwen35/image_features_flat", "/qwen35/slice_start", "/qwen35/slice_end", "/qwen35/slice_axis"], ["/qwen35/scatter_updates"], name="/qwen35/SliceImageUpdates"),
        helper.make_node("ScatterND", [base_output_name, "/qwen35/nonzero_indices", "/qwen35/scatter_updates"], ["inputs_embeds"], name="/qwen35/ScatterImageFeatures"),
    ]
    graph.node.extend(nodes)
    # Do not call check_model here: metadata-only runs intentionally omit
    # external data shards, and ONNX's checker insists those files exist.
    onnx.save(model, path)
    print(f"patched embedding image-feature injection: {path}")


def patch_tokenizer_config(output_dir: Path) -> None:
    path = output_dir / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))

    # The ONNX-OPT repository is packaged for Transformers.js and labels the
    # tokenizer as TokenizersBackend. ORT GenAI routes tokenizer loading through
    # tokenizer_class and supports Qwen through the Qwen2Tokenizer path.
    config["tokenizer_class"] = "Qwen2Tokenizer"
    config["additional_special_tokens"] = QWEN35_SPECIAL_TOKENS
    config.pop("pretokenize_regex", None)

    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched tokenizer_config tokenizer_class for ORT GenAI: {path}")


def patch_tokenizer_json(output_dir: Path) -> None:
    path = output_dir / "tokenizer.json"
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    pre_tokenizer = tokenizer.get("pre_tokenizer", {})
    pretokenizers = pre_tokenizer.get("pretokenizers", [])
    for entry in pretokenizers:
        pattern = entry.get("pattern")
        if isinstance(pattern, dict) and "Regex" in pattern:
            pattern["Regex"] = ORT_GENAI_QWEN_PRETOKENIZER_REGEX
            break

    path.write_text(json.dumps(tokenizer, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"patched tokenizer.json pre-tokenizer regex for ORT GenAI: {path}")


def write_processor_config(output_dir: Path) -> None:
    processor_config = {
        "processor": {
            "name": "qwen3_5_image_processor",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {"color_space": "RGB"},
                    }
                },
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {
                    "operation": {
                        "name": "resize",
                        "type": "Resize",
                        "attrs": {
                            "width": 448,
                            "height": 448,
                            "smart_resize": 1,
                            "min_pixels": 65536,
                            "max_pixels": 16777216,
                            "patch_size": 16,
                            "merge_size": 2,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "rescale",
                        "type": "Rescale",
                        "attrs": {"rescale_factor": 0.00392156862745098},
                    }
                },
                {
                    "operation": {
                        "name": "normalize",
                        "type": "Normalize",
                        "attrs": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                            "qwen3_vl": 1,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": 16,
                            "temporal_patch_size": 2,
                            "merge_size": 2,
                        },
                    }
                },
            ],
        }
    }
    (output_dir / "processor_config.json").write_text(json.dumps(processor_config, indent=2), encoding="utf-8")


def write_genai_config(output_dir: Path, variant_files: dict[str, str]) -> None:
    config = {
        "model": {
            "bos_token_id": 248044,
            "context_length": 262144,
            "decoder": {
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
                "filename": f"onnx/{variant_files['decoder']}.onnx",
                "head_size": 256,
                "hidden_size": 2048,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
            },
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "type": "qwen3_5",
            "vocab_size": 248320,
            "embedding": {
                "filename": f"onnx/{variant_files['embedding']}.onnx",
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {"inputs_embeds": "inputs_embeds"},
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
            },
            "vision": {
                "filename": f"onnx/{variant_files['vision']}.onnx",
                "config_filename": "processor_config.json",
                "spatial_merge_size": 2,
                "tokens_per_second": 2.0,
                "patch_size": 16,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {"image_features": "image_features"},
                "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
            },
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": True,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 262144,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": True,
            "repetition_penalty": 1.0,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        },
    }
    (output_dir / "genai_config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    variant_files = VARIANTS[args.variant]

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_files = [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for file_name in root_files:
        copy_downloaded(args.repo_id, file_name, output_dir / file_name, cache_dir, args.revision)

    copy_downloaded(args.repo_id, "processor_config.json", output_dir / "hf_processor_config.json", cache_dir, args.revision)

    selected_onnx = [
        variant_files["decoder"],
        variant_files["embedding"],
        variant_files["vision"],
    ]
    for stem in selected_onnx:
        copy_downloaded(args.repo_id, f"onnx/{stem}.onnx", output_dir / "onnx" / f"{stem}.onnx", cache_dir, args.revision)
        if not args.metadata_only:
            copy_downloaded(args.repo_id, f"onnx/{stem}.onnx_data", output_dir / "onnx" / f"{stem}.onnx_data", cache_dir, args.revision)

    patch_decoder_graph(output_dir / "onnx" / f"{variant_files['decoder']}.onnx")
    patch_embedding_graph(output_dir / "onnx" / f"{variant_files['embedding']}.onnx")
    patch_tokenizer_config(output_dir)
    patch_tokenizer_json(output_dir)
    write_processor_config(output_dir)
    write_genai_config(output_dir, variant_files)

    if not args.skip_provenance:
        manifest = write_provenance(
            output_dir,
            classification="prototype",
            package_repo_id=args.repo_id,
            package_source_prefix=None,
            package_revision=args.revision,
            upstream_repo_id=args.upstream_repo_id,
            upstream_revision=None,
            cache_dir=str(cache_dir),
            notes=[
                "Experimental ORT GenAI package assembled from ONNX-OPT graphs.",
                "Decoder recurrent I/O names patched for ORT GenAI recurrent-state discovery.",
                "Embedding graph patched to scatter image_features into image token positions.",
            ],
        )
        print(f"wrote {manifest}")

    print(f"prepared experimental Qwen3.5 ORT GenAI package: {output_dir}")


if __name__ == "__main__":
    main()
