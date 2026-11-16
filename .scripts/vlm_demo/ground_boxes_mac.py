#!/usr/bin/env python3
"""VLM grounding on Mac (MPS or CPU).

Loads Qwen3-VL-4B-Instruct, asks for boxes as JSON text, draws them on the
image, and writes an annotated PNG.

Example:
  python3 .scripts/vlm_demo/ground_boxes_mac.py \\
    --image .scripts/vlm_demo/traffic.jpg \\
    --out .scripts/vlm_demo/out_boxes.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QUERY = (
    "Locate every car, bus, truck, taxi, and pedestrian in the image. "
    "Return a JSON array only, with one object per detection. "
    'Example item: {"label": "car", "bbox_2d": [x1, y1, x2, y2]}. '
    "label must be one of: car, bus, truck, taxi, person. "
    "bbox_2d coordinates are integers in 0-1000 relative to image width and height. "
    "Do not put multiple bbox_2d fields in the same object. "
    "Do not invent boxes for distant blobs you cannot resolve."
)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_vlm(
    model,
    processor,
    image: Image.Image,
    query: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out_ids)]
    texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return texts[0].strip()


def _extract_json_blob(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", raw) or re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object/list found in model output:\n{raw}")
    return json.loads(match.group(0))


def _boxes_from_regex(raw: str) -> list[dict]:
    """Recover boxes when the model emits duplicate keys or truncated JSON.

    Qwen3-VL sometimes packs many detections into one object:
      {"label":"car","bbox_2d":[...],"label":"car","bbox_2d":[...], ...}
    Standard json.loads keeps only the last key of each name.
    """
    boxes = []
    # Walk label / bbox_2d pairs in order (works for valid and malformed JSON).
    for m in re.finditer(
        r'"label"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]+)\]',
        raw,
    ):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(2))]
        if len(nums) >= 4:
            boxes.append({"label": m.group(1), "bbox_2d": nums[:4]})
    if boxes:
        return boxes

    # Fallback: bbox-only list under a single earlier label.
    label_m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    label = label_m.group(1) if label_m else "object"
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', raw):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if len(nums) >= 4:
            boxes.append({"label": label, "bbox_2d": nums[:4]})
    return boxes


def parse_boxes(raw: str) -> list[dict]:
    """Parse Qwen-style JSON boxes. Accepts list or single object."""
    cleaned = raw.strip().lower()
    if cleaned in {"", "[]", "none", "null", "there are none.", "there are none"}:
        return []

    # Prefer regex recovery: it keeps every label/bbox pair even when the
    # model packed duplicates into one JSON object.
    recovered = _boxes_from_regex(raw)
    if recovered:
        return recovered

    boxes: list[dict] = []
    try:
        data = _extract_json_blob(raw)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list, got {type(data)}")
        for item in data:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            label = str(item.get("label") or item.get("name") or "object")
            boxes.append({"label": label, "bbox_2d": [x1, y1, x2, y2]})
    except (ValueError, json.JSONDecodeError):
        return []
    return boxes


def norm_to_pixels(bbox_2d, width: int, height: int):
    """Map 0-1000 (or already-pixel) coords to pixel xyxy."""
    x1, y1, x2, y2 = bbox_2d
    # Heuristic: values within 0-1000 are relative; larger means pixel space.
    if max(x1, y1, x2, y2) <= 1000.0 + 1e-3:
        x1 = x1 / 1000.0 * width
        x2 = x2 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        y2 = y2 / 1000.0 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def draw_boxes(image: Image.Image, boxes: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()

    w, h = out.size
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6d597a"]
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = norm_to_pixels(box["bbox_2d"], w, h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = box["label"]
        tw = draw.textlength(label, font=font)
        pad = 3
        ty1 = max(0, y1 - 22)
        draw.rectangle([x1, ty1, x1 + tw + 2 * pad, ty1 + 20], fill=color)
        draw.text((x1 + pad, ty1 + 1), label, fill="white", font=font)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path, help="Input image path")
    p.add_argument("--query", default=DEFAULT_QUERY, help="Grounding instruction")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("out_boxes.png"),
        help="Annotated PNG path",
    )
    p.add_argument("--raw-out", type=Path, default=None, help="Save raw model text")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args(argv)

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    device = pick_device()
    print(f"device={device} model={args.model}")
    model, processor = load_model(args.model, device)

    image = Image.open(args.image).convert("RGB")
    raw = run_vlm(
        model, processor, image, args.query, device, args.max_new_tokens
    )
    print("--- raw ---")
    print(raw)
    print("------------")

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(raw + "\n", encoding="utf-8")

    boxes = parse_boxes(raw)
    print(f"parsed {len(boxes)} box(es)")
    # Drop near-point boxes that usually come from distant hallucination.
    w, h = image.size
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = norm_to_pixels(box["bbox_2d"], w, h)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if bw >= 8 and bh >= 8 and bw * bh >= 80:
            kept.append(box)
    if len(kept) != len(boxes):
        print(f"kept {len(kept)} after size filter")
    annotated = draw_boxes(image, kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.out)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
