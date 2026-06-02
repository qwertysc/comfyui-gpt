#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-compatible GPT image nodes for ComfyUI.
"""

import base64
import json
import re
import time
from io import BytesIO

import numpy as np
from PIL import Image
import requests
import torch


DEFAULT_API_BASE_URL = ""
MAINLINE_MODEL_OPTIONS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]
MAINLINE_DEFAULT_MODEL = "gpt-5.4"
API_BASE_INPUT = (
    "STRING",
    {
        "default": DEFAULT_API_BASE_URL,
        "multiline": False,
        "placeholder": "https://api.openai.com/v1",
    },
)
API_CONNECT_TIMEOUT_SECONDS = 30
HTTP_SESSION = requests.Session()
HTTP_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
HTTP_SESSION.mount("http://", HTTP_ADAPTER)
HTTP_SESSION.mount("https://", HTTP_ADAPTER)
INPUT_IMAGE_WEBP_QUALITY = 100
INPUT_IMAGE_WEBP_METHOD = 0


def api_timeout(timeout_seconds):
    try:
        read_timeout = int(timeout_seconds)
    except (TypeError, ValueError):
        read_timeout = 300
    return (API_CONNECT_TIMEOUT_SECONDS, max(1, read_timeout))


def api_get(url, timeout_seconds, **kwargs):
    return HTTP_SESSION.get(url, timeout=api_timeout(timeout_seconds), **kwargs)


def api_post(url, timeout_seconds, **kwargs):
    return HTTP_SESSION.post(url, timeout=api_timeout(timeout_seconds), **kwargs)


def normalize_api_base(api_base):
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("api_base 不能为空，请填写 OpenAI 兼容接口地址，例如 https://api.openai.com/v1")
    if not re.match(r"^https?://", base, re.I):
        raise ValueError("api_base 必须以 http:// 或 https:// 开头")
    return base


def build_api_url(api_base, endpoint_path):
    """Build an OpenAI-compatible URL without duplicating /v1."""
    base = normalize_api_base(api_base)
    path = "/" + str(endpoint_path or "").strip().lstrip("/")
    if base.endswith(path):
        return base
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


def _tensor_to_pil_image(tensor):
    if tensor is None:
        raise ValueError("输入图像为空")

    single = tensor[0] if len(tensor.shape) == 4 else tensor
    if hasattr(single, "detach"):
        single = single.detach()
    if hasattr(single, "cpu"):
        single = single.cpu()

    arr = np.asarray(single)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")

    channels = arr.shape[-1] if arr.ndim == 3 else 0
    if channels >= 4:
        return Image.fromarray(arr[:, :, :4], mode="RGBA")
    if channels == 1:
        return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    if channels >= 3:
        return Image.fromarray(arr[:, :, :3], mode="RGB")
    raise ValueError(f"不支持的图像 tensor 形状: {getattr(tensor, 'shape', None)}")


def _image_has_transparency(img):
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
        return alpha.getextrema()[0] < 255
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def _base64_length(byte_count):
    return 4 * ((byte_count + 2) // 3)


def data_url_length(image_bytes, mime_type):
    return len(f"data:{mime_type};base64,") + _base64_length(len(image_bytes))


def bytes_to_data_url(image_bytes, mime_type):
    return f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("utf-8")


def tensor_to_image_payload(tensor, name="image"):
    """Encode a ComfyUI IMAGE tensor for OpenAI-compatible image input."""
    img = _tensor_to_pil_image(tensor)
    has_transparency = _image_has_transparency(img)
    buf = BytesIO()
    if has_transparency:
        mime_type = "image/png"
        extension = "png"
        img.convert("RGBA").save(buf, format="PNG")
    else:
        mime_type = "image/webp"
        extension = "webp"
        img.convert("RGB").save(
            buf,
            format="WEBP",
            quality=INPUT_IMAGE_WEBP_QUALITY,
            method=INPUT_IMAGE_WEBP_METHOD,
        )

    image_bytes = buf.getvalue()
    return {
        "filename": f"{name}.{extension}",
        "bytes": image_bytes,
        "mime_type": mime_type,
        "format": extension,
        "has_transparency": has_transparency,
        "byte_count": len(image_bytes),
        "data_url_bytes": data_url_length(image_bytes, mime_type),
    }


def tensor_to_png_bytes(tensor):
    """ComfyUI IMAGE tensor -> PNG bytes."""
    img = _tensor_to_pil_image(tensor)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tensor_to_data_url(tensor):
    """ComfyUI IMAGE tensor -> compact data URL for multimodal prompts."""
    payload = tensor_to_image_payload(tensor)
    return bytes_to_data_url(payload["bytes"], payload["mime_type"])


def mask_to_png_bytes(mask):
    """ComfyUI MASK -> RGBA PNG mask for OpenAI Images edit.

    ComfyUI mask value 1 means edit area. OpenAI-style image masks use
    transparent pixels as edit area, so alpha is inverted.
    """
    if mask is None:
        return None

    if len(mask.shape) == 3:
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask.cpu().numpy()

    alpha = ((1.0 - mask_np) * 255).clip(0, 255).astype(np.uint8)
    height, width = alpha.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = alpha

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def image_bytes_to_tensor(image_bytes):
    """Image bytes -> ComfyUI tensor (1,H,W,3)."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).float()


def b64_json_to_tensor(b64_json):
    """Decode API b64_json, data URL, or plain base64 image content."""
    value = (b64_json or "").strip()
    if not value:
        raise ValueError("base64 图片内容为空")

    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]

    return image_bytes_to_tensor(base64.b64decode(value))


def download_image_url(url, timeout_seconds):
    headers = {
        "User-Agent": "ComfyUI GPT Image/1.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    response = api_get(url, timeout_seconds, headers=headers)
    response.raise_for_status()
    return image_bytes_to_tensor(response.content)


def parse_sse_events(text):
    events = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            if payload.startswith("data:image/"):
                events.append({"b64_json": payload})
    return events


def parse_api_payload(response):
    content_type = response.headers.get("Content-Type", "")
    if content_type.lower().startswith("image/"):
        return {"data": [{"image_bytes": response.content}]}

    try:
        return response.json()
    except ValueError as exc:
        events = parse_sse_events(response.text)
        if events:
            return {"data": events}
        body = response.text[:1000] if response.text else "<empty response>"
        raise RuntimeError(f"API 返回非 JSON 响应: {body}") from exc


def raise_for_api_error(data):
    if not isinstance(data, dict):
        return

    error = data.get("error")
    if not error:
        return

    if isinstance(error, dict):
        message = error.get("message") or str(error)
        error_type = error.get("type")
        code = error.get("code")
        details = []
        if error_type:
            details.append(f"type={error_type}")
        if code:
            details.append(f"code={code}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise RuntimeError(f"API Error: {message}{suffix}")

    raise RuntimeError(f"API Error: {error}")


def extract_image_values(value):
    values = []
    if isinstance(value, (bytes, bytearray)):
        return [bytes(value)]
    if isinstance(value, list):
        for item in value:
            values.extend(extract_image_values(item))
        return values
    if not isinstance(value, dict):
        return values

    for key, item in value.items():
        if key in ("b64_json", "partial_image_b64", "result") and isinstance(item, str) and item.strip():
            values.append(item.strip())
            continue
        if key == "image_bytes" and isinstance(item, (bytes, bytearray)):
            values.append(bytes(item))
            continue
        if key == "url" and isinstance(item, str) and item.strip().lower().startswith(("http://", "https://", "data:image/")):
            values.append(item.strip())
            continue
        if key == "image_url":
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
                continue
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url.strip():
                    values.append(url.strip())
                    continue
        values.extend(extract_image_values(item))
    return values


def image_value_to_tensor(value, timeout_seconds):
    if isinstance(value, (bytes, bytearray)):
        return image_bytes_to_tensor(bytes(value)), "binary_image"

    text = str(value or "").strip()
    if text.lower().startswith(("http://", "https://")):
        return download_image_url(text, timeout_seconds), text
    return b64_json_to_tensor(text), "inline_base64"


GPT_IMAGE2_SIZE_TABLE = {
    "1K": {
        "AUTO": "auto",
        "1:4": "480x1440",
        "4:1": "1440x480",
        "1:8": "480x1440",
        "8:1": "1440x480",
        "1:1": "1024x1024",
        "1:2": "720x1440",
        "2:1": "1440x720",
        "1:3": "480x1440",
        "3:1": "1440x480",
        "2:3": "768x1152",
        "3:2": "1152x768",
        "3:4": "768x1024",
        "4:3": "1024x768",
        "4:5": "768x960",
        "5:4": "960x768",
        "9:16": "720x1280",
        "16:9": "1280x720",
        "9:21": "640x1488",
        "21:9": "1344x576",
    },
    "2K": {
        "AUTO": "auto",
        "1:4": "672x2016",
        "4:1": "2016x672",
        "1:8": "672x2016",
        "8:1": "2016x672",
        "1:1": "2048x2048",
        "1:2": "1024x2048",
        "2:1": "2048x1024",
        "1:3": "672x2016",
        "3:1": "2016x672",
        "2:3": "1440x2160",
        "3:2": "2160x1440",
        "3:4": "1536x2048",
        "4:3": "2048x1536",
        "4:5": "1536x1920",
        "5:4": "1920x1536",
        "9:16": "1152x2048",
        "16:9": "2048x1152",
        "9:21": "960x2240",
        "21:9": "2464x1056",
    },
    "4K": {
        "AUTO": "auto",
        "1:4": "1280x3840",
        "4:1": "3840x1280",
        "1:8": "1280x3840",
        "8:1": "3840x1280",
        "1:1": "2880x2880",
        "1:2": "1920x3840",
        "2:1": "3840x1920",
        "1:3": "1280x3840",
        "3:1": "3840x1280",
        "2:3": "2304x3456",
        "3:2": "3456x2304",
        "3:4": "2448x3264",
        "4:3": "3264x2448",
        "4:5": "2304x2880",
        "5:4": "2880x2304",
        "9:16": "2160x3840",
        "16:9": "3840x2160",
        "9:21": "1648x3840",
        "21:9": "3808x1632",
    },
}


def _validate_gpt_image2_size(size_value):
    if size_value == "auto":
        return size_value

    if not re.fullmatch(r"\d{3,4}x\d{3,4}", size_value):
        raise ValueError("size 必须类似 1600x1200，且宽高都是数字")

    width, height = [int(v) for v in size_value.split("x")]
    max_side = max(width, height)
    min_side = min(width, height)
    total_pixels = width * height

    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("size 的宽和高都必须是 16 的倍数")
    if max_side > 3840:
        raise ValueError("size 最大边不能超过 3840px")
    if max_side / min_side > 3:
        raise ValueError("size 长边/短边不能超过 3:1，因此 3:1 和 1:3 可以，超过不行")
    if total_pixels < 655360 or total_pixels > 8294400:
        raise ValueError("size 总像素需在 655,360 到 8,294,400 之间")

    return f"{width}x{height}"


def _extract_aspect_ratio(value):
    text = str(value or "")
    if text.upper().startswith("AUTO"):
        return "AUTO"
    match = re.search(r"(?:21:9|16:9|9:16|4:5|5:4|4:3|3:4|3:2|2:3|1:8|8:1|1:4|4:1|1:1)", text)
    return match.group(0) if match else "16:9"


def _is_size_tier(value):
    option = str(value or "").strip().lower()
    return option in ("1k", "2k", "4k")


def _tensor_dimensions(tensor):
    if tensor is None:
        return None
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[0])
    return None


def _nearest_aspect_ratio(width, height):
    if not width or not height:
        return "1:1"
    target = width / height
    ratios = [ratio for ratio in GPT_IMAGE2_SIZE_TABLE["1K"] if ratio != "AUTO"]

    def ratio_distance(ratio):
        left, right = [int(part) for part in ratio.split(":")]
        candidate = left / right
        return abs(candidate - target) / max(candidate, target)

    return min(ratios, key=ratio_distance)


def resolve_auto_aspect_ratio(image_size, aspect_ratio, source_image=None):
    ratio = _extract_aspect_ratio(aspect_ratio)
    if ratio != "AUTO" or not _is_size_tier(image_size):
        return ratio

    dimensions = _tensor_dimensions(source_image)
    if dimensions is None:
        return "1:1"
    width, height = dimensions
    return _nearest_aspect_ratio(width, height)


def normalize_size(image_size, aspect_ratio="16:9", custom_size=""):
    option = (image_size or "2K").strip().replace("×", "x")
    option_lower = option.lower()

    if option_lower.startswith("auto"):
        return "auto"

    if option_lower.startswith("custom"):
        custom = (custom_size or "").strip().lower().replace("×", "x")
        if not custom:
            raise ValueError("选择 custom 时，custom_size 必须填写，例如 3072x1024 或 1024x3072")
        return _validate_gpt_image2_size(custom)

    match = re.match(r"(\d{3,4}x\d{3,4})", option_lower)
    if match:
        return _validate_gpt_image2_size(match.group(1))

    tier = None
    if "1k" in option_lower:
        tier = "1K"
    elif "2k" in option_lower:
        tier = "2K"
    elif "4k" in option_lower:
        tier = "4K"

    ratio = _extract_aspect_ratio(aspect_ratio)
    if tier and ratio in GPT_IMAGE2_SIZE_TABLE[tier]:
        return _validate_gpt_image2_size(GPT_IMAGE2_SIZE_TABLE[tier][ratio])

    raise ValueError(f"无法识别尺寸组合: image_size={image_size}, aspect_ratio={aspect_ratio}")


def is_retryable_http_status(status_code):
    return status_code in (408, 429) or status_code >= 500


def safe_choice(value, choices, default):
    return value if value in choices else default


def safe_int(value, default, min_value=None, max_value=None):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    if min_value is not None:
        number = max(min_value, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


def normalize_prompt_text(value):
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def emit_runtime_status(
    node_id,
    status,
    message="",
    elapsed_seconds=0.0,
    attempt=0,
    retry_times=0,
    timeout_seconds=0,
):
    """Send runtime status to the ComfyUI frontend extension."""
    if node_id in (None, ""):
        return
    try:
        from server import PromptServer

        if PromptServer.instance is None:
            return

        PromptServer.instance.send_sync(
            "comfyui_luck_gpt20_status",
            {
                "node_id": str(node_id),
                "status": status,
                "message": message,
                "elapsed_seconds": float(elapsed_seconds),
                "attempt": int(attempt),
                "retry_times": int(retry_times),
                "timeout_seconds": int(timeout_seconds),
                "timestamp": time.time(),
            },
        )
    except Exception:
        pass


class ComfyuiLuckGPTImage2Node:
    """OpenAI-compatible gpt-image-2 node with size, quality, format, and mask controls."""

    MODELS = ["gpt-image-2"]
    IMAGE_SIZES = [
        "auto (不传size)",
        "1K",
        "2K",
        "4K",
        "custom (自定义)",
    ]
    ASPECT_RATIOS = [
        "AUTO",
        "1:4",
        "4:1",
        "1:8",
        "8:1",
        "1:1",
        "1:2",
        "2:1",
        "1:3",
        "3:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "9:21",
        "21:9",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": ("STRING", {"default": "", "multiline": False}),
                "api_base (接口域名)": API_BASE_INPUT,
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "mode (模式)": (["AUTO", "text2img", "img2img"], {"default": "AUTO"}),
                "model (模型)": (cls.MODELS, {"default": "gpt-image-2"}),
                "image_size (分辨率)": (cls.IMAGE_SIZES, {"default": "2K"}),
                "aspect_ratio (宽高比)": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "custom_size (仅custom填写: 宽x高)": ("STRING", {"default": "1600x1200", "multiline": False}),
                "quality (画质)": (["auto", "low", "medium", "high"], {"default": "auto"}),
                "output_format (输出格式)": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression (压缩率)": ("INT", {"default": 85, "min": 0, "max": 100}),
                "stream (流式兼容)": ("BOOLEAN", {"default": False}),
                "seed (种子)": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                    },
                ),
                "timeout_seconds (超时秒数)": ("INT", {"default": 360, "min": 60, "max": 1800}),
                "retry_times (重试次数)": ("INT", {"default": 3, "min": 1, "max": 10}),
                "api_mode (接口模式)": (
                    ["responses_api", "images_api"],
                    {"default": "responses_api"},
                ),
                "mainline_model (Responses主模型)": (
                    MAINLINE_MODEL_OPTIONS,
                    {"default": MAINLINE_DEFAULT_MODEL},
                ),
            },
            "optional": {
                **{f"image_{i:02d}": ("IMAGE",) for i in range(1, 6)},
                "mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-Luck/gpt-image-2"

    def _collect_images(self, kwargs):
        image_payloads = []
        for i in range(1, 6):
            tensor = kwargs.get(f"image_{i:02d}")
            if tensor is None:
                continue
            payload = tensor_to_image_payload(tensor, f"image_{i:02d}")
            payload["slot"] = f"image_{i:02d}"
            image_payloads.append(payload)
        return image_payloads

    def _image_payload_summary(self, image_payloads, mask_bytes=None):
        images = [
            {
                "slot": payload["slot"],
                "filename": payload["filename"],
                "mime_type": payload["mime_type"],
                "format": payload["format"],
                "bytes": payload["byte_count"],
                "data_url_bytes": payload["data_url_bytes"],
            }
            for payload in image_payloads
        ]
        total_image_bytes = sum(item["bytes"] for item in images)
        total_data_url_bytes = sum(item["data_url_bytes"] for item in images)
        if mask_bytes is not None:
            mask_data_url_bytes = data_url_length(mask_bytes, "image/png")
            total_image_bytes += len(mask_bytes)
            total_data_url_bytes += mask_data_url_bytes
        else:
            mask_data_url_bytes = 0
        return {
            "images": images,
            "mask_bytes": len(mask_bytes) if mask_bytes is not None else 0,
            "mask_data_url_bytes": mask_data_url_bytes,
            "total_encoded_bytes": total_image_bytes,
            "total_data_url_bytes": total_data_url_bytes,
        }

    def _responses_image_label(self, filename, index):
        slot = (filename or f"image_{index:02d}.png").rsplit(".", 1)[0]
        return (
            f"图{index} / {slot}: 以下图片是第 {index} 张参考图。"
            f"提示词中提到“图{index}”或“{slot}”时，均指这张图片；请严格按提示词指定的用途引用它。"
        )

    def _payload_fields(self, model, prompt, size, quality, output_format, output_compression, stream):
        fields = {
            "model": model,
            "prompt": prompt,
        }
        if size != "auto":
            fields["size"] = size
        if quality != "auto":
            fields["quality"] = quality
        if output_format != "png":
            fields["output_format"] = output_format
            fields["output_compression"] = output_compression
        if stream:
            fields["stream"] = True
            fields["partial_images"] = 1
        return fields

    def _responses_tool(self, fields, actual_mode, mask_bytes):
        tool = {
            "type": "image_generation",
            "model": fields["model"],
            "action": "edit" if actual_mode == "img2img" else "generate",
        }
        if fields.get("size"):
            tool["size"] = fields["size"]
        if fields.get("quality"):
            tool["quality"] = fields["quality"]
        if fields.get("output_format"):
            tool["output_format"] = fields["output_format"]
        if fields.get("output_compression") is not None:
            tool["output_compression"] = fields["output_compression"]
        if fields.get("stream"):
            tool["partial_images"] = 1
        if mask_bytes is not None:
            tool["input_image_mask"] = {
                "image_url": bytes_to_data_url(mask_bytes, "image/png")
            }
        return tool

    def _responses_input(self, prompt, image_payloads):
        if not image_payloads:
            return prompt

        prompt_with_rules = (
            f"{prompt}\n\n"
            "多图引用规则：输入图片按 image_01、image_02、image_03 的顺序提供，"
            "图1 对应 image_01，图2 对应 image_02，依此类推。"
            "如果提示词要求从不同图片分别参考人物身份、人脸、服装、构图或风格，"
            "必须分别执行，不要把另一张图片的人脸或身份错误迁移到结果中。"
        )
        content = [{"type": "input_text", "text": prompt_with_rules}]
        for index, payload in enumerate(image_payloads, 1):
            filename = payload["filename"]
            content.append({"type": "input_text", "text": self._responses_image_label(filename, index)})
            content.append({
                "type": "input_image",
                "image_url": bytes_to_data_url(payload["bytes"], payload["mime_type"]),
            })
        return [{"role": "user", "content": content}]

    def _request_responses(self, api_base, headers, mainline_model, fields, actual_mode, image_payloads, mask_bytes, timeout_seconds):
        tool = self._responses_tool(fields, actual_mode, mask_bytes)
        payload = {
            "model": mainline_model,
            "input": self._responses_input(fields["prompt"], image_payloads),
            "tools": [tool],
            "tool_choice": {"type": "image_generation"},
        }
        if fields.get("stream"):
            payload["stream"] = True
        return api_post(
            build_api_url(api_base, "/responses"),
            timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            stream=bool(fields.get("stream")),
        )

    def _request_text2img(self, api_base, headers, fields, timeout_seconds):
        return api_post(
            build_api_url(api_base, "/images/generations"),
            timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=fields,
            stream=bool(fields.get("stream")),
        )

    def _request_img2img(self, api_base, headers, fields, image_payloads, mask_bytes, timeout_seconds):
        files = [
            ("image[]", (payload["filename"], BytesIO(payload["bytes"]), payload["mime_type"]))
            for payload in image_payloads
        ]
        if mask_bytes is not None:
            files.append(("mask", ("mask.png", BytesIO(mask_bytes), "image/png")))

        data = {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in fields.items()}
        return api_post(
            build_api_url(api_base, "/images/edits"),
            timeout_seconds,
            headers=headers,
            data=data,
            files=files,
            stream=bool(fields.get("stream")),
        )

    def _parse_response_images(self, data, timeout_seconds):
        image_values = extract_image_values(data)
        if not image_values:
            raise RuntimeError(f"API 未返回可解析图片数据: {str(data)[:1000]}")

        tensors = []
        refs = []
        errors = []
        for value in image_values:
            try:
                tensor, ref = image_value_to_tensor(value, timeout_seconds)
                tensors.append(tensor)
                refs.append(ref)
            except Exception as exc:
                errors.append(str(exc))

        if not tensors:
            raise RuntimeError(f"未能解析 gpt-image-2 响应图片: errors={errors}; response={str(data)[:1000]}")

        return torch.cat(tensors, dim=0), refs

    def generate(self, **kwargs):
        api_key = kwargs.get("api_key (API密钥)", "")
        api_base = kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL)
        prompt = kwargs.get("prompt (提示词)", "")
        mode = kwargs.get("mode (模式)", "AUTO")
        model = kwargs.get("model (模型)", "gpt-image-2")
        image_size = kwargs.get(
            "image_size (分辨率)",
            kwargs.get("size_ratio (尺寸/比例)", kwargs.get("size (尺寸)", "2K")),
        )
        aspect_ratio = kwargs.get("aspect_ratio (宽高比)", "16:9")
        custom_size = kwargs.get(
            "custom_size (仅custom填写: 宽x高)",
            kwargs.get(
                "custom_size (custom时: 宽x高, 例3072x1024)",
                kwargs.get("custom_size (自定义尺寸)", ""),
            ),
        )
        if (
            mode not in ("AUTO", "text2img", "img2img")
            and isinstance(model, str)
            and model.startswith("http")
        ):
            # Old workflows can shift widget values after converting prompt to an input.
            shifted_api_base = model
            shifted_image_size = api_base
            shifted_aspect_ratio = image_size
            shifted_custom_size = aspect_ratio
            shifted_quality = custom_size
            shifted_output_format = kwargs.get("quality (画质)", "png")
            shifted_output_compression = kwargs.get("output_format (输出格式)", 85)

            mode = "AUTO"
            model = "gpt-image-2"
            api_base = shifted_api_base
            image_size = shifted_image_size
            aspect_ratio = shifted_aspect_ratio
            custom_size = shifted_custom_size
            kwargs["quality (画质)"] = shifted_quality
            kwargs["output_format (输出格式)"] = shifted_output_format
            kwargs["output_compression (压缩率)"] = shifted_output_compression
            kwargs["timeout_seconds (超时秒数)"] = 360

        quality = safe_choice(kwargs.get("quality (画质)", "auto"), ["auto", "low", "medium", "high"], "auto")
        output_format = safe_choice(kwargs.get("output_format (输出格式)", "png"), ["png", "jpeg", "webp"], "png")
        output_compression = safe_int(kwargs.get("output_compression (压缩率)", 85), 85, 0, 100)
        stream = bool(kwargs.get("stream (流式兼容)", False))
        seed = safe_int(kwargs.get("seed (种子)", 0), 0, 0, 2147483647)
        timeout_seconds = safe_int(kwargs.get("timeout_seconds (超时秒数)", 360), 360, 60, 1800)
        retry_times = safe_int(kwargs.get("retry_times (重试次数)", 3), 3, 1, 10)
        api_mode = kwargs.get("api_mode (接口模式)", "responses_api")
        mainline_model = safe_choice(
            kwargs.get("mainline_model (Responses主模型)", MAINLINE_DEFAULT_MODEL),
            MAINLINE_MODEL_OPTIONS,
            MAINLINE_DEFAULT_MODEL,
        )
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空", 0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        api_base = normalize_api_base(api_base)
        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        image_payloads = self._collect_images(kwargs)
        mask_bytes = mask_to_png_bytes(kwargs.get("mask"))
        image_encoding_summary = self._image_payload_summary(image_payloads, mask_bytes)
        resolved_aspect_ratio = resolve_auto_aspect_ratio(image_size, aspect_ratio, kwargs.get("image_01"))
        effective_size = normalize_size(image_size, resolved_aspect_ratio, custom_size)

        if mode == "AUTO":
            actual_mode = "img2img" if image_payloads else "text2img"
        else:
            actual_mode = mode

        if actual_mode == "img2img" and not image_payloads:
            emit_runtime_status(unique_id, "error", "img2img 模式需要至少一张参考图", 0.0, 0, retry_times, timeout_seconds)
            raise ValueError("img2img 模式需要至少一张参考图")
        if mask_bytes is not None and not image_payloads:
            raise ValueError("mask 只能和 image_01 一起用于图片编辑")

        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        fields = self._payload_fields(
            model,
            clean_prompt,
            effective_size,
            quality,
            output_format,
            output_compression,
            stream,
        )

        print(f"[ComfyUI GPT Image] api_mode={api_mode}, mode={actual_mode}, image_size={image_size}, aspect_ratio={aspect_ratio}, resolved_aspect_ratio={resolved_aspect_ratio}, resolved_size={effective_size}, fields={fields}, mainline_model={mainline_model}, seed={seed} (not sent to API), input_encoding={image_encoding_summary}")
        emit_runtime_status(unique_id, "running", "开始生成", 0.0, 0, retry_times, timeout_seconds)

        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id,
                    "running",
                    f"{'图片编辑' if actual_mode == 'img2img' else '文生图'}请求中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt,
                    retry_times,
                    timeout_seconds,
                )

                if api_mode.startswith("responses_api"):
                    response = self._request_responses(
                        api_base,
                        headers,
                        mainline_model,
                        fields,
                        actual_mode,
                        image_payloads,
                        mask_bytes,
                        timeout_seconds,
                    )
                elif actual_mode == "img2img":
                    response = self._request_img2img(
                        api_base,
                        headers,
                        fields,
                        image_payloads,
                        mask_bytes,
                        timeout_seconds,
                    )
                else:
                    response = self._request_text2img(api_base, headers, fields, timeout_seconds)

                if not response.ok:
                    last_error = f"API 错误 {response.status_code}: {response.text}"
                    if is_retryable_http_status(response.status_code) and attempt < retry_times:
                        emit_runtime_status(
                            unique_id,
                            "running",
                            f"API 返回 {response.status_code}，重试中 ({attempt}/{retry_times})",
                            time.time() - start_ts,
                            attempt,
                            retry_times,
                            timeout_seconds,
                        )
                        time.sleep(min(2 ** (attempt - 1), 8))
                        continue
                    raise RuntimeError(last_error)

                data = parse_api_payload(response)
                raise_for_api_error(data)
                image_tensor, image_refs = self._parse_response_images(data, timeout_seconds)
                elapsed = time.time() - start_ts
                response_info = {
                    "status": "success",
                    "model": model,
                    "mainline_model": mainline_model if api_mode.startswith("responses_api") else None,
                    "api_mode": api_mode,
                    "mode": actual_mode,
                    "api_base": api_base,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "resolved_aspect_ratio": resolved_aspect_ratio,
                    "resolved_size": effective_size,
                    "request_fields": fields,
                    "input_images": len(image_payloads),
                    "input_encoding": image_encoding_summary,
                    "mask": mask_bytes is not None,
                    "output_images": int(image_tensor.shape[0]),
                    "image_refs": image_refs,
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                    "seed": seed,
                    "seed_note": "seed is a ComfyUI control only and is not sent to gpt-image-2",
                    "elapsed_seconds": round(elapsed, 2),
                }
                emit_runtime_status(
                    unique_id,
                    "success",
                    f"生成成功 (耗时 {elapsed:.1f}s)",
                    elapsed,
                    attempt,
                    retry_times,
                    timeout_seconds,
                )
                return (image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2))

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = str(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id,
                        "running",
                        f"网络或超时，重试中 ({attempt}/{retry_times})",
                        time.time() - start_ts,
                        attempt,
                        retry_times,
                        timeout_seconds,
                    )
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                break
            except Exception as exc:
                last_error = str(exc)
                emit_runtime_status(
                    unique_id,
                    "error",
                    last_error,
                    time.time() - start_ts,
                    attempt,
                    retry_times,
                    timeout_seconds,
                )
                raise

        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id,
            "error",
            f"连续 {retry_times} 次失败",
            elapsed,
            retry_times,
            retry_times,
            timeout_seconds,
        )
        raise RuntimeError(f"ComfyUI GPT Image 连续 {retry_times} 次失败，最后错误: {last_error}")


NODE_CLASS_MAPPINGS = {
    "ComfyuiLuckGPTImage2Node": ComfyuiLuckGPTImage2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiLuckGPTImage2Node": "Comfyui-Luck gpt-image-2",
}
