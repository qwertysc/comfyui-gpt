# comfyui-gpt

OpenAI-compatible GPT Image 2 custom nodes for ComfyUI.

This package is designed for users who want to call their own OpenAI-compatible endpoint and API key from ComfyUI. It does not use ComfyUI credits.

## Install

### Requirements

- ComfyUI is already installed and can start normally.
- Your endpoint supports OpenAI-compatible `POST /v1/responses` for GPT Image 2.
- You have an API key for that endpoint.

No separate dependency installation step is normally needed. In the tested ComfyUI Desktop environment, the required Python packages are already provided by ComfyUI's own Python environment.

### Install With Codex

If you give this repository to another user, they can ask Codex to install it with a prompt like this:

```text
请把这个仓库安装成 ComfyUI 自定义节点：
1. 找到我的 ComfyUI custom_nodes 目录。
2. 如果目录不存在就创建它。
3. 把当前仓库复制或克隆到 custom_nodes/comfyui-gpt。
4. 不要单独创建 Python 虚拟环境。
5. 不要单独安装 Python 依赖，除非 ComfyUI 启动时报缺少依赖。
6. 完成后告诉我需要重启 ComfyUI。
```

Typical ComfyUI Desktop path on macOS:

```text
~/Documents/ComfyUI/custom_nodes/comfyui-gpt
```

Typical manual install command:

```bash
cd ~/Documents/ComfyUI/custom_nodes
git clone https://github.com/qwertysc/comfyui-gpt.git comfyui-gpt
```

Then restart ComfyUI and search for:

```text
Comfyui-Luck
GPT-Image-2
图生图提示词控制器
文本停留编辑器
```

### Install With ComfyUI-Manager

If ComfyUI-Manager supports installing from a Git URL:

1. Open ComfyUI-Manager.
2. Choose install by Git URL or custom node URL.
3. Use:

```text
https://github.com/qwertysc/comfyui-gpt.git
```

4. Restart ComfyUI.

## Endpoint Configuration

Every API node exposes:

```text
api_base (接口域名)
api_key (API密钥)
```

Supported `api_base` examples:

```text
https://api.hjltwt.com:2345
https://api.hjltwt.com:2345/v1
```

Both forms are supported. The node automatically avoids duplicate `/v1`.

The node builds these paths automatically:

```text
/v1/responses
/v1/images/generations
/v1/images/edits
/v1/chat/completions
```

Do not include API keys in shared workflow files. ComfyUI can serialize node widget values into workflows and image metadata.

## Nodes

| Node | Purpose |
| --- | --- |
| `Comfyui-Luck gpt-image-2` | Text-to-image, image edit, multi-image reference, mask edit |
| `GPT-Image-2 文生图提示词控制器` | Optimize text requirements into an image generation prompt |
| `图生图提示词控制器` | Analyze up to 5 reference images and generate an image prompt |
| `文本停留编辑器` | Pause the workflow and manually edit generated prompt text |

### Comfyui-Luck gpt-image-2

This node uses `gpt-image-2` as the image model.

Default API mode:

```text
responses_api
```

In `responses_api` mode, the node calls:

```text
POST /v1/responses
```

The request uses a mainline model plus an image-generation tool:

```json
{
  "model": "gpt-5.4",
  "tools": [
    {
      "type": "image_generation",
      "model": "gpt-image-2"
    }
  ],
  "tool_choice": { "type": "image_generation" }
}
```

You can switch `api_mode (接口模式)`:

- `responses_api`: recommended default for GPT Image 2 on OpenAI-compatible endpoints.
- `images_api`: direct Images API compatibility mode.

Direct Images API compatibility paths:

- no input image: `POST /v1/images/generations`
- with one or more input images: `POST /v1/images/edits`

Inputs:

- `api_key (API密钥)`: your API key.
- `api_base (接口域名)`: your endpoint base URL.
- `prompt (提示词)`: image generation or image editing instruction.
- `mode (模式)`: `AUTO`, `text2img`, or `img2img`.
- `model (模型)`: `gpt-image-2`.
- `image_size (分辨率)`: `auto`, `1K`, `2K`, `4K`, or `custom`.
- `aspect_ratio (宽高比)`: target aspect ratio.
- `custom_size (仅custom填写: 宽x高)`: custom size such as `1600x1200`.
- `quality (画质)`: `auto`, `low`, `medium`, `high`.
- `output_format (输出格式)`: `png`, `jpeg`, `webp`.
- `output_compression (压缩率)`: used for compressed formats.
- `stream (流式兼容)`: sends `stream=true` and `partial_images=1` for compatible endpoints.
- `mainline_model (Responses主模型)`: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`.
- `image_01` to `image_05`: optional reference images.
- `mask`: optional mask for local edits.

Mode behavior:

- `AUTO`: text-to-image when no image is connected; image-to-image when one or more images are connected.
- `text2img`: always text-to-image.
- `img2img`: requires at least one input image.

Multi-image reference:

- Connect up to 5 images to `image_01` through `image_05`.
- Image order matters. You can write prompts such as: `把图1的人物放进图2的场景，参考图3的画风。`
- If one image is the primary subject, connect it to `image_01`.

Mask behavior:

- `mask` only works when at least `image_01` is connected.
- Transparent area means edit area.
- Opaque area means preserved area.

Size presets:

| aspect_ratio | 1K | 2K | 4K |
|---|---:|---:|---:|
| `AUTO` | not sent | not sent | not sent |
| `1:4` | `480x1440` | `672x2016` | `1280x3840` |
| `4:1` | `1440x480` | `2016x672` | `3840x1280` |
| `1:8` | `480x1440` | `672x2016` | `1280x3840` |
| `8:1` | `1440x480` | `2016x672` | `3840x1280` |
| `1:1` | `1024x1024` | `2048x2048` | `2880x2880` |
| `1:2` | `720x1440` | `1024x2048` | `1920x3840` |
| `2:1` | `1440x720` | `2048x1024` | `3840x1920` |
| `1:3` | `480x1440` | `672x2016` | `1280x3840` |
| `3:1` | `1440x480` | `2016x672` | `3840x1280` |
| `2:3` | `768x1152` | `1440x2160` | `2304x3456` |
| `3:2` | `1152x768` | `2160x1440` | `3456x2304` |
| `3:4` | `768x1024` | `1536x2048` | `2448x3264` |
| `4:3` | `1024x768` | `2048x1536` | `3264x2448` |
| `4:5` | `768x960` | `1536x1920` | `2304x2880` |
| `5:4` | `960x768` | `1920x1536` | `2880x2304` |
| `9:16` | `720x1280` | `1152x2048` | `2160x3840` |
| `16:9` | `1280x720` | `2048x1152` | `3840x2160` |
| `9:21` | `640x1488` | `960x2240` | `1648x3840` |
| `21:9` | `1344x576` | `2464x1056` | `3808x1632` |

`custom_size` is only used when `image_size` is `custom`.

`custom_size` constraints:

- Format: `widthxheight`, for example `1600x1200`.
- Max side <= `3840`.
- Width and height must be multiples of `16`.
- Long side / short side <= `3:1`.
- Total pixels must be between `655,360` and `8,294,400`.

Response parsing supports:

- `data[].b64_json`
- `data[].url`
- `partial_image_b64`
- `output[].result`
- `image_url.url`
- direct image responses
- SSE text containing `data: {...}` events

### GPT-Image-2 文生图提示词控制器

This node turns a plain text requirement into a more structured image prompt.

Prompt-control nodes call:

```text
POST /v1/chat/completions
```

Default model:

```text
gpt-5.4
```

Dropdown models:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
```

Typical usage:

```text
user_prompt
  -> GPT-Image-2 文生图提示词控制器
  -> optimized_prompt
  -> Comfyui-Luck gpt-image-2 prompt
```

Useful controls:

- `layout_type`: automatic, pure image, poster, ecommerce main image, social cover.
- `text_policy`: no text, preserve text, optimize text, auto-generate text.
- `aspect_ratio`: target layout ratio.
- `exact_text`: text that should appear in the final image.
- `optimize_strength`: standard or enhanced.

### 图生图提示词控制器

This node analyzes reference images and produces an optimized prompt.

Inputs:

- `reference_image_01`: required.
- `reference_image_02` to `reference_image_05`: optional.
- `subject_image`: optional primary subject image.
- `user_prompt`: your goal or instruction.
- `reference_mode`: automatic, full reference, style only, composition only, color/lighting only, layout only.
- `target_aspect_ratio`: target image ratio.

Recommended multi-image workflow:

```text
reference images
  ├─ connect to 图生图提示词控制器 reference_image_01~reference_image_05
  └─ also connect to Comfyui-Luck gpt-image-2 image_01~image_05

图生图提示词控制器 optimized_prompt
  └─ connect to Comfyui-Luck gpt-image-2 prompt
```

If one image must lock the main subject, also connect it to `subject_image`, and put the same image in `Comfyui-Luck gpt-image-2` as `image_01`.

### 文本停留编辑器

This node pauses a workflow so you can manually edit generated prompt text before image generation continues.

Recommended workflow:

```text
提示词控制器 optimized_prompt
  └─ 文本停留编辑器 text_list

文本停留编辑器 edited_text
  └─ Comfyui-Luck gpt-image-2 prompt
```

Outputs:

- `edited_text`: a single string, suitable for normal prompt inputs.
- `edited_texts`: a list output, useful for batch text workflows.

Run the workflow from the final `PreviewImage` or `SaveImage` node. When execution pauses at `文本停留编辑器`, edit the text in the node and click `Continue`. Do not click the main queue/run button again, or ComfyUI will start a new run and execute upstream prompt optimization again.

## Troubleshooting

### The node is missing after installation

Restart ComfyUI. If it is still missing, confirm the folder is:

```text
ComfyUI/custom_nodes/comfyui-gpt
```

The directory should contain files such as:

```text
__init__.py
gpt_2_0_node.py
luck_prompt_control_nodes.py
```

### `api_base` with `/v1` does not work

Both forms are supported:

```text
https://api.hjltwt.com:2345
https://api.hjltwt.com:2345/v1
```

The node automatically builds `/v1/responses` without duplicating `/v1`.

### Direct Images API returns no image

Use `api_mode=responses_api`. Some OpenAI-compatible endpoints expose GPT Image 2 through the Responses API and may return an upstream error from direct `/v1/images/generations` or `/v1/images/edits`.

### API key leaks in shared workflows

Clear `api_key (API密钥)` before sharing workflow JSON or generated images that may contain ComfyUI metadata.
