import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkflowJSON, WorkflowNode } from "../comfyui/types.js";
import { ValidationError } from "../utils/errors.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const WORKFLOWS_DIR = resolve(__dirname, "../../workflows");

// --- Helpers ---

export function getNextNodeId(workflow: WorkflowJSON): string {
  const ids = Object.keys(workflow).map(Number).filter((n) => !Number.isNaN(n));
  return String(ids.length === 0 ? 1 : Math.max(...ids) + 1);
}

function conn(nodeId: string, outputIndex: number): [string, number] {
  return [nodeId, outputIndex];
}

// --- Template parameter types ---

interface Txt2ImgParams {
  checkpoint?: string;
  positive_prompt?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  seed?: number;
  sampler_name?: string;
  scheduler?: string;
}

interface Img2ImgParams extends Txt2ImgParams {
  image_path?: string;
  denoise?: number;
}

interface UpscaleParams {
  upscale_model?: string;
  image_path?: string;
}

interface InpaintParams extends Img2ImgParams {
  mask_path?: string;
}

interface LtxvI2VParams {
  /** Motion/action prompt — describe what changes, not what's already visible */
  positive_prompt?: string;
  /** Negative prompt for video generation */
  negative_prompt?: string;
  /** Image filename in ComfyUI's input directory */
  image_filename?: string;
  /** Random seed for reproducibility */
  seed?: number;
  /** Path to a custom LTX I2V workflow JSON file (if not using built-in) */
  workflow_file?: string;
}

interface Ltx23I2VParams {
  /** Motion/action prompt — describe what changes in the video */
  positive_prompt?: string;
  /** Negative prompt for video generation */
  negative_prompt?: string;
  /** Image filename in ComfyUI's input directory */
  image_filename?: string;
  /** Final output width in pixels (auto-halved internally, use multiples of 64) */
  width?: number;
  /** Final output height in pixels (auto-halved internally, use multiples of 64) */
  height?: number;
  /** Video duration in seconds */
  duration?: number;
  /** Character LoRA filename, or "None" to skip */
  lora?: string;
  /** Random seed for reproducibility */
  seed?: number;
}

interface Ltx23T2VParams {
  /** Text prompt for video generation */
  positive_prompt?: string;
  /** Negative prompt for video generation */
  negative_prompt?: string;
  /** Final output width in pixels (auto-halved internally, use multiples of 64) */
  width?: number;
  /** Final output height in pixels (auto-halved internally, use multiples of 64) */
  height?: number;
  /** Video duration in seconds */
  duration?: number;
  /** Character LoRA filename from the loras folder */
  lora?: string;
  /** Random seed for reproducibility */
  seed?: number;
}

interface Ltx23Params {
  /** Text prompt for video generation */
  positive_prompt?: string;
  /** Negative prompt for video generation */
  negative_prompt?: string;
  /** Final output width in pixels (auto-divided by 4 internally, must be divisible by 128) */
  width?: number;
  /** Final output height in pixels (auto-divided by 4 internally, must be divisible by 128) */
  height?: number;
  /** Video duration in seconds */
  duration?: number;
  /** Character LoRA filename from the loras folder */
  lora?: string;
  /** Random seed for reproducibility */
  seed?: number;
}

interface Ltx23_3PassParams {
  /** Text/motion prompt for video generation */
  positive_prompt?: string;
  /** Negative prompt for video generation */
  negative_prompt?: string;
  /** true = Text-to-Video (no image), false = Image-to-Video */
  text_to_video?: boolean;
  /** Absolute file path to starting image (only used when text_to_video is false) */
  image_path?: string;
  /** Output width in pixels */
  width?: number;
  /** Output height in pixels */
  height?: number;
  /** Video duration in seconds */
  duration?: number;
  /** Character LoRA filename, or "None" to skip */
  lora?: string;
  /** Random seed for reproducibility */
  seed?: number;
  /** If true, skip Pass 3 (1.5x upscale) — output at ~720p instead of ~1080p */
  skip_pass3?: boolean;
}

type TemplateParams = Txt2ImgParams | Img2ImgParams | UpscaleParams | InpaintParams | LtxvI2VParams | Ltx23I2VParams | Ltx23T2VParams | Ltx23Params | Ltx23_3PassParams;

// --- Templates ---

function buildTxt2Img(p: Txt2ImgParams): WorkflowJSON {
  const ckpt = p.checkpoint ?? "sd_xl_base_1.0.safetensors";
  const positive = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "";
  const width = p.width ?? 1024;
  const height = p.height ?? 1024;
  const steps = p.steps ?? 20;
  const cfg = p.cfg ?? 8.0;
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const sampler = p.sampler_name ?? "euler";
  const scheduler = p.scheduler ?? "normal";

  return {
    "1": {
      class_type: "CheckpointLoaderSimple",
      inputs: { ckpt_name: ckpt },
    },
    "2": {
      class_type: "CLIPTextEncode",
      inputs: { text: positive, clip: conn("1", 1) },
      _meta: { title: "Positive Prompt" },
    },
    "3": {
      class_type: "CLIPTextEncode",
      inputs: { text: negative, clip: conn("1", 1) },
      _meta: { title: "Negative Prompt" },
    },
    "4": {
      class_type: "EmptyLatentImage",
      inputs: { width, height, batch_size: 1 },
    },
    "5": {
      class_type: "KSampler",
      inputs: {
        model: conn("1", 0),
        positive: conn("2", 0),
        negative: conn("3", 0),
        latent_image: conn("4", 0),
        seed,
        steps,
        cfg,
        sampler_name: sampler,
        scheduler,
        denoise: 1.0,
      },
    },
    "6": {
      class_type: "VAEDecode",
      inputs: { samples: conn("5", 0), vae: conn("1", 2) },
    },
    "7": {
      class_type: "SaveImage",
      inputs: { images: conn("6", 0), filename_prefix: "ComfyUI" },
    },
  };
}

function buildImg2Img(p: Img2ImgParams): WorkflowJSON {
  const ckpt = p.checkpoint ?? "sd_xl_base_1.0.safetensors";
  const positive = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "";
  const steps = p.steps ?? 20;
  const cfg = p.cfg ?? 8.0;
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const sampler = p.sampler_name ?? "euler";
  const scheduler = p.scheduler ?? "normal";
  const denoise = p.denoise ?? 0.75;
  const imagePath = p.image_path ?? "input.png";

  return {
    "1": {
      class_type: "CheckpointLoaderSimple",
      inputs: { ckpt_name: ckpt },
    },
    "2": {
      class_type: "LoadImage",
      inputs: { image: imagePath },
    },
    "3": {
      class_type: "VAEEncode",
      inputs: { pixels: conn("2", 0), vae: conn("1", 2) },
    },
    "4": {
      class_type: "CLIPTextEncode",
      inputs: { text: positive, clip: conn("1", 1) },
      _meta: { title: "Positive Prompt" },
    },
    "5": {
      class_type: "CLIPTextEncode",
      inputs: { text: negative, clip: conn("1", 1) },
      _meta: { title: "Negative Prompt" },
    },
    "6": {
      class_type: "KSampler",
      inputs: {
        model: conn("1", 0),
        positive: conn("4", 0),
        negative: conn("5", 0),
        latent_image: conn("3", 0),
        seed,
        steps,
        cfg,
        sampler_name: sampler,
        scheduler,
        denoise,
      },
    },
    "7": {
      class_type: "VAEDecode",
      inputs: { samples: conn("6", 0), vae: conn("1", 2) },
    },
    "8": {
      class_type: "SaveImage",
      inputs: { images: conn("7", 0), filename_prefix: "ComfyUI" },
    },
  };
}

function buildUpscale(p: UpscaleParams): WorkflowJSON {
  const model = p.upscale_model ?? "RealESRGAN_x4plus.pth";
  const imagePath = p.image_path ?? "input.png";

  return {
    "1": {
      class_type: "LoadImage",
      inputs: { image: imagePath },
    },
    "2": {
      class_type: "UpscaleModelLoader",
      inputs: { model_name: model },
    },
    "3": {
      class_type: "ImageUpscaleWithModel",
      inputs: { upscale_model: conn("2", 0), image: conn("1", 0) },
    },
    "4": {
      class_type: "SaveImage",
      inputs: { images: conn("3", 0), filename_prefix: "ComfyUI_upscale" },
    },
  };
}

function buildInpaint(p: InpaintParams): WorkflowJSON {
  const ckpt = p.checkpoint ?? "sd_xl_base_1.0.safetensors";
  const positive = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "";
  const steps = p.steps ?? 20;
  const cfg = p.cfg ?? 8.0;
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const sampler = p.sampler_name ?? "euler";
  const scheduler = p.scheduler ?? "normal";
  const denoise = p.denoise ?? 0.85;
  const imagePath = p.image_path ?? "input.png";
  const maskPath = p.mask_path ?? "mask.png";

  return {
    "1": {
      class_type: "CheckpointLoaderSimple",
      inputs: { ckpt_name: ckpt },
    },
    "2": {
      class_type: "LoadImage",
      inputs: { image: imagePath },
      _meta: { title: "Input Image" },
    },
    "3": {
      class_type: "LoadImage",
      inputs: { image: maskPath },
      _meta: { title: "Mask" },
    },
    "4": {
      class_type: "VAEEncode",
      inputs: { pixels: conn("2", 0), vae: conn("1", 2) },
    },
    "5": {
      class_type: "SetLatentNoiseMask",
      inputs: { samples: conn("4", 0), mask: conn("3", 1) },
    },
    "6": {
      class_type: "CLIPTextEncode",
      inputs: { text: positive, clip: conn("1", 1) },
      _meta: { title: "Positive Prompt" },
    },
    "7": {
      class_type: "CLIPTextEncode",
      inputs: { text: negative, clip: conn("1", 1) },
      _meta: { title: "Negative Prompt" },
    },
    "8": {
      class_type: "KSampler",
      inputs: {
        model: conn("1", 0),
        positive: conn("6", 0),
        negative: conn("7", 0),
        latent_image: conn("5", 0),
        seed,
        steps,
        cfg,
        sampler_name: sampler,
        scheduler,
        denoise,
      },
    },
    "9": {
      class_type: "VAEDecode",
      inputs: { samples: conn("8", 0), vae: conn("1", 2) },
    },
    "10": {
      class_type: "SaveImage",
      inputs: { images: conn("9", 0), filename_prefix: "ComfyUI_inpaint" },
    },
  };
}

function buildLtxvI2V(p: LtxvI2VParams): WorkflowJSON {
  const prompt = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "blurry, oversaturated, pixelated, low resolution, grainy, distorted, watermark, text";
  const imageFilename = p.image_filename ?? "input.png";
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);

  // LTX I2V workflows are complex (50+ nodes). Load from file if provided.
  if (p.workflow_file) {
    try {
      const fs = require("node:fs");
      const wf: WorkflowJSON = JSON.parse(fs.readFileSync(p.workflow_file, "utf-8"));

      // Known node IDs for common LTX I2V workflows — apply overrides
      const KNOWN_PROMPT_NODES = ["121", "6"]; // CLIPTextEncode (Positive)
      const KNOWN_NEG_NODES = ["110", "7"];     // CLIPTextEncode (Negative)
      const KNOWN_IMAGE_NODES = ["167"];         // LoadImage
      const KNOWN_SEED_NODES = ["114", "115", "449"]; // RandomNoise nodes

      for (const nodeId of KNOWN_PROMPT_NODES) {
        if (wf[nodeId]?.class_type === "CLIPTextEncode" && wf[nodeId]._meta?.title?.includes("Positive")) {
          wf[nodeId].inputs.text = prompt;
        }
      }
      for (const nodeId of KNOWN_NEG_NODES) {
        if (wf[nodeId]?.class_type === "CLIPTextEncode" && wf[nodeId]._meta?.title?.includes("Negative")) {
          wf[nodeId].inputs.text = negative;
        }
      }
      for (const nodeId of KNOWN_IMAGE_NODES) {
        if (wf[nodeId]?.class_type === "LoadImage") {
          wf[nodeId].inputs.image = imageFilename;
        }
      }
      for (const nodeId of KNOWN_SEED_NODES) {
        if (wf[nodeId]?.class_type === "RandomNoise") {
          wf[nodeId].inputs.noise_seed = seed;
        }
      }

      // Auto-remove the LTX2SamplingPreviewOverride node (known crash bug)
      if (wf["463"]?.class_type === "LTX2SamplingPreviewOverride") {
        // Rewire any nodes that reference 463 to point to 463's model input source
        const sourceModel = wf["463"].inputs.model;
        if (Array.isArray(sourceModel)) {
          for (const [nid, node] of Object.entries(wf)) {
            if (nid === "463") continue;
            for (const [key, val] of Object.entries(node.inputs)) {
              if (Array.isArray(val) && val.length === 2 && val[0] === "463") {
                node.inputs[key] = [sourceModel[0], val[1]];
              }
            }
          }
        }
        delete wf["463"];
      }

      return wf;
    } catch (err) {
      throw new ValidationError(
        `Failed to load LTX I2V workflow from "${p.workflow_file}": ${err instanceof Error ? err.message : err}`,
      );
    }
  }

  // Without a workflow file, return a minimal error — LTX workflows are too complex to build from scratch
  throw new ValidationError(
    'LTX I2V requires a workflow_file parameter pointing to an exported ComfyUI workflow JSON. ' +
    'Export your working LTX I2V workflow from ComfyUI (Save > API Format) and provide the path.',
  );
}

function loadWorkflowFile(filename: string): WorkflowJSON {
  const filepath = resolve(WORKFLOWS_DIR, filename);
  try {
    return JSON.parse(readFileSync(filepath, "utf-8"));
  } catch (err) {
    throw new ValidationError(
      `Failed to load workflow "${filename}" from ${WORKFLOWS_DIR}: ${err instanceof Error ? err.message : err}`,
    );
  }
}

// --- LoRA bypass utility ---

/**
 * Remove LoraLoader nodes where lora_name is "None" or empty, rewiring
 * downstream consumers to the node's upstream model/clip sources.
 */
function bypassNoneLoras(wf: WorkflowJSON): void {
  const toRemove: string[] = [];
  for (const [nodeId, node] of Object.entries(wf)) {
    if (node.class_type !== "LoraLoader") continue;
    const loraName = node.inputs.lora_name;
    if (loraName && loraName !== "None" && loraName !== "") continue;

    const modelSource = node.inputs.model;
    const clipSource = node.inputs.clip;

    for (const [otherId, otherNode] of Object.entries(wf)) {
      if (otherId === nodeId) continue;
      for (const [key, val] of Object.entries(otherNode.inputs)) {
        if (!Array.isArray(val) || val.length !== 2 || val[0] !== nodeId) continue;
        if (val[1] === 0 && modelSource) otherNode.inputs[key] = modelSource;
        else if (val[1] === 1 && clipSource) otherNode.inputs[key] = clipSource;
      }
    }
    toRemove.push(nodeId);
  }
  for (const id of toRemove) delete wf[id];
}

// --- Config-driven workflow builder for image workflows ---

interface ParamMapping {
  nodeId: string;
  field: string;
}

interface ImageWorkflowConfig {
  file: string;
  defaults: Record<string, unknown>;
  mappings: {
    prompt?: ParamMapping;
    negative_prompt?: ParamMapping;
    model?: ParamMapping;
    seed?: ParamMapping[];
    steps?: ParamMapping;
    cfg?: ParamMapping;
    width?: ParamMapping;
    height?: ParamMapping;
    sampler_name?: ParamMapping;
    scheduler?: ParamMapping;
    denoise?: ParamMapping;
  };
}

interface ImageWorkflowParams {
  positive_prompt?: string;
  negative_prompt?: string;
  model?: string;
  seed?: number;
  steps?: number;
  cfg?: number;
  width?: number;
  height?: number;
  sampler_name?: string;
  scheduler?: string;
  denoise?: number;
}

function buildFromConfig(config: ImageWorkflowConfig, params: ImageWorkflowParams): WorkflowJSON {
  const wf = loadWorkflowFile(config.file);

  const paramSetters: Array<{
    key: keyof ImageWorkflowParams;
    mapping: ParamMapping | ParamMapping[] | undefined;
  }> = [
    { key: "positive_prompt", mapping: config.mappings.prompt },
    { key: "negative_prompt", mapping: config.mappings.negative_prompt },
    { key: "model", mapping: config.mappings.model },
    { key: "steps", mapping: config.mappings.steps },
    { key: "cfg", mapping: config.mappings.cfg },
    { key: "width", mapping: config.mappings.width },
    { key: "height", mapping: config.mappings.height },
    { key: "sampler_name", mapping: config.mappings.sampler_name },
    { key: "scheduler", mapping: config.mappings.scheduler },
    { key: "denoise", mapping: config.mappings.denoise },
  ];

  for (const { key, mapping } of paramSetters) {
    if (!mapping) continue;
    const value = params[key] ?? config.defaults[key];
    if (value === undefined) continue;
    const targets = Array.isArray(mapping) ? mapping : [mapping];
    for (const t of targets) {
      wf[t.nodeId].inputs[t.field] = value;
    }
  }

  // Seed: use provided, or generate random
  if (config.mappings.seed) {
    const seed = params.seed ?? Math.floor(Math.random() * 2 ** 48);
    for (const t of config.mappings.seed) {
      wf[t.nodeId].inputs[t.field] = seed;
    }
  }

  return wf;
}

// --- Image workflow configs ---

const GENERATE_IMAGE_CONFIG: ImageWorkflowConfig = {
  file: "generate_image.json",
  defaults: {
    model: "Illustrious\\perfectionRealisticILXL_60.safetensors",
    steps: 30, cfg: 7.5, sampler_name: "dpmpp_2m", scheduler: "karras",
    denoise: 1.0, width: 1280, height: 720,
    negative_prompt: "blurry, low quality, low resolution, worst quality, normal quality, jpeg artifacts, watermark, text, signature, deformed, ugly, extra limbs, bad hands, bad anatomy",
  },
  mappings: {
    prompt:          { nodeId: "6", field: "text" },
    negative_prompt: { nodeId: "7", field: "text" },
    model:           { nodeId: "4", field: "ckpt_name" },
    seed:           [{ nodeId: "3", field: "seed" }],
    steps:           { nodeId: "3", field: "steps" },
    cfg:             { nodeId: "3", field: "cfg" },
    sampler_name:    { nodeId: "3", field: "sampler_name" },
    scheduler:       { nodeId: "3", field: "scheduler" },
    denoise:         { nodeId: "3", field: "denoise" },
    width:           { nodeId: "5", field: "width" },
    height:          { nodeId: "5", field: "height" },
  },
};

const QWEN_RAPID_IMAGE_CONFIG: ImageWorkflowConfig = {
  file: "qwen_rapid_image.json",
  defaults: {
    model: "AIO\\Qwen-Rapid-AIO-NSFW-v23.safetensors",
    steps: 6, cfg: 1.0, sampler_name: "euler_ancestral", scheduler: "beta",
    denoise: 1.0, width: 832, height: 1216,
    negative_prompt: "blurry, low quality, text, watermark, deformed, ugly, disfigured, extra fingers, bad anatomy, cartoon, anime, painting, oversaturated, plastic skin",
  },
  mappings: {
    prompt:          { nodeId: "6", field: "text" },
    negative_prompt: { nodeId: "7", field: "text" },
    model:           { nodeId: "4", field: "ckpt_name" },
    seed:           [{ nodeId: "3", field: "seed" }],
    steps:           { nodeId: "3", field: "steps" },
    cfg:             { nodeId: "3", field: "cfg" },
    sampler_name:    { nodeId: "3", field: "sampler_name" },
    scheduler:       { nodeId: "3", field: "scheduler" },
    denoise:         { nodeId: "3", field: "denoise" },
    width:           { nodeId: "5", field: "width" },
    height:          { nodeId: "5", field: "height" },
  },
};

const FLUX2_KLEIN_CONFIG: ImageWorkflowConfig = {
  file: "flux2_klein.json",
  defaults: {
    model: "Flux\\flux-2-klein-9b.safetensors",
    steps: 5, cfg: 1, width: 1920, height: 1080,
    negative_prompt: "blurry, low detail, bad teeth, asymmetrical eyes, uncanny valley, over-sharpened, noisy, watermark, text, shiny skin",
  },
  mappings: {
    prompt:          { nodeId: "108", field: "text" },
    negative_prompt: { nodeId: "103", field: "text" },
    model:           { nodeId: "110", field: "unet_name" },
    seed:           [{ nodeId: "106", field: "noise_seed" }],
    steps:           { nodeId: "112", field: "steps" },
    cfg:             { nodeId: "111", field: "cfg" },
    width:           { nodeId: "104", field: "value" },
    height:          { nodeId: "105", field: "value" },
  },
};

const FLUX2_KLEIN_BASE_LORA_CONFIG: ImageWorkflowConfig = {
  file: "flux2_klein_base_lora.json",
  defaults: {
    model: "Flux2\\flux2Klein_9bBase.safetensors",
    steps: 10, cfg: 1.2, width: 1920, height: 1080,
    negative_prompt: "blurry, low detail, bad teeth, asymmetrical eyes, uncanny valley, over-sharpened, noisy, watermark, text, shiny skin",
  },
  mappings: {
    prompt:          { nodeId: "108", field: "text" },
    negative_prompt: { nodeId: "103", field: "text" },
    model:           { nodeId: "110", field: "unet_name" },
    seed:           [{ nodeId: "106", field: "noise_seed" }],
    steps:           { nodeId: "112", field: "steps" },
    cfg:             { nodeId: "111", field: "cfg" },
    width:           { nodeId: "104", field: "value" },
    height:          { nodeId: "105", field: "value" },
  },
};

const QWEN_IMAGE_2512_CONFIG: ImageWorkflowConfig = {
  file: "qwen_image_2512.json",
  defaults: {
    model: "Qwen Remix\\jibMixQwen_v60.safetensors",
    steps: 6, cfg: 1, denoise: 1.0, width: 1920, height: 1088,
    negative_prompt: "((Neck collar)) Nipples, Platstic Barbie-doll skin, blurry, low resolution, distorted limbs, uncanny appearance, ugly, looks AI-generated, noise/grain, watermark, garbled text, mutated, deformed, low quality, malformed",
  },
  mappings: {
    prompt:          { nodeId: "830", field: "text" },
    negative_prompt: { nodeId: "829", field: "text" },
    model:           { nodeId: "807", field: "unet_name" },
    seed:           [{ nodeId: "832", field: "seed" }],
    steps:           { nodeId: "832", field: "steps" },
    cfg:             { nodeId: "832", field: "cfg" },
    denoise:         { nodeId: "832", field: "denoise" },
    width:           { nodeId: "825", field: "width" },
    height:          { nodeId: "825", field: "height" },
  },
};

const QWEN_IMAGE_2512_JIB_CONFIG: ImageWorkflowConfig = {
  file: "qwen_image_2512_jib.json",
  defaults: {
    ...QWEN_IMAGE_2512_CONFIG.defaults,
  },
  mappings: { ...QWEN_IMAGE_2512_CONFIG.mappings },
};

const Z_IMAGE_TURBO_CONFIG: ImageWorkflowConfig = {
  file: "z_image_turbo.json",
  defaults: {
    model: "Z-Image\\spazerZTurbo_spazerZNitro.safetensors",
    steps: 9, cfg: 1, denoise: 1.0,
    sampler_name: "exponential/res_2s", scheduler: "bong_tangent",
    width: 1920, height: 1088,
  },
  mappings: {
    prompt:          { nodeId: "67", field: "text" },
    // No negative_prompt — uses ConditioningZeroOut instead
    model:           { nodeId: "66", field: "unet_name" },
    seed:           [{ nodeId: "71", field: "seed" }],
    steps:           { nodeId: "71", field: "steps" },
    cfg:             { nodeId: "71", field: "cfg" },
    sampler_name:    { nodeId: "71", field: "sampler_name" },
    scheduler:       { nodeId: "71", field: "scheduler" },
    denoise:         { nodeId: "71", field: "denoise" },
    width:           { nodeId: "68", field: "width" },
    height:          { nodeId: "68", field: "height" },
  },
};

function buildLtx23I2V(p: Ltx23I2VParams): WorkflowJSON {
  const wf = loadWorkflowFile("ltx23_i2v.json");
  const prompt = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, compression artifacts, jpeg artifacts, glitches, watermark, text, logo, signature, copyright, subtitles, distorted sound, saturated sound, loud";
  const imageFilename = p.image_filename ?? "input.png";
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const width = p.width ?? 1280;
  const height = p.height ?? 736;
  const duration = p.duration ?? 10;
  const lora = p.lora ?? "None";

  wf["121"].inputs.text = prompt;
  wf["110"].inputs.text = negative;
  wf["167"].inputs.image = imageFilename;
  for (const id of ["114", "115"]) {
    wf[id].inputs.noise_seed = seed;
  }
  wf["292"].inputs.value = width;
  wf["293"].inputs.value = height;
  wf["291"].inputs.value = duration;
  wf["364"].inputs.lora_name = lora;

  bypassNoneLoras(wf);
  return wf;
}

function buildLtx23T2V(p: Ltx23T2VParams): WorkflowJSON {
  const wf = loadWorkflowFile("ltx23_t2v.json");
  const prompt = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, compression artifacts, jpeg artifacts, glitches, watermark, text, logo, signature, copyright, subtitles, distorted sound, saturated sound, loud";
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const width = p.width ?? 1280;
  const height = p.height ?? 736;
  const duration = p.duration ?? 10;
  const lora = p.lora ?? "None";

  wf["121"].inputs.text = prompt;
  wf["110"].inputs.text = negative;
  for (const id of ["114", "115"]) {
    wf[id].inputs.noise_seed = seed;
  }
  wf["292"].inputs.value = width;
  wf["293"].inputs.value = height;
  wf["291"].inputs.value = duration;
  wf["364"].inputs.lora_name = lora;

  bypassNoneLoras(wf);
  return wf;
}

function buildLtx23(p: Ltx23Params): WorkflowJSON {
  const wf = loadWorkflowFile("ltx23.json");
  const prompt = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "";
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const width = p.width ?? 1024;
  const height = p.height ?? 1408;
  const duration = p.duration ?? 5;
  const lora = p.lora ?? "ilya_ltx_step_01000.safetensors";

  // Node 121: positive prompt
  wf["121"].inputs.text = prompt;
  // Node 110: negative prompt
  wf["110"].inputs.text = negative;
  // Nodes 114, 115, 189: seeds
  for (const id of ["114", "115", "189"]) {
    wf[id].inputs.noise_seed = seed;
  }
  // Node 300: width, 301: height, 112: duration
  wf["300"].inputs.value = width;
  wf["301"].inputs.value = height;
  wf["112"].inputs.value = duration;
  // Node 207: character LoRA
  wf["207"].inputs.lora_name = lora;

  return wf;
}

function buildLtx23_3Pass(p: Ltx23_3PassParams): WorkflowJSON {
  const wf = loadWorkflowFile("ltx23_3pass.json");
  const prompt = p.positive_prompt ?? "";
  const negative = p.negative_prompt ?? "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, compression artifacts, jpeg artifacts, glitches, watermark, text, logo, signature, copyright, subtitles, distorted sound, saturated sound, loud";
  const t2v = p.text_to_video ?? false;
  const imagePath = p.image_path ?? "";
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 48);
  const width = p.width ?? 1920;
  const height = p.height ?? 1088;
  const duration = p.duration ?? 10;
  const lora = p.lora ?? "None";
  const skipPass3 = p.skip_pass3 ?? false;

  // Prompt
  wf["98"].inputs.text = prompt;
  // Negative prompt
  wf["40"].inputs.text = negative;
  // T2V switch: true = text-to-video (no image), false = image-to-video
  wf["30"].inputs.value = t2v;
  // Image path (for I2V mode)
  wf["2009"].inputs.image = imagePath;
  // Seeds — all 3 passes
  for (const id of ["88", "89", "1985"]) {
    if (wf[id]) wf[id].inputs.noise_seed = seed;
  }
  // Resolution
  wf["96"].inputs.value = width;
  wf["95"].inputs.value = height;
  // Duration
  wf["77"].inputs.value = duration;
  // Character LoRA (node 122 — Power Lora Loader)
  // Note: node 2007 is the abliterated text encoder LoRA (stays as-is)
  // Note: node 22 is the distill LoRA (stays as-is)
  if (wf["122"]?.inputs?.lora_1) {
    const loraSlot = wf["122"].inputs.lora_1 as Record<string, unknown>;
    if (lora === "None" || lora === "") {
      loraSlot.on = false;
      loraSlot.lora = "None";
    } else {
      loraSlot.on = true;
      loraSlot.lora = lora;
    }
  }

  // Skip Pass 3: mute the 3rd pass nodes to output at ~720p
  // Pass 3 nodes: 1978 (sampler select), 1979 (sigmas), 1980 (upscale 1.5x),
  // 1984 (concat), 1985 (seed), 1986 (separate), 1990 (guider), 1991 (sampler), 1992 (i2v condition)
  if (skipPass3) {
    const pass3Nodes = ["1978", "1979", "1980", "1984", "1985", "1986", "1990", "1991", "1992", "2010"];
    for (const nodeId of pass3Nodes) {
      if (wf[nodeId]) {
        // Remove pass 3 nodes entirely — cleaner than muting
        delete wf[nodeId];
      }
    }
    // Also remove the 1.5x upscale model loader
    if (wf["2010"]) delete wf["2010"];
    // Rewire the final output to read from Pass 2's output instead of Pass 3
    // Pass 2 output: node 54 (LTXVSeparateAVLatent from pass 2 sampler 93)
    // Final decode: node 99 reads from node 1991 (pass 3) — rewire to 93 (pass 2)
    if (wf["99"]) wf["99"].inputs.av_latent = ["93", 0];
  }

  // Auto-remove LTX2SamplingPreviewOverride nodes
  for (const nodeId of Object.keys(wf)) {
    if (wf[nodeId]?.class_type === "LTX2SamplingPreviewOverride") {
      const source = wf[nodeId].inputs.model;
      if (Array.isArray(source)) {
        for (const [nid, node] of Object.entries(wf)) {
          if (nid === nodeId) continue;
          for (const [key, val] of Object.entries(node.inputs)) {
            if (Array.isArray(val) && val.length === 2 && val[0] === nodeId) {
              node.inputs[key] = [source[0] as string, val[1]];
            }
          }
        }
      }
      delete wf[nodeId];
    }
  }

  bypassNoneLoras(wf);
  return wf;
}

const TEMPLATES: Record<string, (params: Record<string, unknown>) => WorkflowJSON> = {
  txt2img: (p) => buildTxt2Img(p as Txt2ImgParams),
  img2img: (p) => buildImg2Img(p as Img2ImgParams),
  upscale: (p) => buildUpscale(p as UpscaleParams),
  inpaint: (p) => buildInpaint(p as InpaintParams),
  ltxv_i2v: (p) => buildLtxvI2V(p as LtxvI2VParams),
  ltx23_i2v: (p) => buildLtx23I2V(p as Ltx23I2VParams),
  ltx23_t2v: (p) => buildLtx23T2V(p as Ltx23T2VParams),
  ltx23: (p) => buildLtx23(p as Ltx23Params),
  ltx23_3pass: (p) => buildLtx23_3Pass(p as Ltx23_3PassParams),
  generate_image: (p) => buildFromConfig(GENERATE_IMAGE_CONFIG, p as ImageWorkflowParams),
  qwen_rapid_image: (p) => buildFromConfig(QWEN_RAPID_IMAGE_CONFIG, p as ImageWorkflowParams),
  flux2_klein: (p) => buildFromConfig(FLUX2_KLEIN_CONFIG, p as ImageWorkflowParams),
  flux2_klein_base_lora: (p) => buildFromConfig(FLUX2_KLEIN_BASE_LORA_CONFIG, p as ImageWorkflowParams),
  qwen_image_2512: (p) => buildFromConfig(QWEN_IMAGE_2512_CONFIG, p as ImageWorkflowParams),
  qwen_image_2512_jib: (p) => buildFromConfig(QWEN_IMAGE_2512_JIB_CONFIG, p as ImageWorkflowParams),
  z_image_turbo: (p) => buildFromConfig(Z_IMAGE_TURBO_CONFIG, p as ImageWorkflowParams),
};

export const TEMPLATE_NAMES = Object.keys(TEMPLATES);

export function createWorkflow(
  template: string,
  params: Record<string, unknown> = {},
): WorkflowJSON {
  const builder = TEMPLATES[template];
  if (!builder) {
    throw new ValidationError(
      `Unknown template "${template}". Available: ${TEMPLATE_NAMES.join(", ")}`,
    );
  }
  return builder(params);
}

// --- Modification operations ---

interface SetInputOp {
  op: "set_input";
  node_id: string;
  input_name: string;
  value: unknown;
}

interface AddNodeOp {
  op: "add_node";
  class_type: string;
  inputs?: Record<string, unknown>;
  id?: string;
}

interface RemoveNodeOp {
  op: "remove_node";
  node_id: string;
}

interface ConnectOp {
  op: "connect";
  source_id: string;
  output_index: number;
  target_id: string;
  input_name: string;
}

interface InsertBetweenOp {
  op: "insert_between";
  source_id: string;
  output_index: number;
  target_id: string;
  input_name: string;
  new_class_type: string;
  new_inputs?: Record<string, unknown>;
}

export type ModifyOperation =
  | SetInputOp
  | AddNodeOp
  | RemoveNodeOp
  | ConnectOp
  | InsertBetweenOp;

function applySetInput(wf: WorkflowJSON, op: SetInputOp): void {
  const node = wf[op.node_id];
  if (!node) throw new ValidationError(`Node "${op.node_id}" not found`);
  node.inputs[op.input_name] = op.value;
}

function applyAddNode(wf: WorkflowJSON, op: AddNodeOp): string {
  const id = op.id ?? getNextNodeId(wf);
  if (wf[id]) throw new ValidationError(`Node ID "${id}" already exists`);
  wf[id] = {
    class_type: op.class_type,
    inputs: op.inputs ?? {},
  };
  return id;
}

function applyRemoveNode(wf: WorkflowJSON, op: RemoveNodeOp): void {
  if (!wf[op.node_id]) throw new ValidationError(`Node "${op.node_id}" not found`);
  delete wf[op.node_id];

  // Clean up any connections pointing to the removed node
  for (const node of Object.values(wf)) {
    for (const [key, val] of Object.entries(node.inputs)) {
      if (
        Array.isArray(val) &&
        val.length === 2 &&
        typeof val[0] === "string" &&
        val[0] === op.node_id
      ) {
        delete node.inputs[key];
      }
    }
  }
}

function applyConnect(wf: WorkflowJSON, op: ConnectOp): void {
  if (!wf[op.source_id]) throw new ValidationError(`Source node "${op.source_id}" not found`);
  if (!wf[op.target_id]) throw new ValidationError(`Target node "${op.target_id}" not found`);
  wf[op.target_id].inputs[op.input_name] = [op.source_id, op.output_index];
}

function applyInsertBetween(wf: WorkflowJSON, op: InsertBetweenOp): string {
  if (!wf[op.source_id]) throw new ValidationError(`Source node "${op.source_id}" not found`);
  if (!wf[op.target_id]) throw new ValidationError(`Target node "${op.target_id}" not found`);

  const newId = getNextNodeId(wf);
  const newInputs: Record<string, unknown> = { ...(op.new_inputs ?? {}) };

  // Connect the new node's first input to the original source
  // Find the first input name that isn't already set -- use a convention-based approach
  // The new node receives the source output on its primary input
  // We'll figure out the right input name by looking for common patterns
  const primaryInputNames = ["model", "clip", "samples", "latent_image", "image", "conditioning", "pixels"];
  let connected = false;
  for (const name of primaryInputNames) {
    if (!(name in newInputs)) {
      newInputs[name] = [op.source_id, op.output_index];
      connected = true;
      break;
    }
  }
  if (!connected) {
    // Fallback: add as first unused slot
    newInputs["input"] = [op.source_id, op.output_index];
  }

  wf[newId] = {
    class_type: op.new_class_type,
    inputs: newInputs,
  };

  // Rewire: target's input now points to the new node's output 0
  wf[op.target_id].inputs[op.input_name] = [newId, 0];

  return newId;
}

export function modifyWorkflow(
  workflow: WorkflowJSON,
  operations: ModifyOperation[],
): { workflow: WorkflowJSON; added_ids: string[] } {
  // Deep clone to avoid mutating the original
  const wf: WorkflowJSON = JSON.parse(JSON.stringify(workflow));
  const addedIds: string[] = [];

  for (const op of operations) {
    switch (op.op) {
      case "set_input":
        applySetInput(wf, op);
        break;
      case "add_node": {
        const id = applyAddNode(wf, op);
        addedIds.push(id);
        break;
      }
      case "remove_node":
        applyRemoveNode(wf, op);
        break;
      case "connect":
        applyConnect(wf, op);
        break;
      case "insert_between": {
        const id = applyInsertBetween(wf, op);
        addedIds.push(id);
        break;
      }
      default:
        throw new ValidationError(`Unknown operation: ${(op as { op: string }).op}`);
    }
  }

  return { workflow: wf, added_ids: addedIds };
}
