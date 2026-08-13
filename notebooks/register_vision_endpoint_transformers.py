# Databricks notebook source
# MAGIC %md
# MAGIC # Register Qwen3-VL as a Databricks Model Serving Endpoint
# MAGIC
# MAGIC One-time (or occasional, if the model changes) infrastructure setup —
# MAGIC **not** part of the bundle-managed ingestion pipeline. Deploys
# MAGIC `Qwen/Qwen3-VL-4B-Instruct` (the Hugging Face Transformers release, *not*
# MAGIC the Ollama GGUF build used for local testing) as a custom, GPU-backed
# MAGIC Model Serving endpoint, so the RAG ingestion pipeline can call it the same
# MAGIC way it called every model in the OCR spike test — an OpenAI-style chat
# MAGIC completions API accepting image input.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - **Serverless GPU compute (A10) is required, not optional** — enforced
# MAGIC   by an explicit `DATABRICKS_ACCELERATOR` check before logging/
# MAGIC   registering. A classic cluster will not work here: the model gets
# MAGIC   packaged with CPU dependencies and the GPU serving endpoint fails to
# MAGIC   start. (This differs from the vLLM notebook's own local-testing step,
# MAGIC   where either compute type is fine — this requirement is specifically
# MAGIC   about the log/register step in *this* notebook.)
# MAGIC - `mlflow>=3.12.0` and `databricks-sdk>=0.102.0` (installed below).
# MAGIC
# MAGIC **Why plain `transformers`, not vLLM.** An earlier version of this
# MAGIC notebook used vLLM (Databricks' documented pattern for custom
# MAGIC vision-language model serving, and it gives an OpenAI-compatible API for
# MAGIC free). In practice it hit a chain of environment-specific issues — a FIPS
# MAGIC crash on classic clusters, a driver-too-old error from an unpinned
# MAGIC dependency pulling a too-new CUDA release, then missing CUDA development
# MAGIC headers needed to JIT-compile one of vLLM's optional accelerated
# MAGIC sampling kernels. That last class of problem kept recurring: this
# MAGIC environment ships pip-installable CUDA *runtime* libraries but not a full
# MAGIC CUDA *development* toolkit, which vLLM's optional kernels assume. vLLM's
# MAGIC actual value — continuous batching, PagedAttention, high-throughput
# MAGIC concurrent serving — matters for busy interactive chat traffic; this is a
# MAGIC low-concurrency batch OCR job (triggered weekly or manually, processing a
# MAGIC few dozen pages at a time), so none of that throughput optimization was
# MAGIC being used anyway. Plain `transformers.generate()` needs no JIT-compiled
# MAGIC kernels at all. (Separately, a real deploy attempt of the vLLM notebook
# MAGIC also hit a hard workspace-level restriction — `entrypoint`-based custom
# MAGIC serving isn't enabled for this workspace at all — making this the only
# MAGIC viable path regardless of the performance tradeoff.)
# MAGIC
# MAGIC **On step 3's request/response handling.** Uses a plain `PythonModel`
# MAGIC with a hand-built signature (`AnyType()` for the `messages` field), not
# MAGIC MLflow's `ChatModel` — see step 3's markdown for why: `ChatModel` turned
# MAGIC out to unconditionally enforce a fixed, text-only chat schema regardless
# MAGIC of `input_example`, a real MLflow 3.15.1 limitation confirmed by reading
# MAGIC its source, not something fixable from this notebook while still
# MAGIC subclassing it.
# MAGIC

# COMMAND ----------

# Exact pins, not minimums -- >=X.Y.Z resolves to whatever the latest
# compatible release is at install time, which can silently drift to a
# newer, differently-behaved version later (this bit us hard earlier in
# this notebook's history: an unpinned vllm>=0.11.0 resolved to a release
# many versions newer that needed a CUDA version this environment didn't
# have). Versions below are the ones actually confirmed working in this
# environment during development -- accelerate and torchvision are the
# exceptions, newly added for this transformers-based rewrite and not
# yet confirmed (torchvision: needed by Qwen3-VL's AutoProcessor, missed
# in the first pass); pin both exactly once you see what versions this
# cell resolves them to.
%pip install "mlflow==3.15.1" "databricks-sdk==0.125.0" "huggingface_hub==0.34.4" "transformers==4.57.1" accelerate torchvision "hf_transfer==0.1.9" "pillow==12.3.0" "pdfplumber==0.11.10"
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "eliao")
dbutils.widgets.text("schema", "wnv_demo")
dbutils.widgets.text("model_name", "qwen3_vl_ocr")
dbutils.widgets.text("hf_model_id", "Qwen/Qwen3-VL-4B-Instruct")
dbutils.widgets.text("endpoint_name", "qwen3-vl-ocr")
# This workspace currently supports GPU_SMALL for this custom endpoint.
# Keep the endpoint on the 16GB T4 and compensate by hard-capping the
# document image before the Qwen vision encoder.
dbutils.widgets.dropdown("workload_type", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"])
# Only needed for the end-to-end validation cell at the bottom.
dbutils.widgets.text(
    "test_pdf_volume_path",
    "/Volumes/eliao/wnv_demo/documents_data/WNV-Outbreak-Communications-Toolkit-2025_508c.pdf",
)
dbutils.widgets.text("test_page", "5")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
model_name = dbutils.widgets.get("model_name").strip()
hf_model_id = dbutils.widgets.get("hf_model_id").strip()
endpoint_name = dbutils.widgets.get("endpoint_name").strip()
workload_type_str = dbutils.widgets.get("workload_type").strip()

uc_model_name = f"{catalog}.{schema}.{model_name}"
print(f"UC model:       {uc_model_name}")
print(f"Endpoint:       {endpoint_name}")
print(f"Source model:   {hf_model_id}")
print(f"GPU workload:   {workload_type_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download model weights
# MAGIC
# MAGIC Downloads the actual Hugging Face Transformers release (safetensors) to
# MAGIC local (non-`/Workspace`) storage — large model weights don't belong on the
# MAGIC workspace filesystem, and this is a genuinely different artifact format
# MAGIC from the Ollama GGUF build already on your machine; that one is for
# MAGIC llama.cpp-style local inference, not compatible with the serving path here.
# MAGIC

# COMMAND ----------

import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

local_model_dir = Path(tempfile.mkdtemp()) / "model"
snapshot_download(repo_id=hf_model_id, local_dir=str(local_model_dir))
print(f"Downloaded {hf_model_id} to {local_model_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Smoke-test locally with `transformers`
# MAGIC
# MAGIC Loads the model directly in-process and runs one generation — no
# MAGIC subprocess, no server, no ports. Confirms the model loads and generates
# MAGIC before wiring it into the pyfunc wrapper below.
# MAGIC

# COMMAND ----------

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# max_pixels caps vision-token count independent of the source image's
# native resolution -- confirmed against the pinned transformers==4.57.1
# source (Qwen2VLImageProcessorFast, which this model's
# preprocessor_config.json also uses) and the model's own shipped config
# (default longest_edge=16777216, ~16.7 megapixels -- far above what a
# 16GB GPU can hold in attention memory for a full 300 DPI page). Without
# this, a real run hit `torch.OutOfMemoryError: Tried to allocate 64.75
# GiB` on a T4 processing an ~8.4 megapixel page image. 602,112
# (28*28*768, ~0.6 megapixel) is the standard balanced default used across
# the Qwen-VL family -- lower than what the OCR spike test validated
# (that test used hosted endpoints and Ollama, which apply their own,
# different internal caps), so treat this as a starting point to get the
# pipeline working, not a re-validation of OCR accuracy at this exact
# resolution.
#
# IMPORTANT: setting this at AutoProcessor.from_pretrained() construction
# time is NOT enough -- confirmed by a real run that still OOM'd with the
# identical 64.75 GiB allocation, meaning it had zero effect. Traced
# through the pinned transformers==4.57.1 source
# (ProcessorMixin.apply_chat_template -> self(text=..., images=...,
# **kwargs) -> Qwen3VLProcessor.__call__'s Qwen3VLImagesKwargs): max_pixels
# must be passed as a kwarg to apply_chat_template() itself, which is what
# actually forwards it into the image processing call.
MAX_PIXELS = 602_112

model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(local_model_dir), dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(str(local_model_dir))

# Plain text smoke test -- the full image-input path gets exercised by
# the end-to-end validation cell in step 5, against the real endpoint.
# Content must be a list of typed parts, even for text-only messages --
# Qwen3-VL's chat template doesn't accept a bare string here (confirmed
# by a real run: TypeError from iterating over the string's characters).
messages = [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    max_pixels=MAX_PIXELS,
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=10)
trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
print(output_text)

# Free GPU memory before the model gets loaded again inside the logged
# pyfunc model's load_context() below.
del model, processor
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log to MLflow
# MAGIC
# MAGIC Plain `PythonModel`, **not** `ChatModel` — despite `ChatModel` being
# MAGIC Databricks' own documented pattern (see the vLLM notebook), and despite
# MAGIC its `ChatMessage.content` type explicitly declaring `str | list[dict] |
# MAGIC None`. Confirmed by reading MLflow 3.15.1's own source
# MAGIC (`mlflow/types/llm.py`): `ChatModel` unconditionally logs a fixed
# MAGIC `CHAT_MODEL_INPUT_SCHEMA` that types `content` as `DataType.string`
# MAGIC *only* — a real inconsistency between what the Python type promises and
# MAGIC what the schema actually enforces at serving time. This schema is
# MAGIC **independent of `input_example`**: a real run with a multimodal
# MAGIC `input_example` still got rejected with `Failed to enforce schema for
# MAGIC key 'content'. Expected type string, received type list` — the example
# MAGIC was silently ignored for that field.
# MAGIC
# MAGIC An earlier version of this notebook used `PythonModel` too, but with
# MAGIC `infer_signature()` on a raw dict example — that built a rigid inferred
# MAGIC `Array`/`Object` schema that also corrupted the nested `content` list
# MAGIC during enforcement (`'int' object is not subscriptable`). This version
# MAGIC instead hand-builds the signature using `AnyType()` for `messages`,
# MAGIC which tells MLflow not to type-check/coerce that field at all. Confirmed
# MAGIC against MLflow's scoring-server source that a raw `{"messages": [...],
# MAGIC ...}` POST (no `dataframe_*`/`instances` wrapper — what the validation
# MAGIC cell below already sends) is detected as "unified LLM input" and handed
# MAGIC to `predict()` as a plain, unmodified dict.
# MAGIC

# COMMAND ----------

import base64
import io
import json
import os
import traceback

import mlflow
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import AnyType, ColSpec, DataType, ParamSchema, ParamSpec, Schema
from PIL import Image

# Must log+register from serverless GPU compute -- otherwise the model
# gets packaged with CPU dependencies and the GPU serving endpoint fails
# to start. Per Databricks' own reference notebook for this serving path.
if not os.environ.get("DATABRICKS_ACCELERATOR"):
    raise RuntimeError(
        "This must be logged+registered from serverless GPU compute, or "
        "the endpoint will be packaged with the wrong dependencies."
    )

mlflow.set_registry_uri("databricks-uc")

# Same cap as the smoke-test cell above -- see that cell's comment for
# why (must be passed at apply_chat_template() call time, not at
# AutoProcessor.from_pretrained() construction time -- confirmed by a
# real run that the latter has zero effect).
MAX_PIXELS = 602_112


class Qwen3VLChatModel(PythonModel):
    """Custom OpenAI-chat-shaped model wrapping Qwen3-VL. Plain
    PythonModel, not ChatModel -- see step 3's markdown for why.
    """

    def load_context(self, context):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = context.artifacts["model_dir"]
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        # transformers 4.57.x can leave the image processor's internal
        # longest_edge at its large default even when max_pixels is passed
        # later. Force the actual processor resize ceiling as well.
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
            self.processor.image_processor.size["longest_edge"] = MAX_PIXELS

    def predict(self, context, model_input, params=None):
        # Kept self-diagnosing (see the earlier commit message): any
        # failure returns as a normal 200 response with debug info and
        # the full traceback, instead of an opaque platform error.
        debug = {}
        try:
            if isinstance(model_input, dict):
                messages = model_input["messages"]
            else:
                messages = model_input.iloc[0]["messages"]
            debug["messages_raw_type"] = type(messages).__name__
            debug["messages_raw_repr"] = repr(messages)[:300]

            if isinstance(messages, str):
                messages = json.loads(messages)
            if isinstance(messages, dict):
                # Confirmed by a real run: for a single-message request,
                # `messages` arrives as that one message dict directly
                # ({'role': 'user', 'content': [...]}), not as a
                # one-element list containing it -- something in
                # Databricks' request handling collapses a length-1 list
                # down to its sole element. Earlier symptoms (string
                # indices must be integers; Expecting value at char 0)
                # were both artifacts of iterating this dict's keys
                # ("role", "content") one at a time, not of any actual
                # JSON-stringification -- that theory was wrong.
                messages = [messages]
            debug["messages_parsed_type"] = type(messages).__name__
            if messages:
                debug["messages_elem0_type"] = type(messages[0]).__name__

            max_new_tokens = int(params.get("max_tokens", 512)) if params else 512
            temperature = float(params.get("temperature", 0.0)) if params else 0.0

            chat_messages = self._to_chat_format(messages)

            inputs = self.processor.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_pixels=MAX_PIXELS,
            )

            # Record what the processor actually produced. These values are
            # returned in the existing debug payload if generation OOMs, so
            # we can verify that the pixel cap is genuinely taking effect.
            if "pixel_values" in inputs:
                debug["pixel_values_shape"] = tuple(inputs["pixel_values"].shape)
            if "image_grid_thw" in inputs:
                debug["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            if hasattr(self.processor, "image_processor"):
                debug["processor_size"] = dict(getattr(self.processor.image_processor, "size", {}) or {})

            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return {
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": output_text}}
                ],
            }
        except Exception as e:
            error_report = (
                f"PREDICT_ERROR: {type(e).__name__}: {e}\n"
                f"debug: {debug}\n"
                f"{traceback.format_exc()}"
            )
            return {
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": error_report}}
                ],
            }

    @staticmethod
    def _to_chat_format(messages: list[dict]) -> list[dict]:
        """OpenAI-style messages (content as a list of text/image_url parts)
        -> Qwen3-VL's chat-template content format
        ({"type": "image", "image": <PIL.Image>} / {"type": "text", ...}).
        """
        chat_messages = []
        for msg in messages:
            if isinstance(msg, str):
                msg = json.loads(msg)

            content = msg["content"]
            if isinstance(content, str):
                # Qwen3-VL's chat template requires a list of typed parts,
                # not a bare string, even for plain text -- confirmed by a
                # real run hitting this exact gap in the smoke test above.
                content = [{"type": "text", "text": content}]

            parts = []
            for part in content:
                if isinstance(part, str):
                    part = json.loads(part)
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        b64_data = url.split(",", 1)[1]
                        image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                        # Hard cap the actual PIL image before it reaches the
                        # Qwen processor. This avoids relying solely on
                        # max_pixels propagation through transformers 4.57.x.
                        # For a portrait document page this yields roughly
                        # 768 x 995 (~0.76 MP) at most.
                        image.thumbnail((768, 1024), Image.Resampling.LANCZOS)
                    else:
                        image = url  # http(s) URL -- processor fetches it directly
                    parts.append({"type": "image", "image": image})
            chat_messages.append({"role": msg["role"], "content": parts})
        return chat_messages


# Explicit signature, not infer_signature() -- AnyType() on `messages`
# tells MLflow not to type-check/coerce that field at all, which is what
# actually avoids both prior failure modes (ChatModel's fixed
# content:string schema, and infer_signature()'s over-eager structural
# typing). Confirmed AnyType exists in mlflow.types.schema (3.15.1) by
# reading the source directly.
input_schema = Schema([ColSpec(AnyType(), name="messages")])
output_schema = Schema([ColSpec(AnyType(), name="choices")])
params_schema = ParamSchema(
    [
        ParamSpec("max_tokens", DataType.long, default=512),
        ParamSpec("temperature", DataType.double, default=0.0),
    ]
)
signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

# Real (decodable) tiny PNG, same request shape the validation cell below
# sends -- exercises the image-decode path during logging's own
# predict()-on-example check, before the ~30 minute log+register round
# trip finishes. Kept even though the signature no longer depends on it
# for typing (AnyType is untyped either way) -- still useful as a real
# smoke test at log time.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
input_example = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this document page as clean markdown."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
                },
            ],
        }
    ],
    "max_tokens": 2000,
    "temperature": 0.0,
}

model_info = mlflow.pyfunc.log_model(
    name=model_name,
    python_model=Qwen3VLChatModel(),
    artifacts={"model_dir": str(local_model_dir)},
    signature=signature,
    input_example=input_example,
    # Matches the pins in the install cell above exactly, so the
    # deployed endpoint's container gets the same versions actually
    # tested locally, not independently-resolved ones.
    extra_pip_requirements=[
        "mlflow==3.15.1",
        "transformers==4.57.1",
        "accelerate",  # pin exactly here too once confirmed above
        "torchvision",  # same -- pin exactly once confirmed above
        "pillow==12.3.0",
    ],
)
print(f"Logged: {model_info.model_uri}")

# COMMAND ----------

# env_pack is required -- Databricks' custom LLM/chat serving depends on
# Serverless Optimized Deployments; without it the endpoint does not work,
# per Databricks' own reference notebook for this exact serving path.
model_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=uc_model_name,
    env_pack="databricks_model_serving",
)
print(f"Registered: {uc_model_name} version {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create the GPU serving endpoint
# MAGIC
# MAGIC `scale_to_zero_enabled=True` — matches the design decision that this
# MAGIC endpoint should cost nothing between ingestion runs (the ingestion job is
# MAGIC scheduled/manually-triggered, not continuous). First call after being idle
# MAGIC will have a cold-start delay while the endpoint reloads the model.
# MAGIC

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=uc_model_name,
        entity_version=str(model_version.version),
        workload_type=getattr(ServingModelWorkloadType, workload_type_str),
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

# Idempotent: re-running this notebook after a signature/code fix re-logs
# a new model version, but the endpoint from an earlier successful run
# likely already exists -- update it to the new version rather than
# failing on an 'already exists' conflict from calling create again.
existing = [
    e for e in w.serving_endpoints.list() if e.name == endpoint_name
]
if existing:
    print(f"Endpoint '{endpoint_name}' already exists -- updating to version {model_version.version}.")
    w.serving_endpoints.update_config_and_wait(
        name=endpoint_name, served_entities=served_entities, timeout=timedelta(minutes=45)
    )
else:
    print(f"Creating endpoint '{endpoint_name}' -- can take 10+ minutes for a first deploy.")
    config = EndpointCoreConfigInput(name=endpoint_name, served_entities=served_entities)
    w.serving_endpoints.create_and_wait(name=endpoint_name, config=config, timeout=timedelta(minutes=45))
print("Endpoint ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate end-to-end
# MAGIC
# MAGIC Same request shape validated in the OCR spike test (`notebooks/rag_ocr_spike.ipynb`)
# MAGIC — confirms this endpoint is a drop-in for `WNV_VISION_ENDPOINT` with no
# MAGIC other code changes needed.
# MAGIC

# COMMAND ----------

import base64
import io
import json
import time
from urllib import request as urlrequest

import pdfplumber

test_pdf_path = dbutils.widgets.get("test_pdf_volume_path").strip()
test_page_num = int(dbutils.widgets.get("test_page").strip())

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_host = context.apiUrl().get().removeprefix("https://").removesuffix("/")
databricks_token = context.apiToken().get()

with pdfplumber.open(test_pdf_path) as pdf:
    page = pdf.pages[test_page_num - 1]
    image = page.to_image(resolution=150)
    buf = io.BytesIO()
    image.original.save(buf, format="PNG")
    image_bytes = buf.getvalue()

b64 = base64.b64encode(image_bytes).decode()
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this document page as clean markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ],
    "max_tokens": 2000,
    "temperature": 0.0,
}
url = f"https://{databricks_host}/serving-endpoints/{endpoint_name}/invocations"
req = urlrequest.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    },
    method="POST",
)
# 120s wasn't enough -- a real run timed out on the client side (not a
# 400 from the model), most likely scale_to_zero_enabled=True cold start
# (spinning up compute + loading the model onto the GPU before generation
# even starts) plus real inference time for a full page image with
# max_tokens=2000 on GPU_SMALL. Generous ceiling for the (slow) first
# call; later calls against an already-warm endpoint should be much
# faster.
print("Sending request -- can take several minutes on a cold start...")
start = time.monotonic()
try:
    with urlrequest.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode())
except urlrequest.HTTPError as e:
    # Bare HTTPError.__str__() drops the response body, which is where
    # the actual rejection reason lives (either Databricks' own
    # request-validation message, or a traceback from inside predict()).
    body = e.read().decode()
    raise RuntimeError(f"HTTP {e.code} {e.reason}. Response body:\n{body}") from e
print(f"Response received after {time.monotonic() - start:.0f}s")

# Since predict() now returns a plain dict (not a ChatCompletionResponse)
# under a hand-built signature, we don't yet have an empirical confirmation
# of whether Databricks wraps the output as {"predictions": {...}} or
# returns our dict at the top level -- handle both, and fail loudly with
# the raw response if neither shape matches, rather than a bare KeyError.
choices = result.get("choices") or result.get("predictions", {}).get("choices")
if choices is None:
    raise RuntimeError(f"Unexpected response shape, no 'choices' found:\n{result}")

print(choices[0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Set `WNV_VISION_ENDPOINT` to the value of `endpoint_name` above (default
# MAGIC `qwen3-vl-ocr`) wherever the ingestion pipeline configures it. The endpoint
# MAGIC is scale-to-zero, so the first call after being idle will be slow (model
# MAGIC reload) — expected, not a bug.
# MAGIC