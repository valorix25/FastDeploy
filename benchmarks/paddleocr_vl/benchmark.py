#!/usr/bin/env python

# Block torch import — both modelscope and transformers try to import torch,
# which SIGBUS-crashes on this MACA environment. The PaddleOCR-VL pipeline
# doesn't use torch (VL model goes through FastDeploy server API), so hiding
# torch from import resolution is safe.
import importlib.util as _iutil
_original_find_spec = _iutil.find_spec
def _patched_find_spec(name, package=None):
    if name == 'torch' or name.startswith('torch.'):
        return None
    return _original_find_spec(name, package)
_iutil.find_spec = _patched_find_spec

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import glob
import json
import sys
import time
import uuid
from operator import itemgetter
from threading import Thread

import pymxsml
from transformers import AutoTokenizer
from tqdm import tqdm

shutdown = False
def monitor_device(gpu_ids, gpu_metrics_list):
    try:
        pymxsml.mxSmlInit()
        time.sleep(5)
        while not shutdown:
            try:
                gpu_util = 0
                mem_bytes = 0
                for gpu_id in gpu_ids:
                    gpu_util += pymxsml.mxSmlGetDeviceIpUsage(gpu_id, pymxsml.MXSML_USAGE_XCORE)
                    mem_bytes += pymxsml.mxSmlGetMemoryInfo(gpu_id).vramUse * 1024  # KB → Byte

                gpu_metrics_list.append(
                    {
                        "utilization": gpu_util,
                        "memory": mem_bytes,
                    }
                )
            except Exception as e:
                print(f"Error monitoring GPUs: {e}")

            time.sleep(0.5)

    except Exception as e:
        print(f"Error initializing the GPU monitor: {e}")

def get_curr_time():
    return time.perf_counter()

def new_task_info():
    task_info = {}
    task_info["id"] = uuid.uuid4().hex
    return task_info

def patch_skip_vlm_labels(pipeline):
    """Monkey-patch: extend IMAGE_LABELS so markdown_ignore_labels blocks skip VL calls.

    In the original pipeline, blocks with labels in markdown_ignore_labels
    (e.g. header, footer, footnote, number, aside_text) are still sent to
    the VL model for recognition, but the results are discarded in markdown
    output. This patch makes them skip VL entirely, reducing latency.

    Mechanism: pipeline.py line 307 checks `block_label not in image_labels`
    to decide whether to call VL. By extending the module-level IMAGE_LABELS
    constant with markdown_ignore_labels, these blocks are excluded from VL
    dispatch. The blocks still appear in parsing_res_list (with empty content),
    and markdown_ignore_labels still filters them from markdown output.
    """
    inner_pipeline = pipeline._pipeline
    skip_labels = set(inner_pipeline.markdown_ignore_labels)
    if not skip_labels:
        return

    from paddlex.inference.pipelines.paddleocr_vl import pipeline as pipeline_mod
    original = pipeline_mod.IMAGE_LABELS
    patched = list(set(original) | skip_labels)
    pipeline_mod.IMAGE_LABELS = patched
    assert pipeline_mod.IMAGE_LABELS == patched, "patch_skip_vlm_labels failed: IMAGE_LABELS not updated"
    print(f"[patch] Skip VL for labels: {sorted(skip_labels)} "
          f"(IMAGE_LABELS: {original} -> {patched})")


class Predictor(object):
    def predict(self, task_info, batch_data):
        task_info["start_time"] = get_curr_time()
        try:
            markdown, num_pages = self._predict(batch_data)
            task_info["successful"] = True
            task_info["processed_pages"] = num_pages
            task_info["generated_tokens"] = len(self.tokenizer.encode(markdown))
            return markdown
        except Exception as e:
            task_info["successful"] = False
            print(e)
            return None
        finally:
            task_info["end_time"] = get_curr_time()

    def _predict(self, batch_data):
        raise NotImplementedError

    def close(self):
        pass

class PaddleXPredictor(Predictor):
    def __init__(self, config_path):
        from paddlex import create_pipeline

        super().__init__()
        self.pipeline = create_pipeline(config_path)
        patch_skip_vlm_labels(self.pipeline)

    def _predict(self, batch_data):
        results = list(self.pipeline.predict(batch_data))
        return "\n\n".join(res._to_markdown(pretty=False)["markdown_texts"] for res in results), len(results)

    def close(self):
        self.pipeline.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=str, nargs="+", metavar="INPUT_DIR")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-o", "--output_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "benchmark_result"))
    parser.add_argument("--paddlex_config_path", type=str, default="PaddleOCR-VL-1.5.yaml")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory for tokenizer-based token counting")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0],
                        help="Physical GPU IDs for monitoring only (not for inference device selection)")
    parser.add_argument("--warmup-rounds", type=int, default=2,
                        help="Number of warmup batches to run before measuring (0 to disable)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    task_info_list = []
    all_input_paths = []
    for input_dir in args.input_dirs:
        all_input_paths += glob.glob(os.path.join(input_dir, "*"))
    all_input_paths.sort()
    if len(all_input_paths) == 0:
        print("No valid data")
        sys.exit(1)

    predictor = PaddleXPredictor(args.paddlex_config_path)
    predictor.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if args.batch_size < 1:
        print("Invalid batch size")
        sys.exit(2)

    gpu_metrics_list = []
    thread_device_monitor = Thread(
        target=monitor_device,
        args=(args.gpu_ids, gpu_metrics_list),
    )
    thread_device_monitor.start()

    # # === Warmup phase: send dummy requests to warm up server (JIT compile, CUDAGraph replay, etc.) ===
    # warmup_rounds = args.warmup_rounds
    # if warmup_rounds > 0 and len(all_input_paths) >= args.batch_size:
    #     warmup_samples = all_input_paths[:args.batch_size * min(warmup_rounds, len(all_input_paths) // args.batch_size)]
    #     print(f"[Warmup] Running {warmup_rounds} warmup batches ({len(warmup_samples)} images)...")
    #     warmup_batch = []
    #     for idx, input_path in enumerate(warmup_samples):
    #         warmup_batch.append(input_path)
    #         if len(warmup_batch) == args.batch_size or idx == len(warmup_samples) - 1:
    #             predictor.predict(new_task_info(), warmup_batch)
    #             warmup_batch.clear()
    #     print("[Warmup] Done.")
    #     # Reset GPU monitor counters (discard warmup data)
    #     gpu_metrics_list.clear()

    # === Benchmark phase ===
    try:
        start_time = get_curr_time()
        batch_data = []
        markdown_path = os.path.join(args.output_dir, "generated_markdown.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            for i, input_path in tqdm(enumerate(all_input_paths), total=len(all_input_paths)):
                batch_data.append(input_path)
                if len(batch_data) == args.batch_size or i == len(all_input_paths) - 1:
                    task_info = new_task_info()
                    task_info["batch_file_count"] = len(batch_data)
                    markdown = predictor.predict(task_info, batch_data)
                    if markdown is not None:
                        f.write(markdown)
                        f.write("\n\n")
                    task_info_list.append(task_info)
                    batch_data.clear()
        end_time = get_curr_time()
    finally:
        shutdown = True
        thread_device_monitor.join()
        predictor.close()

    total_files = len(all_input_paths)
    throughput_file = total_files / (end_time - start_time)
    print(f"Throughput (file): {throughput_file:.4f} files per second")
    duration_list_batch = [info["end_time"] - info["start_time"] for info in task_info_list]
    avg_latency_batch = sum(duration_list_batch) / len(duration_list_batch)
    print(f"Average latency (batch): {avg_latency_batch:.4f} seconds")

    successful_files = sum(x["batch_file_count"] if x["successful"] else 0 for x in task_info_list)
    throughput_file = successful_files / (end_time - start_time) if successful_files else 0
    if successful_files:
        processed_pages = sum(info.get("processed_pages", 0) for info in task_info_list)
        throughput_page = processed_pages / (end_time - start_time)
        print(f"Processed pages: {processed_pages}")
        print(f"Throughput (page): {throughput_page:.4f} pages per second")
        generated_tokens = sum(info.get("generated_tokens", 0) for info in task_info_list)
        throughput_token = generated_tokens / (end_time - start_time)
        print(f"Generated tokens: {generated_tokens}")
        print(f"Throughput (token): {throughput_token:.1f} tokens per second")
    else:
        processed_pages = None
        throughput_page = None
        generated_tokens = None
        throughput_token = None

    if gpu_metrics_list:
        gpu_util_list = list(map(itemgetter("utilization"), gpu_metrics_list))
        print(
            f"GPU utilization (%): {max(gpu_util_list):.1f}, {min(gpu_util_list):.1f}, {sum(gpu_util_list) / len(gpu_util_list):.1f}"
        )
        gpu_mem_list = list(map(itemgetter("memory"), gpu_metrics_list))
        print(
            f"GPU memory usage (MB): {max(gpu_mem_list) / 1024**2:.1f}, {min(gpu_mem_list) / 1024**2:.1f}, {sum(gpu_mem_list) / len(gpu_mem_list) / 1024**2:.1f}"
        )

    dic = {
        "input_dirs": args.input_dirs,
        "batch_size": args.batch_size,
        "total_files": total_files,
        "throughput_file": throughput_file,
        "avg_latency_batch": avg_latency_batch,
        "duration_list": duration_list_batch,
        "successful_files": successful_files,
        "processed_pages": processed_pages,
        "throughput_page": throughput_page,
        "generated_tokens": generated_tokens,
        "throughput_token": throughput_token,
        "gpu_metrics_list": gpu_metrics_list,
    }
    output_json_path = os.path.join(args.output_dir, "benchmark_result.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)
    print(f"Config and results saved to {args.output_dir}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
