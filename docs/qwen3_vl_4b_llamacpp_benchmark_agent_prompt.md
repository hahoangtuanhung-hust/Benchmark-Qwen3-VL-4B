# Prompt triển khai benchmark Qwen3-VL-4B bằng llama.cpp

## Vai trò

Bạn là **Senior AI Inference / Performance Engineer**.  
Nhiệm vụ của bạn là thiết kế, triển khai, chạy và tổng hợp benchmark cho **Qwen3-VL-4B-Instruct** bằng **llama.cpp** trên GPU NVIDIA, với workload **Vision-Language Model (VLM)** và tải thực tế **CCU = 1–2**.

Bạn phải ưu tiên:

1. Tính đúng đắn của benchmark.
2. Khả năng tái lập.
3. Không gán sai tên precision/quantization.
4. Tách rõ **quality**, **latency**, **throughput**, **GPU memory** và **KV cache**.
5. Giữ cùng hardware + software + dataset + generation config khi so sánh.
6. Lưu toàn bộ raw results, không chỉ lưu summary.

---

# 1. Mục tiêu benchmark

Benchmark Qwen3-VL-4B theo yêu cầu business ban đầu:

- FP16
- FP8
- INT4
- W4A16
- W4A8

Các metric bắt buộc:

- Accuracy / Quality
- TTFT — Time To First Token
- TTFV — Time To First Visible Value
- TPS — Token Per Second
- Prefill throughput — tokens/s
- Decode throughput — tokens/s
- TBT / ITL — Inter-Token Latency
- End-to-End latency
- Peak GPU VRAM
- Idle GPU VRAM
- KV cache memory
- GPU utilization
- CCU1 performance
- CCU2 performance
- CCU2 slowdown so với CCU1

Backend bắt buộc:

```text
llama.cpp
```

Model:

```text
Qwen3-VL-4B-Instruct
```

Workload:

```text
Vision + Text -> Text
```

Concurrency:

```text
CCU = 1
CCU = 2
```

---

# 2. RÀNG BUỘC QUAN TRỌNG VỀ PRECISION / QUANTIZATION

## 2.1. Không được giả lập hoặc đổi tên quantization

Yêu cầu business dùng các nhãn:

```text
FP16
FP8
INT4
W4A16
W4A8
```

Nhưng llama.cpp/GGUF có hệ quantization riêng.

Trước khi benchmark, BẮT BUỘC tạo:

```text
results/compatibility_matrix.csv
```

với schema:

```text
requested_label
requested_semantics
llama_cpp_supported
support_type
actual_gguf_type
actual_compute_behavior
model_artifact
mmproj_artifact
evidence
notes
```

Trong đó:

```text
support_type =
SUPPORTED_EXACT
SUPPORTED_PROXY
UNSUPPORTED
UNKNOWN
```

## 2.2. Quy tắc bắt buộc

Không được:

```text
Q8_0 == FP8
Q4_K_M == W4A16
Q4_K_M == INT4
Q4_K_M == W4A8
```

trừ khi có bằng chứng kỹ thuật xác nhận semantics tương ứng.

Không được đổi tên một GGUF quant thành precision khác chỉ để đủ bảng.

Nếu llama.cpp không hỗ trợ exact FP8 hoặc W4A8:

```text
requested_label = FP8
support_type = UNSUPPORTED
```

hoặc:

```text
support_type = SUPPORTED_PROXY
actual_gguf_type = ...
```

và phải báo cáo proxy ở bảng riêng.

## 2.3. Nguồn xác minh ưu tiên

Kiểm tra theo thứ tự:

1. Source code/version llama.cpp đang chạy.
2. `llama-quantize --help`.
3. `llama-server --help`.
4. GGUF metadata thực tế.
5. Qwen official GGUF repository.
6. llama.cpp official docs/source.
7. Không dựa vào tên file community nếu metadata không xác nhận.

Lưu bằng chứng vào:

```text
results/environment/
results/compatibility/
```

## 2.4. Track benchmark

Tạo hai track nếu cần.

### Track A — Requested Precision

Chỉ chứa cấu hình:

```text
SUPPORTED_EXACT
```

### Track B — llama.cpp Native / Proxy

Nếu exact formats không đủ, benchmark thêm các GGUF format thực tế phù hợp, ví dụ:

```text
F16
Q8_0
Q6_K
Q4_K_M
IQ4_XS
```

nhưng KHÔNG được gọi chúng là FP8/W4A8 nếu semantics không đúng.

---

# 3. Baseline VLM phải cố định

Để đo tác động của LLM quantization, trong benchmark chính:

```text
mmproj = FP16
```

cho mọi model variant, nếu phần cứng cho phép.

Không thay đổi đồng thời:

```text
LLM quantization
+
mmproj quantization
```

trong main benchmark.

Sau main benchmark mới được chạy optional experiment:

```text
mmproj FP16 vs Q8_0
```

và phải báo cáo riêng.

---

# 4. Hardware policy

## 4.1. Một hardware cho toàn bộ bảng chính

Tất cả variants trong cùng một bảng phải chạy trên:

```text
same GPU model
same GPU count
same CUDA
same driver
same llama.cpp commit
same server flags
```

Không được so trực tiếp:

```text
FP16 on A100
vs
Q4 on T4
```

## 4.2. Nếu model không fit

Nếu một cấu hình OOM:

1. Ghi OOM vào raw result.
2. Không âm thầm đổi GPU chỉ cho cấu hình đó.
3. Có thể tạo benchmark suite thứ hai trên hardware lớn hơn.
4. Hai suite phải báo cáo độc lập.

Ví dụ:

```text
Suite A: 1x T4
Suite B: 1x L4
```

Không merge TPS của hai suite vào một ranking duy nhất.

---

# 5. Giai đoạn 0 — Thu thập environment

Tạo script:

```text
scripts/collect_environment.sh
```

Thu ít nhất:

```text
date
uname -a
lscpu
free -h
nvidia-smi
nvidia-smi -q
nvcc --version
cmake --version
gcc --version
python --version
git --version
llama.cpp git commit
llama-server --version
llama-server --list-devices
llama-server --help
llama-quantize --help
```

Lưu vào:

```text
results/environment/environment.txt
results/environment/nvidia_smi.txt
results/environment/llama_server_help.txt
results/environment/llama_quantize_help.txt
```

---

# 6. Build llama.cpp

Build CUDA Release.

Baseline:

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build \
  --target llama-server llama-cli llama-quantize \
  -j 2
```

Nếu môi trường Kaggle gặp lỗi:

```text
CUDA::cuda_driver target was not found
```

thì thử build compatibility mode:

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

Nếu GPU là NVIDIA T4, có thể pin architecture:

```text
CMAKE_CUDA_ARCHITECTURES=75
```

Không dùng workaround nếu không cần.

Ghi lại command build thực tế.

---

# 7. Model artifacts

Chuẩn bị thư mục:

```text
models/
├── fp16/
├── fp8/
├── int4/
├── w4a16/
├── w4a8/
├── proxies/
└── mmproj/
```

Mỗi artifact phải có metadata:

```text
model_name
source_repo
source_revision
filename
file_size_bytes
sha256
gguf_file_type
tensor_type_summary
created_by
quantization_command
```

Tạo:

```text
results/model_manifest.csv
```

Không benchmark artifact nếu:

```text
sha256 missing
quantization unknown
GGUF metadata unknown
```

---

# 8. Dataset benchmark

Tạo dataset cố định:

```text
benchmark_data/
├── performance/
│   ├── small/
│   ├── normal/
│   └── large/
├── accuracy/
└── manifest.jsonl
```

## 8.1. Performance image sets

### Small

```text
resolution target: ~512x512
samples: 10
```

### Normal — workload chính

```text
resolution target: ~1280x720
samples: 30
```

### Large

```text
resolution target: ~1920x1080
samples: 10
```

Không tải ảnh từ Internet trong khi benchmark.

Ảnh phải tồn tại local trước khi test.

Lưu:

```text
image_id
path
width
height
bytes
sha256
category
```

## 8.2. Prompt performance cố định

Prompt mặc định:

```text
Describe the important objects, text, and events visible in this image. Be concise and factual.
```

Generation:

```text
temperature = 0
seed = fixed if supported
max_tokens = 256
stream = true
```

Decode-heavy optional:

```text
max_tokens = 512
```

---

# 9. Accuracy dataset

Benchmark accuracy phải dùng cùng input và evaluator cho mọi variants.

Ưu tiên:

```text
TextVQA
DocVQA
ChartQA
```

Có thể bổ sung:

```text
custom production dataset
```

## Quick mode

```text
50 samples / public dataset
50 custom samples nếu có
```

## Full mode

```text
>= 200 samples / dataset
```

Ưu tiên official metric của từng dataset.

Ví dụ:

```text
TextVQA -> official VQA-style accuracy nếu available
DocVQA  -> ANLS nếu evaluator chuẩn available
ChartQA -> official/relaxed accuracy nếu available
```

Nếu official evaluator không dùng được:

1. Không tự tạo metric tùy tiện rồi gọi là official accuracy.
2. Dùng fallback metric.
3. Đổi tên rõ ràng.
4. Ghi limitation.

Không dùng LLM-as-a-Judge làm metric primary trừ khi người dùng yêu cầu.

---

# 10. Cấu hình llama-server chính

Main benchmark cố định:

```text
GPU offload        = all layers
Flash Attention    = on nếu backend hỗ trợ ổn định
batch-size         = 512
ubatch-size        = 256
KV K type          = f16
KV V type          = f16
prompt cache       = OFF
continuous batch   = ON
metrics            = ON
slots endpoint     = ON
perf               = ON
mmproj             = FP16
```

Dùng các flag thực tế có trong version `llama-server --help`.

Không assume flag tồn tại nếu version hiện tại không có.

---

# 11. Context configuration cho CCU 1–2

Mục tiêu:

```text
8192 context tokens / slot
```

Ưu tiên:

```bash
--kv-unified-per-slot 8192
```

nếu version llama.cpp hỗ trợ.

### CCU1

```text
-np 1
context per slot = 8192
```

### CCU2

```text
-np 2
context per slot = 8192
```

Nếu bất kỳ variant hợp lệ nào OOM với 8192:

1. thử common configuration:

```text
4096 tokens / slot
```

2. dùng cùng context cho TẤT CẢ variants trong main table.

Không dùng:

```text
F16 ctx=4096
Q4 ctx=8192
```

trong cùng bảng so sánh chính.

---

# 12. Prompt cache policy

Main benchmark:

```text
--no-cache-prompt
```

hoặc request:

```json
{
  "cache_prompt": false
}
```

nếu API/version hỗ trợ.

Mục tiêu là đo cold-request behavior.

Có thể chạy optional production experiment:

```text
cache ON
```

nhưng kết quả phải ở bảng riêng.

---

# 13. Warm-up policy

Mỗi lần load model:

```text
server health check
↓
5 warm-up requests
↓
discard results
↓
reset / snapshot metrics
↓
measured run
```

Không đưa warm-up vào summary.

---

# 14. Workload matrix

Main benchmark:

| Scenario | CCU | Image | Output | Runs |
|---|---:|---|---:|---:|
| S1 | 1 | 512x512 | 256 | 10 |
| S2 | 1 | 1280x720 | 256 | 30 |
| S3 | 1 | 1920x1080 | 256 | 10 |
| S4 | 2 | 1280x720 | 256 | 30 pairs |
| S5 optional | 1 | 1280x720 | 512 | 10 |

Scenario trọng tâm:

```text
S2 = CCU1 normal
S4 = CCU2 normal
```

---

# 15. Cách tạo CCU2

CCU2 phải là hai request đồng thời.

Không chạy tuần tự.

Ví dụ logic:

```python
await asyncio.gather(
    request(user_a),
    request(user_b),
)
```

Mỗi pair phải lưu:

```text
pair_id
request_a_run_id
request_b_run_id
pair_start_time
pair_end_time
```

Hai request nên dùng hai ảnh khác nhau để tránh cache/reuse ngoài ý muốn.

---

# 16. Metric definitions

## 16.1. TTFT

```text
TTFT = timestamp(first generated token) - timestamp(request sent)
```

Unit:

```text
ms
```

## 16.2. TTFV

```text
TTFV = timestamp(first user-visible answer token/value)
       - timestamp(request sent)
```

Nếu response có reasoning/internal content:

```text
reasoning token != visible token
```

TTFV phải bỏ qua hidden reasoning.

Nếu model không có hidden reasoning:

```text
TTFV ~= TTFT
```

nhưng vẫn lưu cả hai field.

## 16.3. E2E latency

```text
E2E = final response timestamp - request timestamp
```

## 16.4. TBT / ITL

Cho timestamps:

```text
t1, t2, t3, ..., tn
```

tính:

```text
ITL_i = t_i - t_(i-1)
```

Report:

```text
mean
median/P50
P95
P99
```

### Cảnh báo quan trọng

SSE event không phải lúc nào cũng tương ứng đúng 1 token.

Nếu một streamed event chứa nhiều token:

- Không được gọi event-to-event latency là exact token ITL.
- Gọi nó là:

```text
stream_event_itl
```

- Nếu `timings_per_token` / token-level data của llama.cpp version đang chạy cung cấp thông tin đủ chính xác, ưu tiên dữ liệu đó.
- Ghi `itl_measurement_method` trong output.

## 16.5. Decode TPS

Primary source:

```text
llama.cpp timings.predicted_per_second
```

hoặc server metric tương đương.

Client fallback:

```text
(output_tokens - 1) /
(last_token_time - first_token_time)
```

Ghi nguồn:

```text
decode_tps_source
```

## 16.6. Prefill TPS

Primary source:

```text
llama.cpp timings.prompt_per_second
```

hoặc:

```text
llamacpp:prompt_tokens_seconds
```

Không tính vision preprocessing vào `prefill_tps` nếu backend báo text/model prefill riêng.

Nếu không tách được:

```text
prefill_tps_scope = llama.cpp reported prompt processing
```

## 16.7. TPS tổng

Lưu riêng:

```text
decode_tps
prefill_tps
output_tps
aggregate_decode_tps
```

Không gộp chúng thành một field mơ hồ tên `TPS`.

---

# 17. VLM latency breakdown

Nếu instrumentation cho phép, đo:

```text
image_load_ms
image_preprocess_ms
vision_encode_ms
llm_prefill_ms
first_decode_ms
```

Mục tiêu:

```text
TTFT ≈ media processing + vision encode + LLM prefill + first decode
```

Nếu llama.cpp không expose một thành phần:

```text
NULL
```

Không ước lượng rồi ghi như measured value.

---

# 18. GPU monitoring

Tạo:

```text
scripts/gpu_monitor.py
```

hoặc shell monitor.

Sampling interval:

```text
100–250 ms
```

Thu:

```text
timestamp
gpu_index
memory_used_mb
memory_total_mb
gpu_util_pct
memory_util_pct
power_w
temperature_c
```

Nguồn có thể là:

```text
nvidia-smi
NVML
```

Ưu tiên NVML nếu thuận tiện.

---

# 19. VRAM metrics

Mỗi model variant phải đo:

```text
baseline_vram_mb
model_loaded_idle_vram_mb
peak_vram_mb
delta_model_load_vram_mb
```

Definitions:

```text
baseline_vram =
GPU memory trước khi start llama-server

model_loaded_idle_vram =
server ready, chưa chạy measured request

peak_vram =
max(memory_used) trong benchmark
```

Lưu CCU1 và CCU2 riêng.

---

# 20. KV cache memory

KV cache là metric bắt buộc.

Main test:

```text
K cache = f16
V cache = f16
```

Agent phải tìm cách đo theo thứ tự ưu tiên:

1. llama.cpp startup/runtime log nếu có exact allocation.
2. server properties/slots/log.
3. difference-based measurement có kiểm soát.
4. theoretical calculation chỉ dùng như `estimated_kv_cache`, không được gọi là measured.

Output fields:

```text
kv_cache_k_type
kv_cache_v_type
kv_cache_context_per_slot
kv_cache_slots
kv_cache_measured_mb
kv_cache_estimated_mb
kv_cache_measurement_method
```

Nếu chỉ có estimate:

```text
kv_cache_measured_mb = NULL
```

---

# 21. Optional KV cache experiment

Chỉ chạy SAU main benchmark.

Chọn production candidate, ví dụ Q4_K_M.

Benchmark:

```text
K=f16 V=f16
K=q8_0 V=q8_0
K=q4_0 V=q4_0
```

chỉ khi llama.cpp version + backend hỗ trợ.

Giữ:

```text
same model
same workload
same CCU
same context
```

So sánh:

```text
TTFT
decode TPS
peak VRAM
KV memory
accuracy smoke test
```

Không trộn KV experiment vào main quantization table.

---

# 22. Server metrics

Start server với:

```text
--metrics
--slots
--perf
```

Thu `/metrics` trước và sau mỗi measured batch.

Quan tâm ít nhất:

```text
llamacpp:prompt_tokens_total
llamacpp:prompt_seconds_total
llamacpp:prompt_tokens_seconds
llamacpp:tokens_predicted_total
llamacpp:tokens_predicted_seconds_total
llamacpp:predicted_tokens_seconds
llamacpp:requests_processing
llamacpp:requests_deferred
llamacpp:n_tokens_max
llamacpp:n_decode_total
llamacpp:n_busy_slots_per_decode
```

Thu `/slots` trong CCU2 để xác nhận:

```text
2 slots processing
```

khi hai request overlap.

---

# 23. Raw request schema

Lưu mỗi request thành một row:

```text
results/raw/requests.csv
```

Schema tối thiểu:

```text
run_id
suite_id
timestamp
model_name
model_revision
requested_precision
actual_quant_type
support_type
gguf_sha256
mmproj_type
mmproj_sha256
llama_cpp_commit
gpu_name
gpu_count
cuda_version
ccu
pair_id
scenario
image_id
image_width
image_height
input_tokens
output_tokens
context_per_slot
kv_k_type
kv_v_type
cache_prompt
temperature
max_tokens
http_status
success
error_type
ttft_ms
ttfv_ms
e2e_ms
itl_mean_ms
itl_p50_ms
itl_p95_ms
itl_p99_ms
itl_measurement_method
prefill_tps
prefill_tps_source
decode_tps
decode_tps_source
baseline_vram_mb
idle_vram_mb
peak_vram_mb
gpu_util_mean_pct
gpu_util_peak_pct
kv_cache_measured_mb
kv_cache_estimated_mb
kv_cache_measurement_method
answer_text
```

---

# 24. Accuracy schema

Tạo:

```text
results/accuracy/predictions.jsonl
results/accuracy/summary.csv
```

Mỗi sample:

```text
sample_id
dataset
requested_precision
actual_quant_type
question
ground_truth
prediction
normalized_prediction
metric_name
score
error
```

Summary:

```text
dataset
requested_precision
actual_quant_type
samples
success_rate
metric_name
score
delta_vs_fp16
```

---

# 25. Statistics

Không chỉ report mean.

Latency:

```text
P50
P95
P99
mean
std
```

TPS:

```text
mean
median
std
```

VRAM:

```text
idle
peak
```

Accuracy:

```text
score
delta vs FP16
```

CCU2:

```text
per-user TPS
aggregate TPS
P50 TTFT
P95 TTFT
```

---

# 26. Derived metrics

## Speedup

```text
decode_speedup =
decode_tps_variant / decode_tps_fp16
```

## Prefill speedup

```text
prefill_speedup =
prefill_tps_variant / prefill_tps_fp16
```

## VRAM saving

```text
vram_saving_pct =
(1 - peak_vram_variant / peak_vram_fp16) * 100
```

## Quality loss

```text
quality_loss =
accuracy_fp16 - accuracy_variant
```

## CCU2 latency slowdown

```text
ccu2_ttft_slowdown =
TTFT_P50_CCU2 / TTFT_P50_CCU1
```

## CCU2 per-user TPS loss

```text
ccu2_tps_loss_pct =
(1 - TPS_per_user_CCU2 / TPS_CCU1) * 100
```

---

# 27. Benchmark validity checks

Một run được coi là hợp lệ chỉ khi:

```text
HTTP status = success
model did not OOM
model did not restart
output_tokens > 0
image successfully parsed
response not empty
context not truncated unexpectedly
cache not reused in cold benchmark
GPU same as suite configuration
```

Loại bỏ / flag:

```text
server startup request
warm-up request
OOM run
HTTP error
timeout
corrupt response
```

Không âm thầm delete failed rows.

---

# 28. Reproducibility

Mỗi run phải gắn:

```text
suite_id
run_id
git_commit
config_hash
dataset_hash
model_sha256
mmproj_sha256
```

Tạo:

```text
results/reproducibility_manifest.json
```

---

# 29. Cấu trúc repository cần tạo

```text
qwen3_vl_benchmark/
├── README.md
├── Makefile
├── requirements.txt
├── configs/
│   ├── benchmark.yaml
│   ├── models.yaml
│   └── datasets.yaml
├── models/
├── benchmark_data/
│   ├── performance/
│   └── accuracy/
├── scripts/
│   ├── build_llamacpp.sh
│   ├── collect_environment.sh
│   ├── inspect_model.py
│   ├── start_server.sh
│   ├── wait_for_server.py
│   ├── stop_server.sh
│   ├── gpu_monitor.py
│   ├── benchmark_client.py
│   ├── benchmark_ccu1.py
│   ├── benchmark_ccu2.py
│   ├── run_accuracy.py
│   ├── collect_server_metrics.py
│   ├── summarize.py
│   └── validate_results.py
├── results/
│   ├── environment/
│   ├── compatibility/
│   ├── raw/
│   ├── gpu/
│   ├── accuracy/
│   ├── logs/
│   └── report/
└── run_benchmark.sh
```

---

# 30. `configs/benchmark.yaml`

Tạo config trung tâm, ví dụ:

```yaml
benchmark:
  warmup_requests: 5

  generation:
    temperature: 0.0
    max_tokens: 256
    stream: true
    cache_prompt: false

  server:
    batch_size: 512
    ubatch_size: 256
    flash_attention: true
    kv_k_type: f16
    kv_v_type: f16
    metrics: true
    slots: true
    perf: true

  context:
    preferred_per_slot: 8192
    fallback_per_slot: 4096

  workloads:
    - id: ccu1_small
      ccu: 1
      dataset: small
      runs: 10

    - id: ccu1_normal
      ccu: 1
      dataset: normal
      runs: 30

    - id: ccu1_large
      ccu: 1
      dataset: large
      runs: 10

    - id: ccu2_normal
      ccu: 2
      dataset: normal
      pairs: 30

accuracy:
  quick_samples_per_dataset: 50
  full_samples_per_dataset: 200
```

---

# 31. Orchestration

Tạo một command duy nhất:

```bash
./run_benchmark.sh
```

Workflow:

```text
collect environment
        ↓
build / verify llama.cpp
        ↓
inspect supported quantization
        ↓
build compatibility matrix
        ↓
validate model artifacts
        ↓
validate dataset
        ↓
for each SUPPORTED model:
    start server CCU1
    wait health
    warmup
    monitor GPU
    run S1/S2/S3
    collect metrics
    stop server

    start server CCU2
    wait health
    warmup
    monitor GPU
    run S4
    collect metrics
    stop server

    run accuracy
        ↓
summarize
        ↓
validate results
        ↓
generate report
```

---

# 32. Fail-fast rules

Dừng benchmark variant nếu:

```text
server health never becomes ready
CUDA OOM
mmproj fails to load
vision capability missing
GGUF metadata incompatible
more than 5% requests fail
```

Không dừng toàn bộ suite nếu chỉ một variant fail.

Ghi status:

```text
PASS
FAIL
OOM
UNSUPPORTED
INVALID
```

---

# 33. Kaggle-specific development mode

Nếu đang chạy trên Kaggle:

## Development objective

Chỉ cần chứng minh trước:

```text
Q4_K_M
+
FP16 mmproj
+
1 NVIDIA GPU
+
CCU1
+
VLM request
+
streaming
+
metrics
```

Smoke test:

```text
5 requests
```

Sau đó:

```text
CCU2 smoke = 2 requests đồng thời
```

Khi pipeline chạy ổn mới chạy full benchmark trên platform chính.

Không dùng Kaggle development numbers làm final benchmark nếu hardware/session khác với final suite.

---

# 34. Performance vs accuracy isolation

Không chạy accuracy trong lúc performance load test đang chạy.

Flow:

```text
performance suite
stop
accuracy suite
```

GPU phải idle trước mỗi measured performance suite.

---

# 35. Reporting

Tạo:

```text
results/report/benchmark_summary.md
results/report/summary.csv
results/report/compatibility_matrix.csv
```

## Main compatibility table

```text
Requested | Exact support | Actual llama.cpp config | Status | Notes
```

## Main performance table

```text
Variant
Actual GGUF
CCU
TTFT P50
TTFT P95
TTFV P50
E2E P50
Prefill TPS
Decode TPS
ITL P50
ITL P95
Peak VRAM
KV Cache
GPU Util
```

## Quality table

```text
Variant
Dataset
Metric
Score
Delta vs FP16
```

## CCU table

```text
Variant
CCU1 TPS
CCU2 TPS/user
CCU2 aggregate TPS
CCU1 TTFT P50
CCU2 TTFT P50
TTFT slowdown
TPS loss/user
```

---

# 36. Charts

Tạo ít nhất:

```text
1. Quantization vs Decode TPS
2. Quantization vs Prefill TPS
3. Quantization vs TTFT P50/P95
4. Quantization vs ITL P50/P95
5. Quantization vs Peak VRAM
6. Quantization vs Accuracy
7. CCU1 vs CCU2 TPS/user
8. CCU1 vs CCU2 TTFT
9. Accuracy vs Decode TPS
10. Accuracy vs Peak VRAM
```

Không dùng biểu đồ gây hiểu nhầm.

---

# 37. Pareto analysis

Không tự động tuyên bố "best model" chỉ dựa trên TPS.

Xác định Pareto frontier theo:

```text
higher accuracy
higher decode TPS
lower TTFT
lower VRAM
```

Báo cáo các trade-off.

Ví dụ:

```text
Variant A:
+ nhanh hơn
+ ít VRAM
- quality giảm

Variant B:
+ quality gần FP16
- chậm hơn
```

---

# 38. Acceptance criteria

Benchmark hoàn thành khi:

- [ ] llama.cpp commit được pin và lưu.
- [ ] GPU/CUDA environment được lưu.
- [ ] Compatibility matrix đã tạo.
- [ ] Không dùng sai nhãn FP8/W4A16/W4A8.
- [ ] Model hash được lưu.
- [ ] mmproj hash được lưu.
- [ ] Dataset hash được lưu.
- [ ] Prompt/generation config cố định.
- [ ] KV F16 cố định cho main benchmark.
- [ ] Prompt cache OFF cho cold benchmark.
- [ ] CCU1 chạy đủ samples.
- [ ] CCU2 chạy đủ concurrent pairs.
- [ ] TTFT đo được.
- [ ] TTFV đo được.
- [ ] Prefill TPS đo được.
- [ ] Decode TPS đo được.
- [ ] ITL/TBT có measurement method rõ.
- [ ] Peak VRAM đo được.
- [ ] KV memory measured hoặc clearly estimated.
- [ ] Accuracy chạy cùng dataset/evaluator.
- [ ] Raw results được giữ nguyên.
- [ ] Summary được sinh tự động.
- [ ] Failed/OOM/unsupported variants được ghi rõ.
- [ ] Report không trộn hardware khác nhau.

---

# 39. Quy tắc làm việc của agent

## Không hỏi lại nếu có thể tự kiểm tra

Agent phải tự:

```text
inspect repo
inspect GPU
inspect llama.cpp support
inspect model metadata
inspect existing files
```

trước khi hỏi người dùng.

## Không fake số liệu

Nếu chưa benchmark:

```text
value = NULL / NOT_RUN
```

Không điền số giả.

## Không fake support

Nếu backend không hỗ trợ exact requested precision:

```text
UNSUPPORTED
```

Không thay bằng một quant gần giống rồi gọi cùng tên.

## Không benchmark khi config chưa frozen

Trước measured run phải in:

```text
BENCHMARK CONFIG FROZEN
```

và dump config ra:

```text
results/config_snapshot.yaml
```

---

# 40. Deliverables cuối cùng

Agent phải bàn giao:

```text
1. Source code benchmark
2. Build scripts
3. Model compatibility matrix
4. Model manifest + hashes
5. Environment manifest
6. Dataset manifest
7. Raw request results
8. GPU monitoring logs
9. Accuracy predictions
10. Summary CSV
11. Charts
12. benchmark_summary.md
13. README hướng dẫn reproduce
```

README phải có một lệnh để reproduce:

```bash
./run_benchmark.sh
```

hoặc equivalent.

---

# 41. Thứ tự triển khai bắt buộc

Hãy thực thi theo thứ tự sau:

```text
PHASE 0
Inspect environment
        ↓
PHASE 1
Build/validate llama.cpp CUDA
        ↓
PHASE 2
Validate Qwen3-VL VLM smoke test
        ↓
PHASE 3
Build requested-precision compatibility matrix
        ↓
PHASE 4
Prepare/freeze model artifacts
        ↓
PHASE 5
Prepare/freeze benchmark dataset
        ↓
PHASE 6
Implement benchmark instrumentation
        ↓
PHASE 7
Run CCU1
        ↓
PHASE 8
Run CCU2
        ↓
PHASE 9
Run accuracy
        ↓
PHASE 10
Collect/validate metrics
        ↓
PHASE 11
Generate report + charts
```

Không nhảy thẳng vào chạy benchmark nếu chưa hoàn thành compatibility matrix.

---

# 42. Output đầu tiên agent phải trả về

Trước khi sửa code hoặc chạy full benchmark, hãy trả:

```text
A. Detected hardware
B. llama.cpp commit/version
C. Qwen3-VL artifact availability
D. Requested precision compatibility matrix
E. Proposed actual benchmark matrix
F. Detected blockers
G. Files/scripts sẽ tạo
```

Sau đó tự tiếp tục triển khai nếu không có blocker bắt buộc cần user input.

---

# 43. Kết quả benchmark mong muốn

Cuối cùng phải trả lời được các câu hỏi:

```text
1. FP16 baseline đạt TTFT/TPS/VRAM/accuracy bao nhiêu?
2. Các requested precisions nào llama.cpp thực sự hỗ trợ exact?
3. Các requested precisions nào chỉ có proxy hoặc không hỗ trợ?
4. Quantization nào giảm VRAM nhiều nhất?
5. Quantization nào giữ quality gần FP16 nhất?
6. Quantization nào decode nhanh nhất?
7. Quantization nào prefill nhanh nhất?
8. CCU2 làm giảm TPS/user bao nhiêu?
9. CCU2 làm tăng TTFT bao nhiêu?
10. KV cache chiếm bao nhiêu memory ở CCU1 và CCU2?
11. Bottleneck nằm ở vision encode, prefill hay decode?
12. Cấu hình nào nằm trên Pareto frontier cho deployment CCU 1–2?
```

Không đưa ra kết luận deployment nếu dữ liệu benchmark chưa đủ hoặc benchmark không cùng hardware/config.
