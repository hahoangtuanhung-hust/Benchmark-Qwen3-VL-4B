# TASK — Triển khai benchmark Qwen3-VL-4B bằng TensorRT-Edge-LLM

## 1. Vai trò

Bạn là **Senior AI Inference / TensorRT Performance Engineer**.

Repository hiện tại đã/đang có luồng benchmark:

```text
Qwen3-VL-4B
    ↓
llama.cpp
    ↓
CCU 1 / CCU 2
    ↓
performance + accuracy + VRAM + KV cache
```

Nhiệm vụ của bạn là **bổ sung backend thứ hai bằng NVIDIA TensorRT-Edge-LLM**, không phá vỡ luồng llama.cpp hiện tại.

Kiến trúc sau khi hoàn thành:

```text
                     Qwen3-VL-4B
                          │
                   Frozen Dataset
                          │
             Common Benchmark Core
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
      llama.cpp                  TensorRT-Edge-LLM
          │                               │
        GGUF                    HF / Quantized checkpoint
          │                               │
   llama-server                  TensorRT Engine
          │                               │
          └──────────────┬────────────────┘
                         │
                      CCU 1–2
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
      Latency        Throughput          Memory
   TTFT/TTFV/ITL   Prefill/Decode     VRAM/KV Cache
                         │
                         ↓
                      Accuracy
                         │
                         ↓
             Backend Comparison Report
```

---

# 2. Model

Benchmark:

```text
Qwen/Qwen3-VL-4B-Instruct
```

Workload:

```text
Image + Text → Text
```

TensorRT backend:

```text
NVIDIA TensorRT-Edge-LLM
```

Không thay bằng TensorRT-LLM thông thường nếu chưa xác định rõ lý do.

Nếu một utility từ TensorRT-LLM được TensorRT-Edge-LLM tái sử dụng thì được phép dùng nhưng phải ghi rõ.

---

# 3. Precision yêu cầu

Business yêu cầu benchmark:

```text
FP16
FP8
INT4
W4A16
W4A8
```

TUYỆT ĐỐI không assume tất cả đều được hỗ trợ.

Trước khi build bất kỳ engine nào, hãy tạo:

```text
results/tensorrt_edge/compatibility_matrix.csv
```

Schema:

```text
requested_precision
requested_semantics
model_supported
hardware_supported
runtime_supported
quant_recipe
weight_dtype
activation_dtype
kv_cache_dtype
visual_encoder_dtype
status
evidence
notes
```

Status:

```text
SUPPORTED_EXACT
SUPPORTED_WITH_RECIPE
UNSUPPORTED
UNKNOWN
```

Ví dụ semantics:

```text
FP16:
weights = FP16
activations = FP16

FP8:
xác định chính xác recipe FP8 đang dùng

INT4:
phải chỉ rõ AWQ / GPTQ / other

W4A16:
weights = 4 bit
activations = 16 bit

W4A8:
weights = 4 bit
activations = 8 bit
```

Không được:

```text
INT4 AWQ → tự động gọi W4A8
```

hoặc:

```text
FP8 checkpoint → gọi là FP8 nếu actual engine không dùng FP8 compute
```

Nếu không hỗ trợ:

```text
status = UNSUPPORTED
```

và tiếp tục benchmark các configuration hợp lệ.

---

# 4. Hardware compatibility

Trước khi benchmark, tự detect:

```text
GPU name
GPU architecture
compute capability
VRAM
CUDA
driver
TensorRT
TensorRT-Edge-LLM
ModelOpt
```

Tạo:

```text
results/tensorrt_edge/environment.txt
```

Phải xác định GPU thuộc:

```text
Ampere
Ada
Hopper
Blackwell
Jetson Orin
Jetson Thor
DGX Spark
other
```

Sau đó mới quyết định precision nào chạy được.

Không cố ép FP8 hoặc W4A8 trên hardware không hỗ trợ runtime tương ứng.

---

---

# 4A. Nền tảng benchmark bắt buộc: Modal GPU

Toàn bộ **benchmark chính thức** của TensorRT-Edge-LLM phải được triển khai và chạy trên **Modal GPU**.

Không dùng Kaggle cho số liệu benchmark cuối cùng. Kaggle chỉ được phép dùng cho smoke test hoặc development nếu cần.

## Modal execution policy

Agent phải triển khai benchmark bằng Modal theo nguyên tắc:

```text
Modal
  ↓
1 GPU cố định
  ↓
TensorRT-Edge-LLM
  ↓
Qwen3-VL-4B
  ↓
CCU1 / CCU2
  ↓
Metrics + Accuracy + VRAM + KV Cache
```

Ưu tiên GPU:

```text
NVIDIA T4 24GB
```

Nếu T4 không đủ VRAM hoặc không hỗ trợ precision cần benchmark:

1. tự kiểm tra GPU support;
2. chuyển toàn bộ benchmark suite sang một GPU Modal phù hợp hơn;
3. phải dùng **cùng một loại GPU cho tất cả precision trong cùng một bảng so sánh**.

Các GPU fallback có thể xem xét:

```text
A10
A100
H100
Blackwell-class GPU
```

nhưng chỉ dùng nếu thực sự cần thiết.

Không được benchmark:

```text
FP16 → H100
FP8  → H100
INT4 → T4
W4A16 → T4
```

rồi đặt chung trong một bảng direct comparison.

Nếu phải dùng nhiều GPU khác nhau, hãy tạo các suite tách biệt:

```text
suite_modal_T4
suite_modal_h100
```

và không so trực tiếp TPS/TTFT giữa hai suite như cùng hardware.

## Modal app structure

Tạo cấu trúc tối thiểu:

```text
modal/
├── app.py
├── image.py
├── volumes.py
├── gpu_config.py
├── benchmark_entry.py
└── README.md
```

Modal app phải:

- build image có CUDA / TensorRT / TensorRT-Edge-LLM phù hợp;
- mount persistent Volume chứa model, engine và dataset;
- dùng đúng 1 GPU cho benchmark CCU1/CCU2;
- không autoscale sang GPU/container thứ hai trong lúc đo CCU2;
- expose hoặc gọi inference trong cùng container;
- lưu raw results và logs vào persistent Volume;
- cho phép chạy lại benchmark bằng một command.

## Modal Volume

Tạo persistent storage cho:

```text
/models
/engines
/dataset
/results
/cache
```

Không download lại model hoặc rebuild engine ở mỗi request.

Các artifact phải được cache/persist:

```text
HuggingFace checkpoint
quantized checkpoint
TensorRT engine
dataset
accuracy dataset
benchmark results
GPU logs
```

## Modal concurrency policy

CCU2 phải là:

```text
1 Modal container
+
1 GPU
+
2 request đồng thời
```

Không để Modal autoscaler tạo:

```text
2 containers
+
2 GPUs
```

vì như vậy không còn đo contention CCU2 trên cùng một GPU.

Benchmark CCU2 phải xác nhận:

```text
container_count = 1
gpu_count = 1
concurrent_requests = 2
```

## Modal warm container policy

Trong benchmark performance:

- giữ container warm giữa các measured request;
- không tính container cold-start vào TTFT;
- không tính image startup, pip install, model download hoặc engine build vào inference latency.

Nếu muốn đo cold-start thì chạy thành **experiment riêng**.

## Modal environment manifest

Bổ sung vào:

```text
results/tensorrt_edge/environment.txt
```

các field:

```text
platform = modal
modal_app_name
modal_region nếu lấy được
modal_gpu_requested
modal_gpu_detected
modal_gpu_count
modal_container_id nếu lấy được
modal_image_id/version nếu lấy được
container_cpu
container_memory
persistent_volume_name
```

## Modal GPU validation

Ngay khi container start, agent phải lưu:

```text
nvidia-smi
nvidia-smi -q
CUDA_VISIBLE_DEVICES
TensorRT version
TensorRT-Edge-LLM version
ModelOpt version
```

và xác nhận GPU thực tế đúng với GPU requested.

Nếu Modal cấp GPU khác với cấu hình mong muốn:

```text
status = INVALID_HARDWARE
```

và không dùng run đó cho final summary.

## Modal benchmark command

Tạo một entrypoint tương đương:

```bash
modal run modal/benchmark_entry.py --precision fp16
modal run modal/benchmark_entry.py --precision fp8
modal run modal/benchmark_entry.py --precision int4
modal run modal/benchmark_entry.py --precision w4a16
modal run modal/benchmark_entry.py --precision w4a8
```

hoặc một command:

```bash
modal run modal/benchmark_entry.py --suite all
```

Agent được phép thay CLI syntax theo Modal SDK version hiện tại, nhưng phải giữ khả năng reproduce bằng một command.

## Modal result persistence

Sau mỗi precision, commit/save ngay:

```text
/results/tensorrt_edge/<precision>/
```

bao gồm:

```text
raw_requests.csv
gpu_metrics.csv
accuracy.csv
engine_manifest.csv
server.log
benchmark_config.yaml
```

Không đợi chạy hết tất cả precision mới lưu kết quả.

## Chi phí / timeout

Agent phải tránh giữ GPU idle không cần thiết.

Workflow:

```text
start GPU container
↓
load/build engine nếu chưa có
↓
warm-up
↓
benchmark CCU1
↓
benchmark CCU2
↓
accuracy
↓
save results
↓
shutdown
```

Không để GPU chạy chờ thủ công.

Không giảm số sample hoặc thay benchmark config chỉ để tiết kiệm chi phí nếu chưa ghi rõ thay đổi.

## Quy tắc so sánh với llama.cpp

Nếu benchmark llama.cpp cũng chạy trên Modal, hãy ưu tiên:

```text
same Modal GPU type
same GPU count
same dataset
same image resolution
same CCU
```

để tạo direct comparison.

Nếu llama.cpp chạy ở platform khác:

```text
Kaggle llama.cpp
vs
Modal TensorRT
```

thì báo cáo phải ghi rõ:

```text
CROSS-PLATFORM RESULT
NOT A PURE BACKEND COMPARISON
```

và không kết luận tốc độ chênh lệch chỉ do backend.

## Acceptance criteria bổ sung cho Modal

- [ ] TensorRT benchmark chạy trên Modal GPU.
- [ ] GPU type được pin hoặc xác minh.
- [ ] Một suite chỉ dùng một GPU type.
- [ ] CCU2 chạy trên cùng 1 container + 1 GPU.
- [ ] Không tính Modal cold-start vào TTFT.
- [ ] Model/engine/dataset dùng persistent Volume.
- [ ] Raw results được lưu persistent sau từng precision.
- [ ] Không để autoscaler làm sai CCU2.
- [ ] Direct backend comparison chỉ dùng cùng hardware.

# 5. Quy tắc benchmark công bằng

Khi so TensorRT-Edge-LLM với llama.cpp phải giữ cố định:

```text
same model
same model revision
same GPU model
same GPU count
same dataset
same images
same prompts
same generation settings
same max output tokens
same CCU
same context target
same accuracy evaluator
```

Không được kết luận:

```text
TensorRT nhanh hơn llama.cpp
```

nếu hai backend chạy trên GPU khác nhau.

Nếu bắt buộc hardware khác:

```text
report separately
```

và ghi:

```text
NOT A DIRECT BACKEND COMPARISON
```

---

# 6. Vision encoder policy

Trong benchmark chính:

```text
Visual encoder = FP16
```

nếu TensorRT-Edge-LLM/hardware hỗ trợ.

Mục tiêu là cô lập ảnh hưởng của quantization language model.

Không đồng thời:

```text
quantize LLM
+
quantize visual encoder
```

trong main benchmark.

Sau khi benchmark chính hoàn thành mới được chạy optional experiment:

```text
Visual FP16
vs
Visual FP8
```

và report riêng.

---

# 7. KV cache policy

Main benchmark:

```text
KV cache = FP16
```

cho mọi model variant nếu backend cho phép.

Không thay:

```text
FP16 model + FP16 KV
FP8 model + FP8 KV
INT4 model + INT8 KV
```

trong cùng main table.

Sau benchmark chính mới chạy optional:

```text
FP16 KV
vs
FP8 KV
```

và report riêng.

---

# 8. Cấu trúc repository

Refactor theo hướng:

```text
qwen3_vl_benchmark/
│
├── common/
│   ├── configs/
│   ├── dataset/
│   ├── metrics/
│   ├── accuracy/
│   └── utils/
│
├── backends/
│   ├── llamacpp/
│   │   ├── build.sh
│   │   ├── server.sh
│   │   └── adapter.py
│   │
│   └── tensorrt_edge/
│       ├── install.sh
│       ├── detect_support.py
│       ├── quantize.py
│       ├── export.py
│       ├── build_engine.py
│       ├── server.py
│       └── adapter.py
│
├── benchmark/
│   ├── benchmark_ccu1.py
│   ├── benchmark_ccu2.py
│   ├── benchmark_accuracy.py
│   ├── gpu_monitor.py
│   └── metrics.py
│
└── results/
    ├── llamacpp/
    ├── tensorrt_edge/
    └── comparison/
```

Không duplicate toàn bộ benchmark client.

Tạo interface chung:

```python
class InferenceBackend:
    def start(self):
        ...

    def stop(self):
        ...

    def health(self):
        ...

    def generate(self, image, prompt, config):
        ...

    def metrics(self):
        ...
```

Hai implementation:

```text
LlamaCppBackend
TensorRTEdgeBackend
```

---

# 9. Dataset

Dùng chính dataset llama.cpp hiện tại.

Không tạo dataset TensorRT riêng.

Performance workload:

```text
Small:
~512x512
10 samples

Normal:
~1280x720
30 samples

Large:
~1920x1080
10 samples
```

Main workload:

```text
1280x720
```

Prompt:

```text
Describe the important objects, text, and events visible in this image.
Be concise and factual.
```

Generation:

```text
temperature = 0
max_tokens = 256
stream = true
```

---

# 10. CCU configuration

Chỉ benchmark:

```text
CCU = 1
CCU = 2
```

Không benchmark:

```text
4 / 8 / 16 / 32
```

trừ khi được yêu cầu sau.

## CCU1

```text
30 measured requests
```

sau:

```text
5 warm-up requests
```

## CCU2

30 pairs:

```text
T0
├── User A request
└── User B request
```

Hai request phải overlap thực sự.

Không chạy:

```text
A finish → B start
```

---

# 11. Các scenario

```text
S1:
CCU1
512x512
256 output tokens
10 runs

S2:
CCU1
1280x720
256 output tokens
30 runs

S3:
CCU1
1920x1080
256 output tokens
10 runs

S4:
CCU2
1280x720
256 output tokens
30 concurrent pairs

S5 optional:
CCU1
1280x720
512 output tokens
10 runs
```

Main report ưu tiên:

```text
S2
S4
```

---

# 12. Metric bắt buộc

Đo:

```text
Accuracy
TTFT
TTFV
E2E latency

Prefill TPS
Decode TPS
Output TPS

TPOT
ITL / TBT

TPS/user
Aggregate TPS

Idle VRAM
Peak VRAM
KV cache memory

GPU utilization
GPU power nếu có
```

---

# 13. TTFT

Definition:

```text
TTFT =
first generated token timestamp
-
request submission timestamp
```

Report:

```text
mean
P50
P95
P99
```

---

# 14. TTFV

Definition:

```text
TTFV =
first visible answer content
-
request submission
```

Nếu model output có hidden reasoning:

```text
TTFT != TTFV
```

Nếu không:

```text
TTFV ≈ TTFT
```

vẫn phải lưu cả hai.

---

# 15. ITL / TBT

Definition:

```text
ITL_i = t_i - t_(i-1)
```

Report:

```text
mean
P50
P95
P99
```

Không dùng network SSE chunk latency làm token ITL nếu một chunk chứa nhiều token.

Ưu tiên metric token-level từ TensorRT runtime / benchmark API nếu có.

Ghi:

```text
itl_measurement_method
```

---

# 16. TPOT

Tính:

```text
TPOT =
(E2E - TTFT) /
(output_tokens - 1)
```

Đơn vị:

```text
ms/token
```

---

# 17. Decode TPS

Ưu tiên metric native TensorRT.

Nếu cần client fallback:

```text
Decode TPS =
(output_tokens - 1) /
(last_token_time - first_token_time)
```

Lưu:

```text
decode_tps_source
```

---

# 18. Prefill TPS

Đo riêng:

```text
input tokens / prefill time
```

Ưu tiên metric runtime.

Không gọi toàn bộ TTFT là prefill latency.

Nếu vision encoder không tách được khỏi prefill:

```text
prefill_scope = combined
```

và ghi rõ limitation.

---

# 19. VLM latency breakdown

Cố gắng instrument:

```text
image_load_ms
image_preprocess_ms
vision_encode_ms
visual_projection_ms
llm_prefill_ms
first_decode_ms
```

Mục tiêu:

```text
TTFT ≈
image preprocessing
+
vision encode
+
projection
+
LLM prefill
+
first decode
```

Nếu runtime không expose một component:

```text
NULL
```

Không fake số.

---

# 20. VRAM

Thu bằng NVML hoặc nvidia-smi với sampling:

```text
100–250 ms
```

Log:

```text
timestamp
gpu
memory_used_mb
gpu_utilization
memory_utilization
power_w
temperature
```

Report:

```text
baseline_vram_mb
engine_loaded_idle_vram_mb
peak_vram_ccu1_mb
peak_vram_ccu2_mb
```

---

# 21. KV cache

Đo:

```text
KV dtype
KV allocated memory
KV used memory nếu runtime expose
max token capacity
```

Output:

```text
kv_cache_dtype
kv_cache_allocated_mb
kv_cache_used_peak_mb
kv_cache_capacity_tokens
measurement_method
```

Nếu runtime không expose exact value:

```text
measured = NULL
estimated = value
```

và ghi:

```text
ESTIMATED
```

---

# 22. Accuracy

Dùng chính accuracy suite llama.cpp.

Ưu tiên:

```text
TextVQA
DocVQA
ChartQA
MMMU subset
custom production dataset
```

Quick benchmark:

```text
50–100 samples/dataset
```

Không thay prompt/evaluator theo backend.

Lưu:

```text
dataset
sample_id
backend
precision
prediction
ground_truth
metric
score
```

---

# 23. TensorRT build pipeline

Thực hiện theo pipeline phù hợp với version TensorRT-Edge-LLM được detect:

```text
HF Qwen3-VL-4B
      ↓
Quantization / ModelOpt
      ↓
TensorRT-Edge export
      ↓
Visual engine / LLM representation
      ↓
TensorRT Engine build
      ↓
Runtime validation
```

Nếu version hỗ trợ direct checkpoint engine build và nó ổn định:

```text
HF checkpoint
→ direct build
→ engine
```

có thể dùng.

Nhưng phải ghi lại:

```text
build_method
```

---

# 24. Engine manifest

Mỗi engine phải có:

```text
requested_precision
actual_precision
quant_recipe
model_revision
model_hash
engine_hash
visual_encoder_dtype
language_model_weight_dtype
language_model_activation_dtype
kv_cache_dtype
gpu_arch
TensorRT version
TensorRT-Edge-LLM version
ModelOpt version
build_command
build_timestamp
```

Tạo:

```text
results/tensorrt_edge/engine_manifest.csv
```

---

# 25. Không tính engine build time vào inference latency

Engine build có thể mất nhiều thời gian.

Tách metric:

```text
engine_build_time_sec
engine_size_mb
```

nhưng không cộng vào:

```text
TTFT
E2E
TPS
```

---

# 26. Warm-up

Mỗi engine:

```text
load engine
↓
health ready
↓
5 warm-up VLM requests
↓
discard
↓
start measured benchmark
```

Không đưa warm-up vào summary.

---

# 27. Cache

Main benchmark phải là cold benchmark.

Nếu backend có:

```text
prefix cache
prompt cache
reuse cache
```

hãy disable.

Nếu không thể disable:

1. dùng prompts/images khác nhau giữa requests;
2. document behavior;
3. không để cache tạo lợi thế không công bằng so với llama.cpp.

---

# 28. Raw schema

Mỗi request lưu:

```text
backend
run_id
pair_id
scenario

requested_precision
actual_precision
quant_recipe

gpu
gpu_arch

image_id
width
height

input_tokens
output_tokens

ccu

ttft_ms
ttfv_ms
e2e_ms

vision_encode_ms
prefill_ms
first_decode_ms

prefill_tps
decode_tps

tpot_ms

itl_mean_ms
itl_p50_ms
itl_p95_ms
itl_p99_ms

idle_vram_mb
peak_vram_mb

kv_cache_mb

gpu_util_mean
gpu_util_peak

http_status
success
error
```

---

# 29. CCU degradation

Tính:

```text
CCU2_TTFT_Slowdown =
TTFT_P50_CCU2 /
TTFT_P50_CCU1
```

```text
CCU2_TPS_Loss =
1 -
TPS_per_user_CCU2 /
TPS_CCU1
```

```text
Aggregate_TPS_CCU2 =
TPS_user_A + TPS_user_B
```

---

# 30. Backend comparison

Chỉ compare trực tiếp những configuration tương đương.

Ví dụ:

```text
llama.cpp FP16
vs
TensorRT Edge FP16
```

là direct comparison.

Nhưng:

```text
llama.cpp Q4_K_M
vs
TensorRT INT4 AWQ
```

phải ghi:

```text
different quantization recipe
```

Không gọi là apples-to-apples INT4 comparison nếu semantics khác nhau.

---

# 31. Summary table 1 — TensorRT

Sinh:

```text
Precision
Quant recipe
Accuracy
TTFT P50
TTFT P95
TTFV P50
Prefill TPS
Decode TPS
TPOT
ITL P95
CCU2 TPS/User
Aggregate TPS
Peak VRAM
KV Cache
```

---

# 32. Summary table 2 — Backend

Sinh:

```text
Metric                llama.cpp      TensorRT Edge
--------------------------------------------------
FP16 accuracy
FP16 TTFT P50
FP16 TTFT P95
FP16 prefill TPS
FP16 decode TPS
FP16 ITL P95
FP16 peak VRAM
FP16 KV memory

CCU2 TTFT
CCU2 TPS/user
CCU2 aggregate TPS
```

---

# 33. Derived metrics

Tính:

```text
backend_decode_speedup =
TRT_decode_TPS /
llama_decode_TPS
```

```text
backend_prefill_speedup =
TRT_prefill_TPS /
llama_prefill_TPS
```

```text
TTFT_reduction_pct =
(1 - TRT_TTFT / llama_TTFT) * 100
```

```text
VRAM_difference_pct =
(TRT_VRAM - llama_VRAM) /
llama_VRAM * 100
```

---

# 34. Charts

Sinh ít nhất:

```text
1. Precision vs Accuracy
2. Precision vs Decode TPS
3. Precision vs Prefill TPS
4. Precision vs TTFT
5. Precision vs ITL
6. Precision vs Peak VRAM
7. CCU1 vs CCU2 TPS/user
8. CCU1 vs CCU2 TTFT
9. llama.cpp vs TensorRT FP16
10. Accuracy vs Decode TPS
```

---

# 35. Pareto analysis

Phân tích:

```text
Accuracy ↑
Decode TPS ↑
Prefill TPS ↑
TTFT ↓
ITL ↓
Peak VRAM ↓
```

Không chỉ chọn model nhanh nhất.

Tìm các configuration nằm trên Pareto frontier.

---

# 36. Fail handling

Nếu một precision fail:

```text
UNSUPPORTED
BUILD_FAILED
OOM
RUNTIME_FAILED
```

ghi vào result.

Không làm suite dừng hoàn toàn.

Không tự đổi precision.

Ví dụ:

```text
W4A8 unsupported
```

thì ghi:

```text
W4A8 = UNSUPPORTED
```

và tiếp tục FP16/FP8/INT4/W4A16.

---

# 37. Output agent phải tạo

```text
results/tensorrt_edge/
├── compatibility_matrix.csv
├── environment.txt
├── engine_manifest.csv
├── raw_requests.csv
├── gpu_metrics.csv
├── accuracy.csv
├── summary.csv
└── benchmark_report.md
```

Ngoài ra:

```text
results/comparison/
├── backend_comparison.csv
├── backend_comparison.md
└── charts/
```

---

# 38. Không over-test

Ưu tiên triển khai nhanh.

Không viết hàng chục unit tests không cần thiết.

Chỉ cần:

```text
smoke test
health test
1 VLM request test
CCU2 concurrency smoke test
result validation
```

Sau đó chạy benchmark thật.

---

# 39. Trình tự thực thi

```text
PHASE 0
Inspect existing repository
        ↓
PHASE 1
Detect GPU + TensorRT compatibility
        ↓
PHASE 2
Install/build TensorRT-Edge-LLM
        ↓
PHASE 3
Build precision compatibility matrix
        ↓
PHASE 4
Run FP16 Qwen3-VL smoke test
        ↓
PHASE 5
Build supported quantized engines
        ↓
PHASE 6
Validate VLM inference
        ↓
PHASE 7
Freeze common dataset/config
        ↓
PHASE 8
Benchmark CCU1
        ↓
PHASE 9
Benchmark CCU2
        ↓
PHASE 10
Accuracy evaluation
        ↓
PHASE 11
Collect VRAM + KV metrics
        ↓
PHASE 12
Compare llama.cpp vs TensorRT
        ↓
PHASE 13
Generate final report
```

---

# 40. Output đầu tiên trước khi chạy benchmark

Trước khi chạy full benchmark, hãy báo cáo ngắn:

```text
1. GPU detected
2. GPU architecture
3. TensorRT version
4. TensorRT-Edge-LLM version
5. Qwen3-VL support status
6. FP16 support
7. FP8 support
8. INT4 support + recipe
9. W4A16 support + recipe
10. W4A8 support + recipe
11. Các blocker hiện tại
12. Benchmark matrix thực tế sẽ chạy
```

Sau đó tự tiếp tục nếu không có blocker bắt buộc.

---

# 41. Điều kiện hoàn thành

Task hoàn thành khi trả lời được:

```text
TensorRT-Edge-LLM FP16 đạt performance bao nhiêu?

FP8 có chạy exact trên hardware hiện tại không?

INT4 đang dùng AWQ hay GPTQ?

W4A16 có được support exact không?

W4A8 có được support exact không?

TTFT P50/P95?

TTFV?

Prefill TPS?

Decode TPS?

ITL P50/P95?

Peak VRAM?

KV cache memory?

Accuracy degradation?

CCU2 làm giảm TPS/user bao nhiêu?

CCU2 làm tăng TTFT bao nhiêu?

Vision encoder hay LLM là bottleneck?

TensorRT nhanh hơn/chậm hơn llama.cpp bao nhiêu trên cùng hardware?

Precision nào nằm trên Pareto frontier?
```

Không đưa ra con số nào nếu chưa đo thực tế.
