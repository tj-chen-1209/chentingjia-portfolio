# Placeholder Guide

This document tracks all placeholders (`{{VARIABLE}}` format) used throughout the portfolio documentation and provides instructions for filling them with actual measured values.

## Global Placeholders (Used in Multiple Files)

| Placeholder | Current Value | Where to Update | Priority |
|-------------|---------------|-----------------|----------|
| `{{PDF_PORTFOLIO_URL}}` | - | README.md, index.md | High |
| `{{ONE_PAGE_URL}}` | - | README.md, index.md | High |
| `{{DEMO_URL}}` | - | README.md, index.md | Medium |
| `{{YOUR_EMAIL}}` | - | mkdocs.yml, index.md | High |

### How to Fill Global Placeholders

1. **PDF_PORTFOLIO_URL:** Upload compiled PDF portfolio to GitHub Releases or external hosting; paste URL
2. **ONE_PAGE_URL:** Create 1-page research highlights PDF; host and paste URL
3. **DEMO_URL:** Upload demo videos to YouTube/Vimeo or GitHub Releases; paste URL
4. **YOUR_EMAIL:** Replace with actual contact email

---

## Project 1: OptiTrack ROS2 Streaming Node

**File:** `docs/projects/teleop-mocap.md`

### System Interfaces & Timing

| Placeholder | Description | Measurement Method |
|-------------|-------------|-------------------|
| `{{MOCAP_HZ}}` | Motion capture frame rate | Check Motive settings (likely 120Hz) |
| `{{NETWORK_LATENCY_MS}}` | Network propagation delay | `ping` test between Motive server and ROS2 workstation |
| `{{FRAME_PROCESS_MS}}` | Per-frame callback processing time | Add `std::chrono` timers in `DataHandler`; compute mean over 1000 frames |
| `{{END_TO_END_LATENCY_MS}}` | Total latency from capture to subscriber | Compare Motive frame timestamp vs. ROS2 subscriber receipt time |

**Measurement Script (Latency):**

```cpp
// In DataHandler callback
auto start = std::chrono::high_resolution_clock::now();
// ... process frame ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
                     "Frame process time: %.3f ms", duration.count() / 1000.0);
```

### Performance Metrics

| Placeholder | Description | Measurement Method |
|-------------|-------------|-------------------|
| `{{DATA_LOSS_RATE}}` | Percentage of dropped frames | Monitor frame counter discontinuities; compute: (dropped / total) × 100% |
| `{{TEST_DURATION}}` | Duration of stability test | Record test session length (e.g., 30 min) |
| `{{CPU_USAGE}}` | CPU utilization | `top -p $(pgrep motive_streamer_node)` average over 5 min |
| `{{CONTINUOUS_HOURS}}` | Longest continuous operation | Log uptime during extended session; use `ros2 node info` |
| `{{RECONNECT_TESTS}}` | Number of reconnection tests | Manually disconnect/reconnect; count successful recoveries |
| `{{NUM_ASSETS}}` | Simultaneous rigid bodies tracked | Count active rigid bodies in test environment |

**Measurement Protocol:**

1. Launch node with logging enabled
2. Run for 30+ minutes continuously
3. Use `ros2 topic hz /rigid_body_pose` to verify rate
4. Monitor frame counter in Motive vs. ROS2 received frames
5. Compute statistics: mean, std, min, max for timing metrics

---

## Project 2: GR00T-N1.6 SFT on LIBERO (Reproduction)

**File:** `docs/projects/groot-libero-reproduction.md`

### Status: All Metrics Filled ✅

This project page has **zero text placeholders** - all performance numbers are actual measured values from experiments:

- Success rates: 98.5% (Spatial), 97.0% (Goal), 100.0% (Object), 95.5% (LIBERO-10), 97.8% (Overall)
- Training configuration: 8 GPUs, batch 1600, 20K-40K steps, 14-28h training time
- Evaluation: 5 parallel envs, 20 episodes/task, 782/800 successful episodes
- Inference latency: ~50ms/step (7× faster than RDT)

**Only Visual Assets Missing:**

| Asset | Status | Priority | Instructions |
|-------|--------|----------|--------------|
| `project3_overview.png` | ⏳ TODO | High | System architecture + LIBERO loop |
| `groot_trainable_modules.png` | ⏳ TODO | Medium | Frozen vs. trainable module diagram |
| `groot_results_table.png` | ⏳ TODO | High | Bar chart with 4 task suite results |

---

## Project 3: RDT SFT & Evaluation on LIBERO

**File:** `docs/projects/rdt-libero.md`

### Status: Partial Metrics Filled

**Filled Exact Numbers:**
- Success rates: 92% (object), 94% (goal), 76% (spatial), 38% (long)
- LoRA upper bound: ~20% (vs. 76%+ full-parameter)
- Inference latency: 375ms/step on A100 (69% of eval time)
- Diffusion steps ablation: H=1 (78%) → H=8 (88%) on libero_object
- Action chunk ablation: N=1 (78%) → N=10 (90%) on libero_object
- Language embedding cache speedup: ~30%
- Single demo overfitting: 1000+ epochs, ~20% validation
- Task 8 language confusion: 0% success
- Random seed: 42
- Image preprocessing: 128×128 → 336×336
- LoRA tested ranks: 8-32
- peft version: 0.10.0

**Placeholders Remaining:** Training configs, software versions, parameter counts, resource usage

### Training Configuration

| Placeholder | Description | Where to Find |
|-------------|-------------|---------------|
| `{{BATCH_SIZE}}` | Training batch size per GPU | Training config bash script |
| `{{LEARNING_RATE}}` | Learning rate | Training config (likely 1e-4 to 5e-4) |
| `{{NUM_EPOCHS}}` | Number of training epochs | Training config |
| `{{NUM_GPUS}}` | Number of GPUs used | Training cluster config (likely 8) |
| `{{GPU_TYPE}}` | GPU hardware model | A100 (mentioned in profiler) |

### Model Parameters

| Placeholder | Description | Measurement Method |
|-------------|-------------|-------------------|
| `{{PARAMS_TOTAL}}` | Total RDT model parameters | Count: `sum(p.numel() for p in model.parameters()) / 1e6` |
| `{{PARAMS_LORA}}` | LoRA trainable parameters | Count LoRA adapter parameters only |
| `{{MODEL_SIZE_GB}}` | Checkpoint file size | Check .pt file size in GB |

### Training Efficiency

| Placeholder | Description | Measurement Method |
|-------------|-------------|-------------------|
| `{{PARAMS_FULL}}` | Trainable parameters (full) | Count: `sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6` |
| `{{PARAMS_LORA}}` | Trainable parameters (LoRA) | Count LoRA adapter parameters only |
| `{{TIME_FULL}}` | Training wall-clock time (full) | Log training start/end timestamps; compute hours |
| `{{TIME_LORA}}` | Training wall-clock time (LoRA) | Log training start/end timestamps; compute hours |
| `{{MEM_FULL}}` | Peak GPU memory (full) | Monitor `nvidia-smi` during training; record max |
| `{{MEM_LORA}}` | Peak GPU memory (LoRA) | Monitor `nvidia-smi` during training; record max |
| `{{LORA_RANK}}` | LoRA rank hyperparameter | Training config YAML (typically 8, 16, or 32) |
| `{{LORA_ALPHA}}` | LoRA alpha hyperparameter | Training config YAML (typically 16 or 32) |
| `{{LORA_MODULES}}` | LoRA target modules | Training config (e.g., "q_proj,v_proj,k_proj") |

### Software Versions

| Placeholder | Description | Command to Find |
|-------------|-------------|-----------------|
| `{{PYTORCH_VERSION}}` | PyTorch version | `pip show torch | grep Version` |
| `{{DEEPSPEED_VERSION}}` | DeepSpeed version | `pip show deepspeed | grep Version` |
| `{{TRANSFORMERS_VERSION}}` | HuggingFace version | `pip show transformers | grep Version` |
| `{{LIBERO_VERSION}}` | LIBERO version | `pip show libero | grep Version` |

### Ablation Experiments

| Placeholder | Description | Experiment Needed |
|-------------|-------------|-------------------|
| `{{SUCCESS_H10}}` | Success rate with H=10 diffusion steps | Run ablation on libero_object |
| `{{SUCCESS_H1_N10}}` | Success with H=1, N=10 | Run ablation experiment |
| `{{CONTEXT_LEN}}` | Current action history context | Check model config |
| `{{CONTEXT_LEN_EXTENDED}}` | Proposed extended context | Design decision (e.g., 2× current) |
| `{{STUDENT_SIZE}}` | Distilled model size target | Design decision (e.g., 1.0B params) |

### Path Placeholders

| Placeholder | Description | Usage |
|-------------|-------------|-------|
| `{{LIBERO_DATA_PATH}}` | LIBERO dataset location | Training script argument |
| `{{CHECKPOINT_PATH}}` | Checkpoint file path | Evaluation script argument |
| `{{EXPERIMENT_DIR}}` | Experiment output directory | Auto-generated by training script |
| `{{CONFIG}}` | Config file path | Training script argument |
| `{{TASKS}}` | Task list for evaluation | Evaluation script argument |

---

## Project 4: RLinf × RDT Integration

**File:** `docs/projects/rlinf-rdt-integration.md`

### Status: Interface Contracts Defined

**Filled Facts:**
- joint_states dimension: 9 (7 joints + 2 gripper)
- Image rotation correction: 180° (k=2 in np.rot90)
- Dependency patch: os.environ["_CHECK_PEFT"]="0"
- peft version: 0.10.0
- Status: Forward loop works ✅, Training blocked ❌
- Blocker: Diffusion policy no log_prob for policy gradient

**Placeholders Remaining:** Observation/action dimensions, batch sizes, throughput, versions

### Interface Dimensions

| Placeholder | Description | How to Obtain |
|-------------|-------------|---------------|
| `{{IMAGE_SHAPE}}` | Single image shape | Check LIBERO obs; likely (128, 128, 3) per view |
| `{{IMAGE_BATCH_SHAPE}}` | Batched images shape | (batch, num_views, H, W, C) |
| `{{EEF_STATE_DIM}}` | EEF state dimension | Count: position(3) + orientation(4) = 7? |
| `{{ACTION_DIM}}` | Action dimension | LIBERO: 6D EEF delta + gripper = 7 or 8 |
| `{{BATCH_SIZE}}` | Parallel environments | RLinf config |
| `{{THROUGHPUT}}` | Effective steps/second | Measure in integration test |

### Configuration Placeholders

| Placeholder | Description | Where to Find |
|-------------|-------------|---------------|
| `{{DIFFUSION_STEPS}}` | H denoising steps | RDT config (likely 8) |
| `{{ACTION_CHUNK}}` | N action chunk length | RDT config (likely 10) |
| `{{NUM_PARALLEL_ENVS}}` | Parallel environment count | RLinf config |
| `{{DIFFUSERS_VERSION}}` | diffusers library version | `pip show diffusers` |
| `{{RDT_CHECKPOINT_PATH}}` | RDT checkpoint location | User config |
| `{{LIBERO_DATA_PATH}}` | LIBERO dataset path | User config |
| `{{RLINF_REPO_URL}}` | RLinf repository URL | Private (contact lab) |
| `{{TEST_OBS_PATH}}` | Test observation file | Generated for testing |

---

## Visual Assets (Figures & GIFs)

**Status:** All figure files currently have `.placeholder` extensions with creation instructions.

### Project 1: OptiTrack ROS2 Streaming Node

| Asset | Status | Priority | Instructions File |
|-------|--------|----------|-------------------|
| `optitrack_ros2_arch.png` | ⏳ TODO | High | `docs/assets/figures/optitrack_ros2_arch.png.placeholder` |
| `id_mapping_flow.png` | ⏳ TODO | Medium | `docs/assets/figures/id_mapping_flow.png.placeholder` |
| `ik_discontinuity.png` | ⏳ TODO | Medium | `docs/assets/figures/ik_discontinuity.png.placeholder` |
| `optitrack_demo.gif` | ⏳ TODO | High | `docs/assets/gifs/optitrack_demo.gif.placeholder` |

### Project 2: GR00T-N1.6 Reproduction

| Asset | Status | Priority | Instructions File |
|-------|--------|----------|-------------------|
| `project3_overview.png` | ⏳ TODO | High | `docs/assets/figures/project3_overview.png.placeholder` |
| `groot_trainable_modules.png` | ⏳ TODO | Medium | `docs/assets/figures/groot_trainable_modules.png.placeholder` |
| `groot_results_table.png` | ⏳ TODO | High | `docs/assets/figures/groot_results_table.png.placeholder` |

### Project 3: RDT SFT & Evaluation

| Asset | Status | Priority | Instructions File |
|-------|--------|----------|-------------------|
| `libero_pipeline.png` | ⏳ TODO | High | `docs/assets/figures/libero_pipeline.png.placeholder` (or use Mermaid) |
| `libero_curve.png` | ⏳ TODO | High | `docs/assets/figures/libero_curve.png.placeholder` |
| `libero_ablation_hn.png` | ⏳ TODO | Medium | `docs/assets/figures/libero_ablation_hn.png.placeholder` |
| `libero_profiler_breakdown.png` | ⏳ TODO | Medium | `docs/assets/figures/libero_profiler_breakdown.png.placeholder` |
| `libero_language_confusion.png` | ⏳ TODO | Low | `docs/assets/figures/libero_language_confusion.png.placeholder` |

### Project 4: RLinf × RDT Integration

| Asset | Status | Priority | Instructions File |
|-------|--------|----------|-------------------|
| `project4_overview.png` | ⏳ TODO | High | `docs/assets/figures/project4_overview.png.placeholder` |
| `rlinf_rdt_dataflow.png` | ⏳ TODO | Medium | `docs/assets/figures/rlinf_rdt_dataflow.png.placeholder` |
| `diffusion_vs_policy_gradient.png` | ⏳ TODO | High | `docs/assets/figures/diffusion_vs_policy_gradient.png.placeholder` |

**Note:** MkDocs will still build successfully with placeholder files (broken image links). Replace with actual figures before final deployment.

**Quick Alternative:** Use Mermaid diagrams (embedded in Markdown) for system architecture instead of PNG files. Mermaid is already configured in `mkdocs.yml`.

---

## Workflow for Updating Placeholders

### Step 1: Collect Data

Run experiments and measurements according to instructions above.

### Step 2: Update Files

Use find-and-replace to update placeholders:

```bash
# Example: Replace MOCAP_HZ with actual value
find docs -type f -name "*.md" -exec sed -i 's/{{MOCAP_HZ}}/120/g' {} +

# Or use your IDE's find-and-replace across files
```

### Step 3: Verify Changes

```bash
# Check for remaining placeholders
grep -r "{{" docs/ --include="*.md"

# Count remaining placeholders
grep -r "{{" docs/ --include="*.md" | wc -l
```

### Step 4: Build and Review

```bash
mkdocs serve
# Open browser to http://127.0.0.1:8000
# Verify all placeholders filled and rendering correctly
```

### Step 5: Deploy

```bash
git add .
git commit -m "Fill measurement placeholders with experimental data"
git push
# GitHub Actions auto-deploys to Pages
```

---

## Priority Filling Order

**Phase 1 (Essential for PI Review):**
1. Contact information: `{{YOUR_EMAIL}}`
2. PDF/Demo links: `{{PDF_PORTFOLIO_URL}}`, `{{DEMO_URL}}`
3. Core performance metrics: `{{MOCAP_HZ}}`, `{{END_TO_END_LATENCY_MS}}`, `{{FRAME_PROCESS_MS}}`
4. High-level results: `{{SUCCESS_AVG_FULL}}`, `{{SUCCESS_AVG_LORA}}`
5. System architecture figures

**Phase 2 (Detail Enhancement):**
1. Training efficiency metrics: `{{PARAMS_FULL}}`, `{{TIME_FULL}}`, `{{MEM_FULL}}`
2. Hyperparameter values: `{{LEARNING_RATE}}`, `{{BATCH_SIZE}}`, `{{LORA_RANK}}`
3. Software versions: `{{PYTORCH_VERSION}}`, `{{CUDA_VERSION}}`
4. Learning curve figures

**Phase 3 (Optional Polish):**
1. Extended metrics: `{{DATA_LOSS_RATE}}`, `{{CPU_USAGE}}`, `{{CONTINUOUS_HOURS}}`
2. Per-task-suite breakdowns: `{{SUCCESS_SPATIAL_FULL}}`, etc.
3. Demo GIFs and videos

---

## Extracting Data from Logs

### Weights & Biases

```python
import wandb

api = wandb.Api()
run = api.run("username/project/run_id")

# Extract metrics
history = run.scan_history(keys=["success_rate", "step"])
final_success = [row["success_rate"] for row in history][-1]

print(f"Final success rate: {final_success:.1f}%")
```

### Model Parameters

```python
import torch

model = ...  # Load your model
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable_params / 1e6:.1f}M")
print(f"Total: {total_params / 1e6:.1f}M")
```

### GPU Memory

```bash
# Monitor during training
watch -n 1 nvidia-smi

# Or log to file
while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> gpu_mem.log
    sleep 1
done

# Find peak
sort -n gpu_mem.log | tail -1
```

---

## Status Summary

**Total Text Placeholders:** ~50  
**Files with Text Placeholders:** 4 (README.md, docs/index.md, teleop-mocap.md, rdt-libero.md, rlinf-rdt-integration.md)  
**Files with ZERO Text Placeholders:** groot-libero-reproduction.md (Project 2) ✅  
**Files with Partial Fill:**
- rdt-libero.md (Project 3) - core numbers filled, configs pending
- rlinf-rdt-integration.md (Project 4) - interface contracts filled, dimensions/configs pending  
**Visual Assets:** 15 pending (14 PNG, 1 GIF)

**Recommended Action:**  
Start with Phase 1 priorities to make the portfolio PI-ready for initial review, then progressively fill Phase 2 and 3 details as data becomes available.
