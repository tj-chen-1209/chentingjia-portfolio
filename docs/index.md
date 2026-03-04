# Tingjia Chen - Embodied AI Portfolio

**简介：** 机器人学习与具身智能研究工程师，专注于端到端系统搭建、大模型微调和可复现实验流程。本站记录工程细节、性能指标和技术权衡。

---

## About

I am a research engineer specializing in **Embodied AI**, with hands-on experience building reproducible robot learning systems from low-level infrastructure to foundation model fine-tuning.

**Research Interests:**

- Robot manipulation and teleoperation systems
- Vision-language-action model adaptation and evaluation
- Real-time sensor integration and control loops
- Reproducible benchmarking pipelines

## What I Can Do Immediately in a Lab

**System Integration & Deployment**

- Set up hardware-software pipelines (motion capture, sensors, robot control) with documented interfaces and timing analysis
- Debug multi-component systems (networking, threading, protocol parsing) with systematic failure mode analysis
- Build ROS/ROS2 nodes with parameter management, lifecycle handling, and clean message interfaces

**Reproducible Experimentation**

- Implement training/evaluation pipelines with versioned configs, deterministic seeding, and automated logging
- Establish experiment tracking infrastructure (W&B, MLflow) with checkpoint management and artifact versioning
- Write clear technical documentation with system diagrams, performance metrics, and placeholder tables for missing data

**Model Training & Adaptation**

- Fine-tune foundation models (full-parameter, LoRA, adapter methods) with distributed training (DeepSpeed, FSDP)
- Integrate models into standard benchmarks (LIBERO, RLBench) with observation/action space handling
- Run ablation studies with multi-seed evaluation and statistical analysis

## Experience

### Standard - Embodiment Team
**Research Intern | 2024**

Built motion capture streaming infrastructure and humanoid robot teleoperation systems for demonstration data collection.

### PKU Lingchu Lab
**Research Intern | 2024**

Developed fine-tuning and evaluation pipelines for large-scale robot foundation models; contributed to ICML submission.

## Featured Projects

<div class="grid cards" markdown>

-   :material-network-outline: **OptiTrack ROS2 Streaming Node**

    ---

    Real-time motion capture data bridge: NatNet SDK → ROS2 node converting 6DoF poses to standard messages at {{MOCAP_HZ}}Hz. Solved async callback bridging, dynamic ID mapping, and multi-thread safety.

    **Stack:** C++17, ROS2 Humble, NatNet SDK 4.2/4.3, CMake

    [:octicons-arrow-right-24: Technical Details](projects/teleop-mocap.md)

-   :material-check-circle: **GR00T-N1.6 SFT on LIBERO**

    ---

    Fine-tuned NVIDIA GR00T-N1.6 (3.2B params) achieving 97.8% success on 40 LIBERO tasks. Reproduced official benchmark within ±1.6%. Config-driven pipeline with DeepSpeed, parallel eval, and batch checkpoint comparison.

    **Stack:** PyTorch, DeepSpeed, LIBERO, LeRobot, uv

    [:octicons-arrow-right-24: Technical Details](projects/groot-libero-reproduction.md)

-   :material-robot-outline: **RDT SFT & Evaluation on LIBERO**

    ---

    Full-parameter and LoRA fine-tuning comparison on LIBERO. Achieved 92% (object), 94% (goal), 76% (spatial). Found LoRA ceiling at ~20% vs. 76%+ full-parameter. Built profiler identifying 375ms inference (69% eval time).

    **Stack:** PyTorch, DeepSpeed, LIBERO, peft, T5-XXL, SigLIP

    [:octicons-arrow-right-24: Technical Details](projects/rdt-libero.md)

-   :material-link-variant: **RLinf × RDT Integration**

    ---

    Integrated RDT into RLinf RL framework for LIBERO. Extended env to extract joint_states (9D), added Experience field, built RDT ActionModel wrapper. Forward inference loop complete; RL training blocked by diffusion-policy log_prob mismatch.

    **Stack:** RLinf Framework, RDT, LIBERO, peft, diffusers

    [:octicons-arrow-right-24: Technical Details](projects/rlinf-rdt-integration.md)

-   :material-chart-timeline-variant: **ResMerge (ICML Project)**

    ---

    Residual policy merging framework for continual learning with parameter efficiency. Contributed scientific figures (method pipeline, teaser comparison), assisted experiments (consistency analysis, τ ablation), supported reproducibility infrastructure.

    **Stack:** PyTorch, peft (LoRA), Continual Learning

    [:octicons-arrow-right-24: Technical Details](projects/icml-resmerge.md)

</div>

## Technical Skills

**Languages:** C++17, Python 3.10+, CMake, Bash

**ML Frameworks:** PyTorch, HuggingFace Transformers, DeepSpeed, PEFT (LoRA)

**Robotics:** ROS2 (Humble/Iron), MoveIt2, Motion capture systems (OptiTrack), Inverse kinematics

**Benchmarks & Simulation:** LIBERO, RLBench, PyBullet, MuJoCo

**Infrastructure:** Git, Docker, Weights & Biases, Linux (Ubuntu 22.04), Multi-GPU training

## Philosophy

Research engineering in robotics requires:

1. **Quantitative Verification** - Measure latency, throughput, success rates; report with statistical rigor
2. **Reproducibility First** - Versioned dependencies, deterministic seeding, documented failure modes
3. **Honest Failure Analysis** - Document what didn't work and why; technical problems over polished narratives

This portfolio demonstrates my approach to building reliable, well-documented systems.

---

**Portfolio Links:**

- [📄 PDF Portfolio]({{{PDF_PORTFOLIO_URL}}})
- [📋 1-Page Research Highlights]({{{ONE_PAGE_URL}}})
- [🎥 Demo Videos]({{{DEMO_URL}}})

**Contact:** [GitHub Issues](https://github.com/chentingjia/chentingjia-portfolio/issues) | Email: {{YOUR_EMAIL}}

**Note:** This portfolio contains public technical documentation only. All proprietary code and internal implementation details remain private.
