# llm-d Feature Summary

A summary of the main features in llm-d, based on the guides directory. Guides
are "well-lit paths" — documented, tested, and benchmarked recipes to serve LLMs
with best-practices for high performance — organized by category.

| # | Feature (llm-d) | Category | Description | Comparable feature in NVIDIA Dynamo |
|---|---------|----------|-------------|-------------------------------------|
| 1 | [**Optimized Baseline**](./guides/optimized-baseline/README.md) | Intelligent Routing | Deploy vLLM with prefix-cache and load-aware routing enabled by the llm-d EPP (Endpoint Picker). | **KV Cache Aware Routing** (Router) — routes on worker load + KV-cache overlap to cut redundant prefill (~2× faster TTFT). |
| 2 | [**Predicted Latency-Based Routing**](./guides/predicted-latency-routing/README.md) | Intelligent Routing | Enhances baseline with real-time request-latency predictions via a live-trained XGBoost model, rather than heuristic utilization metrics (queue depth, KV-cache use). | *Partial* — Router cost/latency-aware policies (WSPT scheduling) and the **SLA Planner/Profiler** target TTFT/ITL, but there is no live-trained predictive latency model equivalent. |
| 3 | [**Multi-Model Routing**](./guides/multi-model-routing/README.md) | Intelligent Routing | Deploys the Inference Payload Processor (IPP) to serve multiple LLMs behind a single OpenAI-compatible Gateway endpoint; also supports LoRA adapter routing. | **Multi-model serving** via the Frontend — multiple models registered behind one OpenAI-compatible endpoint. |
| 4 | [**Precise Prefix Cache Routing**](./guides/precise-prefix-cache-routing/README.md) | KV-Cache Management | Enhances baseline with precise global indexing of the vLLM KV-cache state. | **KV-Aware Routing with global KV indexer** (KV events + standalone indexer) — precise cluster-wide KV block tracking. |
| 5 | [**Tiered Prefix Cache**](./guides/tiered-prefix-cache/README.md) | KV-Cache Management | Offloads KV caches beyond accelerator memory (to CPU/disk), increasing KV working-set size for multi-turn request patterns. | **KV Block Manager (KVBM)** — offloads KV cache GPU → CPU → SSD → remote/object storage (S3/Azure); LMCache/FlexKV integrations. |
| 6 | [**Prefill/Decode Disaggregation**](./guides/pd-disaggregation/README.md) | Serving Large Models | Splits inference into specialized prefill and decode instances, improving throughput and QoS for medium/large models (e.g. `gpt-oss-120b`). | **Disaggregated Serving** — independently scalable prefill and decode GPU pools (NIXL-based KV transfer). |
| 7 | [**Wide Expert-Parallelism (Wide-EP)**](./guides/wide-ep-lws/README.md) | Serving Large Models | Deploys large MoE models (e.g. `DeepSeek-R1`) across multiple nodes via DP/EP config, boosting KV-cache space and throughput. | **Wide Expert Parallelism (WideEP)** + **Multinode Deployment** — spreads MoE experts across many GPUs (e.g. 64-way EP) with Grove/KAI-Scheduler. |
| 8 | [**Flow Control**](./guides/flow-control/README.md) | Operational Excellence | Intelligent request queuing for multi-tenant deployments and managing traffic spikes. | **Router queue scheduling** — Deficit Round Robin (DRR) arbitration + priority scheduling / strict-priority tiers across policy classes. |
| 9 | [**Workload Autoscaling**](./guides/workload-autoscaling/README.md) | Operational Excellence | Proactive, SLO-aware autoscaling using queue depth, in-flight counts, and KV-cache pressure to add capacity before latency degrades. | **Planner** — SLA-driven autoscaler that profiles workloads and right-sizes prefill/decode pools to meet TTFT/ITL targets. |
| 10 | [**Rollouts**](./guides/rollouts/README.md) | Operational Excellence | Incremental rollouts of LoRA adapters, base models, and model-server versions using traffic splitting and gradual deployment. | *Partial* — Kubernetes operator (DGDR) handles deployment/updates; **ModelExpress** (GPU-to-GPU weight streaming) speeds new-replica cold starts. No dedicated traffic-splitting rollout guide. |
| 11 | [**Agentic Serving**](./guides/agentic-serving/README.md) | Workloads | Serves long, multi-turn, tool-using agentic workloads (e.g. coding agents) by composing prefix-aware routing, KV offloading, and P/D disaggregation. | **Agentic inference** — per-request `agent_hints` (priority, expected output length) + session metadata, composing KV routing / offload / disaggregation. |
| 12 | [**Multimodal Serving**](./guides/multimodal-serving/README.md) | Workloads | Deploys multimodal (image/audio/video) serving via aggregated routing or dedicated encode-disaggregation topologies. | **Multimodal Serving** — native multimodal with embedding cache; also video generation (FastVideo / SGLang Diffusion). |
| 13 | [**Asynchronous Processing**](./guides/asynchronous-processing/README.md) | Experimental | Processes inference requests asynchronously via a queue-based architecture; ideal for latency-insensitive batch or "slack"-capacity workloads. | *No direct equivalent* — no queue-based async/slack-capacity path found in docs. |
| 14 | [**Batch Gateway**](./guides/batch-gateway/README.md) | Experimental | Submit/track/manage large-scale batch inference jobs via an OpenAI-compatible Batch API, coexisting with interactive workloads. | *No direct equivalent* — no OpenAI-compatible Batch API found in docs. |
| 15 | [**Encode Disaggregation**](./guides/multimodal-serving/e-disaggregation/README.md) | Experimental | Offloads multimodal encoding to dedicated workers via E/PD or E/P/D topologies, freeing prefill/decode resources for text. | **Disaggregated encode/prefill/decode** for multimodal (E/PD, E/P/D) with embedding cache. |
| 16 | [**No-Kubernetes Deployment**](./guides/no-kubernetes-deployment/README.md) | Deployment | Runs the routing stack (EPP, Envoy, vLLM workers) without a cluster — EPP gets endpoints from a YAML file via the file-discovery plugin. | **Local installation** with `--discovery-backend file` — file-based discovery, no Kubernetes/NATS (ZMQ event plane). |
| 17 | [**verl Integration (RL)**](./guides/rl/verl-integration.md) | RL Training | Replaces verl's RLHF/GRPO rollout routing with the llm-d scheduler, bringing production scoring, filtering, and flow-control to RL training. | *No direct equivalent* — no RL-training / verl rollout-routing integration found in docs. |

## Notes

- **Items 1–12** are the primary "well-lit path" features (Intelligent Routing,
  KV-Cache Management, Serving Large Models, Operational Excellence, Workloads),
  highlighted in `guides/README.md`.
- **Items 13–15** are labeled **Experimental**.
- **Items 16–17** exist as guide directories (`no-kubernetes-deployment`, `rl`)
  but are not listed in the top-level guides README index.
- The guides README references a **Rollouts** guide (`./rollouts/README.md`), but
  that directory does not currently exist in the repo — a possible broken link.

## NVIDIA Dynamo comparison

- The **Comparable feature in NVIDIA Dynamo** column maps each llm-d feature to
  the closest Dynamo capability, based on the Dynamo docs (`~/dynamo/docs`).
- *Partial* means Dynamo has related functionality but not a direct 1:1 match.
- *No direct equivalent* means no matching feature was found in the Dynamo docs
  (notably: async/slack-capacity processing, an OpenAI-compatible Batch API, and
  an RL-training rollout-routing integration like llm-d's verl integration).
- Dynamo also ships capabilities without a dedicated llm-d guide analog, e.g.
  **Request Migration** (in-flight fault tolerance), **speculative decoding**,
  **tool calling / reasoning**, and **ModelExpress** weight streaming.
