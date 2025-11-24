# Disaggregated vLLM Benchmark Suite

This directory contains a Docker Compose configuration and a benchmark script for testing disaggregated vLLM inference with separate prefill and decode containers on XPU devices.


## Overview

### Objective

The goal of this setup is to demonstrate and benchmark **disaggregated inference** using vLLM, where:

- **Prefill containers** handle prompt processing (KV cache generation)
- **Decode containers** handle token generation (autoregressive decoding)
- **Router** coordinates request routing and KV cache transfer between prefill and decode stages

This generally follows the [llm-d XPU Guide for PD-Disaggregation](https://github.com/llm-d/llm-d/blob/v0.3.1/guides/pd-disaggregation/README.xpu.md), and is the equivalent of the configuration captured by [values_xpu.yaml](https://github.com/llm-d/llm-d/blob/v0.3.1/guides/pd-disaggregation/ms-pd/values_xpu.yaml)

The benchmark script (`benchmark.py`) automates testing of this disaggregated setup by:
- Discovering running containers automatically
- Sending requests through the router with proper headers
- Measuring throughput and KV transfer metrics
- Supporting both synchronous and asynchronous request patterns

## Docker Setup

### Prerequisites

1. **Intel XPU** (Data Center GPU Max Series)
2. **Docker** with Compose v2.x
3. **Network Access** to Docker registries and Hugging Face

### Environment Configuration

Before launching the containers, set up the required environment variables:

```bash
# Proxy configuration (adjust for your network)
export NO_PROXY=172.26.47.132,172.26.47.133,192.168.58.0/32,172.26.44.0/22,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,habana-labs.com,.habana-labs.com,fake-validation-service.kubevirt.svc,virt-api.kubevirt.svc,kubevirt-operator-webhook.kubevirt.svc,cert-manager-webhook.cert-manager.svc,
export HTTPS_PROXY=http://proxy-dmz.intel.com:912
export HTTP_PROXY=http://proxy-dmz.intel.com:912
export http_proxy=http://proxy-dmz.intel.com:912

# Hugging Face token for model downloads
export HUGGING_FACE_HUB_TOKEN=hf_YourHuggingFaceToken
```

**Note:** Replace `hf_YourHuggingFaceToken` with your actual Hugging Face token from https://huggingface.co/settings/tokens

### Launching the Containers

```bash
# Stop any existing containers and start fresh
docker compose down && docker compose up -d
```

### Verifying the Setup

Check that all containers are running:

```bash
docker ps --filter name=xpu_pd
```

You should see:
- `xpu_pd-router-1` (port 8000)
- `xpu_pd-prefill1-1` (port 8001)
- `xpu_pd-decode-1` (port 8002)

Check container logs for startup completion:

```bash
# Router logs
docker logs xpu_pd-router-1

# Prefill container logs
docker logs xpu_pd-prefill1-1

# Decode container logs
docker logs xpu_pd-decode-1
```

Look for "Application startup complete" in each container's logs.

## Benchmark Script

### Setup

#### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv ~/venv/xpu_pd

# Activate virtual environment
source ~/venv/xpu_pd/bin/activate
```

#### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Basic Usage

#### Simple Benchmark (10 requests, 5 second gaps)

```bash
python benchmark.py
```

#### Custom Request Count

```bash
./benchmark.py --request-count 20
```

#### Faster Request Rate

```bash
./benchmark.py --request-gap 2.0
```

#### Async Mode (Concurrent Requests)

```bash
./benchmark.py --enable-async-requests --request-gap 1.0
```

**Warning:** Async mode launches requests concurrently which can overwhelm single-XPU setups. Use with caution.

#### Custom Prompt

```bash
./benchmark.py --prompt "Explain quantum computing in simple terms."
```


### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--router` | `localhost:8000` | Router address and port |
| `--request-count` | `10` | Number of requests to send |
| `--scheduler-strategy` | `round-robin` | Prefill container selection strategy |
| `--prompt` | `"Describe how an LLM works."` | Prompt text for requests |
| `--request-gap` | `5.0` | Time gap between requests (seconds) |
| `--print-response` | `False` | Print full response content |
| `--enable-async-requests` | `False` | Enable async/concurrent mode |

### Sample Output

```
================================================================================
DISAGGREGATED vLLM BENCHMARK
================================================================================

Step 1: Discovering containers...
container={'name': 'xpu_pd-router-1', 'status': 'Up 5 minutes'} .. name='xpu_pd-router-1'
container={'name': 'xpu_pd-prefill1-1', 'status': 'Up 5 minutes'} .. name='xpu_pd-prefill1-1'
container={'name': 'xpu_pd-decode-1', 'status': 'Up 5 minutes'} .. name='xpu_pd-decode-1'
✓ Router: ROUTER(xpu_pd-router-1): localhost:8000
✓ Prefill containers: 1
  - PREFILL(xpu_pd-prefill1-1): localhost:8001
✓ Decode containers: 1
  - DECODE(xpu_pd-decode-1): localhost:8002

Step 2: Initializing scheduler (strategy: round-robin)...
✓ Scheduler ready with 1 prefill container(s)

Step 3: Starting benchmark at 2025-11-24 10:30:15.123456
  - Request count: 10
  - Request gap: 5.0s
  - Prompt: 'Describe how an LLM works.'

Step 4: Sending requests...sync
  [1/10] Using xpu_pd-prefill1-1 -> ✓ (150 tokens, took 2.34s)
  [2/10] Using xpu_pd-prefill1-1 -> ✓ (148 tokens, took 2.41s)
  [3/10] Using xpu_pd-prefill1-1 -> ✓ (152 tokens, took 2.38s)
  [4/10] Using xpu_pd-prefill1-1 -> ✓ (149 tokens, took 2.45s)
  [5/10] Using xpu_pd-prefill1-1 -> ✓ (151 tokens, took 2.37s)
  [6/10] Using xpu_pd-prefill1-1 -> ✓ (147 tokens, took 2.42s)
  [7/10] Using xpu_pd-prefill1-1 -> ✓ (150 tokens, took 2.39s)
  [8/10] Using xpu_pd-prefill1-1 -> ✓ (153 tokens, took 2.36s)
  [9/10] Using xpu_pd-prefill1-1 -> ✓ (148 tokens, took 2.44s)
  [10/10] Using xpu_pd-prefill1-1 -> ✓ (149 tokens, took 2.40s)

Step 5: Waiting for all requests to complete...
✓ Done

Step 6: Analyzing container logs...

--------------------------------------------------------------------------------
PREFILL CONTAINERS:
--------------------------------------------------------------------------------

xpu_pd-prefill1-1:
  Avg Prompt Throughput: 245.67 tokens/s
  Avg Generation Throughput: 0.2 tokens/s
  Samples: 10

--------------------------------------------------------------------------------
DECODE CONTAINERS:
--------------------------------------------------------------------------------

xpu_pd-decode-1:
  Avg Prompt Throughput: 248.34 tokens/s
  Avg Generation Throughput: 62.15 tokens/s
  Samples: 10

  KV Transfer Metrics:
    Successful Transfers: 10
    Avg Transfer Time: 3.45 ms
    Transfer Throughput: 1823.56 MB/s

--------------------------------------------------------------------------------
SUMMARY:
--------------------------------------------------------------------------------
Total Requests: 10
Successful: 10
Failed: 0
Success Rate: 100.0%

```


## Architecture Details

### Container Configuration

- **Router** (`llm-d-routing-sidecar:v0.3.0`).
  - Port: 8000
  - Routes requests to prefill containers
  - Uses `nixlv2` connector for KV cache coordination
  - Points to decode container at port 8002

- **Prefill1** (`llm-d-xpu:v0.3.1`)
  - Port: 8001
  - NIXL port: 5556
  - Role: `kv_both` (can send/receive KV cache)
  - Processes prompts and generates KV cache

- **Decode** (`llm-d-xpu:v0.3.1`)
  - Port: 8002
  - NIXL port: 5555
  - Role: `kv_both` (can send/receive KV cache)
  - Receives KV cache and performs generation

### Network Configuration

All containers use **host networking** mode for optimal performance and to enable NIXL (UCX-based) communication between prefill and decode containers.

This is necessitated by the [requirements of the `llm-d-routing-sidecar` container](https://github.com/llm-d/llm-d/blob/v0.3.1/guides/pd-disaggregation/README.xpu.md).

### Model

Default model: **Qwen/Qwen3-0.6B** (small model for testing)

To use a different model, modify the `docker-compose.yml` file and update the `--model` argument in both prefill and decode containers.

## Troubleshooting

### Containers Not Starting

1. Check XPU availability:
   ```bash
   ls -l /dev/dri/renderD128
   ```

2. Verify Docker can access GPU:
   ```bash
   docker run --rm --device=/dev/dri ghcr.io/llm-d/llm-d-xpu:v0.3.1 ls -l /dev/dri
   ```

3. Check environment variables are set (especially `HUGGING_FACE_HUB_TOKEN`)

### Benchmark Script Errors

1. **Port extraction fails**: Containers may not be fully started. Wait 30-60 seconds after `docker compose up -d`

2. **Connection refused**: Verify containers are running and ports are correct

3. **Timeout errors in async mode**: Single XPU may be overloaded. Use sync mode or reduce `--request-count`

### Low Throughput

1. Check XPU utilization:
   ```bash
   xpu-smi dump -m 0,18
   ```

2. Verify KV transfers are occurring (check decode container logs for KV transfer metrics)

3. Ensure no other processes are using the XPU

## Advanced Usage

### Multiple Prefill Containers

To add more prefill containers, modify `docker-compose.yml` using `extends`:

```yaml
prefill2:
  extends: prefill1
  container_name: xpu_pd-prefill2-1
  environment:
    - VLLM_PORT=8003
    - NIXL_PORT=5557
```

Update the ports to avoid conflicts. The benchmark script will automatically discover and round-robin between all prefill containers.

### Custom Scheduling Strategy

Currently only `round-robin` is supported. Future versions may include:
- Least-loaded scheduling
- Random selection
- Weighted distribution

## References

### Project Documentation
- [llm-d GitHub Repository](https://github.com/llm-d/llm-d)
- [llm-d XPU Prefill-Decode Disaggregation Guide](https://github.com/llm-d/llm-d/blob/v0.3.1/guides/pd-disaggregation/README.xpu.md)
- [llm-d Kubernetes Multi-Service Example](https://github.com/llm-d/llm-d/blob/v0.3.1/guides/pd-disaggregation/ms-pd/values_xpu.yaml)
- [llm-d Routing Sidecar](https://github.com/llm-d/llm-d/tree/v0.3.1/llm-d-routing-sidecar)

### Related Technologies
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Intel Extension for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch)
- [Intel XPU Documentation](https://dgpu-docs.intel.com/)

### Communication Layer
- [UCX (Unified Communication X)](https://github.com/openucx/ucx) - High-performance communication framework used by NIXL
