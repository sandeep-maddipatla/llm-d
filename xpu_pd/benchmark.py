#!/usr/bin/env python3
"""
Benchmark script for disaggregated vLLM inference testing.
Tests router-based prefill/decode separation with configurable scheduling.
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class ContainerInfo:
    """Information about a discovered container."""
    def __init__(self, name: str, host: str, port: int, role: str):
        self.name = name
        self.host = host
        self.port = port
        self.role = role  # 'router', 'prefill', or 'decode'

    def __repr__(self):
        return f"{self.role.upper()}({self.name}): {self.host}:{self.port}"


class ContainerDiscovery:
    """Discovers running vLLM and router containers."""

    @staticmethod
    def get_running_containers() -> List[Dict[str, str]]:
        """Get list of running docker containers with xpu_pd prefix."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=xpu_pd', '--format', '{{.Names}}\t{{.Status}}'],
                capture_output=True, text=True, check=True
            )
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    name, status = line.split('\t')
                    if 'Up' in status:
                        containers.append({'name': name, 'status': status})
            return containers
        except subprocess.CalledProcessError:
            return []

    @staticmethod
    def extract_port_from_logs(container_name: str) -> Optional[int]:
        """Extract the port number from container logs."""
        try:
            result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True, text=True, check=True
            )
            logs = result.stdout + result.stderr

            # Check if application started successfully
            if 'Application startup complete' not in logs:
                print(f"  ⚠ {container_name}: Application has not completed startup")
                return None

            # Find the position of "Application startup complete"
            startup_pos = logs.rfind('Application startup complete')
            logs_after_startup = logs[startup_pos:]

            # Check for errors after startup
            if re.search(r'\berror\b', logs_after_startup, re.IGNORECASE):
                print(f"  ✗ {container_name}: Errors detected after startup")
                # Show first error line
                error_match = re.search(r'.*\berror\b.*', logs_after_startup, re.IGNORECASE)
                if error_match:
                    print(f"    Error: {error_match.group(0).strip()}")
                return None

            # Look for "Starting vLLM API server N on http://0.0.0.0:PORT"
            port_match = re.search(r'Starting vLLM API server \d+ on http://[^:]+:(\d+)', logs)
            if port_match:
                return int(port_match.group(1))

            print(f"  ⚠ {container_name}: Could not extract port from logs")
        except Exception as e:
            print(f"  ✗ {container_name}: Failed to read logs - {e}")
        return None

    @staticmethod
    def discover() -> Tuple[Optional[ContainerInfo], List[ContainerInfo], List[ContainerInfo]]:
        """
        Discover running containers and categorize them.
        Returns: (router, prefill_containers, decode_containers)
        """
        containers = ContainerDiscovery.get_running_containers()

        router = None
        prefills = []
        decodes = []

        for container in containers:
            name = container['name']
            print(f'{container=} .. {name=}')

            if 'router' in name:
                # Router typically runs on port 8000
                router = ContainerInfo(name, 'localhost', 8000, 'router')
            elif 'prefill' in name:
                port = ContainerDiscovery.extract_port_from_logs(name)
                if port:
                    prefills.append(ContainerInfo(name, 'localhost', port, 'prefill'))
            elif 'decode' in name:
                port = ContainerDiscovery.extract_port_from_logs(name)
                if port:
                    decodes.append(ContainerInfo(name, 'localhost', port, 'decode'))

        return router, prefills, decodes


class Scheduler:
    """Schedules prefill container selection based on strategy."""

    def __init__(self, prefill_containers: List[ContainerInfo], strategy: str = 'round-robin'):
        self.prefill_containers = prefill_containers
        self.strategy = strategy
        self.current_index = 0

    def get_next_prefill(self) -> ContainerInfo:
        """Get the next prefill container based on the scheduling strategy."""
        if not self.prefill_containers:
            raise ValueError("No prefill containers available")

        if self.strategy == 'round-robin':
            container = self.prefill_containers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.prefill_containers)
            return container
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")


class ThroughputAnalyzer:
    """Analyzes container logs for throughput metrics."""

    @staticmethod
    def get_logs_since(container_name: str, since_time: str) -> str:
        """Get container logs since a specific timestamp."""
        try:
            result = subprocess.run(
                ['docker', 'logs', '--since', since_time, container_name],
                capture_output=True, text=True, check=True
            )
            return result.stdout + result.stderr
        except Exception as e:
            print(f"  Warning: Failed to get logs for {container_name}: {e}")
            return ""

    @staticmethod
    def extract_throughput_metrics(logs: str) -> List[Dict[str, float]]:
        """Extract throughput metrics from vLLM logs."""
        metrics = []
        pattern = r'Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s'

        for match in re.finditer(pattern, logs):
            metrics.append({
                'prompt_throughput': float(match.group(1)),
                'generation_throughput': float(match.group(2))
            })

        return metrics

    @staticmethod
    def get_kv_transfer_metrics(logs: str) -> Optional[Dict[str, float]]:
        """Extract KV transfer metrics from decode container logs."""
        pattern = r'KV Transfer metrics: Num successful transfers=(\d+).*?Avg xfer time \(ms\)=([\d.]+).*?Throughput \(MB/s\)=([\d.]+)'
        matches = re.finditer(pattern, logs)

        all_metrics = []
        for match in matches:
            all_metrics.append({
                'num_transfers': int(match.group(1)),
                'avg_xfer_time_ms': float(match.group(2)),
                'throughput_mbps': float(match.group(3))
            })

        if not all_metrics:
            return None

        # Aggregate metrics: sum transfers, average times and throughput
        total_transfers = sum(m['num_transfers'] for m in all_metrics)
        avg_xfer_time = sum(m['avg_xfer_time_ms'] for m in all_metrics) / len(all_metrics)
        avg_throughput = sum(m['throughput_mbps'] for m in all_metrics) / len(all_metrics)

        return {
            'num_transfers': total_transfers,
            'avg_xfer_time_ms': avg_xfer_time,
            'throughput_mbps': avg_throughput
        }

    @staticmethod
    def analyze_and_print_container_metrics(container_name: str, log_since: str):
        """Analyze and print metrics for a single container."""
        logs = ThroughputAnalyzer.get_logs_since(container_name, log_since)
        metrics = ThroughputAnalyzer.extract_throughput_metrics(logs)
        kv_metrics = ThroughputAnalyzer.get_kv_transfer_metrics(logs)

        if metrics:
            # Get the last few metrics
            recent = metrics[-3:] if len(metrics) >= 3 else metrics
            avg_prompt = sum(m['prompt_throughput'] for m in recent) / len(recent)
            avg_gen = sum(m['generation_throughput'] for m in recent) / len(recent)

            print(f"\n{container_name}:")
            print(f"  Avg Prompt Throughput: {avg_prompt:.2f} tokens/s")
            print(f"  Avg Generation Throughput: {avg_gen:.2f} tokens/s")
            print(f"  Samples: {len(metrics)}")

            if kv_metrics:
                print(f"\n  KV Transfer Metrics:")
                print(f"    Successful Transfers: {kv_metrics['num_transfers']}")
                print(f"    Avg Transfer Time: {kv_metrics['avg_xfer_time_ms']:.2f} ms")
                print(f"    Transfer Throughput: {kv_metrics['throughput_mbps']:.2f} MB/s")
        else:
            print(f"\n{container_name}: No throughput metrics found")


async def send_chat_completion_async(
    session,
    router_url: str,
    prefiller_url: str,
    prompt: str,
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 100
) -> Dict:
    """Send an async chat completion request to the router with prefiller specification."""
    import aiohttp

    url = f"{router_url}/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'x-prefiller-host-port': prefiller_url
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7
    }

    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
        return await response.json()


def send_chat_completion_sync(
    router_url: str,
    prefiller_url: str,
    prompt: str,
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 100
) -> Dict:
    """Send a synchronous chat completion request to the router with prefiller specification."""
    import requests

    url = f"{router_url}/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'x-prefiller-host-port': prefiller_url
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    return response.json()


def send_requests_sync(args, router_url: str, scheduler: Scheduler) -> List[Dict]:
    """Send requests synchronously - wait for each to complete before sending next."""
    results = []

    for i in range(args.request_count):
        prefill_container = scheduler.get_next_prefill()
        prefiller_url = f"http://{prefill_container.host}:{prefill_container.port}"

        print(f"  [{i+1}/{args.request_count}] Using {prefill_container.name} -> ", end='', flush=True)

        request_start = time.time()
        try:
            result = send_chat_completion_sync(router_url, prefiller_url, args.prompt)
            request_time = time.time() - request_start

            if 'error' in result:
                print(f"❌ ERROR: {result['error']} (took {request_time:.2f}s)")
            else:
                tokens = result.get('usage', {}).get('total_tokens', 0)
                print(f"✓ ({tokens} tokens, took {request_time:.2f}s)")

                if args.print_response:
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"    Response: {content}")

            results.append({'success': 'error' not in result, 'result': result})
        except Exception as e:
            request_time = time.time() - request_start
            print(f"❌ FAILED: {e} (took {request_time:.2f}s)")
            results.append({'success': False, 'error': str(e)})

        # Wait before next request (except for last one)
        if i < args.request_count - 1:
            remaining_time = args.request_gap - request_time
            if remaining_time > 0:
                time.sleep(remaining_time)

    return results


async def send_requests_async(args, router_url: str, scheduler: Scheduler) -> List[Dict]:
    """Send requests asynchronously - launch with gaps but don't wait for completion."""
    import aiohttp

    async def send_single_request(session, i, prefill_container):
        """Send a single request and return the result."""
        prefiller_url = f"http://{prefill_container.host}:{prefill_container.port}"
        print(f"  [{i+1}/{args.request_count}] Using {prefill_container.name} -> ", end='', flush=True)

        request_start = time.time()
        try:
            result = await send_chat_completion_async(session, router_url, prefiller_url, args.prompt)
            request_time = time.time() - request_start

            if 'error' in result:
                print(f"❌ ERROR: {result['error']} (took {request_time:.2f}s)")
            else:
                tokens = result.get('usage', {}).get('total_tokens', 0)
                print(f"✓ ({tokens} tokens, took {request_time:.2f}s)")

                if args.print_response:
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"    Response: {content}")

            return {'success': 'error' not in result, 'result': result}
        except Exception as e:
            request_time = time.time() - request_start
            print(f"❌ FAILED: {e} (took {request_time:.2f}s)")
            return {'success': False, 'error': str(e)}

    # Create session and launch requests with staggered start times
    async with aiohttp.ClientSession() as session:
        async def launch_request_with_delay(delay, i, prefill_container):
            """Wait for the specified delay, then send the request."""
            if delay > 0:
                await asyncio.sleep(delay)
            return await send_single_request(session, i, prefill_container)

        tasks = []
        for i in range(args.request_count):
            prefill_container = scheduler.get_next_prefill()
            # Calculate delay: request 0 starts immediately, request 1 after request_gap, etc.
            delay = i * args.request_gap
            task = asyncio.create_task(launch_request_with_delay(delay, i, prefill_container))
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)

    return results


async def run_benchmark(args):
    """Main benchmark execution."""

    print("=" * 80)
    print("DISAGGREGATED vLLM BENCHMARK")
    print("=" * 80)
    print()

    # Step 1: Discover containers
    print("Step 1: Discovering containers...")
    router, prefills, decodes = ContainerDiscovery.discover()

    if not router:
        print("❌ ERROR: No router container found!")
        sys.exit(1)

    print(f"✓ Router: {router}")
    print(f"✓ Prefill containers: {len(prefills)}")
    for pf in prefills:
        print(f"  - {pf}")
    print(f"✓ Decode containers: {len(decodes)}")
    for dc in decodes:
        print(f"  - {dc}")
    print()

    if not prefills:
        print("❌ ERROR: No prefill containers found!")
        sys.exit(1)

    if not decodes:
        print("⚠ WARNING: No decode containers found!")

    # Step 2: Initialize scheduler
    print(f"Step 2: Initializing scheduler (strategy: {args.scheduler_strategy})...")
    scheduler = Scheduler(prefills, args.scheduler_strategy)
    print(f"✓ Scheduler ready with {len(prefills)} prefill container(s)")
    print()

    # Step 3: Record start time
    start_time = datetime.now()
    # Docker accepts relative time like "5m" or ISO format like "2025-11-23T19:42:33"
    # Using a buffer of 1 minute to ensure we catch all logs
    start_timestamp = "1m"
    print(f"Step 3: Starting benchmark at {start_time}")
    print(f"  - Request count: {args.request_count}")
    print(f"  - Request gap: {args.request_gap}s")
    print(f"  - Prompt: '{args.prompt}'")
    print()

    # Step 4: Send requests
    print("Step 4: Sending requests..." + ("async" if args.enable_async_requests else "sync"))
    router_url = f"http://{args.router}"

    if args.enable_async_requests:
        results = await send_requests_async(args, router_url, scheduler)
    else:
        results = send_requests_sync(args, router_url, scheduler)

    print()

    # Step 5: Wait for processing to complete
    print("Step 5: Waiting for all requests to complete...")
    await asyncio.sleep(5)
    print("✓ Done")
    print()

    # Step 6: Analyze logs and report metrics
    print("Step 6: Analyzing container logs...")
    print()

    # Calculate elapsed time and add a buffer
    elapsed_seconds = int((datetime.now() - start_time).total_seconds()) + 10
    log_since = f"{elapsed_seconds}s"

    print("-" * 80)
    print("PREFILL CONTAINERS:")
    print("-" * 80)
    for prefill in prefills:
        ThroughputAnalyzer.analyze_and_print_container_metrics(prefill.name, log_since)

    print()
    print("-" * 80)
    print("DECODE CONTAINERS:")
    print("-" * 80)
    for decode in decodes:
        ThroughputAnalyzer.analyze_and_print_container_metrics(decode.name, log_since)

    print()
    print("-" * 80)
    print("SUMMARY:")
    print("-" * 80)
    successful = sum(1 for r in results if r['success'])
    print(f"Total Requests: {args.request_count}")
    print(f"Successful: {successful}")
    print(f"Failed: {args.request_count - successful}")
    print(f"Success Rate: {successful/args.request_count*100:.1f}%")
    print()


def get_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark disaggregated vLLM inference with router-based scheduling'
    )
    parser.add_argument(
        '--router',
        default='localhost:8000',
        help='Router address (default: localhost:8000)'
    )
    parser.add_argument(
        '--request-count',
        type=int,
        default=10,
        help='Number of requests to send (default: 10)'
    )
    parser.add_argument(
        '--scheduler-strategy',
        default='round-robin',
        choices=['round-robin'],
        help='Scheduling strategy for prefill container selection (default: round-robin)'
    )
    parser.add_argument(
        '--prompt',
        default='Describe how an LLM works.',
        help='Prompt to send in requests (default: "Describe how an LLM works.")'
    )
    parser.add_argument(
        '--request-gap',
        type=float,
        default=5.0,
        help='Time gap between requests in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--print-response',
        action='store_true',
        help='Print the full response content for each request (default: False)'
    )
    parser.add_argument(
        '--enable-async-requests',
        action='store_true',
        help='Enable async requests for concurrent execution (default: False, sync mode)'
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    asyncio.run(run_benchmark(args))


if __name__ == '__main__':
    main()
