# Flashcrowd Workload Configuration

This folder contains configuration files for generating and executing flashcrowd workloads using µBench.

## Files

- **TrafficGeneratorParameters.json** - Configuration for generating flashcrowd workload files
- **RunnerParameters.json** - Configuration for executing flashcrowd workloads
- **workload_example.json** - Example of generated flashcrowd workload format
- **variants/** - Different flashcrowd intensity configurations

## Quick Start

### 1. Generate Flashcrowd Workload

Inside the µBench container:

```bash
python3 Benchmarks/TrafficGenerator/RunTrafficGen.py -c /path/to/TrafficGeneratorParameters.json
```

### 2. Execute Flashcrowd

```bash
python3 Benchmarks/Runner/Runner.py -c /path/to/RunnerParameters.json
```

### 3. Monitor

```bash
kubectl get work -A
kubectl get pods -A -w
```

## Customization

Adjust `mean_interarrival_time` in TrafficGeneratorParameters.json:
- **10ms** - Extreme flashcrowd (~100 req/s)
- **50ms** - Intense flashcrowd (~20 req/s)
- **100ms** - Moderate flashcrowd (~10 req/s)
- **200ms** - Light flashcrowd (~5 req/s)
