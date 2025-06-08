---
id: performance-tuning
title: Performance Optimization
sidebar_position: 4
---

# Performance Optimization

Achieve blazing-fast search performance with these optimization techniques. Learn how to tune Oboyu for your hardware, scale to millions of documents, and maintain responsive search times.

## Performance Baseline

### Measuring Performance

```bash
# Benchmark current performance
oboyu benchmark --db-path ~/indexes/personal.db

# Detailed performance metrics
oboyu query "test search" --metrics

# Performance report
oboyu performance report --days 7
```

### Key Performance Indicators

- **Indexing Speed**: Documents per second
- **Search Latency**: Time to first result
- **Memory Usage**: RAM consumption
- **Disk I/O**: Read/write patterns
- **CPU Usage**: Core utilization

## Hardware Optimization

### Storage Performance

#### SSD vs HDD
```bash
# Check disk performance
oboyu benchmark disk --path ~/.oboyu

# Move index to SSD
mv ~/.oboyu/index.db /ssd/oboyu/
ln -s /ssd/oboyu/index.db ~/.oboyu/index.db
```

**SSD Benefits:**
- 10-50x faster random reads
- Better concurrent access
- Lower latency

#### RAID Configuration
```bash
# RAID 0 for speed (no redundancy)
oboyu config set indexer.db_path /raid0/oboyu/index.db

# RAID 10 for speed + redundancy
oboyu config set indexer.db_path /raid10/oboyu/index.db
```

### Memory Optimization

#### System Memory
```bash
# Check available memory
free -h

# Configure memory limits
export OBOYU_MEMORY_LIMIT=8G

# Preallocate memory
oboyu start --preload-memory 4G
```

#### Memory-Mapped Files
```yaml
# Enable memory mapping
indexer:
  use_mmap: true
  mmap_size: 4GB
```

### CPU Optimization

#### Multi-Core Usage
```bash
# Use all cores
oboyu index ~/Documents --threads $(nproc)

# Leave cores for system
oboyu index ~/Documents --threads $(($(nproc) - 2))

# CPU affinity
taskset -c 0-7 oboyu index ~/Documents
```

#### SIMD Optimization
```bash
# Check CPU features
lscpu | grep -i avx

# Enable AVX optimizations
export OBOYU_USE_AVX2=1
```

## Indexing Performance

### Batch Processing

```yaml
# Optimal batch sizes by RAM
# 4GB RAM
indexer:
  batch_size: 16
  
# 8GB RAM  
indexer:
  batch_size: 32
  
# 16GB+ RAM
indexer:
  batch_size: 64
```

### Parallel Indexing

```bash
# Split large collections
find ~/Documents -type f -name "*.md" | \
  split -l 1000 - batch_

# Index in parallel
ls batch_* | parallel -j 8 'oboyu index --file-list {}'

# Merge indices
oboyu index merge batch_*.db --output main.db
```

### Incremental Strategies

```bash
# Index only changes
oboyu index ~/Documents \
  --update \
  --change-detection mtime

# Use checksums for accuracy
oboyu index ~/Documents \
  --update \
  --change-detection sha256

# Fast change detection
oboyu index ~/Documents \
  --update \
  --change-detection size
```

## Search Performance

### Query Optimization

#### Cache Configuration
```yaml
query:
  cache_enabled: true
  cache_size: 1000  # Number of queries
  cache_ttl: 3600   # Seconds
```

```bash
# Warm up cache
oboyu cache warm --common-queries

# Monitor cache hits
oboyu cache stats
```

#### Index Optimization
```bash
# Optimize vector index
oboyu optimize vector --db-path ~/indexes/personal.db

# Optimize BM25 index
oboyu optimize bm25 --db-path ~/indexes/personal.db

# Full optimization
oboyu optimize all --db-path ~/indexes/personal.db
```

### Result Processing

#### Lazy Loading
```yaml
query:
  lazy_load: true
  initial_load: 10
  load_increment: 10
```

#### Streaming Results
```bash
# Stream results as found
oboyu query "search term" --stream

# Process results in pipeline
oboyu query "search term" --format jsonl | \
  jq '.score > 0.8' | \
  head -20
```

## Scaling Strategies

### Horizontal Scaling

#### Sharding
```bash
# Create shards by date
oboyu index ~/Documents/2023 --db-path ~/indexes/shard-2023.db
oboyu index ~/Documents/2024 --db-path ~/indexes/shard-2024.db

# Query across shards
oboyu query "search term" --shards shard-2023,shard-2024
```

#### Distributed Setup
```yaml
# Master node
cluster:
  role: master
  port: 9200
  
# Worker nodes
cluster:
  role: worker
  master: master.host:9200
```

### Vertical Scaling

#### Large Memory Systems
```yaml
# For 64GB+ systems
indexer:
  buffer_size: 8GB
  cache_size: 16GB
  mmap_size: 32GB
```

#### GPU Acceleration
```bash
# Check GPU availability
oboyu gpu info

# Enable GPU for embeddings
export OBOYU_USE_GPU=1
export CUDA_VISIBLE_DEVICES=0

# GPU batch processing
oboyu index ~/Documents --gpu --gpu-batch-size 256
```

## Network Performance

### Remote Storage

```bash
# NFS optimization
mount -o rsize=1048576,wsize=1048576,noatime nfs-server:/oboyu /mnt/oboyu

# S3-compatible storage
oboyu config set storage.type s3
oboyu config set storage.bucket oboyu-indices
```

### API Performance

```bash
# Enable HTTP/2
oboyu server --http2

# Connection pooling
oboyu server --max-connections 1000

# Response compression
oboyu server --enable-gzip
```

## Monitoring and Profiling

### Performance Monitoring

```bash
# Real-time monitoring
oboyu monitor --interval 1s

# Export metrics
oboyu metrics export --format prometheus

# Performance dashboard
oboyu dashboard --port 8080
```

### Profiling Tools

```bash
# CPU profiling
oboyu profile cpu --duration 60s

# Memory profiling
oboyu profile memory --interval 5s

# I/O profiling
oboyu profile io --trace
```

### Bottleneck Analysis

```bash
# Find slow queries
oboyu query slowlog --threshold 100ms

# Analyze index hotspots
oboyu analyze hotspots --db-path ~/indexes/personal.db

# Resource usage report
oboyu report resources --detailed
```

## Configuration Templates

### Minimum Latency
```yaml
# For &lt;50ms search latency
indexer:
  db_path: /nvme/oboyu/index.db  # NVMe SSD
  cache_size: 4GB
  preload_index: true

query:
  default_mode: bm25  # Fastest
  use_reranker: false
  cache_enabled: true
  connection_pool: 100
```

### Maximum Throughput
```yaml
# For high concurrent users
indexer:
  db_path: /ssd/oboyu/index.db
  read_replicas: 4
  
query:
  max_concurrent: 1000
  queue_size: 10000
  timeout: 5s
  circuit_breaker: true
```

### Balanced Performance
```yaml
# Good performance with quality
indexer:
  db_path: /ssd/oboyu/index.db
  chunk_size: 1024
  use_mmap: true
  
query:
  default_mode: hybrid
  use_reranker: true
  reranker_batch: 32
  cache_enabled: true
```

## Performance Troubleshooting

### Common Issues

**High Memory Usage**
```bash
# Check memory leaks
oboyu diagnose memory

# Limit memory usage
export OBOYU_MEMORY_LIMIT=4G

# Enable memory profiling
oboyu index ~/Documents --profile-memory
```

**Slow Searches**
```bash
# Check index fragmentation
oboyu index analyze --fragmentation

# Rebuild if needed
oboyu index rebuild --db-path ~/indexes/personal.db

# Optimize query patterns
oboyu query optimize "slow query"
```

**High CPU Usage**
```bash
# Check thread count
oboyu config get indexer.threads

# Reduce parallelism
oboyu config set indexer.threads 4

# Enable CPU throttling
cpulimit -l 50 oboyu index ~/Documents
```

## Performance Best Practices

### Do's
- ✅ Use SSD storage for indices
- ✅ Enable query caching
- ✅ Optimize indices regularly
- ✅ Monitor performance metrics
- ✅ Update incrementally
- ✅ Use appropriate chunk sizes

### Don'ts
- ❌ Over-parallelize on limited hardware
- ❌ Ignore memory limits
- ❌ Use network storage for indices
- ❌ Skip index optimization
- ❌ Use huge chunk sizes
- ❌ Disable all caching

## Performance Checklist

Before deployment:
- [ ] Benchmark baseline performance
- [ ] Move indices to fast storage
- [ ] Configure memory limits
- [ ] Enable query caching
- [ ] Set up monitoring
- [ ] Plan optimization schedule
- [ ] Test under load
- [ ] Configure backups

## Advanced Techniques

### Custom Index Structures
```python
# Create specialized index
from oboyu import CustomIndex

index = CustomIndex(
    vector_index="faiss",
    vector_params={"nlist": 100},
    text_index="tantivy",
    text_params={"memory_budget": "4GB"}
)
```

### Query Planning
```bash
# Analyze query plan
oboyu query explain "complex search query"

# Force query plan
oboyu query "search" --plan "vector_first"

# Optimize query plan
oboyu query optimize-plan "repeated query"
```

## Next Steps

- Implement [Automation](../integration/automation.md) for performance monitoring
- Explore [MCP Integration](../integration/mcp-integration.md) for distributed search
- Review [Troubleshooting](../reference-troubleshooting/troubleshooting.md) for performance issues