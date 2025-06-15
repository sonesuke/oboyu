---
id: automation
title: Automation Examples
sidebar_position: 3
---

# Automation Examples

Automate your document indexing and search workflows with these practical examples. Learn how to set up scheduled tasks, monitor changes, and create intelligent automation systems.

## Scheduled Indexing

### Cron Jobs

#### Basic Daily Update
```bash
# Add to crontab with: crontab -e

# Daily update at 2 AM
0 2 * * * /usr/local/bin/oboyu index ~/Documents --update --quiet

# Hourly update during work hours
0 9-17 * * 1-5 /usr/local/bin/oboyu index ~/work --update

# Weekly full reindex on Sunday
0 3 * * 0 /usr/local/bin/oboyu index ~/Documents --force

# Monthly optimization
0 4 1 * * /usr/local/bin/oboyu index optimize --all
```

#### Advanced Cron Script
```bash
#!/bin/bash
# /usr/local/bin/oboyu-cron.sh

# Load environment
source ~/.bashrc

# Logging
LOG_DIR=~/.oboyu/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cron-$(date +%Y%m%d).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if another instance is running
LOCKFILE=/tmp/oboyu-cron.lock
if [ -e "$LOCKFILE" ]; then
    log "Another instance is running, skipping"
    exit 0
fi

# Create lock file
touch "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# Update indices
log "Starting scheduled index update"

for index_name in $(oboyu index list --format names); do
    log "Updating index: $index_name"
    
    if oboyu index update --db-path ~/indexes/example.db >> "$LOG_FILE" 2>&1; then
        log "Successfully updated $index_name"
    else
        log "Failed to update $index_name"
        # Send notification
        echo "Oboyu index update failed for $index_name" | \
            mail -s "Oboyu Cron Error" user@example.com
    fi
done

log "Scheduled update complete"

# Cleanup old logs
find "$LOG_DIR" -name "*.log" -mtime +30 -delete
```

### Systemd Timers

#### Service File
```ini
# /etc/systemd/system/oboyu-index.service
[Unit]
Description=Oboyu Index Update
After=network.target

[Service]
Type=oneshot
User=username
ExecStart=/usr/local/bin/oboyu index /home/username/Documents --update
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Timer File
```ini
# /etc/systemd/system/oboyu-index.timer
[Unit]
Description=Run Oboyu Index Update every 4 hours
Requires=oboyu-index.service

[Timer]
OnBootSec=10min
OnUnitActiveSec=4h
Persistent=true

[Install]
WantedBy=timers.target
```

#### Enable Timer
```bash
# Enable and start the timer
sudo systemctl enable oboyu-index.timer
sudo systemctl start oboyu-index.timer

# Check status
systemctl status oboyu-index.timer
systemctl list-timers
```

## File System Monitoring

### inotify Watcher (Linux)

```bash
#!/bin/bash
# watch-and-index.sh - Real-time indexing

WATCH_DIR="$HOME/Documents"
INDEX_NAME="personal"

# Install inotify-tools if needed
# sudo apt-get install inotify-tools

log() {
    logger -t oboyu-watch "$1"
}

# Watch for file changes
inotifywait -m -r \
    -e create -e modify -e delete -e moved_to \
    --exclude '(\.tmp|\.swp|~$)' \
    "$WATCH_DIR" |
while read path event file; do
    case $event in
        CREATE|MODIFY|MOVED_TO)
            log "Indexing new/modified file: $path$file"
            oboyu index "$path$file" --db-path ~/indexes/example.db --update
            ;;
        DELETE)
            log "Removing deleted file: $path$file"
            oboyu index remove "$path$file" --db-path ~/indexes/example.db
            ;;
    esac
done
```

### FSEvents Watcher (macOS)

```bash
#!/bin/bash
# fswatch-index.sh - macOS file watching

WATCH_DIR="$HOME/Documents"

# Install fswatch if needed
# brew install fswatch

fswatch -0 -e ".*\.tmp$" -e ".*\.swp$" "$WATCH_DIR" | \
while IFS= read -r -d '' path; do
    echo "File changed: $path"
    
    # Debounce - wait for file to stabilize
    sleep 1
    
    # Update index
    oboyu index "$path" --update --quiet
done
```

### Python Watcher (Cross-platform)

```python
#!/usr/bin/env python3
# watch_and_index.py - Cross-platform file watcher

import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class OboyuIndexHandler(FileSystemEventHandler):
    def __init__(self, index_name="personal"):
        self.index_name = index_name
        self.pending_files = set()
        
    def on_created(self, event):
        if not event.is_directory:
            self.index_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            self.index_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.remove_file(event.src_path)
    
    def index_file(self, filepath):
        # Skip temporary files
        if filepath.endswith(('.tmp', '.swp', '~')):
            return
        
        print(f"Indexing: {filepath}")
        subprocess.run([
            'oboyu', 'index', filepath,
            '--name', self.index_name,
            '--update'
        ])
    
    def remove_file(self, filepath):
        print(f"Removing: {filepath}")
        subprocess.run([
            'oboyu', 'index', 'remove', filepath,
            '--name', self.index_name
        ])

if __name__ == "__main__":
    path = Path.home() / "Documents"
    event_handler = OboyuIndexHandler()
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=True)
    
    print(f"Watching {path} for changes...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

## Smart Automation

### Conditional Indexing

```bash
#!/bin/bash
# smart-index.sh - Index based on conditions

# Only index if system is idle
check_system_idle() {
    local load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1)
    local threshold=2.0
    
    (( $(echo "$load < $threshold" | bc -l) ))
}

# Only index if on AC power (laptops)
check_ac_power() {
    if [[ -f /sys/class/power_supply/AC/online ]]; then
        [[ $(cat /sys/class/power_supply/AC/online) == "1" ]]
    else
        true  # Assume desktop
    fi
}

# Only index if enough disk space
check_disk_space() {
    local min_space_gb=5
    local available=$(df -BG ~/.oboyu | tail -1 | awk '{print $4}' | sed 's/G//')
    
    [[ $available -gt $min_space_gb ]]
}

# Smart indexing decision
if check_system_idle && check_ac_power && check_disk_space; then
    echo "Conditions met, starting indexing..."
    oboyu index ~/Documents --update
else
    echo "Skipping indexing due to system conditions"
fi
```

### Adaptive Scheduling

```python
#!/usr/bin/env python3
# adaptive_scheduler.py - Adjust indexing based on usage patterns

import json
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

class AdaptiveScheduler:
    def __init__(self, stats_file="~/.oboyu/usage_stats.json"):
        self.stats_file = Path(stats_file).expanduser()
        self.load_stats()
    
    def load_stats(self):
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                self.stats = json.load(f)
        else:
            self.stats = defaultdict(list)
    
    def record_search(self):
        """Record when searches happen"""
        hour = datetime.now().hour
        self.stats['search_hours'].append(hour)
        self.save_stats()
    
    def get_optimal_index_times(self):
        """Determine best times to index based on usage"""
        if not self.stats['search_hours']:
            return [2, 14]  # Default: 2 AM and 2 PM
        
        # Find hours with least searches
        hour_counts = defaultdict(int)
        for hour in self.stats['search_hours']:
            hour_counts[hour] += 1
        
        # Get 4 hours with least activity
        quiet_hours = sorted(hour_counts.items(), 
                           key=lambda x: x[1])[:4]
        
        return [hour for hour, _ in quiet_hours]
    
    def should_index_now(self):
        """Check if now is a good time to index"""
        current_hour = datetime.now().hour
        optimal_hours = self.get_optimal_index_times()
        
        return current_hour in optimal_hours
    
    def run_adaptive_index(self):
        """Run indexing if appropriate"""
        if self.should_index_now():
            print(f"Running adaptive index at {datetime.now()}")
            subprocess.run(['oboyu', 'index', '--update', '--all'])
        else:
            print("Not an optimal time for indexing")

# Usage
scheduler = AdaptiveScheduler()
scheduler.run_adaptive_index()
```

## Event-Driven Automation

### Git Hook Integration

```bash
#!/bin/bash
# .git/hooks/post-commit - Index after commits

# Index changed files
git diff-tree --no-commit-id --name-only -r HEAD | \
while read file; do
    if [[ -f "$file" ]]; then
        oboyu index "$file" --update --quiet
    fi
done

echo "Oboyu index updated with committed files"
```

### Email Trigger

```python
#!/usr/bin/env python3
# email_trigger.py - Index documents from email

import email
import imaplib
import subprocess
from pathlib import Path

def check_email_for_documents():
    # Connect to email
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login('user@gmail.com', 'app_password')
    mail.select('inbox')
    
    # Search for emails with attachments
    _, message_numbers = mail.search(None, 'UNSEEN', 'HAS attachment')
    
    for num in message_numbers[0].split():
        _, msg_data = mail.fetch(num, '(RFC822)')
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        # Process attachments
        for part in email_message.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                if filename:
                    # Save attachment
                    filepath = Path.home() / 'Downloads' / filename
                    with open(filepath, 'wb') as f:
                        f.write(part.get_payload(decode=True))
                    
                    # Index the file
                    subprocess.run([
                        'oboyu', 'index', str(filepath),
                        '--update'
                    ])
                    
                    print(f"Indexed email attachment: {filename}")
    
    mail.close()
    mail.logout()

if __name__ == "__main__":
    check_email_for_documents()
```

### Calendar Integration

```bash
#!/bin/bash
# calendar_index.sh - Index based on calendar events

# Get today's meetings from calendar
get_meetings() {
    # Example using gcalcli
    gcalcli agenda --nocolor --tsv | grep "$(date +%Y-%m-%d)"
}

# Index meeting-related documents
index_meeting_docs() {
    local meeting_title="$1"
    
    # Search for related documents
    find ~/Documents -type f -name "*${meeting_title}*" -mtime -7 | \
    while read file; do
        oboyu index "$file" --update --priority high
    done
}

# Process today's meetings
get_meetings | while read -r line; do
    meeting=$(echo "$line" | cut -f5)
    echo "Preparing documents for: $meeting"
    index_meeting_docs "$meeting"
done
```

## Cloud Sync Automation

### Dropbox Integration

```bash
#!/bin/bash
# dropbox_sync.sh - Index Dropbox changes

DROPBOX_PATH="$HOME/Dropbox"
LAST_SYNC_FILE="$HOME/.oboyu/last_dropbox_sync"

# Get last sync time
if [[ -f "$LAST_SYNC_FILE" ]]; then
    LAST_SYNC=$(cat "$LAST_SYNC_FILE")
else
    LAST_SYNC="1970-01-01"
fi

# Find and index new/modified files
find "$DROPBOX_PATH" -type f -newermt "$LAST_SYNC" | \
while read file; do
    echo "Indexing: $file"
    oboyu index "$file" --update
done

# Update last sync time
date +%Y-%m-%d > "$LAST_SYNC_FILE"
```

### Google Drive Sync

```python
#!/usr/bin/env python3
# gdrive_sync.py - Sync Google Drive documents

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import subprocess
import os

def sync_google_drive():
    # Authenticate
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('drive', 'v3', credentials=creds)
    
    # Get recent files
    results = service.files().list(
        q="modifiedTime > '2024-01-01T00:00:00'",
        fields="files(id, name, mimeType)"
    ).execute()
    
    items = results.get('files', [])
    
    for item in items:
        # Download text-based files
        if 'text' in item['mimeType'] or 'document' in item['mimeType']:
            # Download file
            file_path = download_file(service, item['id'], item['name'])
            
            # Index file
            subprocess.run([
                'oboyu', 'index', file_path, '--update'
            ])
            
            print(f"Indexed: {item['name']}")

def download_file(service, file_id, filename):
    # Implementation for downloading files
    pass

if __name__ == "__main__":
    sync_google_drive()
```

## Monitoring and Alerts

### Health Check Script

```bash
#!/bin/bash
# health_check.sh - Monitor Oboyu health

# Configuration
ALERT_EMAIL="admin@example.com"
MAX_INDEX_AGE_DAYS=7
MIN_FREE_SPACE_GB=5

# Check index age
check_index_age() {
    local index_name="$1"
    local last_update=$(oboyu index info --db-path ~/indexes/example.db --format json | \
                       jq -r '.last_updated')
    local age_days=$(( ($(date +%s) - $(date -d "$last_update" +%s)) / 86400 ))
    
    if [[ $age_days -gt $MAX_INDEX_AGE_DAYS ]]; then
        send_alert "Index '$index_name' is $age_days days old"
    fi
}

# Check disk space
check_disk_space() {
    local available_gb=$(df -BG ~/.oboyu | tail -1 | awk '{print $4}' | sed 's/G//')
    
    if [[ $available_gb -lt $MIN_FREE_SPACE_GB ]]; then
        send_alert "Low disk space: ${available_gb}GB available"
    fi
}

# Check search performance
check_search_performance() {
    local avg_time=$(oboyu query history --days 1 --format json | \
                    jq -r 'map(.duration_ms) | add/length')
    
    if (( $(echo "$avg_time > 1000" | bc -l) )); then
        send_alert "Slow search performance: ${avg_time}ms average"
    fi
}

# Send alert
send_alert() {
    local message="$1"
    echo "ALERT: $message"
    echo "$message" | mail -s "Oboyu Alert" "$ALERT_EMAIL"
}

# Run all checks
for index in $(oboyu index list --format names); do
    check_index_age "$index"
done

check_disk_space
check_search_performance
```

### Prometheus Exporter

```python
#!/usr/bin/env python3
# oboyu_exporter.py - Prometheus metrics exporter

from prometheus_client import start_http_server, Gauge
import subprocess
import json
import time

# Define metrics
index_size = Gauge('oboyu_index_size_bytes', 'Index size in bytes', ['index_name'])
document_count = Gauge('oboyu_document_count', 'Number of documents', ['index_name'])
search_latency = Gauge('oboyu_search_latency_ms', 'Search latency in milliseconds')
index_age = Gauge('oboyu_index_age_seconds', 'Index age in seconds', ['index_name'])

def collect_metrics():
    # Get index stats
    result = subprocess.run(
        ['oboyu', 'index', 'stats', '--all', '--format', 'json'],
        capture_output=True, text=True
    )
    stats = json.loads(result.stdout)
    
    for index_name, index_stats in stats.items():
        index_size.labels(index_name=index_name).set(index_stats['size_bytes'])
        document_count.labels(index_name=index_name).set(index_stats['document_count'])
        index_age.labels(index_name=index_name).set(index_stats['age_seconds'])
    
    # Get search performance
    perf_result = subprocess.run(
        ['oboyu', 'query', 'history', '--days', '1', '--format', 'json'],
        capture_output=True, text=True
    )
    
    if perf_result.stdout:
        searches = json.loads(perf_result.stdout)
        if searches:
            avg_latency = sum(s['duration_ms'] for s in searches) / len(searches)
            search_latency.set(avg_latency)

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(9090)
    
    # Collect metrics every 60 seconds
    while True:
        collect_metrics()
        time.sleep(60)
```

## CSV Enrichment Automation

### Automated Data Pipeline

Set up automated enrichment pipelines for regular data processing:

```bash
#!/bin/bash
# enrichment_pipeline.sh - Automated CSV enrichment workflow

# Configuration
DATA_DIR="$HOME/data/raw"
OUTPUT_DIR="$HOME/data/enriched"
SCHEMA_DIR="$HOME/schemas"
LOG_FILE="$HOME/.oboyu/logs/enrichment-$(date +%Y%m%d).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Process all CSV files in data directory
process_csv_files() {
    for csv_file in "$DATA_DIR"/*.csv; do
        if [[ -f "$csv_file" ]]; then
            local basename=$(basename "$csv_file" .csv)
            local schema_file="$SCHEMA_DIR/${basename}-schema.json"
            local output_file="$OUTPUT_DIR/${basename}-enriched.csv"
            
            if [[ -f "$schema_file" ]]; then
                log "Processing: $csv_file with schema: $schema_file"
                
                if oboyu enrich "$csv_file" "$schema_file" \
                   --output "$output_file" \
                   --batch-size 10 \
                   --confidence 0.6 2>>"$LOG_FILE"; then
                    log "Successfully enriched: $basename"
                    
                    # Archive processed file
                    mv "$csv_file" "$DATA_DIR/processed/"
                else
                    log "Failed to enrich: $basename"
                    mv "$csv_file" "$DATA_DIR/failed/"
                fi
            else
                log "No schema found for: $basename"
            fi
        fi
    done
}

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$DATA_DIR/processed" "$DATA_DIR/failed"

# Run enrichment pipeline
log "Starting automated enrichment pipeline"
process_csv_files
log "Enrichment pipeline completed"
```

### Scheduled Enrichment

```bash
# Add to crontab for daily enrichment
# Run every night at 1 AM
0 1 * * * /usr/local/bin/enrichment_pipeline.sh

# Process new data every 4 hours during business hours
0 9,13,17 * * 1-5 /usr/local/bin/enrichment_pipeline.sh
```

### Email-Triggered Enrichment

```python
#!/usr/bin/env python3
# email_enrichment.py - Process CSV files from email attachments

import email
import imaplib
import subprocess
import tempfile
import json
from pathlib import Path

def process_email_attachments():
    """Process CSV attachments from emails and enrich them"""
    
    # Email configuration
    IMAP_SERVER = 'imap.gmail.com'
    EMAIL = 'your-email@gmail.com'
    PASSWORD = 'app-password'
    
    # Connect to email
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')
    
    # Search for emails with CSV attachments
    _, message_numbers = mail.search(None, 
        'UNSEEN SUBJECT "Data for enrichment"')
    
    for num in message_numbers[0].split():
        _, msg_data = mail.fetch(num, '(RFC822)')
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        # Process CSV attachments
        for part in email_message.walk():
            if (part.get_content_disposition() == 'attachment' and 
                part.get_filename().endswith('.csv')):
                
                filename = part.get_filename()
                
                # Save attachment to temp directory
                with tempfile.NamedTemporaryFile(
                    mode='wb', suffix='.csv', delete=False
                ) as tmp_file:
                    tmp_file.write(part.get_payload(decode=True))
                    csv_path = tmp_file.name
                
                # Determine schema based on filename
                schema_name = filename.replace('.csv', '-schema.json')
                schema_path = Path.home() / 'schemas' / schema_name
                
                if schema_path.exists():
                    output_path = Path.home() / 'data/enriched' / \
                                 f"{filename.replace('.csv', '')}-enriched.csv"
                    
                    # Run enrichment
                    result = subprocess.run([
                        'oboyu', 'enrich', csv_path, str(schema_path),
                        '--output', str(output_path),
                        '--batch-size', '5',
                        '--confidence', '0.7'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"Successfully enriched: {filename}")
                        send_completion_email(filename, str(output_path))
                    else:
                        print(f"Failed to enrich: {filename}")
                        send_error_email(filename, result.stderr)
                else:
                    print(f"No schema found for: {filename}")
                
                # Cleanup temp file
                Path(csv_path).unlink()
    
    mail.close()
    mail.logout()

def send_completion_email(filename, output_path):
    """Send completion notification"""
    # Implementation for sending completion email
    pass

def send_error_email(filename, error):
    """Send error notification"""
    # Implementation for sending error email
    pass

if __name__ == "__main__":
    process_email_attachments()
```

### Dynamic Schema Selection

```python
#!/usr/bin/env python3
# smart_enrichment.py - Automatically select appropriate schemas

import pandas as pd
import json
import subprocess
from pathlib import Path

class SmartEnrichmentSystem:
    def __init__(self, schema_dir="~/schemas"):
        self.schema_dir = Path(schema_dir).expanduser()
        self.schema_templates = self.load_schema_templates()
    
    def load_schema_templates(self):
        """Load available schema templates"""
        templates = {}
        for schema_file in self.schema_dir.glob("*.json"):
            with open(schema_file) as f:
                schema = json.load(f)
                templates[schema_file.stem] = schema
        return templates
    
    def analyze_csv_structure(self, csv_path):
        """Analyze CSV to determine appropriate schema"""
        df = pd.read_csv(csv_path, nrows=5)  # Sample first 5 rows
        
        columns = df.columns.tolist()
        
        # Heuristics for schema selection
        if 'company_name' in columns or 'company' in columns:
            return 'company-enrichment'
        elif 'person_name' in columns or 'name' in columns:
            return 'person-enrichment'
        elif 'product_name' in columns or 'product' in columns:
            return 'product-enrichment'
        else:
            return 'generic-enrichment'
    
    def auto_enrich(self, csv_path, output_path=None):
        """Automatically enrich CSV with best matching schema"""
        # Determine schema
        schema_type = self.analyze_csv_structure(csv_path)
        schema_path = self.schema_dir / f"{schema_type}.json"
        
        if not schema_path.exists():
            print(f"Schema not found: {schema_path}")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            csv_file = Path(csv_path)
            output_path = csv_file.parent / f"{csv_file.stem}-enriched.csv"
        
        print(f"Using schema: {schema_type}")
        print(f"Input: {csv_path}")
        print(f"Output: {output_path}")
        
        # Run enrichment
        result = subprocess.run([
            'oboyu', 'enrich', csv_path, str(schema_path),
            '--output', str(output_path),
            '--batch-size', '8',
            '--confidence', '0.6'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Enrichment completed successfully")
            return True
        else:
            print(f"Enrichment failed: {result.stderr}")
            return False

# Usage example
enricher = SmartEnrichmentSystem()

# Process multiple files
for csv_file in Path("~/data/incoming").glob("*.csv"):
    enricher.auto_enrich(csv_file)
```

### Performance Monitoring for Enrichment

```bash
#!/bin/bash
# monitor_enrichment.sh - Monitor enrichment performance

METRICS_FILE="$HOME/.oboyu/enrichment_metrics.json"
ALERT_THRESHOLD_MS=30000  # 30 seconds

log_enrichment_metrics() {
    local csv_file="$1"
    local schema_file="$2"
    local start_time="$3"
    local end_time="$4"
    local status="$5"
    
    local duration_ms=$(( (end_time - start_time) * 1000 ))
    local timestamp=$(date -Iseconds)
    
    # Create metrics entry
    local metric_entry=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "csv_file": "$csv_file",
  "schema_file": "$schema_file",
  "duration_ms": $duration_ms,
  "status": "$status"
}
EOF
    )
    
    # Append to metrics file
    if [[ -f "$METRICS_FILE" ]]; then
        # Read existing metrics and append new one
        jq ". += [$metric_entry]" "$METRICS_FILE" > "${METRICS_FILE}.tmp"
        mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
    else
        echo "[$metric_entry]" > "$METRICS_FILE"
    fi
    
    # Check for performance issues
    if [[ $duration_ms -gt $ALERT_THRESHOLD_MS ]]; then
        send_performance_alert "$csv_file" "$duration_ms"
    fi
}

send_performance_alert() {
    local csv_file="$1"
    local duration_ms="$2"
    
    local message="Slow enrichment detected:
File: $csv_file
Duration: ${duration_ms}ms
Threshold: ${ALERT_THRESHOLD_MS}ms"
    
    echo "$message" | mail -s "Oboyu Enrichment Performance Alert" admin@example.com
}

# Enhanced enrichment wrapper
monitored_enrich() {
    local csv_file="$1"
    local schema_file="$2"
    local output_file="$3"
    
    local start_time=$(date +%s)
    
    if oboyu enrich "$csv_file" "$schema_file" --output "$output_file"; then
        local end_time=$(date +%s)
        log_enrichment_metrics "$csv_file" "$schema_file" "$start_time" "$end_time" "success"
        return 0
    else
        local end_time=$(date +%s)
        log_enrichment_metrics "$csv_file" "$schema_file" "$start_time" "$end_time" "failed"
        return 1
    fi
}

# Generate performance report
generate_enrichment_report() {
    if [[ -f "$METRICS_FILE" ]]; then
        echo "Enrichment Performance Report"
        echo "============================="
        
        # Average duration
        local avg_duration=$(jq '[.[] | select(.status == "success") | .duration_ms] | add / length' "$METRICS_FILE")
        echo "Average Duration: ${avg_duration}ms"
        
        # Success rate
        local total=$(jq 'length' "$METRICS_FILE")
        local successful=$(jq '[.[] | select(.status == "success")] | length' "$METRICS_FILE")
        local success_rate=$(echo "scale=2; $successful * 100 / $total" | bc)
        echo "Success Rate: ${success_rate}%"
        
        # Slowest files
        echo ""
        echo "Slowest Enrichments:"
        jq -r 'sort_by(.duration_ms) | reverse | .[0:5] | .[] | "\(.duration_ms)ms - \(.csv_file)"' "$METRICS_FILE"
    fi
}

# Usage
# monitored_enrich input.csv schema.json output.csv
# generate_enrichment_report
```

### Integration with Data Warehouses

```python
#!/usr/bin/env python3
# warehouse_enrichment.py - Enrich data from/to data warehouse

import pandas as pd
import sqlalchemy
import subprocess
import tempfile
from pathlib import Path

class WarehouseEnrichmentPipeline:
    def __init__(self, db_url, oboyu_schemas_dir):
        self.engine = sqlalchemy.create_engine(db_url)
        self.schemas_dir = Path(oboyu_schemas_dir)
    
    def extract_and_enrich(self, query, schema_name, table_name):
        """Extract data from warehouse, enrich, and load back"""
        
        # Extract data
        df = pd.read_sql(query, self.engine)
        
        # Save to temporary CSV
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
            csv_path = tmp_csv.name
        
        # Enrich data
        enriched_csv = csv_path.replace('.csv', '-enriched.csv')
        schema_path = self.schemas_dir / f"{schema_name}.json"
        
        result = subprocess.run([
            'oboyu', 'enrich', csv_path, str(schema_path),
            '--output', enriched_csv,
            '--batch-size', '20',
            '--confidence', '0.7'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Load enriched data back to warehouse
            enriched_df = pd.read_csv(enriched_csv)
            enriched_df.to_sql(
                table_name, 
                self.engine, 
                if_exists='replace', 
                index=False
            )
            print(f"Successfully enriched and loaded {len(enriched_df)} rows")
        else:
            print(f"Enrichment failed: {result.stderr}")
        
        # Cleanup
        Path(csv_path).unlink()
        Path(enriched_csv).unlink()

# Usage example
pipeline = WarehouseEnrichmentPipeline(
    'postgresql://user:pass@localhost/db',
    '~/schemas'
)

# Enrich customer data
pipeline.extract_and_enrich(
    "SELECT company_name, industry FROM customers WHERE enriched_at IS NULL",
    "company-enrichment",
    "enriched_customers"
)
```

## Backup Automation

### Incremental Backup

```bash
#!/bin/bash
# backup_indices.sh - Incremental backup of indices

BACKUP_DIR="/backup/oboyu"
RETENTION_DAYS=30

# Create backup directory with date
TODAY=$(date +%Y%m%d)
BACKUP_PATH="$BACKUP_DIR/$TODAY"
mkdir -p "$BACKUP_PATH"

# Backup each index
for index in $(oboyu index list --format names); do
    echo "Backing up index: $index"
    
    # Get index file path
    index_path=$(oboyu index info --db-path ~/indexes/example.db --format json | \
                jq -r '.path')
    
    # Incremental backup using rsync
    rsync -av --link-dest="$BACKUP_DIR/latest/$index" \
          "$index_path" "$BACKUP_PATH/$index"
done

# Update latest symlink
rm -f "$BACKUP_DIR/latest"
ln -s "$BACKUP_PATH" "$BACKUP_DIR/latest"

# Cleanup old backups
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
```

## Integration Examples

### Slack Notifications

```python
#!/usr/bin/env python3
# slack_notifier.py - Send indexing updates to Slack

import requests
import subprocess
import json

SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

def send_slack_message(message):
    requests.post(SLACK_WEBHOOK, json={"text": message})

def index_with_notifications():
    # Start indexing
    send_slack_message("🔄 Starting document indexing...")
    
    result = subprocess.run(
        ['oboyu', 'index', '~/Documents', '--update', '--format', 'json'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        stats = json.loads(result.stdout)
        message = f"✅ Indexing complete!\n" \
                 f"Files processed: {stats['files_processed']}\n" \
                 f"New documents: {stats['new_documents']}\n" \
                 f"Updated: {stats['updated_documents']}"
    else:
        message = f"❌ Indexing failed: {result.stderr}"
    
    send_slack_message(message)

if __name__ == "__main__":
    index_with_notifications()
```

## Next Steps

- Review [Performance Tuning](../reference/configuration.md) for optimization
- Explore [CLI Workflows](cli-workflows.md) for manual automation
- Set up [MCP Integration](mcp-integration.md) for AI-powered automation
- Learn about [CSV Enrichment Use Case](../use-cases/csv-enrichment.md) for detailed enrichment workflows
- Explore enrichment in [Basic Workflow](../basic-usage/basic-workflow.md) for daily usage patterns