# FRED Supreme Litigation OS - Quick Start Guide

**Master Workflow Orchestration System**

---

## 🚀 5-Minute Quick Start

### Step 1: Install & Configure

```bash
# Clone/cd to workspace
cd /workspaces/fredprime-legal-system

# Install dependencies
pip install -e ".[dev]"
pip install click rich pyyaml pytest pytest-asyncio

# Create directories
mkdir -p config state output
```

### Step 2: Prepare Evidence

```bash
# Create evidence directory
mkdir -p evidence

# Add your case files (PDFs, documents, images, etc.)
cp /path/to/evidence/* evidence/
```

### Step 3: Launch & Execute

```bash
# Interactive menu (recommended for first run)
python -m src.master_cli interactive

# OR command-line execution
python -m src.master_cli new-case \
  --case-type custody \
  --case-number "2025-001234-CZ"

# Execute complete workflow
python -m src.master_cli execute \
  --case-number "2025-001234-CZ" \
  --evidence-dir ./evidence
```

### Step 4: Review Output

```bash
# Output organized in:
output/
├── exhibits/              # Labeled exhibits A-Z
├── documents/             # Generated motions & affidavits
├── warboards/             # Timeline visualizations
├── discovery/             # Discovery requests
├── filing/                # Court filing bundle
└── state/                 # Case state & checkpoints
```

---

## 🎯 Common Workflows

### Custody Case (Standard)

```bash
# Full execution with progress
python -m src.master_cli execute \
  --case-number "2025-001234-CZ" \
  --case-type custody \
  --evidence-dir ./evidence \
  --output-dir ./output

# Output: Motion for Modification + Exhibits + Warboard
```

**Duration**: 2-5 minutes (depending on evidence volume)
**Output Files**:

- `Motion_for_Modification_of_Custody.docx`
- 26 labeled exhibits (A-Z)
- `timeline_warboard.svg`
- Custody interference analysis

### Housing Emergency (Rapid Response)

```bash
# Expedited workflow for emergency injunctions
python -m src.master_cli execute \
  --case-number "2025-004567-CZ" \
  --case-type housing \
  --evidence-dir ./emergency_evidence

# Output: Emergency motion + Affidavit + Filing bundle
```

**Duration**: <2 minutes
**Output Files**:

- `Emergency_Injunction_Motion.docx`
- `Supporting_Affidavit.docx`
- Filing package for immediate submission

### PPO Defense (Personal Protection Order)

```bash
# PPO response workflow
python -m src.master_cli execute \
  --case-number "2025-003456-CZ" \
  --case-type ppo \
  --evidence-dir ./ppo_evidence

# Output: PPO response + Rebuttal evidence + Warboard
```

**Duration**: 3-4 minutes
**Output Files**:

- `Response_to_Personal_Protection_Order.docx`
- PPO misuse pattern analysis
- `case_map_warboard.svg`

---

## 📋 CLI Commands Reference

### Case Management

#### Create New Case

```bash
python -m src.master_cli new-case \
  --case-type {custody|housing|ppo|contempt} \
  --case-number "YYYY-NNNNNNN-CC" \
  [--parties "Plaintiff vs. Defendant"]
```

#### Open Existing Case

```bash
python -m src.master_cli open-case \
  --case-number "2025-001234-CZ"
```

#### View Case Status

```bash
python -m src.master_cli status \
  --case-number "2025-001234-CZ"
```

### Workflow Operations

#### List Available Workflows

```bash
python -m src.master_cli workflows
```

Output:

```
┌──────────────────────────┬──────────┐
│ Workflow                 │ Stages   │
├──────────────────────────┼──────────┤
│ custody_modification     │ 12       │
│ housing_emergency        │ 7        │
│ ppo_defense              │ 8        │
└──────────────────────────┴──────────┘
```

#### Get Workflow Details

```bash
python -m src.master_cli workflow-info custody_modification
```

#### Execute Workflow

```bash
python -m src.master_cli execute \
  --case-number "2025-001234-CZ" \
  --case-type custody \
  --evidence-dir ./evidence \
  [--resume]                 # Resume from last checkpoint
  [--dry-run]               # Preview without execution
```

### Evidence Management

#### Ingest Evidence Files

```bash
python -m src.master_cli ingest \
  --case-number "2025-001234-CZ" \
  --evidence-dir ./raw_evidence
```

#### Organize & Label Exhibits

```bash
python -m src.master_cli organize \
  --case-number "2025-001234-CZ" \
  --output-dir ./exhibits
```

### Document Generation

#### Generate Motion & Affidavits

```bash
python -m src.master_cli generate-motion \
  --case-number "2025-001234-CZ" \
  --case-type custody \
  --output-dir ./documents
```

#### Validate Documents for Court Compliance

```bash
python -m src.master_cli validate \
  --case-number "2025-001234-CZ" \
  --document-dir ./documents
```

### Visualizations

#### Generate Warboards & Timelines

```bash
python -m src.master_cli warboard \
  --case-number "2025-001234-CZ" \
  --case-type custody \
  --output-dir ./warboards
```

### System

#### Interactive Menu System

```bash
python -m src.master_cli interactive
```

#### Application Info

```bash
python -m src.master_cli about
```

---

## 🔄 Workflow Execution States

### Normal Flow

```
START
  ↓
[INTAKE] - Scan evidence files
  ↓
[ANALYSIS] - Score & analyze evidence
  ↓
[ORGANIZATION] - Label exhibits A-Z
  ↓
[GENERATION] - Create motions/affidavits
  ↓
[VALIDATION] - Check MCR compliance
  ↓
[WARBOARDING] - Generate visualizations
  ↓
[DISCOVERY] - Prepare discovery docs
  ↓
[FILING] - Bundle for court submission
  ↓
COMPLETE ✓
```

### Resume from Checkpoint

```bash
# If workflow interrupted, resume from last completed stage
python -m src.master_cli execute \
  --case-number "2025-001234-CZ" \
  --resume

# System automatically:
# 1. Loads last checkpoint
# 2. Resumes from next incomplete stage
# 3. Preserves all prior results
```

---

## 📂 Output Directory Structure

```
output/
├── exhibits/
│   ├── Exhibit_A.pdf
│   ├── Exhibit_B.pdf
│   └── ... (up to Z, AA, AB, etc.)
│
├── documents/
│   ├── Motion_for_Modification_of_Custody.docx
│   └── Supporting_Affidavit.docx
│
├── warboards/
│   ├── timeline_warboard.svg
│   └── case_map_warboard.svg
│
├── discovery/
│   ├── discovery_request.docx
│   └── interrogatories.docx
│
├── filing/
│   ├── FILING_MANIFEST.json
│   └── MiFile_bundle.zip
│
└── state/
    ├── case_2025001234.json
    └── checkpoints/
        ├── checkpoint_0_intake.json
        └── checkpoint_1_analysis.json
```

---

## ⚙️ Configuration

### Workflows Configuration

Edit `config/workflows.yaml`:

```yaml
custody_modification:
  name: 'Custody Modification'
  description: 'Motion for modification of custody arrangement'
  case_types:
    - CUSTODY
    - CHILD_SUPPORT

  stages:
    - name: 'intake_evidence'
      type: 'intake'
      enabled: true
      timeout: 300

    - name: 'organize_exhibits'
      type: 'organization'
      dependencies: ['intake_evidence']
      enabled: true
      config:
        max_exhibits: 100
```

### User Preferences

Edit `config/user_settings.json`:

```json
{
  "default_case_type": "custody",
  "default_output_dir": "./output",
  "auto_backup": true,
  "log_level": "INFO",
  "rich_ui_enabled": true,
  "theme": "dark"
}
```

---

## 🧪 Testing & Validation

### Run Test Suite

```bash
# All tests
pytest tests/test_master_integration.py -v

# Specific test class
pytest tests/test_master_integration.py::TestStageHandlers -v

# With performance metrics
pytest tests/test_master_integration.py -v --durations=10
```

### Manual Testing

```bash
# Dry run (preview without changes)
python -m src.master_cli execute \
  --case-number "2025-TEST-001" \
  --dry-run

# Verbose logging
export LOG_LEVEL=DEBUG
python -m src.master_cli execute \
  --case-number "2025-001234-CZ" \
  --verbose
```

---

## 🔧 Advanced Usage

### Custom Stage Handler

```python
from src.master_integration_bridge import get_handler_registry

registry = get_handler_registry()

async def my_custom_handler(context, config):
    """Custom stage implementation."""
    # Your logic here
    return {
        'status': 'completed',
        'custom_metric': 42,
    }

# Register
registry.register('custom_stage', my_custom_handler)
```

### Programmatic API

```python
import asyncio
from src.master_workflow_engine import WorkflowEngine, CaseContext, CaseType
from pathlib import Path

async def main():
    # Create engine
    engine = WorkflowEngine()

    # Create case
    case = CaseContext(
        case_id="2025001234",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[Path("evidence")],
    )

    # Execute workflow
    result = await engine.execute_workflow("custody_modification", case)

    # Process results
    print(f"Completed stages: {result['completed_count']}")
    print(f"Artifacts: {result['artifacts']}")

asyncio.run(main())
```

### Batch Case Processing

```python
# Process multiple cases
cases = [
    "2025-001234-CZ",
    "2025-001235-CZ",
    "2025-001236-CZ",
]

for case_num in cases:
    result = subprocess.run([
        "python", "-m", "src.master_cli", "execute",
        "--case-number", case_num,
        "--case-type", "custody",
        "--evidence-dir", f"evidence/{case_num}"
    ])
```

---

## 🐛 Troubleshooting

### Issue: "No evidence files found"

**Solution**: Check evidence directory path and file types

```bash
# Verify files exist
ls -la evidence/

# Ensure .txt, .pdf, .docx, .msg files present
find evidence -type f | head -20
```

### Issue: "Workflow interrupted"

**Solution**: Resume from checkpoint

```bash
# Check case status
python -m src.master_cli status --case-number "2025-001234-CZ"

# Resume execution
python -m src.master_cli execute --case-number "2025-001234-CZ" --resume
```

### Issue: "Document validation failed"

**Solution**: Check MCR compliance

```bash
# Validate specific document
python -m src.master_cli validate \
  --case-number "2025-001234-CZ" \
  --document-dir ./documents

# Review errors in output
cat output/validation_report.json
```

### Issue: "Memory exhausted with large evidence set"

**Solution**: Process in batches or increase virtual memory

```bash
# Batch process large directories
split -n l/10 evidence_list.txt evidence_batch_

for batch in evidence_batch_*; do
  python -m src.master_cli execute \
    --case-number "2025-001234-CZ" \
    --evidence-dir "$batch"
done
```

---

## 📊 Performance Tips

### Optimize Evidence Ingestion

```bash
# Exclude large files
find evidence -size +100M -delete

# Use SSD for evidence directories
ln -s /ssd/evidence ./evidence

# Enable parallel processing (future feature)
export PARALLEL_WORKERS=4
```

### Improve Document Generation

```bash
# Pre-cache motion templates
python -m src.master_cli --prepare-templates

# Reduce warboard complexity
python -m src.master_cli warboard \
  --max-nodes 500 \
  --simplify-timeline
```

---

## 📚 Additional Resources

### Documentation Files

- `MASTER_WORKFLOW_ARCHITECTURE.md` - Complete system overview
- `AGENTS.md` - Module directory
- `README.md` - Project overview

### Code Examples

- `src/master_workflow_engine.py` - Orchestration engine
- `src/master_cli.py` - CLI interface
- `src/master_integration_bridge.py` - Stage handlers
- `tests/test_master_integration.py` - Test suite

### Configuration Files

- `config/workflows.yaml` - Workflow definitions
- `config/user_settings.json` - User preferences
- `config/system_enforcement.json` - System rules

---

## ✅ Checklist for First-Time Use

- [ ] Install dependencies: `pip install -e ".[dev]"`
- [ ] Create directories: `mkdir -p config state output`
- [ ] Prepare evidence files in `evidence/` directory
- [ ] Review `config/workflows.yaml` for your case type
- [ ] Run test: `pytest tests/test_master_integration.py`
- [ ] Execute first workflow: `python -m src.master_cli interactive`
- [ ] Review generated outputs in `output/`
- [ ] Check case state in `state/case_*.json`

---

## 🎓 Next Steps

1. **Customize Workflows**: Edit `config/workflows.yaml` for your specific workflow
2. **Integrate Evidence**: Add your case files to `evidence/` directory
3. **Generate Documents**: Run workflow to create motions and affidavits
4. **Review Output**: Check generated documents for accuracy
5. **Validate**: Use `validate` command to check court compliance
6. **File**: Use `filing` stage to prepare court submission package

---

**Status**: 🟢 Production Ready
**Support**: See troubleshooting section above
**Last Updated**: January 14, 2026

---

_Master Workflow Orchestration System - Fully Offline, Michigan-Compliant, Enterprise-Grade_
