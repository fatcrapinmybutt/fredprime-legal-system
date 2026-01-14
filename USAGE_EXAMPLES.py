"""
Master Workflow System - Usage Examples

Real-world examples of using the master workflow orchestration engine
for various litigation scenarios.
"""

# ============================================================================
# Example 1: Custody Case with Interactive Menu
# ============================================================================

def example_1_interactive_custody():
    """
    Simplest usage: Launch interactive menu and work through TUI.

    Best for: First-time users, complex case analysis, manual oversight
    """
    # User runs:
    # python -m src.master_cli interactive

    # System presents menu:
    # ┌─────────────────────────────────┐
    # │ FRED SUPREME LITIGATION OS       │
    # │ Master Workflow Orchestrator    │
    # └─────────────────────────────────┘
    #
    # Main Menu
    # ├─ 🏛️  New Case
    # ├─ 📂 Open Case
    # ├─ 🚀 Execute Workflow
    # ├─ 📋 View Workflows
    # └─ ❌ Exit

    # User selects "New Case":
    # - Prompts for case type (custody, housing, ppo, contempt)
    # - Prompts for case number (e.g., 2025-001234-CZ)
    # - Prompts for parties

    # User selects "Execute Workflow":
    # - Shows available workflows for case type
    # - Prompts for evidence directory
    # - Starts execution with real-time progress

    # Output written to:
    # - output/exhibits/Exhibit_A.pdf, Exhibit_B.pdf, etc.
    # - output/documents/Motion_for_Modification_of_Custody.docx
    # - output/warboards/timeline_warboard.svg
    # - state/case_2025001234.json


# ============================================================================
# Example 2: Batch Custody Case Processing
# ============================================================================

def example_2_batch_custody_processing():
    """
    Process multiple similar cases programmatically.

    Best for: Multiple cases, automation, batch processing
    """
    import subprocess
    import json

    # Process multiple cases in batch
    cases = [
        {"number": "2025-001234-CZ", "type": "custody", "dir": "evidence_case_001"},
        {"number": "2025-001235-CZ", "type": "custody", "dir": "evidence_case_002"},
        {"number": "2025-001236-CZ", "type": "custody", "dir": "evidence_case_003"},
    ]

    for case in cases:
        # Create case
        subprocess.run([
            "python", "-m", "src.master_cli", "new-case",
            "--case-type", case["type"],
            "--case-number", case["number"],
        ])

        # Execute workflow
        result = subprocess.run([
            "python", "-m", "src.master_cli", "execute",
            "--case-number", case["number"],
            "--case-type", case["type"],
            "--evidence-dir", case["dir"],
        ], capture_output=True, text=True)

        print(f"Case {case['number']}: {result.returncode == 0 and 'SUCCESS' or 'FAILED'}")


# ============================================================================
# Example 3: Emergency Housing with Resumable Execution
# ============================================================================

def example_3_emergency_housing_resumable():
    """
    Emergency housing workflow with resume capability.

    Best for: Long-running cases, unreliable environments, checkpoint safety
    """
    # User initiates emergency case
    # python -m src.master_cli new-case \
    #   --case-type housing \
    #   --case-number "2025-004567-CZ"

    # Start execution
    # python -m src.master_cli execute \
    #   --case-number "2025-004567-CZ" \
    #   --case-type housing \
    #   --evidence-dir ./emergency_evidence

    # Workflow progresses through stages:
    # ✓ [1/7] Rapid Intake (scan evidence)
    # ✓ [2/7] Assess Harm (irreparability analysis)
    # ✓ [3/7] Organize Evidence (label exhibits)
    # ⏳ [4/7] Generate Motion (in progress...)

    # If interrupted (Ctrl+C):
    # - System saves checkpoint
    # - Audit log records where it stopped
    # - State persisted to disk

    # User later resumes
    # python -m src.master_cli execute \
    #   --case-number "2025-004567-CZ" \
    #   --resume

    # System automatically:
    # - Loads checkpoint from state/case_2025004567.json
    # - Verifies completed stages (1-3)
    # - Continues from stage 4 (Generate Motion)
    # - Completes remaining stages 5-7
    # - Final output ready for filing


# ============================================================================
# Example 4: Programmatic API Usage
# ============================================================================

async def example_4_programmatic_api():
    """
    Use WorkflowEngine directly in Python code.

    Best for: Custom automation, integration with other systems
    """
    import asyncio
    from pathlib import Path
    from src.master_workflow_engine import (
        WorkflowEngine, CaseContext, CaseType
    )

    # Create engine
    engine = WorkflowEngine()

    # Create case context
    case = CaseContext(
        case_id="2025001234",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[Path("evidence")],
        parties={
            "plaintiff": "John Doe",
            "defendant": "Jane Doe",
        },
    )

    # Execute workflow
    result = await engine.execute_workflow(
        "custody_modification",
        case,
        resume=False
    )

    # Access results
    print(f"Status: {result['status']}")
    print(f"Completed: {result['completed_count']} stages")
    print(f"Artifacts: {result['artifacts']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")

    # Results include:
    # {
    #   'status': 'completed',
    #   'completed_count': 12,
    #   'total_count': 12,
    #   'duration_seconds': 234.5,
    #   'artifacts': [
    #     'exhibits/',
    #     'documents/',
    #     'warboards/',
    #     'filing/',
    #   ],
    # }


# ============================================================================
# Example 5: Custom Workflow Handler
# ============================================================================

async def example_5_custom_handler():
    """
    Register custom stage handler for specialized processing.

    Best for: Custom workflows, specialized analysis, domain-specific logic
    """
    from src.master_integration_bridge import get_handler_registry
    from src.master_workflow_engine import CaseContext

    # Get handler registry
    registry = get_handler_registry()

    # Define custom handler
    async def analyze_custody_interference(context, config):
        """Custom handler: Analyze custody interference patterns."""
        interference_events = []

        # Analyze evidence files for custody interference
        for file_info in context.evidence_files:
            # Custom logic: check for interference patterns
            if 'custody' in file_info['name'].lower():
                if 'interference' in file_info['name'].lower():
                    interference_events.append({
                        'file': file_info['name'],
                        'type': 'direct_interference',
                    })

        return {
            'status': 'completed',
            'interference_events': interference_events,
            'interference_count': len(interference_events),
        }

    # Register custom handler
    registry.register('custom_interference_analysis', analyze_custody_interference)

    # Use in workflow:
    # 1. Add custom stage to config/workflows.yaml:
    #    - name: "analyze_interference"
    #      type: "custom_interference_analysis"
    #      dependencies: ["intake_evidence"]
    #
    # 2. Handler is called automatically when stage executes


# ============================================================================
# Example 6: Evidence Scoring and Organization
# ============================================================================

async def example_6_evidence_workflow():
    """
    Detailed evidence processing workflow.

    Best for: Understanding the evidence pipeline
    """
    from src.master_integration_bridge import (
        get_handler_registry, CaseContext
    )
    from pathlib import Path

    registry = get_handler_registry()

    # Create case
    case = CaseContext(
        case_id="2025001234",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[Path("evidence")],
    )

    # Stage 1: INTAKE - Scan and hash all files
    result = await registry.handle_intake_stage(case, {})
    # Result: {
    #   'status': 'completed',
    #   'files_ingested': 145,
    #   'total_size_mb': 523.4,
    #   'hashes_computed': 145,
    # }

    # Stage 2: ANALYSIS - Score evidence by relevance
    result = await registry.handle_analysis_stage(case, {})
    # Result: {
    #   'status': 'completed',
    #   'unique_files': 143,  # 2 duplicates found
    #   'duplicate_files': 2,
    #   'dedup_ratio': 0.013,
    #   'avg_relevance_score': 0.67,
    # }

    # Stage 3: ORGANIZATION - Label exhibits A-Z
    result = await registry.handle_organization_stage(
        case,
        {'output_dir': Path('exhibits')}
    )
    # Result: {
    #   'status': 'completed',
    #   'exhibits_created': 143,
    #   'output_dir': 'exhibits',
    #   'labels_used': ['A', 'B', 'C', ...],
    # }

    # Exhibits now labeled:
    # exhibits/Exhibit_A.pdf (highest relevance)
    # exhibits/Exhibit_B.pdf
    # exhibits/Exhibit_C.pdf
    # ...
    # exhibits/Exhibit_ET.pdf (lowest relevance)


# ============================================================================
# Example 7: State Management and Recovery
# ============================================================================

def example_7_state_management():
    """
    Demonstrate state management and recovery.

    Best for: Understanding checkpointing and resumable execution
    """
    from src.state_manager import get_state_manager

    # Get state manager
    state_mgr = get_state_manager()

    # Create case state
    state = state_mgr.create_case_state(
        case_id="2025001234",
        case_number="2025-001234-CZ",
        case_type="custody",
        workflow_name="custody_modification",
    )

    # Start workflow
    state_mgr.start_workflow("2025001234")

    # Simulate workflow execution with checkpoints
    stages = [
        ("intake_evidence", 0, {"files": 145}),
        ("analyze_evidence", 1, {"unique": 143, "score": 0.67}),
        ("organize_exhibits", 2, {"count": 143}),
        ("generate_motion", 3, {"words": 1247}),
    ]

    for stage_name, stage_idx, state_data in stages:
        # Add checkpoint
        state_mgr.add_checkpoint(
            case_id="2025001234",
            stage_name=stage_name,
            stage_index=stage_idx,
            state=state_data,
            metrics={"duration": 45.2}
        )

        # Log action
        state_mgr.add_audit_log(
            case_id="2025001234",
            action=f"completed_{stage_name}",
            details={"stage_index": stage_idx, "data": state_data}
        )

    # Complete workflow
    state_mgr.complete_workflow("2025001234")

    # Later: Check if can resume
    last_checkpoint = state_mgr.resume_workflow("2025001234")
    if last_checkpoint:
        print(f"Resume from: {last_checkpoint['stage_name']}")

    # Get summary
    summary = state_mgr.get_case_summary("2025001234")
    # Summary includes:
    # {
    #   'case_id': '2025001234',
    #   'status': 'completed',
    #   'stage_count': 4,
    #   'progress': 100.0,
    #   'audit_log': [...],
    # }


# ============================================================================
# Example 8: Error Handling and Recovery
# ============================================================================

async def example_8_error_handling():
    """
    Demonstrate error handling and recovery.

    Best for: Resilient workflow execution
    """
    from src.master_integration_bridge import get_handler_registry
    from src.master_workflow_engine import CaseContext
    from pathlib import Path

    registry = get_handler_registry()

    # Case with missing evidence directory
    case = CaseContext(
        case_id="2025001234",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[Path("nonexistent_dir")],
    )

    # Handler gracefully handles missing directory
    result = await registry.handle_intake_stage(case, {})
    # Result: {
    #   'status': 'completed',
    #   'files_ingested': 0,
    #   'total_size_mb': 0,
    #   'hashes_computed': 0,
    # }
    # (Logs warning but doesn't fail)

    # Workflow continues with downstream handlers


# ============================================================================
# Example 9: CLI Command-Line Usage Patterns
# ============================================================================

def example_9_cli_patterns():
    """
    Various command-line usage patterns.

    Best for: Shell scripting, automation, batch processing
    """

    # Pattern 1: Simple execution with defaults
    # python -m src.master_cli execute \
    #   --case-number "2025-001234-CZ"

    # Pattern 2: Full specification
    # python -m src.master_cli execute \
    #   --case-number "2025-001234-CZ" \
    #   --case-type custody \
    #   --evidence-dir ./evidence \
    #   --output-dir ./output \
    #   --verbose

    # Pattern 3: Dry run (preview)
    # python -m src.master_cli execute \
    #   --case-number "2025-001234-CZ" \
    #   --dry-run

    # Pattern 4: Resume from checkpoint
    # python -m src.master_cli execute \
    #   --case-number "2025-001234-CZ" \
    #   --resume

    # Pattern 5: Check workflow info
    # python -m src.master_cli workflow-info custody_modification

    # Pattern 6: List available workflows
    # python -m src.master_cli workflows

    # Pattern 7: Generate just documents
    # python -m src.master_cli generate-motion \
    #   --case-number "2025-001234-CZ" \
    #   --case-type custody

    # Pattern 8: Validate documents
    # python -m src.master_cli validate \
    #   --case-number "2025-001234-CZ"


# ============================================================================
# Example 10: Complete End-to-End Workflow
# ============================================================================

async def example_10_complete_workflow():
    """
    Complete end-to-end custody case workflow.

    Best for: Understanding complete system operation
    """
    import asyncio
    from pathlib import Path
    from src.master_workflow_engine import WorkflowEngine, CaseContext
    from src.state_manager import get_state_manager

    # Step 1: Initialize
    engine = WorkflowEngine()
    state_mgr = get_state_manager()

    # Step 2: Create case
    case = CaseContext(
        case_id="2025001234",
        case_type="custody",
        case_number="2025-001234-CZ",
        root_directories=[Path("evidence")],
        parties={
            "plaintiff": "John Doe",
            "defendant": "Jane Doe",
        },
    )

    # Step 3: Create case state
    state = state_mgr.create_case_state(
        case_id="2025001234",
        case_number="2025-001234-CZ",
        case_type="custody",
        workflow_name="custody_modification",
    )

    # Step 4: Execute workflow
    print("Starting custody modification workflow...")
    result = await engine.execute_workflow(
        "custody_modification",
        case,
        resume=False
    )

    # Step 5: Process results
    print(f"✓ Workflow completed in {result['duration_seconds']:.1f}s")
    print(f"✓ {result['completed_count']} stages completed")
    print(f"✓ Artifacts generated: {', '.join(result['artifacts'])}")

    # Step 6: Complete case state
    state_mgr.complete_workflow("2025001234")

    # Step 7: Verify output files
    output_files = {
        'exhibits': (Path('output') / 'exhibits').glob('Exhibit_*.pdf'),
        'documents': (Path('output') / 'documents').glob('*.docx'),
        'warboards': (Path('output') / 'warboards').glob('*.svg'),
        'filing': (Path('output') / 'filing').glob('*'),
    }

    for category, files in output_files.items():
        count = len(list(files))
        print(f"✓ {category}: {count} files")

    # Output:
    # Starting custody modification workflow...
    # ✓ Workflow completed in 234.5s
    # ✓ 12 stages completed
    # ✓ Artifacts generated: exhibits, documents, warboards, discovery, filing
    # ✓ exhibits: 143 files
    # ✓ documents: 2 files
    # ✓ warboards: 2 files
    # ✓ filing: 1 file


# ============================================================================
# Usage Guide Summary
# ============================================================================

"""
QUICK REFERENCE

1. Interactive Menu (Easiest):
   python -m src.master_cli interactive

2. Command-Line (Standard):
   python -m src.master_cli execute --case-number "2025-001234-CZ"

3. Programmatic (Advanced):
   from src.master_workflow_engine import WorkflowEngine
   engine = WorkflowEngine()
   result = await engine.execute_workflow("custody_modification", case)

4. Resume Interrupted Workflow:
   python -m src.master_cli execute --case-number "2025-001234-CZ" --resume

5. Check Workflow Available:
   python -m src.master_cli workflows

6. Get Workflow Details:
   python -m src.master_cli workflow-info custody_modification

For complete documentation, see:
- MASTER_WORKFLOW_ARCHITECTURE.md (system overview)
- QUICK_START.md (practical guide)
- src/master_cli.py (CLI implementation)
- src/master_workflow_engine.py (engine implementation)
"""
