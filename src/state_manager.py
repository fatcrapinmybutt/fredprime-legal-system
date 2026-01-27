"""
FRED Supreme Litigation OS - State Management System

Persistent state management for case lifecycle tracking, checkpoint resumption,
and comprehensive audit logging.

Features:
- JSON-based state persistence
- Checkpoint system for resumable workflows
- Full audit trail with timestamps
- State validation and integrity checking
- Automatic backup and recovery
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StateCheckpoint:
    """A single checkpoint in the workflow execution."""
    id: str
    timestamp: str
    stage_name: str
    stage_index: int
    state: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class CaseState:
    """Complete case state snapshot."""
    case_id: str
    case_number: str
    case_type: str
    workflow_name: str
    status: str  # "pending", "in_progress", "completed", "failed"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoints: List[StateCheckpoint] = field(default_factory=list)
    current_stage: Optional[str] = None
    current_stage_index: int = 0
    evidence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Add a workflow checkpoint."""
        self.checkpoints.append(checkpoint)

    def add_audit_log(self, action: str, details: Optional[str] = None) -> None:
        """Add audit log entry."""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details,
        })

    def add_error(self, stage: str, error_message: str) -> None:
        """Add error to the error log."""
        self.errors.append({
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "error": error_message,
        })

    def get_progress_percentage(self, total_stages: int) -> float:
        """Calculate workflow progress percentage."""
        if total_stages == 0:
            return 0.0
        return (self.current_stage_index / total_stages) * 100

    def can_resume(self) -> bool:
        """Check if workflow can be resumed."""
        return len(self.checkpoints) > 0 and self.status == "in_progress"


class StateManager:
    """Manages persistent case state."""

    def __init__(self, state_dir: Path = Path("state")):
        self.state_dir = state_dir
        self.state_dir.mkdir(exist_ok=True)
        self.cases: Dict[str, CaseState] = {}
        self._load_all_cases()

    def _load_all_cases(self) -> None:
        """Load all case states from disk."""
        for state_file in self.state_dir.glob("case_*.json"):
            try:
                with open(state_file) as f:
                    case_data = json.load(f)
                    # Deserialize
                    checkpoints = [
                        StateCheckpoint(**cp) for cp in case_data.get("checkpoints", [])
                    ]
                    case_data["checkpoints"] = checkpoints
                    case_state = CaseState(**case_data)
                    self.cases[case_state.case_id] = case_state
                    logger.info(f"Loaded case state: {case_state.case_number}")
            except Exception as e:
                logger.error(f"Failed to load {state_file}: {e}")

    def create_case_state(
        self,
        case_id: str,
        case_number: str,
        case_type: str,
        workflow_name: str,
    ) -> CaseState:
        """Create a new case state."""
        case_state = CaseState(
            case_id=case_id,
            case_number=case_number,
            case_type=case_type,
            workflow_name=workflow_name,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
        )
        self.cases[case_id] = case_state
        self.save_case_state(case_id)
        logger.info(f"Created case state: {case_number}")
        return case_state

    def get_case_state(self, case_id: str) -> Optional[CaseState]:
        """Get case state by ID."""
        return self.cases.get(case_id)

    def start_workflow(self, case_id: str) -> None:
        """Mark workflow as started."""
        if case_state := self.get_case_state(case_id):
            case_state.status = "in_progress"
            case_state.started_at = datetime.utcnow().isoformat()
            case_state.add_audit_log("workflow_started")
            self.save_case_state(case_id)

    def add_checkpoint(
        self,
        case_id: str,
        stage_name: str,
        stage_index: int,
        state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a workflow checkpoint."""
        if case_state := self.get_case_state(case_id):
            checkpoint = StateCheckpoint(
                id=f"{case_id}_{stage_index}_{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow().isoformat(),
                stage_name=stage_name,
                stage_index=stage_index,
                state=state,
                metrics=metrics or {},
            )
            case_state.add_checkpoint(checkpoint)
            case_state.current_stage = stage_name
            case_state.current_stage_index = stage_index
            case_state.add_audit_log(f"checkpoint_saved", stage_name)
            self.save_case_state(case_id)
            logger.info(f"Checkpoint saved for {case_id}: {stage_name}")

    def complete_workflow(self, case_id: str) -> None:
        """Mark workflow as completed."""
        if case_state := self.get_case_state(case_id):
            case_state.status = "completed"
            case_state.completed_at = datetime.utcnow().isoformat()
            case_state.add_audit_log("workflow_completed")
            self.save_case_state(case_id)
            logger.info(f"Workflow completed: {case_state.case_number}")

    def fail_workflow(self, case_id: str, error: str) -> None:
        """Mark workflow as failed."""
        if case_state := self.get_case_state(case_id):
            case_state.status = "failed"
            case_state.add_error(case_state.current_stage or "unknown", error)
            case_state.add_audit_log("workflow_failed", error)
            self.save_case_state(case_id)
            logger.error(f"Workflow failed: {case_state.case_number} - {error}")

    def save_case_state(self, case_id: str) -> Path:
        """Save case state to disk."""
        if not (case_state := self.get_case_state(case_id)):
            raise ValueError(f"Case not found: {case_id}")

        state_file = self.state_dir / f"case_{case_id}.json"

        # Create backup before overwriting
        if state_file.exists():
            backup_file = self.state_dir / f"case_{case_id}.backup.json"
            backup_file.write_text(state_file.read_text())

        # Serialize case state
        case_dict = {
            "case_id": case_state.case_id,
            "case_number": case_state.case_number,
            "case_type": case_state.case_type,
            "workflow_name": case_state.workflow_name,
            "status": case_state.status,
            "created_at": case_state.created_at,
            "started_at": case_state.started_at,
            "completed_at": case_state.completed_at,
            "current_stage": case_state.current_stage,
            "current_stage_index": case_state.current_stage_index,
            "evidence_count": case_state.evidence_count,
            "checkpoints": [asdict(cp) for cp in case_state.checkpoints],
            "metadata": case_state.metadata,
            "audit_log": case_state.audit_log,
            "errors": case_state.errors,
        }

        with open(state_file, "w") as f:
            json.dump(case_dict, f, indent=2, default=str)

        logger.info(f"Case state saved: {state_file}")
        return state_file

    def resume_workflow(self, case_id: str) -> Optional[StateCheckpoint]:
        """Get the last checkpoint for resuming workflow."""
        if case_state := self.get_case_state(case_id):
            if case_state.can_resume() and case_state.checkpoints:
                return case_state.checkpoints[-1]
        return None

    def get_case_summary(self, case_id: str) -> Dict[str, Any]:
        """Get summary statistics for a case."""
        if not (case_state := self.get_case_state(case_id)):
            return {}

        total_stages = case_state.current_stage_index + 1
        completed_checkpoints = len(case_state.checkpoints)

        return {
            "case_number": case_state.case_number,
            "case_type": case_state.case_type,
            "status": case_state.status,
            "workflow": case_state.workflow_name,
            "created_at": case_state.created_at,
            "started_at": case_state.started_at,
            "completed_at": case_state.completed_at,
            "current_stage": case_state.current_stage,
            "progress_percentage": case_state.get_progress_percentage(total_stages),
            "checkpoints_saved": completed_checkpoints,
            "audit_log_entries": len(case_state.audit_log),
            "errors_encountered": len(case_state.errors),
        }

    def list_cases(self) -> List[Dict[str, Any]]:
        """List all cases with summaries."""
        return [
            self.get_case_summary(case_id)
            for case_id in sorted(self.cases.keys())
        ]

    def cleanup_old_cases(self, days: int = 30) -> int:
        """Remove case states older than specified days."""
        import time
        cutoff_time = time.time() - (days * 86400)
        removed = 0

        for state_file in self.state_dir.glob("case_*.json"):
            if state_file.stat().st_mtime < cutoff_time:
                state_file.unlink()
                removed += 1

        logger.info(f"Cleaned up {removed} old case states")
        return removed

    def verify_state_integrity(self, case_id: str) -> Dict[str, Any]:
        """Verify integrity of case state."""
        if not (case_state := self.get_case_state(case_id)):
            return {"valid": False, "error": "Case not found"}

        issues = []

        # Check timestamps are in order
        prev_timestamp = case_state.created_at
        for checkpoint in case_state.checkpoints:
            if checkpoint.timestamp < prev_timestamp:
                issues.append(f"Out-of-order checkpoint: {checkpoint.id}")
            prev_timestamp = checkpoint.timestamp

        # Check stage indices are sequential
        prev_index = -1
        for checkpoint in case_state.checkpoints:
            if checkpoint.stage_index <= prev_index:
                issues.append(f"Non-sequential stage index: {checkpoint.stage_index}")
            prev_index = checkpoint.stage_index

        return {
            "valid": len(issues) == 0,
            "case_id": case_id,
            "checkpoints": len(case_state.checkpoints),
            "issues": issues,
        }


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager(state_dir: Path = Path("state")) -> StateManager:
    """Get or create global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(state_dir)
    return _state_manager
