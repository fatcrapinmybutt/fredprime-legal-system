"""
FRED Supreme Litigation OS - Master Workflow Orchestration Engine

Unified orchestration for all litigation workflows with:
- Declarative workflow definitions
- Intelligent state management
- Multi-stage pipeline orchestration
- Smart routing and dependency resolution
- Rich CLI with TUI menus
- Full offline capability (no external APIs)
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import yaml


# ============================================================================
# ENUMS & TYPES
# ============================================================================


class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageType(Enum):
    """Types of workflow stages."""
    INTAKE = "intake"                    # Evidence ingestion
    ANALYSIS = "analysis"                # Evidence analysis
    ORGANIZATION = "organization"        # File organization & labeling
    GENERATION = "generation"            # Document generation
    VALIDATION = "validation"            # Compliance validation
    WARBOARDING = "warboarding"          # Timeline & visualization
    DISCOVERY = "discovery"              # Discovery preparation
    FILING = "filing"                    # Court filing
    NOTIFICATION = "notification"        # Alerts & notifications
    ARCHIVAL = "archival"                # Case archival


class CaseType(Enum):
    """Michigan litigation case types."""
    CUSTODY = "custody"                  # Custody/parenting time
    HOUSING = "housing"                  # Housing/eviction
    CHILD_SUPPORT = "child_support"      # Child support modification
    PPO = "ppo"                          # Personal Protection Order
    CONTEMPT = "contempt"                # Contempt proceedings
    ENFORCEMENT = "enforcement"          # Judgment enforcement
    APPEAL = "appeal"                    # Appeal/Extraordinary writs
    FAMILY_OTHER = "family_other"        # Other family law


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class FileRecord:
    """Evidence file metadata."""
    id: str = field(default_factory=lambda: str(uuid4()))
    path: Path = field(default_factory=Path)
    sha256: str = ""
    size_bytes: int = 0
    ext: str = ""
    bates_number: str = ""
    exhibit_label: str = ""
    case_id: str = ""
    evidence_type: str = ""
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result of a workflow stage execution."""
    stage_name: str
    state: WorkflowState
    duration_seconds: float = 0.0
    records_processed: int = 0
    records_failed: int = 0
    output_artifacts: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CaseContext:
    """Runtime context for a litigation case."""
    case_id: str
    case_type: CaseType
    case_number: str = ""
    court: str = "michigan"
    parties: Dict[str, str] = field(default_factory=dict)
    root_directories: List[Path] = field(default_factory=list)
    output_directory: Path = field(default_factory=Path)
    evidence_files: List[FileRecord] = field(default_factory=list)
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    stage_results: List[StageResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: StageResult) -> None:
        """Add a stage result to the case context."""
        self.stage_results.append(result)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get workflow state."""
        return self.workflow_state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set workflow state."""
        self.workflow_state[key] = value


@dataclass
class WorkflowStage:
    """Definition of a workflow stage."""
    name: str
    stage_type: StageType
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    handler: Optional[Callable[[CaseContext], StageResult]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 3600
    retry_count: int = 3
    skip_on_error: bool = False


@dataclass
class LitigationWorkflow:
    """Complete litigation workflow definition."""
    name: str
    case_types: List[CaseType] = field(default_factory=list)
    description: str = ""
    stages: List[WorkflowStage] = field(default_factory=list)
    parallel: bool = False
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_stage(self, name: str) -> Optional[WorkflowStage]:
        """Get stage by name."""
        return next((s for s in self.stages if s.name == name), None)

    def get_execution_order(self) -> List[str]:
        """Topological sort of stages by dependencies."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            stage = self.get_stage(name)
            if stage:
                for dep in stage.dependencies:
                    visit(dep)
            order.append(name)

        for stage in self.stages:
            visit(stage.name)
        return order


# ============================================================================
# WORKFLOW ENGINE (CORE ORCHESTRATOR)
# ============================================================================


class WorkflowEngine:
    """Master orchestration engine for litigation workflows."""

    def __init__(self, config_dir: Path = Path("config"), log_dir: Path = Path("logs")):
        self.config_dir = config_dir
        self.log_dir = log_dir
        self.workflows: Dict[str, LitigationWorkflow] = {}
        self.case_contexts: Dict[str, CaseContext] = {}
        self._setup_logging()
        self._load_workflows()

    def _setup_logging(self) -> None:
        """Configure logging."""
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / f"fredprime_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("WorkflowEngine")

    def _load_workflows(self) -> None:
        """Load workflow definitions from YAML files."""
        self.config_dir.mkdir(exist_ok=True)
        for yaml_file in self.config_dir.glob("workflows/*.yaml"):
            try:
                with open(yaml_file) as f:
                    workflow_def = yaml.safe_load(f)
                    workflow = self._parse_workflow_def(workflow_def)
                    self.workflows[workflow.name] = workflow
                    self.logger.info(f"Loaded workflow: {workflow.name}")
            except Exception as e:
                self.logger.error(f"Failed to load workflow {yaml_file}: {e}")

    def _parse_workflow_def(self, workflow_def: Dict[str, Any]) -> LitigationWorkflow:
        """Parse workflow definition from YAML."""
        case_types = [CaseType[t] for t in workflow_def.get("case_types", [])]
        stages = [
            WorkflowStage(
                name=s["name"],
                stage_type=StageType[s.get("type", "analysis").upper()],
                description=s.get("description", ""),
                dependencies=s.get("dependencies", []),
                enabled=s.get("enabled", True),
                config=s.get("config", {}),
                timeout_seconds=s.get("timeout", 3600),
                retry_count=s.get("retry", 3),
                skip_on_error=s.get("skip_on_error", False),
            )
            for s in workflow_def.get("stages", [])
        ]
        return LitigationWorkflow(
            name=workflow_def["name"],
            case_types=case_types,
            description=workflow_def.get("description", ""),
            stages=stages,
            parallel=workflow_def.get("parallel", False),
            version=workflow_def.get("version", "1.0"),
            metadata=workflow_def.get("metadata", {}),
        )

    async def execute_workflow(
        self,
        workflow_name: str,
        case_context: CaseContext,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Execute a workflow for a case."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")

        workflow = self.workflows[workflow_name]
        self.case_contexts[case_context.case_id] = case_context

        self.logger.info(f"Executing workflow '{workflow_name}' for case {case_context.case_id}")

        execution_order = workflow.get_execution_order()
        start_time = datetime.utcnow()

        for stage_name in execution_order:
            stage = workflow.get_stage(stage_name)
            if not stage or not stage.enabled:
                continue

            # Check if stage was already completed (resume mode)
            if resume and any(r.stage_name == stage_name and r.state == WorkflowState.COMPLETED
                            for r in case_context.stage_results):
                self.logger.info(f"Skipping completed stage: {stage_name}")
                continue

            try:
                self.logger.info(f"Running stage: {stage_name}")
                result = await self._run_stage(stage, case_context)
                case_context.add_result(result)
            except Exception as e:
                self.logger.error(f"Stage failed: {stage_name}: {e}")
                if not stage.skip_on_error:
                    raise

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        return {
            "case_id": case_context.case_id,
            "workflow": workflow_name,
            "duration_seconds": elapsed,
            "status": "completed",
            "stage_results": [asdict(r) for r in case_context.stage_results],
        }

    async def _run_stage(self, stage: WorkflowStage, context: CaseContext) -> StageResult:
        """Execute a single workflow stage."""
        start_time = datetime.utcnow()
        result = StageResult(
            stage_name=stage.name,
            state=WorkflowState.RUNNING,
        )

        try:
            # Use custom handler if provided
            if stage.handler:
                result = await asyncio.wait_for(
                    asyncio.to_thread(stage.handler, context),
                    timeout=stage.timeout_seconds,
                )
            else:
                # Default: run built-in handler for stage type
                result = await self._run_builtin_stage(stage, context)

            result.state = WorkflowState.COMPLETED
        except asyncio.TimeoutError:
            result.state = WorkflowState.FAILED
            result.errors.append(f"Stage timeout after {stage.timeout_seconds} seconds")
        except Exception as e:
            result.state = WorkflowState.FAILED
            result.errors.append(str(e))

        result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        return result

    async def _run_builtin_stage(self, stage: WorkflowStage, context: CaseContext) -> StageResult:
        """Run built-in stage handler based on stage type."""
        handlers = {
            StageType.INTAKE: self._stage_intake,
            StageType.ORGANIZATION: self._stage_organization,
            StageType.ANALYSIS: self._stage_analysis,
            StageType.GENERATION: self._stage_generation,
            StageType.VALIDATION: self._stage_validation,
            StageType.WARBOARDING: self._stage_warboarding,
            StageType.DISCOVERY: self._stage_discovery,
            StageType.FILING: self._stage_filing,
        }

        handler = handlers.get(stage.stage_type)
        if handler:
            return await asyncio.to_thread(handler, context, stage.config)

        return StageResult(
            stage_name=stage.name,
            state=WorkflowState.SKIPPED,
            errors=["No handler for stage type"],
        )

    # Built-in stage handlers
    def _stage_intake(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Evidence ingestion stage."""
        result = StageResult(stage_name="intake", state=WorkflowState.RUNNING)
        # Implementation: scan root directories, hash files, build manifest
        result.records_processed = len(context.evidence_files)
        return result

    def _stage_organization(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """File organization and labeling stage."""
        result = StageResult(stage_name="organization", state=WorkflowState.RUNNING)
        # Implementation: label exhibits, organize by category
        result.records_processed = len(context.evidence_files)
        return result

    def _stage_analysis(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Evidence analysis stage."""
        result = StageResult(stage_name="analysis", state=WorkflowState.RUNNING)
        # Implementation: extract text, build search indices, score relevance
        result.records_processed = len(context.evidence_files)
        return result

    def _stage_generation(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Document generation stage."""
        result = StageResult(stage_name="generation", state=WorkflowState.RUNNING)
        # Implementation: generate motions, affidavits, notices from templates
        return result

    def _stage_validation(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Compliance validation stage."""
        result = StageResult(stage_name="validation", state=WorkflowState.RUNNING)
        # Implementation: check MCR compliance, verify signatures, validate exhibits
        return result

    def _stage_warboarding(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Timeline and visualization stage."""
        result = StageResult(stage_name="warboarding", state=WorkflowState.RUNNING)
        # Implementation: build timeline, generate warboards, create visualizations
        return result

    def _stage_discovery(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Discovery preparation stage."""
        result = StageResult(stage_name="discovery", state=WorkflowState.RUNNING)
        # Implementation: generate discovery requests, build privilege logs
        return result

    def _stage_filing(self, context: CaseContext, config: Dict[str, Any]) -> StageResult:
        """Court filing stage."""
        result = StageResult(stage_name="filing", state=WorkflowState.RUNNING)
        # Implementation: prepare MiFile bundles, generate notices of service
        return result

    def list_workflows(self) -> List[str]:
        """List all available workflows."""
        return list(self.workflows.keys())

    def get_workflow_info(self, workflow_name: str) -> Dict[str, Any]:
        """Get detailed information about a workflow."""
        if workflow_name not in self.workflows:
            return {}
        workflow = self.workflows[workflow_name]
        return {
            "name": workflow.name,
            "description": workflow.description,
            "case_types": [ct.value for ct in workflow.case_types],
            "stages": [
                {
                    "name": s.name,
                    "type": s.stage_type.value,
                    "description": s.description,
                    "dependencies": s.dependencies,
                }
                for s in workflow.stages
            ],
            "version": workflow.version,
        }

    def save_case_state(self, case_id: str, output_dir: Path) -> Path:
        """Save case context to disk."""
        if case_id not in self.case_contexts:
            raise ValueError(f"Case not found: {case_id}")

        output_dir.mkdir(exist_ok=True)
        context = self.case_contexts[case_id]
        state_file = output_dir / f"case_{case_id}_state.json"

        with open(state_file, "w") as f:
            json.dump(
                {
                    "case_id": context.case_id,
                    "case_type": context.case_type.value,
                    "case_number": context.case_number,
                    "court": context.court,
                    "parties": context.parties,
                    "evidence_files": [asdict(ef) for ef in context.evidence_files],
                    "stage_results": [asdict(sr) for sr in context.stage_results],
                    "workflow_state": context.workflow_state,
                    "created_at": context.created_at,
                },
                f,
                indent=2,
                default=str,
            )

        self.logger.info(f"Case state saved to {state_file}")
        return state_file


# ============================================================================
# PRE-BUILT WORKFLOW TEMPLATES
# ============================================================================


def create_custody_workflow() -> LitigationWorkflow:
    """Create a standard custody/parenting time modification workflow."""
    return LitigationWorkflow(
        name="custody_modification",
        case_types=[CaseType.CUSTODY],
        description="Complete custody modification workflow with motion, exhibits, and filing",
        stages=[
            WorkflowStage(
                name="intake_evidence",
                stage_type=StageType.INTAKE,
                description="Scan and ingest evidence files",
                enabled=True,
            ),
            WorkflowStage(
                name="organize_exhibits",
                stage_type=StageType.ORGANIZATION,
                description="Label exhibits A-Z by relevance",
                dependencies=["intake_evidence"],
                enabled=True,
            ),
            WorkflowStage(
                name="analyze_timeline",
                stage_type=StageType.ANALYSIS,
                description="Build chronological timeline",
                dependencies=["organize_exhibits"],
                enabled=True,
            ),
            WorkflowStage(
                name="generate_motion",
                stage_type=StageType.GENERATION,
                description="Generate custody modification motion",
                dependencies=["analyze_timeline"],
                enabled=True,
            ),
            WorkflowStage(
                name="build_warboard",
                stage_type=StageType.WARBOARDING,
                description="Create visual timeline and custody interference map",
                dependencies=["analyze_timeline"],
                enabled=True,
            ),
            WorkflowStage(
                name="validate_compliance",
                stage_type=StageType.VALIDATION,
                description="Verify MCR/MCL compliance",
                dependencies=["generate_motion", "build_warboard"],
                enabled=True,
            ),
            WorkflowStage(
                name="prepare_filing",
                stage_type=StageType.FILING,
                description="Prepare MiFile bundle for court filing",
                dependencies=["validate_compliance"],
                enabled=True,
            ),
        ],
        version="1.0",
    )


def create_housing_workflow() -> LitigationWorkflow:
    """Create a standard housing/eviction defense workflow."""
    return LitigationWorkflow(
        name="housing_emergency",
        case_types=[CaseType.HOUSING],
        description="Emergency housing intervention with injunction motion",
        stages=[
            WorkflowStage(
                name="intake_evidence",
                stage_type=StageType.INTAKE,
                description="Rapid ingestion of evidence",
                enabled=True,
            ),
            WorkflowStage(
                name="organize_exhibits",
                stage_type=StageType.ORGANIZATION,
                description="Quick exhibit organization",
                dependencies=["intake_evidence"],
                enabled=True,
            ),
            WorkflowStage(
                name="analyze_harm",
                stage_type=StageType.ANALYSIS,
                description="Analyze irreparable harm and imminent danger",
                dependencies=["organize_exhibits"],
                enabled=True,
            ),
            WorkflowStage(
                name="generate_injunction",
                stage_type=StageType.GENERATION,
                description="Generate emergency injunction motion",
                dependencies=["analyze_harm"],
                enabled=True,
            ),
            WorkflowStage(
                name="validate_compliance",
                stage_type=StageType.VALIDATION,
                description="Verify MCR/MCL emergency filing requirements",
                dependencies=["generate_injunction"],
                enabled=True,
            ),
            WorkflowStage(
                name="prepare_filing",
                stage_type=StageType.FILING,
                description="Prepare for immediate court filing",
                dependencies=["validate_compliance"],
                enabled=True,
            ),
        ],
        version="1.0",
    )


def create_ppo_workflow() -> LitigationWorkflow:
    """Create a Personal Protection Order (PPO) defense workflow."""
    return LitigationWorkflow(
        name="ppo_defense",
        case_types=[CaseType.PPO],
        description="Comprehensive PPO defense with evidence and testimony preparation",
        stages=[
            WorkflowStage(
                name="intake_evidence",
                stage_type=StageType.INTAKE,
                description="Ingest evidence supporting defense",
                enabled=True,
            ),
            WorkflowStage(
                name="organize_exhibits",
                stage_type=StageType.ORGANIZATION,
                description="Organize exhibits by category",
                dependencies=["intake_evidence"],
                enabled=True,
            ),
            WorkflowStage(
                name="analyze_allegations",
                stage_type=StageType.ANALYSIS,
                description="Analyze and refute allegations",
                dependencies=["organize_exhibits"],
                enabled=True,
            ),
            WorkflowStage(
                name="build_warboard",
                stage_type=StageType.WARBOARDING,
                description="Create PPO interference/false allegation warboard",
                dependencies=["analyze_allegations"],
                enabled=True,
            ),
            WorkflowStage(
                name="prepare_response",
                stage_type=StageType.GENERATION,
                description="Generate PPO response motion",
                dependencies=["build_warboard"],
                enabled=True,
            ),
            WorkflowStage(
                name="validate_compliance",
                stage_type=StageType.VALIDATION,
                description="Verify MCR/MCL compliance",
                dependencies=["prepare_response"],
                enabled=True,
            ),
            WorkflowStage(
                name="prepare_filing",
                stage_type=StageType.FILING,
                description="Prepare filing package",
                dependencies=["validate_compliance"],
                enabled=True,
            ),
        ],
        version="1.0",
    )


# ============================================================================
# MAIN & CLI
# ============================================================================


async def main():
    """Main entry point."""
    engine = WorkflowEngine()

    # Register pre-built workflows
    for workflow in [
        create_custody_workflow(),
        create_housing_workflow(),
        create_ppo_workflow(),
    ]:
        engine.workflows[workflow.name] = workflow

    print("\n" + "=" * 80)
    print("FRED SUPREME LITIGATION OS - Master Workflow Orchestration Engine")
    print("=" * 80 + "\n")

    # Create sample case context
    case_context = CaseContext(
        case_id=str(uuid4()),
        case_type=CaseType.CUSTODY,
        case_number="2025-001234-CZ",
        court="michigan",
        parties={"plaintiff": "You", "defendant": "Other Party"},
        root_directories=[Path("evidence")],
        output_directory=Path("output"),
    )

    # Execute custody workflow
    print(f"Executing custody workflow for case: {case_context.case_id}")
    result = await engine.execute_workflow("custody_modification", case_context)

    print(f"\nWorkflow execution completed!")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")
    print(f"Stages executed: {len(result['stage_results'])}")

    # Save case state
    engine.save_case_state(case_context.case_id, Path("output"))


if __name__ == "__main__":
    asyncio.run(main())
