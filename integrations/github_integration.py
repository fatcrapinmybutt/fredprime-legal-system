"""
GitHub Integration Module
Provides API connectivity for repository management, CI/CD integration,
and collaborative development features.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from github import Github, GithubException
    HAS_PYGITHUB = True
except ImportError:
    HAS_PYGITHUB = False

logger = logging.getLogger(__name__)


class IssueState(Enum):
    """Issue states"""
    OPEN = "open"
    CLOSED = "closed"


class PRState(Enum):
    """Pull request states"""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


@dataclass
class Repository:
    """Repository information"""
    name: str
    owner: str
    url: str
    description: Optional[str] = None
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    is_private: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Issue:
    """GitHub issue"""
    number: int
    title: str
    body: str
    state: IssueState
    author: str
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    comments_count: int = 0
    url: Optional[str] = None


@dataclass
class PullRequest:
    """GitHub pull request"""
    number: int
    title: str
    body: str
    state: PRState
    author: str
    source_branch: str
    target_branch: str
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    merged_at: Optional[str] = None
    commits_count: int = 0
    additions: int = 0
    deletions: int = 0
    url: Optional[str] = None


@dataclass
class WorkflowRun:
    """GitHub Actions workflow run"""
    id: int
    name: str
    status: str  # queued, in_progress, completed
    conclusion: Optional[str]  # success, failure, neutral, cancelled
    branch: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    duration_seconds: int = 0
    artifacts: List[str] = field(default_factory=list)


class GitHubAPIClient:
    """
    GitHub API client for repository management and CI/CD integration.
    Supports both PyGithub and direct API calls.
    """

    def __init__(self, token: Optional[str] = None, owner: str = "", repo: str = ""):
        """
        Initialize GitHub client

        Args:
            token: GitHub authentication token
            owner: Repository owner
            repo: Repository name
        """
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.github_client = None

        if HAS_PYGITHUB and self.token:
            try:
                self.github_client = Github(self.token)
                logger.info("Initialized PyGithub client")
            except Exception as e:
                logger.warning(f"Failed to initialize PyGithub: {e}")
                self.github_client = None
        elif HAS_PYGITHUB:
            logger.warning("GITHUB_TOKEN not set; using unauthenticated client")
            try:
                self.github_client = Github()
            except Exception as e:
                logger.warning(f"Failed to initialize unauthenticated client: {e}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make API request"""
        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return None

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        url = f"{self.base_url}/{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, params=params)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return None

            if response.status_code >= 400:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_repository(self) -> Optional[Repository]:
        """Get repository information"""
        if not self.owner or not self.repo:
            logger.warning("Owner or repo not set")
            return None

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                return Repository(
                    name=gh_repo.name,
                    owner=self.owner,
                    url=gh_repo.html_url,
                    description=gh_repo.description,
                    language=gh_repo.language,
                    stars=gh_repo.stargazers_count,
                    forks=gh_repo.forks_count,
                    open_issues=gh_repo.open_issues_count,
                    is_private=gh_repo.private,
                    created_at=str(gh_repo.created_at),
                    updated_at=str(gh_repo.updated_at)
                )
            except GithubException as e:
                logger.error(f"GitHub error: {e}")
                return None
        else:
            # Fallback to REST API
            data = self._make_request("GET", f"repos/{self.owner}/{self.repo}")
            if data:
                return Repository(
                    name=data.get("name", ""),
                    owner=self.owner,
                    url=data.get("html_url", ""),
                    description=data.get("description"),
                    language=data.get("language"),
                    stars=data.get("stargazers_count", 0),
                    forks=data.get("forks_count", 0),
                    open_issues=data.get("open_issues_count", 0),
                    is_private=data.get("private", False),
                    created_at=data.get("created_at"),
                    updated_at=data.get("updated_at")
                )

        return None

    def list_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        limit: int = 30
    ) -> List[Issue]:
        """List issues"""
        issues = []

        if not self.owner or not self.repo:
            return issues

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                gh_issues = gh_repo.get_issues(state=state)

                for gh_issue in gh_issues[:limit]:
                    issue_labels = [label.name for label in gh_issue.labels]

                    # Skip if has specific labels and none match
                    if labels and not any(l in issue_labels for l in labels):
                        continue

                    issues.append(
                        Issue(
                            number=gh_issue.number,
                            title=gh_issue.title,
                            body=gh_issue.body or "",
                            state=IssueState(state),
                            author=gh_issue.user.login,
                            labels=issue_labels,
                            assignees=[a.login for a in gh_issue.assignees],
                            created_at=str(gh_issue.created_at),
                            updated_at=str(gh_issue.updated_at),
                            closed_at=str(gh_issue.closed_at) if gh_issue.closed_at else None,
                            comments_count=gh_issue.comments,
                            url=gh_issue.html_url
                        )
                    )
            except GithubException as e:
                logger.error(f"GitHub error: {e}")
        else:
            # Fallback to REST API
            params = {"state": state, "per_page": limit}
            data = self._make_request("GET", f"repos/{self.owner}/{self.repo}/issues", params=params)
            if data:
                for issue_data in data:
                    issue_labels = [label["name"] for label in issue_data.get("labels", [])]

                    # Skip pull requests
                    if "pull_request" in issue_data:
                        continue

                    if labels and not any(l in issue_labels for l in labels):
                        continue

                    issues.append(
                        Issue(
                            number=issue_data.get("number", 0),
                            title=issue_data.get("title", ""),
                            body=issue_data.get("body", ""),
                            state=IssueState(state),
                            author=issue_data.get("user", {}).get("login", ""),
                            labels=issue_labels,
                            assignees=[a.get("login", "") for a in issue_data.get("assignees", [])],
                            created_at=issue_data.get("created_at"),
                            updated_at=issue_data.get("updated_at"),
                            closed_at=issue_data.get("closed_at"),
                            comments_count=issue_data.get("comments", 0),
                            url=issue_data.get("html_url")
                        )
                    )

        return issues

    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Optional[Issue]:
        """Create a new issue"""
        if not self.owner or not self.repo:
            logger.warning("Owner or repo not set")
            return None

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                gh_issue = gh_repo.create_issue(
                    title=title,
                    body=body,
                    labels=labels or [],
                    assignees=assignees or []
                )

                return Issue(
                    number=gh_issue.number,
                    title=gh_issue.title,
                    body=gh_issue.body or "",
                    state=IssueState.OPEN,
                    author=gh_issue.user.login,
                    labels=[label.name for label in gh_issue.labels],
                    assignees=[a.login for a in gh_issue.assignees],
                    created_at=str(gh_issue.created_at),
                    url=gh_issue.html_url
                )
            except GithubException as e:
                logger.error(f"Failed to create issue: {e}")
                return None
        else:
            # Fallback to REST API
            data = {
                "title": title,
                "body": body,
                "labels": labels or [],
                "assignees": assignees or []
            }
            response = self._make_request(
                "POST",
                f"repos/{self.owner}/{self.repo}/issues",
                data=data
            )
            if response:
                return Issue(
                    number=response.get("number", 0),
                    title=response.get("title", ""),
                    body=response.get("body", ""),
                    state=IssueState.OPEN,
                    author=response.get("user", {}).get("login", ""),
                    labels=[label["name"] for label in response.get("labels", [])],
                    assignees=[a.get("login", "") for a in response.get("assignees", [])],
                    created_at=response.get("created_at"),
                    url=response.get("html_url")
                )

        return None

    def list_pull_requests(
        self,
        state: str = "open",
        limit: int = 30
    ) -> List[PullRequest]:
        """List pull requests"""
        prs = []

        if not self.owner or not self.repo:
            return prs

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                gh_prs = gh_repo.get_pulls(state=state)

                for gh_pr in gh_prs[:limit]:
                    prs.append(
                        PullRequest(
                            number=gh_pr.number,
                            title=gh_pr.title,
                            body=gh_pr.body or "",
                            state=PRState(gh_pr.state),
                            author=gh_pr.user.login,
                            source_branch=gh_pr.head.ref,
                            target_branch=gh_pr.base.ref,
                            labels=[label.name for label in gh_pr.labels],
                            reviewers=[r.login for r in gh_pr.get_review_requests()[0]],
                            created_at=str(gh_pr.created_at),
                            updated_at=str(gh_pr.updated_at),
                            merged_at=str(gh_pr.merged_at) if gh_pr.merged_at else None,
                            commits_count=gh_pr.commits,
                            additions=gh_pr.additions,
                            deletions=gh_pr.deletions,
                            url=gh_pr.html_url
                        )
                    )
            except GithubException as e:
                logger.error(f"GitHub error: {e}")

        return prs

    def get_workflow_runs(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 10
    ) -> List[WorkflowRun]:
        """Get workflow runs"""
        runs = []

        if not self.owner or not self.repo:
            return runs

        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return runs

        try:
            endpoint = f"repos/{self.owner}/{self.repo}/actions/runs"
            data = self._make_request("GET", endpoint, params={"per_page": limit})

            if data and "workflow_runs" in data:
                for run_data in data["workflow_runs"]:
                    if workflow_name and workflow_name not in run_data.get("name", ""):
                        continue

                    runs.append(
                        WorkflowRun(
                            id=run_data.get("id", 0),
                            name=run_data.get("name", ""),
                            status=run_data.get("status", "unknown"),
                            conclusion=run_data.get("conclusion"),
                            branch=run_data.get("head_branch", ""),
                            created_at=run_data.get("created_at"),
                            updated_at=run_data.get("updated_at"),
                            duration_seconds=0
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to get workflow runs: {e}")

        return runs

    def trigger_workflow(
        self,
        workflow_id: str,
        ref: str = "main"
    ) -> bool:
        """Trigger a workflow run"""
        if not self.owner or not self.repo:
            return False

        if not HAS_REQUESTS:
            logger.warning("requests library not available")
            return False

        try:
            endpoint = f"repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches"
            data = {"ref": ref}
            response = self._make_request("POST", endpoint, data=data)
            return response is not None
        except Exception as e:
            logger.error(f"Failed to trigger workflow: {e}")
            return False

    def add_labels_to_issue(self, issue_number: int, labels: List[str]) -> bool:
        """Add labels to an issue"""
        if not self.owner or not self.repo:
            return False

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                issue = gh_repo.get_issue(issue_number)
                issue.add_to_labels(*labels)
                return True
            except GithubException as e:
                logger.error(f"Failed to add labels: {e}")
                return False
        else:
            # Fallback to REST API
            endpoint = f"repos/{self.owner}/{self.repo}/issues/{issue_number}/labels"
            data = {"labels": labels}
            response = self._make_request("POST", endpoint, data=data)
            return response is not None

    def update_issue(
        self,
        issue_number: int,
        state: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None
    ) -> bool:
        """Update an issue"""
        if not self.owner or not self.repo:
            return False

        update_data = {}
        if state:
            update_data["state"] = state
        if title:
            update_data["title"] = title
        if body:
            update_data["body"] = body

        if not update_data:
            return False

        if self.github_client:
            try:
                gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
                issue = gh_repo.get_issue(issue_number)
                if state:
                    issue.edit(state=state)
                if title:
                    issue.edit(title=title)
                if body:
                    issue.edit(body=body)
                return True
            except GithubException as e:
                logger.error(f"Failed to update issue: {e}")
                return False
        else:
            # Fallback to REST API
            endpoint = f"repos/{self.owner}/{self.repo}/issues/{issue_number}"
            response = self._make_request("PATCH", endpoint, data=update_data)
            return response is not None

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """Create a new branch"""
        if not self.owner or not self.repo:
            return False

        if not self.github_client:
            logger.warning("PyGithub not available")
            return False

        try:
            gh_repo = self.github_client.get_user(self.owner).get_repo(self.repo)
            base = gh_repo.get_branch(from_branch)
            gh_repo.create_git_ref(f"refs/heads/{branch_name}", base.commit.sha)
            return True
        except GithubException as e:
            logger.error(f"Failed to create branch: {e}")
            return False

    def export_issues(self, issues: List[Issue], format: str = "json") -> str:
        """Export issues in various formats"""
        if format == "json":
            return json.dumps(
                [asdict(issue) for issue in issues],
                indent=2,
                default=str
            )
        elif format == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            if issues:
                fieldnames = ["number", "title", "state", "author", "labels", "created_at"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

                for issue in issues:
                    writer.writerow({
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state.value,
                        "author": issue.author,
                        "labels": ",".join(issue.labels),
                        "created_at": issue.created_at
                    })

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # Example usage
    client = GitHubAPIClient(
        owner="username",
        repo="litigation-system"
    )

    # Get repository
    repo = client.get_repository()
    if repo:
        print(f"Repository: {repo.name}")
        print(f"Stars: {repo.stars}")

    # List issues
    issues = client.list_issues(state="open", limit=5)
    print(f"\nOpen Issues: {len(issues)}")
    for issue in issues:
        print(f"- #{issue.number}: {issue.title}")
