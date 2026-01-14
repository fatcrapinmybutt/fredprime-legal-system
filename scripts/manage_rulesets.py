#!/usr/bin/env python3
"""
GitHub Repository Ruleset Management Tool

This script helps create and manage GitHub repository rulesets via the GitHub API.
Requires: gh CLI with authentication, admin access to the repository
"""

import json
import subprocess
import sys
from typing import Dict, Any, Optional


class GitHubRulesetManager:
    """Manages GitHub repository rulesets"""
    
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.repo_path = f"{owner}/{repo}"
    
    def _run_gh_api(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Run GitHub CLI API command"""
        cmd = ["gh", "api", "--method", method, endpoint]
        
        if data:
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    cmd.extend(["-f", f"{key}={json.dumps(value)}"])
                else:
                    cmd.extend(["-f", f"{key}={value}"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout) if result.stdout else {}
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}", file=sys.stderr)
            return {"error": e.stderr}
    
    def create_ruleset(self, ruleset_config: Dict[str, Any]) -> Dict:
        """Create a new repository ruleset"""
        endpoint = f"repos/{self.repo_path}/rulesets"
        
        return self._run_gh_api("POST", endpoint, ruleset_config)
    
    def list_rulesets(self) -> Dict:
        """List all rulesets for the repository"""
        endpoint = f"repos/{self.repo_path}/rulesets"
        
        return self._run_gh_api("GET", endpoint)
    
    def delete_ruleset(self, ruleset_id: int) -> Dict:
        """Delete a ruleset by ID"""
        endpoint = f"repos/{self.repo_path}/rulesets/{ruleset_id}"
        
        return self._run_gh_api("DELETE", endpoint)
    
    def update_ruleset(self, ruleset_id: int, updates: Dict[str, Any]) -> Dict:
        """Update an existing ruleset"""
        endpoint = f"repos/{self.repo_path}/rulesets/{ruleset_id}"
        
        return self._run_gh_api("PATCH", endpoint, updates)


def load_rulesets_config(config_path: str = ".github/rulesets.json") -> Dict:
    """Load ruleset configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 manage_rulesets.py [list|create|delete|update]")
        print("  list              - List all rulesets")
        print("  create            - Create rulesets from .github/rulesets.json")
        print("  delete <id>       - Delete ruleset by ID")
        print("  update <id>       - Update ruleset by ID")
        sys.exit(1)
    
    manager = GitHubRulesetManager("fatcrapinmybutt", "fredprime-legal-system")
    command = sys.argv[1]
    
    if command == "list":
        result = manager.list_rulesets()
        print(json.dumps(result, indent=2))
    
    elif command == "create":
        config = load_rulesets_config()
        for ruleset in config.get("rulesets", []):
            print(f"Creating ruleset: {ruleset['name']}")
            result = manager.create_ruleset(ruleset)
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Created: {result.get('id', 'N/A')}")
    
    elif command == "delete" and len(sys.argv) > 2:
        ruleset_id = int(sys.argv[2])
        result = manager.delete_ruleset(ruleset_id)
        print(f"Deleted ruleset {ruleset_id}: {result}")
    
    elif command == "update" and len(sys.argv) > 2:
        ruleset_id = int(sys.argv[2])
        config = load_rulesets_config()
        if config.get("rulesets"):
            result = manager.update_ruleset(ruleset_id, config["rulesets"][0])
            print(json.dumps(result, indent=2))
    
    else:
        print("Invalid command")
        sys.exit(1)


if __name__ == "__main__":
    main()
