"""Agent 006: Michigan court specialist scaffold."""

from typing import Optional
import agents.michigan_reference as ref
from pathlib import Path

class Agent:
    def __init__(self, config_path: Optional[str] = None):
        self.id = "agent_006"
        self.config = {}
        if config_path:
            p = Path(config_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        self.config[k.strip()] = v.strip().strip('"')

    def summarize_rule(self, rule_key: str) -> str:
        data = ref.get_rule(rule_key)
        if not data:
            return f"Rule {rule_key} not found in local reference."
        return f"{rule_key} — {data['title']}: {data['summary']} (source: {data['source']})"

    def search(self, query: str):
        return ref.search_rules(query)

    def run(self, context: Optional[dict] = None):
        print(f"Agent: {self.id} (Michigan specialist)")
        print("Known jurisdiction: Michigan")
        print(self.summarize_rule("MRE 403"))


def main():
    a = Agent(config_path=str(Path(__file__).parent / "config.yaml"))
    a.run()

if __name__ == "__main__":
    main()
