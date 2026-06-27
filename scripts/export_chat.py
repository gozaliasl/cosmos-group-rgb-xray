#!/usr/bin/env python3
"""
Export a Claude Code conversation (.jsonl) to a readable Markdown file.

Usage
-----
  python scripts/export_chat.py <session.jsonl> [output.md]

  # Export this project's latest session
  python scripts/export_chat.py \
      ~/.claude/projects/-Users-gozalig1-Projects-PHOTOSHOP-scripting/*.jsonl \
      chat_export.md

If output path is omitted, prints to stdout.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def jsonl_to_markdown(jsonl_path: Path) -> str:
    lines = []
    lines.append(f"# Claude Code Chat Export\n")
    lines.append(f"**Source:** `{jsonl_path.name}`  ")
    lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")

    def extract_text(content) -> str:
        """Pull plain text out of string or list-of-blocks content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "tool_use":
                    name = block.get("name", "tool")
                    inp  = json.dumps(block.get("input", {}), indent=2)
                    parts.append(f"*[Tool: `{name}`]*\n```json\n{inp}\n```")
                elif t == "tool_result":
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = "\n".join(
                            b.get("text", "") for b in result
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    parts.append(f"*[Tool result]*\n```\n{str(result)[:1000]}\n```")
            return "\n\n".join(parts)
        return ""

    with open(jsonl_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = entry.get("type", "")

            # Claude Code format: user/assistant turns stored under "message" key
            if etype == "user":
                msg = entry.get("message", {})
                content = extract_text(msg.get("content", ""))
                if content.strip():
                    lines.append(f"## 👤 User\n\n{content}\n")

            elif etype == "assistant":
                msg = entry.get("message", {})
                content = extract_text(msg.get("content", ""))
                if content.strip():
                    lines.append(f"## 🤖 Assistant\n\n{content}\n")

            elif etype == "summary":
                content = extract_text(entry.get("summary", entry.get("content", "")))
                if content.strip():
                    lines.append(
                        f"---\n*\\[Context summary — conversation compressed here\\]*\n\n{content}\n"
                    )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    jsonl_path = Path(sys.argv[1]).expanduser()
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    md = jsonl_to_markdown(jsonl_path)

    if len(sys.argv) >= 3:
        out = Path(sys.argv[2])
        out.write_text(md, encoding="utf-8")
        print(f"Saved → {out}  ({out.stat().st_size / 1024:.0f} KB)")
    else:
        print(md)


if __name__ == "__main__":
    main()
