#!/usr/bin/env python3
"""
GLOBAL AI MEMORY EXTRACTOR / MERGER
Python 3.12

Importa exportaciones de ChatGPT, DeepSeek, Grow, Dola u otras IAs,
extrae chats/memorias/proyectos/prompts/código, normaliza contenido,
deduplica con hash + similitud Jaccard y genera un repositorio global.

Uso:
    python chat_project_merger.py --input ./imports --output ./global_ai_memory

Formatos soportados:
    JSON, JSONL, TXT, MD, HTML
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_ALIASES = {
    "chatgpt": "chatgpt",
    "openai": "chatgpt",
    "deepseek": "deepseek",
    "grow": "grow",
    "dola": "dola",
    "dola ia": "dola",
}


@dataclass
class MemoryItem:
    type: str
    title: str
    content: str
    source: str
    tags: list[str] = field(default_factory=list)
    hash: str = ""
    similarity_group: str = ""


@dataclass
class ExtractionResult:
    source: str
    file: str
    memories: list[MemoryItem] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    code: list[dict[str, Any]] = field(default_factory=list)
    conversations: list[dict[str, Any]] = field(default_factory=list)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"```[\w+-]*", " ", text)
    text = text.replace("`", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[\w+#.-]{2,}\b", normalize(text)))


def jaccard_similarity(a: str, b: str) -> float:
    set_a = tokens(a)
    set_b = tokens(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def detect_source(path: Path) -> str:
    name = path.name.lower()
    for alias, source in SOURCE_ALIASES.items():
        if alias in name:
            return source
    parent = path.parent.name.lower()
    for alias, source in SOURCE_ALIASES.items():
        if alias in parent:
            return source
    return "unknown"


def extract_code_blocks(text: str) -> list[dict[str, Any]]:
    results = []
    pattern = re.compile(r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```", re.S)

    for match in pattern.finditer(text):
        language = match.group(1).strip().lower() or "text"
        code = match.group(2).strip()
        if not code:
            continue
        results.append({
            "language": language,
            "description": "Código extraído automáticamente",
            "code": code,
            "hash": content_hash(code),
        })
    return results


def extract_prompts(text: str) -> list[dict[str, Any]]:
    prompts = []
    patterns = [
        r"(?im)^\s*(?:prompt|prompts|instrucción|instrucciones)\s*:\s*(.+)$",
        r"(?im)^\s*#\s*prompt\s*\n(.+?)(?=\n#|\Z)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.S):
            value = clean_text(match.group(1))
            if len(value) >= 15:
                prompts.append({
                    "tag": "imported",
                    "prompt": value,
                    "hash": content_hash(value),
                })
    return prompts


def infer_tags(text: str) -> list[str]:
    lower = normalize(text)
    candidates = [
        "python", "javascript", "typescript", "node", "fastapi", "react",
        "json", "jaccard", "ai", "ia", "android", "kotlin", "security",
        "ciberseguridad", "database", "mongodb", "sql", "api", "web",
        "docker", "memory", "chat", "project",
    ]
    return [tag for tag in candidates if tag in lower]


def memory_from_text(text: str, source: str, title: str) -> MemoryItem | None:
    content = clean_text(text)
    if len(content) < 20:
        return None

    item = MemoryItem(
        type="conversation",
        title=title or "Imported conversation",
        content=content,
        source=source,
        tags=infer_tags(content),
    )
    item.hash = content_hash(content)
    return item


def flatten_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from flatten_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten_json(child)


def parse_json(data: Any, source: str, file_name: str) -> ExtractionResult:
    result = ExtractionResult(source=source, file=file_name)

    # Detect explicit memory/project/prompt/code structures.
    if isinstance(data, dict):
        for pref in data.get("preferences", []):
            value = clean_text(pref)
            if value:
                item = memory_from_text(value, source, "Preference")
                if item:
                    item.type = "preference"
                    result.memories.append(item)

        for mem in data.get("memories", []):
            if isinstance(mem, dict):
                content = clean_text(mem.get("content", ""))
                title = clean_text(mem.get("title", mem.get("name", "Memory")))
                if content:
                    item = memory_from_text(content, source, title)
                    if item:
                        item.type = clean_text(mem.get("type", "memory")) or "memory"
                        item.tags = list(dict.fromkeys(mem.get("tags", []) or item.tags))
                        result.memories.append(item)
            elif isinstance(mem, str):
                item = memory_from_text(mem, source, "Memory")
                if item:
                    item.type = "memory"
                    result.memories.append(item)

        for prompt in data.get("prompts", []):
            if isinstance(prompt, dict):
                p = clean_text(prompt.get("prompt", prompt.get("content", "")))
                if p:
                    result.prompts.append({
                        "tag": clean_text(prompt.get("tag", "imported")) or "imported",
                        "prompt": p,
                        "hash": content_hash(p),
                        "source": source,
                    })
            elif isinstance(prompt, str):
                p = clean_text(prompt)
                if p:
                    result.prompts.append({
                        "tag": "imported",
                        "prompt": p,
                        "hash": content_hash(p),
                        "source": source,
                    })

        for block in data.get("code", []):
            if isinstance(block, dict):
                code = str(block.get("code", "")).strip()
                if code:
                    result.code.append({
                        "language": clean_text(block.get("language", "text")),
                        "description": clean_text(block.get("description", "")),
                        "code": code,
                        "hash": content_hash(code),
                        "source": source,
                    })

    # Generic conversation extraction.
    for obj in flatten_json(data):
        role = obj.get("role")
        content = obj.get("content")
        if isinstance(content, dict):
            content = content.get("parts") or content.get("text") or content.get("content")
        if isinstance(content, list):
            content = "\n".join(clean_text(x) for x in content)

        if role and content:
            text = clean_text(content)
            if len(text) >= 5:
                result.conversations.append({
                    "role": clean_text(role),
                    "content": text,
                    "source": source,
                })

    return result


def parse_file(path: Path) -> ExtractionResult:
    source = detect_source(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return parse_json(data, source, path.name)

    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return parse_json(rows, source, path.name)

    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    text = clean_text(raw)

    result = ExtractionResult(source=source, file=path.name)
    item = memory_from_text(text, source, path.stem)
    if item:
        result.memories.append(item)
    result.code.extend(extract_code_blocks(raw))
    for code in result.code:
        code["source"] = source
    result.prompts.extend(extract_prompts(raw))
    for prompt in result.prompts:
        prompt["source"] = source
    result.conversations.append({
        "role": "document",
        "content": text,
        "source": source,
    })
    return result


def unique_by_hash(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        value = item.get(field, "")
        h = item.get("hash") or content_hash(value)
        item["hash"] = h
        if h in seen:
            continue
        seen.add(h)
        output.append(item)
    return output


def deduplicate_memories(
    items: list[MemoryItem],
    threshold: float = 0.88,
) -> list[MemoryItem]:
    unique: list[MemoryItem] = []
    groups: list[str] = []

    for item in items:
        if any(existing.hash == item.hash for existing in unique):
            continue

        duplicate_index = None
        for index, existing in enumerate(unique):
            similarity = jaccard_similarity(item.content, existing.content)
            if similarity >= threshold:
                duplicate_index = index
                break

        if duplicate_index is not None:
            existing = unique[duplicate_index]
            existing.tags = sorted(set(existing.tags + item.tags))
            if len(item.content) > len(existing.content):
                existing.content = item.content
                existing.hash = content_hash(existing.content)
            continue

        group_id = f"group-{len(groups) + 1:05d}"
        item.similarity_group = group_id
        groups.append(group_id)
        unique.append(item)

    return unique


def build_global(results: list[ExtractionResult], threshold: float) -> dict[str, Any]:
    memories = []
    prompts = []
    code = []
    conversations = []

    for result in results:
        memories.extend(result.memories)
        prompts.extend(result.prompts)
        code.extend(result.code)
        conversations.extend(result.conversations)

    merged_memories = deduplicate_memories(memories, threshold)
    prompts = unique_by_hash(prompts, "prompt")
    code = unique_by_hash(code, "code")

    return {
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sorted({r.source for r in results}),
        "statistics": {
            "files_processed": len(results),
            "memories": len(merged_memories),
            "prompts": len(prompts),
            "code_blocks": len(code),
            "conversation_messages": len(conversations),
        },
        "memories": [asdict(x) for x in merged_memories],
        "prompts": prompts,
        "code": code,
        "conversations": conversations,
    }


def write_outputs(global_data: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "GLOBAL_AI_MEMORY.json").write_text(
        json.dumps(global_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for key in ("memories", "prompts", "code", "conversations"):
        (output / f"{key}.json").write_text(
            json.dumps(global_data[key], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "generated_at": global_data["generated_at"],
        "sources": global_data["sources"],
        "statistics": global_data["statistics"],
    }
    (output / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Global AI memory/chat extractor")
    parser.add_argument("--input", required=True, help="Carpeta con exportaciones")
    parser.add_argument("--output", default="./global_ai_memory")
    parser.add_argument("--threshold", type=float, default=0.88)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    extensions = {".json", ".jsonl", ".txt", ".md", ".html", ".htm"}
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not files:
        raise SystemExit("No se encontraron archivos compatibles.")

    results = []
    for path in files:
        try:
            results.append(parse_file(path))
            print(f"[OK] {path}")
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")

    global_data = build_global(results, args.threshold)
    write_outputs(global_data, output_dir)

    print("\nGLOBAL AI MEMORY")
    print(json.dumps(global_data["statistics"], indent=2, ensure_ascii=False))
    print(f"\nSalida: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
