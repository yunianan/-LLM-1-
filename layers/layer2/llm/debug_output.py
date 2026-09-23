"""可选的本机 MI 输出诊断；不保存请求、密钥或模型思考内容。"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import uuid


DEBUG_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "debug" / "l2"


def debug_output_enabled() -> bool:
    return os.getenv("L2_DEBUG_OUTPUT", "false").strip().casefold() == "true"


def record_mi_output(
    *, stage: str, raw: str | None, metadata: dict,
    parse_success: bool | None, output_valid: bool, error: Exception | None,
) -> None:
    """记录诊断元数据；仅 JSON 解析失败保存原文。写盘失败不影响评估结果。"""
    now = datetime.now(timezone.utc)
    record_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
    details = {
        "stage": stage,
        "timestamp": now.isoformat(),
        "finish_reason": metadata.get("finish_reason"),
        "content_length": len(raw) if isinstance(raw, str) else None,
        "parse_success": parse_success,
        "output_valid": output_valid,
        "exception": type(error).__name__ if error is not None else None,
    }
    if isinstance(error, json.JSONDecodeError):
        details["json_error"] = {
            "message": error.msg, "line": error.lineno,
            "column": error.colno, "position": error.pos,
        }
        # 行列针对去除外层空白/完整 Markdown 围栏后的解析文本。
        details["position_basis"] = "normalized_json_text"
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        if isinstance(error, json.JSONDecodeError) and isinstance(raw, str):
            response_path = DEBUG_DIR / f"{record_id}.response.txt"
            response_path.write_text(raw, encoding="utf-8")
            details["response_file"] = response_path.name
        meta_path = DEBUG_DIR / f"{record_id}.meta.json"
        meta_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "[开发调试] L2 调试记录写入失败：%s", type(exc).__name__
        )
        return
    print(f"[开发调试] L2 MI 输出记录：{meta_path}", flush=True)
