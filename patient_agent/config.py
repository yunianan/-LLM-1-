"""运行配置：环境变量 → Settings。

刻意只用标准库：不引入 python-dotenv，避免给别的层增加依赖。
优先级：显式构造函数参数 > 真实环境变量 > 项目根目录 ``.env`` 文件 > 默认值。

三种 LLM 模式：
- ``offline``  纯规则渲染，确定性、零依赖，pytest 与 CI 默认走这条。
- ``auto``     有 API key 就走真实模型，没有就退回 offline（本地开发友好）。
- ``api``      强制真实模型，缺 key 直接报错，防止「以为在测真模型其实在测规则」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

LLMMode = str  # "offline" | "auto" | "api"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def parse_env_file(path: Path) -> dict[str, str]:
    """解析极简 ``.env``：``KEY=VALUE`` 每行一条，``#`` 起注释，值可带引号。"""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _as_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on", "y")


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 9】配置总开关：所有外部接口的地址 / 密钥在这里汇聚
#   * 真实编排器：SYSTEM_API_MODE, SYSTEM_API_BASE_URL, SYSTEM_API_TOKEN, SYSTEM_API_TIMEOUT
#   * 真实大模型：PATIENT_AGENT_LLM_MODE, OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL,
#                 OPENAI_TIMEOUT, OPENAI_TEMPERATURE
#   * 数据与输出路径：PATIENT_AGENT_SCENARIO_DIR（病例库）, PATIENT_AGENT_OUT_DIR（报告）
# 新增外部依赖时的约定：优先复用已有字段；确实需要新字段就加在 Settings 上，在
# from_env 里接一个环境变量，并在 validate() 里补上「缺了就直接报错」的校验，
# 保证「以为连上了、其实还在用桩」这类错误在启动时就暴露。
# 取值优先级：显式传参 > 真实环境变量 > 项目根 .env > 默认值。
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Settings:
    """一次运行需要的全部外部配置。"""

    llm_mode: LLMMode = "offline"
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_timeout: float = 60.0
    openai_temperature: float = 0.7
    # 生成器单轮最多重试次数（触发一致性守卫时重说）。
    max_generation_retries: int = 2

    system_api_base_url: str = ""
    system_api_timeout: float = 30.0
    system_api_token: str = ""
    # "mock" 用内置五层桩；"http" 连真实编排器。
    system_api_mode: str = "mock"

    out_dir: str = "runs"
    scenario_dir: str = "scenarios"

    extra: dict[str, str] = field(default_factory=dict)

    @property
    def use_real_llm(self) -> bool:
        """本轮是否真的调 LLM。``api`` 缺 key 会在 ``validate`` 里直接报错。"""

        if self.llm_mode == "api":
            return True
        if self.llm_mode == "auto":
            return bool(self.openai_api_key)
        return False

    def validate(self) -> None:
        if self.llm_mode not in ("offline", "auto", "api"):
            raise ValueError(f"PATIENT_AGENT_LLM_MODE 取值非法：{self.llm_mode}")
        if self.system_api_mode not in ("mock", "http"):
            raise ValueError(f"SYSTEM_API_MODE 取值非法：{self.system_api_mode}")
        if self.llm_mode == "api" and not self.openai_api_key:
            raise ValueError("llm_mode=api 需要 OPENAI_API_KEY；离线请用 offline 或 auto")
        if self.system_api_mode == "http" and not self.system_api_base_url:
            raise ValueError("system_api_mode=http 需要 SYSTEM_API_BASE_URL")

    @classmethod
    def from_env(
        cls,
        env_file: Path | str | None = None,
        environ: dict[str, str] | None = None,
        **overrides: object,
    ) -> "Settings":
        """读配置。``environ`` 默认取 ``os.environ``；``overrides`` 优先级最高。"""

        base_dir = Path(__file__).resolve().parent.parent
        env_path = Path(env_file) if env_file is not None else base_dir / ".env"
        file_values = parse_env_file(env_path)
        env_values = dict(os.environ) if environ is None else environ

        def get(key: str, default: str | None = None) -> str | None:
            if key in env_values and str(env_values[key]).strip():
                return env_values[key]
            if key in file_values and file_values[key].strip():
                return file_values[key]
            return default

        settings = cls(
            llm_mode=(get("PATIENT_AGENT_LLM_MODE", "offline") or "offline"),
            openai_base_url=(get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL) or DEFAULT_OPENAI_BASE_URL),
            openai_api_key=get("OPENAI_API_KEY", "") or "",
            openai_model=get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL,
            openai_timeout=_as_float(get("OPENAI_TIMEOUT"), 60.0),
            openai_temperature=_as_float(get("OPENAI_TEMPERATURE"), 0.7),
            max_generation_retries=_as_int(get("PATIENT_AGENT_MAX_RETRIES"), 2),
            system_api_base_url=get("SYSTEM_API_BASE_URL", "") or "",
            system_api_timeout=_as_float(get("SYSTEM_API_TIMEOUT"), 30.0),
            system_api_token=get("SYSTEM_API_TOKEN", "") or "",
            system_api_mode=(get("SYSTEM_API_MODE", "mock") or "mock"),
            out_dir=(get("PATIENT_AGENT_OUT_DIR", "runs") or "runs"),
            scenario_dir=(get("PATIENT_AGENT_SCENARIO_DIR", "scenarios") or "scenarios"),
        )
        ignored = {"PATIENT_AGENT_LLM_MODE", "OPENAI_BASE_URL"}
        for key, value in file_values.items():
            if key not in ignored:
                settings.extra.setdefault(key, value)
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(settings, key):
                raise TypeError(f"Settings 没有字段 {key}")
            setattr(settings, key, value)
        return settings
