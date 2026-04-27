#!/usr/bin/env python3
"""
LLM响应缓存 - 基于SHA256的文件缓存系统
避免重复调用LLM API，加速重跑流程
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
from project_config import get_config


class LLMCache:
    """基于文件的LLM响应缓存"""

    def __init__(self):
        config = get_config()
        self.enabled = config.get("cache.enable_cache", True)
        self.cache_dir = Path(config.get("cache.cache_dir", "cache"))
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)

    def _hash(self, prompt: str) -> str:
        """生成prompt的SHA256哈希作为缓存key"""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        """从缓存获取LLM响应，未命中返回None"""
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"{self._hash(prompt)}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except Exception:
            return None

    def set(self, prompt: str, response: str) -> None:
        """将LLM响应写入缓存"""
        if not self.enabled:
            return

        cache_file = self.cache_dir / f"{self._hash(prompt)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"response": response}, f, ensure_ascii=False)
        except Exception:
            pass

    def clear(self) -> None:
        """清空缓存"""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
