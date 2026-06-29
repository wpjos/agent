# -*- coding: utf-8 -*-

"""
随机数工具模块

提供安全的随机字符串生成功能，包含以下核心内容：
1. UPPER_ID_CHARACTERS: 大写安全ID字符集定义
2. generate_secure_upper_case_id: 生成指定长度的大写安全随机ID
使用secrets模块确保生成的随机数，适用于安全敏感场景
"""

import secrets
import string

__all__ = [
    "generate_secure_upper_case_id"
]

# 安全随机ID字符集（大写字母[A-Z]+数字[0-9]）
UPPER_ID_CHARACTERS: str = string.ascii_uppercase + string.digits


def generate_secure_upper_case_id(bit: int) -> str:
    """
    生成指定长度的大写安全随机ID字符串

    Args:
        bit(int): 随机ID的字符长度

    Returns:
        str: 由大写字母和数字组成的随机字符串

    Raises:
        ValueError: 当bit为负数时抛出异常

    Notes:
        使用secrets模块确保生成的随机数，适用于安全敏感场景
    """
    if bit < 0:
        raise ValueError("bit不能为负数")
    # 使用安全随机数生成器采样（避免使用random模块以确保安全性）
    return "".join(secrets.choice(UPPER_ID_CHARACTERS) for _ in range(bit))