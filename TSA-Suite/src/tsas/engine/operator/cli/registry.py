# -*- coding: utf-8 -*-

"""
通用算子注册中心

提供自动发现和注册算子类的通用机制，一个类可实例化多次分别服务于
feature.construction、detection、evaluation 三个模块。

核心能力:
    - 自动扫描指定包下的所有模块，发现符合条件的算子类
    - 以算子的 ``name()`` 返回值为键进行注册
    - 同名算子自动保留最高版本，低版本被覆盖（debug 日志记录）
    - 手动注册时若版本不占优，记录 warning 日志提示注册无效
    - 支持通过名称查找算子类
    - 支持可选的过滤函数，控制注册哪些算子

使用示例::

    from tsas.engine.operator.cli.registry import OperatorRegistry
    from tsas.engine.operator.base import BaseOperator

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['tsas.engine.operator.detection'],
        filter_fn=lambda cls: hasattr(cls, 'name'),
    )
    registry.discover()
    all_operators = registry.list_all()
"""

import importlib
import inspect
import pkgutil
from typing import Callable

from loguru import logger

from tsas.engine.operator.base import BaseOperator

__all__ = [
    'OperatorRegistry',
]


class OperatorRegistry:
    """
    通用算子注册中心

    通过自动扫描指定包发现并注册算子类，以 ``name()`` 返回值为键。
    同一个类可实例化多次，分别用于不同模块的算子注册。

    Attributes:
        _base_class (type): 算子基类，用于 ``issubclass`` 过滤
        _scan_packages (list[str]): 需要扫描的包路径列表（点分格式）
        _filter_fn (Callable[[type], bool] | None): 额外的过滤函数，
            返回 True 表示该类应被注册
        _registry (dict[str, type]): 已注册的算子 {name: class} 映射
        _discovered (bool): 是否已执行过 discover 扫描
    """

    # 同名冲突时的版本覆盖策略:
    # - 高版本始终胜出，与注册入口无关
    # - 自动扫描发生覆盖时记录 debug 日志
    # - 手动注册若版本不占优（实际未生效）记录 warning 日志
    # - 同名同版本不同类 → raise ValueError

    def __init__(
        self,
        base_class: type,
        scan_packages: list[str],
        filter_fn: Callable[[type], bool] | None = None,
    ) -> None:
        """
        初始化注册中心

        Args:
            base_class (type): 算子基类，仅该类的非抽象子类会被注册
            scan_packages (list[str]): 需要扫描的包路径列表，如
                ``['tsas.engine.operator.detection']``
            filter_fn (Callable[[type], bool] | None): 额外的过滤函数。
                当提供时，只有 ``filter_fn(cls)`` 返回 True 的类才会被注册。
                默认为 None，表示不做额外过滤
        """
        self._base_class = base_class
        self._scan_packages = scan_packages
        self._filter_fn = filter_fn
        self._registry: dict[str, type] = {}
        self._discovered: bool = False

    def discover(self) -> None:
        """
        扫描指定包，自动发现并注册所有符合条件的算子类

        扫描规则:
            1. 递归遍历 ``_scan_packages`` 中所有包的子模块
            2. 对每个模块中的类，检查是否为 ``_base_class`` 的子类
            3. 排除抽象类（含未实现的抽象方法）
            4. 排除没有 ``name`` 类方法的类
            5. 如有 ``_filter_fn``，额外调用过滤
            6. 以 ``cls.name()`` 为键注册到 ``_registry``

        重复调用时会增量合并，不会清空已有注册。

        Raises:
            ImportError: 扫描的包路径不存在或无法导入时
        """
        for package_path in self._scan_packages:
            # 导入顶层包
            package = importlib.import_module(package_path)

            # 递归遍历所有子模块
            package_paths = getattr(package, '__path__', None)
            if package_paths is None:
                # 不是包（是普通模块），直接扫描其中的类
                self._scan_module(package)
                continue

            for _importer, modname, _ispkg in pkgutil.walk_packages(
                package_paths, prefix=package_path + '.'
            ):
                try:
                    module = importlib.import_module(modname)
                except ImportError:
                    # 跳过无法导入的模块（可能有缺失依赖）
                    continue
                self._scan_module(module)

        self._discovered = True

    def _scan_module(self, module) -> None:
        """
        扫描单个模块中的类并注册符合条件的算子

        同名算子按版本覆盖策略处理：高版本胜出，低版本被覆盖。
        覆盖时记录 debug 级别日志。同名同版本不同类则抛出 ValueError。

        Args:
            module: 已导入的 Python 模块对象
        """
        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            # 必须是 base_class 的子类（但不是 base_class 本身）
            if not issubclass(obj, self._base_class) or obj is self._base_class:
                continue

            # 排除抽象类
            if inspect.isabstract(obj):
                continue

            # 必须有 name 类方法
            if not hasattr(obj, 'name') or not callable(getattr(obj, 'name')):
                continue

            # 额外过滤
            if self._filter_fn is not None and not self._filter_fn(obj):
                continue

            # 以 name() 为键注册，按版本覆盖策略处理同名冲突
            try:
                op_name = obj.name()
            except Exception:
                # name() 调用失败则跳过
                continue

            self._register_with_version(
                name=op_name, cls=obj,
                is_manual=False,
            )

    def get(self, name: str) -> type:
        """
        通过算子名称获取已注册的算子类

        Args:
            name (str): 算子名称，即 ``cls.name()`` 的返回值

        Returns:
            type: 对应的算子类

        Raises:
            KeyError: 指定名称的算子未注册时
        """
        if not self._discovered:
            self.discover()

        if name not in self._registry:
            available = ', '.join(sorted(self._registry.keys()))
            raise KeyError(
                f"未找到名为 '{name}' 的算子。可用算子: [{available}]"
            )
        return self._registry[name]

    def list_all(self) -> dict[str, type]:
        """
        返回所有已注册算子的名称到类的映射

        Returns:
            dict[str, type]: {算子名称: 算子类} 字典，按名称排序
        """
        if not self._discovered:
            self.discover()

        return dict(sorted(self._registry.items()))

    @property
    def discovered(self) -> bool:
        """
        是否已执行过 discover 扫描

        Returns:
            bool: True 表示已扫描
        """
        return self._discovered

    def register(self, cls: type, name: str | None = None) -> None:
        """
        手动注册一个算子类

        按版本覆盖策略处理同名冲突：高版本胜出。若手动注册的版本低于已注册的版本，
        则本次注册实际无效，记录 warning 日志。

        Args:
            cls (type): 要注册的算子类
            name (str | None): 注册名称。为 None 时使用 ``cls.name()``

        Raises:
            ValueError: 类没有 ``name`` 方法且未提供 name 参数时
            ValueError: 同名同版本但实际类不同时
        """
        if name is None:
            if not hasattr(cls, 'name') or not callable(getattr(cls, 'name')):
                raise ValueError(
                    f"类 {cls.__name__} 没有 name() 方法，请显式提供 name 参数"
                )
            name = cls.name()

        self._register_with_version(name=name, cls=cls, is_manual=True)

    def _register_with_version(
            self, name: str, cls: type, *, is_manual: bool
    ) -> None:
        """
        按版本覆盖策略注册算子（内部统一入口）

        策略规则：
        - 同名同类：幂等跳过
        - 同名不同类，新版本 > 旧版本：新版本覆盖旧版本
        - 同名不同类，新版本 < 旧版本：拒绝注册，保留旧版本
        - 同名不同类，版本相同：raise ValueError

        日志分级：
        - 自动扫描（``is_manual=False``）发生覆盖时记录 debug 日志
        - 手动注册（``is_manual=True``）版本不占优被拒绝时记录 warning 日志

        Args:
            name(str): 注册名称
            cls(type): 待注册的算子类
            is_manual(bool): 是否为手动注册（影响日志级别）

        Raises:
            ValueError: 同名同版本不同类时
        """
        if name not in self._registry:
            # 无冲突，直接注册
            self._registry[name] = cls
            return

        existing = self._registry[name]

        # 同一个类的重复注册是幂等的，跳过
        if existing is cls:
            return

        # 提取版本号（优先使用 version()，回退为 (0,) 以保证向后兼容）
        new_version = cls.version() if hasattr(cls, 'version') and callable(cls.version) else (0,)
        old_version = existing.version() if hasattr(existing, 'version') and callable(existing.version) else (0,)

        if new_version > old_version:
            # 新版本胜出，覆盖旧版本
            self._registry[name] = cls
            old_label = f"{existing.__module__}.{existing.__qualname__}"
            new_label = f"{cls.__module__}.{cls.__qualname__}"
            logger.debug(
                f"算子 '{name}' 版本覆盖: "
                f"{BaseOperator._format_version(old_version)} ({old_label}) "
                f"→ {BaseOperator._format_version(new_version)} ({new_label})"
            )
        elif new_version == old_version:
            # 同名同版本不同类 → 冲突
            raise ValueError(
                f"算子名称冲突: '{name}' 已被 {existing.__module__}.{existing.__qualname__} 注册，"
                f"当前尝试注册 {cls.__module__}.{cls.__qualname__}"
            )
        else:
            # 新版本低于已注册版本，本次注册无效
            if is_manual:
                # 手动注册被拒绝，记录 warning
                logger.warning(
                    f"手动注册算子 '{name}' 无效: "
                    f"当前已注册版本 {BaseOperator._format_version(old_version)}"
                    f"({self._registry[name].__module__}.{self._registry[name].__qualname__})，"
                    f"尝试注册的版本 {BaseOperator._format_version(new_version)}"
                    f"({cls.__module__}.{cls.__qualname__}) 不占优"
                )
            # 自动扫描场景不需要额外日志（已在覆盖侧记录了 debug）
