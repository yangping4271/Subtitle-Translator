"""
单模型缓存管理器 - 优化转录模型加载性能

设计原则:
1. 只缓存一个模型实例，避免内存占用过多
2. 单文件处理后立即释放，批量处理完成后释放
3. 线程安全的缓存管理
"""
import threading
import time
from typing import Optional, Tuple, Dict, Any, Callable
import mlx.core as mx
from contextlib import contextmanager

from .parakeet import BaseParakeet
from ..logger import setup_logger

logger = setup_logger(__name__)


class SingleModelCache:
    """单模型缓存管理器 - 内存中只保持一个模型实例"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # 缓存状态
        self._cached_model: Optional[BaseParakeet] = None
        self._cached_model_id: Optional[str] = None
        self._cached_dtype: Optional[mx.Dtype] = None
        self._load_time: Optional[float] = None
        self._access_count: int = 0
        self._last_access: Optional[float] = None
        
        # 线程安全锁
        self._cache_lock = threading.RLock()
        
        # 批量处理状态管理
        self._batch_mode: bool = False
        self._batch_ref_count: int = 0
        
        self._initialized = True
        
    def is_cached(self, model_id: str, dtype: mx.Dtype) -> bool:
        """检查指定模型是否已缓存"""
        with self._cache_lock:
            return (
                self._cached_model is not None 
                and self._cached_model_id == model_id 
                and self._cached_dtype == dtype
            )
    
    def get_model(self, model_id: str, dtype: mx.Dtype) -> Optional[BaseParakeet]:
        """获取缓存的模型实例"""
        with self._cache_lock:
            if self.is_cached(model_id, dtype):
                self._access_count += 1
                self._last_access = time.time()
                logger.info(f"缓存命中: {model_id} ({dtype})")
                return self._cached_model
            return None
    
    def set_model(self, model_id: str, dtype: mx.Dtype, model: BaseParakeet) -> None:
        """缓存模型实例"""
        with self._cache_lock:
            # 如果已有不同的模型，先释放
            if (self._cached_model is not None 
                and (self._cached_model_id != model_id or self._cached_dtype != dtype)):
                logger.info(f"释放旧模型: {self._cached_model_id}")
                self._clear_cache()
            
            # 缓存新模型
            self._cached_model = model
            self._cached_model_id = model_id
            self._cached_dtype = dtype
            self._load_time = time.time()
            self._access_count = 1
            self._last_access = time.time()
            
            logger.info(f"模型已缓存: {model_id} ({dtype})")
    
    def _clear_cache(self) -> None:
        """内部清理缓存（不加锁）"""
        if self._cached_model is not None:
            # 强制清理模型占用的显存
            try:
                mx.clear_cache()
            except Exception as e:
                logger.warning(f"清理MLX缓存时出错: {e}")
        
        self._cached_model = None
        self._cached_model_id = None
        self._cached_dtype = None
        self._load_time = None
        self._access_count = 0
        self._last_access = None
    
    def clear_cache(self) -> None:
        """手动清理缓存"""
        with self._cache_lock:
            if self._cached_model is not None:
                logger.info(f"手动清理缓存: {self._cached_model_id}")
                self._clear_cache()
    
    def enter_batch_mode(self) -> None:
        """进入批量处理模式"""
        with self._cache_lock:
            self._batch_mode = True
            self._batch_ref_count += 1
            logger.info(f"进入批量模式，引用计数: {self._batch_ref_count}")
    
    def exit_batch_mode(self) -> None:
        """退出批量处理模式"""
        with self._cache_lock:
            if self._batch_ref_count > 0:
                self._batch_ref_count -= 1
                logger.info(f"退出批量模式，引用计数: {self._batch_ref_count}")
                
                if self._batch_ref_count == 0:
                    self._batch_mode = False
                    # 批量处理完成，清理缓存
                    if self._cached_model is not None:
                        logger.info("批量处理完成，清理模型缓存")
                        self._clear_cache()
    
    def should_release(self) -> bool:
        """判断是否应该释放缓存"""
        with self._cache_lock:
            # 批量模式下不释放
            return not self._batch_mode
    
    def auto_release_if_needed(self) -> None:
        """根据策略自动释放缓存"""
        with self._cache_lock:
            if self.should_release() and self._cached_model is not None:
                logger.info("单文件处理完成，自动释放模型缓存")
                self._clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        with self._cache_lock:
            if self._cached_model is None:
                return {"status": "empty"}
            
            return {
                "status": "cached",
                "model_id": self._cached_model_id,
                "dtype": str(self._cached_dtype),
                "load_time": self._load_time,
                "access_count": self._access_count,
                "last_access": self._last_access,
                "batch_mode": self._batch_mode,
                "batch_ref_count": self._batch_ref_count
            }


# 全局缓存实例
_model_cache = SingleModelCache()


def get_model_cache() -> SingleModelCache:
    """获取全局模型缓存实例"""
    return _model_cache


@contextmanager
def model_context(batch_mode: bool = False):
    """
    模型生命周期管理上下文管理器
    
    Args:
        batch_mode: 是否为批量处理模式
        
    Usage:
        # 单文件处理
        with model_context():
            model = load_cached_model(model_id, dtype)
            # 使用模型...
        # 自动释放
        
        # 批量处理
        with model_context(batch_mode=True):
            for file in files:
                model = load_cached_model(model_id, dtype)
                # 处理文件...
        # 批量完成后自动释放
    """
    cache = get_model_cache()
    
    try:
        if batch_mode:
            cache.enter_batch_mode()
        yield cache
    finally:
        if batch_mode:
            cache.exit_batch_mode()
        else:
            # 单文件模式，立即检查是否需要释放
            cache.auto_release_if_needed()


def load_cached_model(
    model_id: str,
    dtype: mx.Dtype,
    loader_func: Callable[[], BaseParakeet]
) -> Tuple[BaseParakeet, bool]:
    """
    加载缓存的模型实例

    Args:
        model_id: 模型ID
        dtype: 数据类型
        loader_func: 实际的模型加载函数，signature: () -> BaseParakeet

    Returns:
        tuple: (模型实例, 是否从缓存加载)
    """
    cache = get_model_cache()

    # 尝试从内存缓存获取
    model = cache.get_model(model_id, dtype)
    if model is not None:
        # 内存缓存命中
        return model, True

    # 缓存未命中，需要实际加载模型
    logger.info(f"缓存未命中，开始加载模型: {model_id} ({dtype})")
    start_time = time.time()

    model = loader_func()

    load_time = time.time() - start_time
    logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")

    # 缓存新模型
    cache.set_model(model_id, dtype, model)

    return model, False


def clear_model_cache() -> None:
    """清理模型缓存（提供给外部调用）"""
    cache = get_model_cache()
    cache.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """获取缓存信息（提供给外部调用）"""
    cache = get_model_cache()
    return cache.get_cache_info()