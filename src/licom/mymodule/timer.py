"""
File: simple_mpi_timer.py
Description: Refactored MPI-aware timing tool with parameter-controlled decorators

Author: Chtholly
Created: 2025-09-03
Updated: 2025-01-03

REVISION HISTORY:
    03/09/2025 - Initial implementation of MPI-aware timing tool
    04/01/2026 - Refactored to use dataclasses and improved aggregation
"""

import time
import jax
import numpy as np
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class TimingStats:
    """统一的计时统计数据结构"""
    name: str
    total_time: float = 0.0
    count: int = 0
    mean: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std: float = 0.0
    percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'total_time': self.total_time,
            'count': self.count,
            'mean': self.mean,
            'min': self.min_time,
            'max': self.max_time,
            'std': self.std,
            'percent': self.percent
        }


@dataclass
class MPITimingData:
    """MPI计时数据"""
    timings: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于MPI传输"""
        return {
            'timings': self.timings.copy(),
            'counts': self.counts.copy()
        }


class SimpleMPITimer:
    """轻量级MPI计时器"""
    
    def __init__(self):
        self._data = MPITimingData()
        self._start_times: Dict[str, float] = {}
        
    def reset(self) -> None:
        """重置所有计时数据"""
        self._data.timings.clear()
        self._data.counts.clear()
        self._start_times.clear()
        
    def start(self, name: str) -> None:
        """开始计时"""
        jax.block_until_ready(jax.numpy.array(0))
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> None:
        """停止计时"""
        jax.block_until_ready(jax.numpy.array(0))
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self._data.timings[name] = self._data.timings.get(name, 0.0) + elapsed
            self._data.counts[name] = self._data.counts.get(name, 0) + 1
            del self._start_times[name]
    
    def gather_mpi_results(self, comm) -> Optional[List[TimingStats]]:
        """
        收集所有MPI进程的计时结果并聚合
        
        Args:
            comm: MPI通信器 (如 MPI.COMM_WORLD)
        
        Returns:
            List[TimingStats]: 在rank 0上返回聚合后的统计数据，其他rank返回None
        """
        rank = comm.Get_rank()
        
        # 收集所有进程的数据到rank 0
        all_data = comm.gather(self._data.to_dict(), root=0)
        
        if rank == 0:
            return self._aggregate_mpi_data(all_data)
        return None
    
    def _aggregate_mpi_data(self, all_data: List[Dict]) -> List[TimingStats]:
        """聚合所有进程的数据"""
        # 收集所有组件名称
        all_names = set()
        for data in all_data:
            all_names.update(data['timings'].keys())
        
        # rank 0 的数据（第一个进程）
        rank0_data = all_data[0] if all_data else {}
        
        # 聚合统计
        results = []
        for name in all_names:
            times = [data['timings'].get(name, 0.0) for data in all_data]
            # count 只使用 rank 0 的值，不进行 MPI 求和
            count = rank0_data['counts'].get(name, 0)
            
            stats = TimingStats(
                name=name,
                total_time=float(np.sum(times)),
                count=int(count),
                mean=float(np.mean(times)),
                min_time=float(np.min(times)),
                max_time=float(np.max(times)),
                std=float(np.std(times))
            )
            results.append(stats)
        
        return results
    
    def _get_all_time(self) -> Optional[List[TimingStats]]:
        """
        打印MPI聚合摘要（仅在rank 0上）
        """
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        
        results = self.gather_mpi_results(MPI.COMM_WORLD)

        return results

    def _print_all_time(self) -> None:
        """
        打印MPI聚合摘要（仅在rank 0上）
        """
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        
        results = self.gather_mpi_results(MPI.COMM_WORLD)

        if rank == 0 and results:
            self._print_summary_table(results, size)
    
    def _print_summary_table(self, results: List[TimingStats], size: int) -> None:
        """打印格式化的摘要表格"""
        # 查找并提取 "simulation" 组件作为总时间
        simulation_stat = None
        filtered_results = []
        for stat in results:
            if stat.name == "simulation":
                simulation_stat = stat
            elif stat.name != "initial":
                filtered_results.append(stat)
        
        # 使用 simulation 的 mean 作为总时间，如果没有则使用所有组件的总和
        if simulation_stat is not None:
            total_time = simulation_stat.mean
        else:
            total_time = sum(stat.mean for stat in results)
        
        # 计算各组件相对于总时间的百分比
        for stat in filtered_results:
            stat.percent = (stat.mean / total_time * 100.0) if total_time > 0 else 0.0
        
        # 按平均时间排序
        filtered_results.sort(key=lambda x: x.mean, reverse=True)
        
        # 打印表头
        print(f"\n{'='*85}")
        print(f"MPI AGGREGATED TIMING SUMMARY (across {size} processes)")
        print(f"{'='*85}")
        print(f"{'Component':<20} {'Mean(s)':>10} {'Min(s)':>10} {'Max(s)':>10} "
              f"{'Std(s)':>10} {'Percent(%)':>8} {'Count':>8}")
        print(f"{'-'*85}")
        
        # 打印各组件（不包括 simulation）
        for stat in filtered_results:
            print(f"{stat.name:<20} {stat.mean:>10.4f} {stat.min_time:>10.4f} "
                  f"{stat.max_time:>10.4f} {stat.std:>10.4f} {stat.percent:>7.2f}% {stat.count:>8}")
        
        # 打印总计（使用 simulation 的时间）
        print(f"{'-'*85}")
        total_count = simulation_stat.count if simulation_stat is not None else ''
        print(f"{'TOTAL':<20} {total_time:>10.4f} {'':>10} {'':>10} "
                f"{'':>10} {100.0:>7.2f}% {total_count:>8}")
        
        # 计算通信和计算部分的汇总（使用过滤后的结果）
        #self._print_category_summary(filtered_results, total_time)
        
        print(f"{'='*85}\n")
    
    def _print_category_summary(self, results: List[TimingStats], 
                                total_time: float) -> None:
        """打印分类汇总（通信和计算）"""
        comm_components = ['ssh.communication', 'uv.communication', 'remap.communication']
        calc_components = ['ssh.calculation', 'uv.calculation', 'remap.calculation']
        
        stats_dict = {stat.name: stat for stat in results}
        
        # 计算通信总时间
        comm_time = sum(stats_dict.get(name, TimingStats(name=name)).mean 
                       for name in comm_components)
        comm_percent = (comm_time / total_time * 100.0) if total_time > 0 else 0.0
        
        # 计算计算总时间
        calc_time = sum(stats_dict.get(name, TimingStats(name=name)).mean 
                       for name in calc_components)
        calc_percent = (calc_time / total_time * 100.0) if total_time > 0 else 0.0
        
        if comm_time > 0:
            print(f"{'Communication':<30} {comm_time:>10.4f} {'':>10} {'':>10} "
                  f"{'':>10} {comm_percent:>7.2f}%")
        if calc_time > 0:
            print(f"{'Calculation':<30} {calc_time:>10.4f} {'':>10} {'':>10} "
                  f"{'':>10} {calc_percent:>7.2f}%")


# 全局计时器实例
_timer = SimpleMPITimer()


def timed(name: Optional[str] = None, enabled: bool = True):
    """
    计时装饰器
    
    Args:
        name: 计时器名称，默认使用函数名
        enabled: 是否启用计时，默认为False。可以是布尔值或可调用对象（在运行时获取值）
    
    Example:
        @timed(name="my_function", enabled=True)
        def my_function():
            pass
        
        @timed(name="my_function", enabled=lambda: some_condition())
        def my_function():
            pass
    """
    def decorator(func):
        timing_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果 enabled 是可调用对象，在运行时获取值
            if callable(enabled):
                actual_enabled = enabled()
            else:
                actual_enabled = enabled
            
            if not actual_enabled:
                return func(*args, **kwargs)
            
            _timer.start(timing_name)
            try:
                return func(*args, **kwargs)
            finally:
                _timer.stop(timing_name)
        return wrapper
    return decorator


# 便捷函数
def reset_timer() -> None:
    """重置计时器"""
    _timer.reset()

def print_all_time(comm=None) -> None:
    """打印MPI聚合摘要"""
    _timer._print_all_time()

def stop_timer(name: str) -> None:
    """停止计时器"""
    _timer.stop(name)

def get_all_time() -> Optional[List[TimingStats]]:
    """
    获取MPI聚合摘要
    
    """
    results = _timer._get_all_time()    
    return results

class Timer:
    """计时上下文管理器"""
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Args:
            name: 计时器名称
            enabled: 是否启用计时
        """
        self.name = name
        self.enabled = enabled
    
    def __enter__(self):
        if self.enabled:
            _timer.start(self.name)
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            _timer.stop(self.name)