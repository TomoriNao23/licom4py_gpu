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
# Standard library imports
import time
from typing import Dict, List, Optional, Any
from functools import wraps
from dataclasses import dataclass, field

# Third-party imports
import jax


@dataclass
class TimingStats:
    """Unified timing statistics data structure"""
    name: str
    total_time: float = 0.0
    count: int = 0
    mean: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std: float = 0.0
    percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
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
    """Timing data container (MPI-free)"""
    timings: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            'timings': self.timings.copy(),
            'counts': self.counts.copy()
        }


class SimpleMPITimer:
    """Lightweight timer (rank0-only reporting)"""
    
    def __init__(self):
        self._data = MPITimingData()
        self._start_times: Dict[str, float] = {}
        
    def reset(self) -> None:
        """Reset all timing data"""
        self._data.timings.clear()
        self._data.counts.clear()
        self._start_times.clear()
        
    def start(self, name: str) -> None:
        """Start timing a named section"""
        jax.block_until_ready(jax.numpy.array(0))
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> None:
        """Stop timing a named section"""
        jax.block_until_ready(jax.numpy.array(0))
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self._data.timings[name] = self._data.timings.get(name, 0.0) + elapsed
            self._data.counts[name] = self._data.counts.get(name, 0) + 1
            del self._start_times[name]

    def get_results(self) -> List[TimingStats]:
        """
        Get timing summary from local process only.

        Notes:
            - This project no longer aggregates timings via mpi4py.
            - min/max/std are reported at "per-section average" granularity because
              we only keep running totals (not per-call samples).
        """
        results: List[TimingStats] = []
        for name, total_time in self._data.timings.items():
            count = int(self._data.counts.get(name, 0))
            mean = float(total_time / count) if count > 0 else 0.0
            results.append(
                TimingStats(
                    name=name,
                    total_time=float(total_time),
                    count=count,
                    mean=mean,
                    min_time=mean,
                    max_time=mean,
                    std=0.0,
                )
            )
        return self._stable_order(results)

    @staticmethod
    def _stable_order(results: List[TimingStats]) -> List[TimingStats]:
        """
        Make result ordering deterministic for callers.

        Compatibility:
            - Some callers use get_all_time()[0] and expect "initial" to be first.
        """
        by_name = {s.name: s for s in results}
        ordered: List[TimingStats] = []
        for preferred in ("initial", "simulation"):
            if preferred in by_name:
                ordered.append(by_name.pop(preferred))
        ordered.extend(sorted(by_name.values(), key=lambda s: s.name))
        return ordered
    
    def _get_all_time(self) -> Optional[List[TimingStats]]:
        """
        Get timing summary from main process only.
        """
        results = self.get_results()
        return results if results else None

    def _print_all_time(self) -> None:
        """
        Print timing summary from main process only.
        """
        results = self.get_results()
        if results:
            self._print_summary_table(results, size=1)
    
    def _print_summary_table(self, results: List[TimingStats], size: int) -> None:
        """Print a formatted summary table"""
        # Extract the "simulation" component as the overall time
        simulation_stat = None
        filtered_results = []
        for stat in results:
            if stat.name == "simulation":
                simulation_stat = stat
            elif stat.name != "initial":
                filtered_results.append(stat)
        
        # Use simulation's mean as total time, otherwise sum all means
        if simulation_stat is not None:
            total_time = simulation_stat.mean
        else:
            total_time = sum(stat.mean for stat in results)
        
        # Compute percentage of total time for each component
        for stat in filtered_results:
            stat.percent = (stat.mean / total_time * 100.0) if total_time > 0 else 0.0
        
        # Sort by mean time descending
        filtered_results.sort(key=lambda x: x.mean, reverse=True)
        
        # Print header
        print(f"\n{'='*85}")
        print(f"Successfully gathered timing summary from {size} processes")
        print(f"{'='*85}")
        print(f"{'Component':<20} {'Mean(s)':>10} {'Min(s)':>10} {'Max(s)':>10} "
              f"{'Std(s)':>10} {'Percent(%)':>8} {'Count':>8}")
        print(f"{'-'*85}")
        
        # Print each component (excluding simulation)
        for stat in filtered_results:
            print(f"{stat.name:<20} {stat.mean:>10.4f} {stat.min_time:>10.4f} "
                  f"{stat.max_time:>10.4f} {stat.std:>10.4f} {stat.percent:>7.2f}% {stat.count:>8}")
        
        # Print totals (using simulation time)
        print(f"{'-'*85}")
        total_count = simulation_stat.count if simulation_stat is not None else ''
        print(f"{'TOTAL':<20} {total_time:>10.4f} {'':>10} {'':>10} "
                f"{'':>10} {100.0:>7.2f}% {total_count:>8}")
        
        # Print communication and computation summaries (based on filtered results)
        #self._print_category_summary(filtered_results, total_time)
        
        print(f"{'='*85}\n")
    
    def _print_category_summary(self, results: List[TimingStats], 
                                total_time: float) -> None:
        """Print category summaries (communication and computation)"""
        comm_components = ['ssh.communication', 'uv.communication', 'remap.communication']
        calc_components = ['ssh.calculation', 'uv.calculation', 'remap.calculation']
        
        stats_dict = {stat.name: stat for stat in results}
        
        # Compute total communication time
        comm_time = sum(stats_dict.get(name, TimingStats(name=name)).mean 
                       for name in comm_components)
        comm_percent = (comm_time / total_time * 100.0) if total_time > 0 else 0.0
        
        # Compute total computation time
        calc_time = sum(stats_dict.get(name, TimingStats(name=name)).mean 
                       for name in calc_components)
        calc_percent = (calc_time / total_time * 100.0) if total_time > 0 else 0.0
        
        if comm_time > 0:
            print(f"{'Communication':<30} {comm_time:>10.4f} {'':>10} {'':>10} "
                  f"{'':>10} {comm_percent:>7.2f}%")
        if calc_time > 0:
            print(f"{'Calculation':<30} {calc_time:>10.4f} {'':>10} {'':>10} "
                  f"{'':>10} {calc_percent:>7.2f}%")


# Global timer instance
_timer = SimpleMPITimer()


def timed(name: Optional[str] = None, enabled: bool = True):
    """
    Timing decorator

    Args:
        name: Timer name, defaults to the function name.
        enabled: Whether timing is enabled. Defaults to True. Can be a boolean or a callable
                 that returns a boolean at runtime.

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
            # If enabled is callable, evaluate it at runtime
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


# Convenience functions
def reset_timer() -> None:
    """Reset the timer"""
    _timer.reset()

def print_all_time(comm=None) -> None:
    """Print timing summary from main process only"""
    _timer._print_all_time()

def stop_timer(name: str) -> None:
    """Stop the timer for a given name"""
    _timer.stop(name)

def get_all_time() -> Optional[List[TimingStats]]:
    """
    Get timing summary from main process only
    """
    results = _timer._get_all_time()    
    return results

class Timer:
    """Timing context manager"""
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Args:
            name: Timer name
            enabled: Whether timing is enabled
        """
        self.name = name
        if callable(enabled):
            self.enabled = enabled()
        else:
            self.enabled = enabled
    
    def __enter__(self):
        if self.enabled:
            _timer.start(self.name)
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            _timer.stop(self.name)