"""
File: simple_mpi_timer.py
Description: Simplified MPI-aware timing tool for barotropic module
    Focus on total time and percentage only, with MPI gather support

Author: Simplified Performance Tool
Created: 2025-10-30
"""

import jax
import time
import numpy as np
from collections import defaultdict
from functools import wraps


class SimpleMPITimer:
    """Lightweight timer with MPI support"""
    
    def __init__(self):
        self.timings = defaultdict(float)  # name -> total_time
        self.counts = defaultdict(int)     # name -> call_count
        self.enabled = True
        self._start_times = {}
        
    def reset(self):
        """Reset all timing data"""
        self.timings.clear()
        self.counts.clear()
        self._start_times.clear()
        
    def start(self, name: str):
        """Start timing"""
        if not self.enabled:
            return
        jax.block_until_ready(jax.numpy.array(0))
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str):
        """Stop timing"""
        if not self.enabled:
            return
        jax.block_until_ready(jax.numpy.array(0))
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.timings[name] += elapsed
            self.counts[name] += 1
            del self._start_times[name]
    
    def get_local_summary(self):
        """Get summary for current process"""
        total_time = sum(self.timings.values())
        
        results = []
        for name in sorted(self.timings.keys()):
            t = self.timings[name]
            percent = (t / total_time * 100) if total_time > 0 else 0
            results.append({
                'name': name,
                'time': t,
                'percent': percent,
                'count': self.counts[name]
            })
        
        results.sort(key=lambda x: x['time'], reverse=True)
        return results, total_time
    
    def print_local_summary(self, rank=None):
        """Print summary for current process"""
        results, total_time = self.get_local_summary()
        
        rank_str = f" (Rank {rank})" if rank is not None else ""
        print(f"\n{'='*70}")
        print(f"TIMING SUMMARY{rank_str}")
        print(f"{'='*70}")
        print(f"{'Component':<35} {'Time(s)':>12} {'Percent':>10} {'Calls':>8}")
        print(f"{'-'*70}")
        
        for item in results:
            print(f"{item['name']:<35} {item['time']:>12.4f} "
                  f"{item['percent']:>9.2f}% {item['count']:>8}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<35} {total_time:>12.4f} {100.0:>9.2f}%")
        print(f"{'='*70}\n")
    
    def gather_mpi_results(self, comm):
        """
        Gather timing results across all MPI processes
        
        Args:
            comm: MPI communicator (e.g., MPI.COMM_WORLD)
        
        Returns:
            dict: Aggregated results on rank 0, None on other ranks
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Prepare local data
        local_data = {
            'timings': dict(self.timings),
            'counts': dict(self.counts)
        }
        
        # Gather all data to rank 0
        all_data = comm.gather(local_data, root=0)
        
        if rank == 0:
            return self._aggregate_mpi_data(all_data, size)
        return None
    
    def _aggregate_mpi_data(self, all_data, size):
        """Aggregate data from all processes"""
        # Collect all component names
        all_names = set()
        for data in all_data:
            all_names.update(data['timings'].keys())
        
        # Aggregate statistics
        aggregated = {}
        for name in all_names:
            times = [data['timings'].get(name, 0) for data in all_data]
            counts = [data['counts'].get(name, 0) for data in all_data]
            
            aggregated[name] = {
                'mean': np.mean(times),
                'min': np.min(times),
                'max': np.max(times),
                'std': np.std(times),
                'total_mean': np.mean(times),
                'total_calls': np.sum(counts)
            }
        
        return aggregated
    
    def print_mpi_summary(self, comm):
        """
        Print aggregated MPI summary (only on rank 0)
        
        Args:
            comm: MPI communicator
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        aggregated = self.gather_mpi_results(comm)
        
        if rank == 0:
            # Calculate total time
            if 'total' in aggregated:
                total_time = aggregated.get('total', {}).get('mean', 0)
                del aggregated['total']
                total_percet = sum(item['mean'] for item in aggregated.values())/total_time * 100.0
            else:
                total_time = sum(item['mean'] for item in aggregated.values())
                total_percet = 100.0
            
            # Sort by mean time
            results = []
            for name, stats in aggregated.items():
                percent = (stats['mean'] / total_time * 100) if total_time > 0 else 0
                results.append({
                    'name': name,
                    'mean': stats['mean'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'std': stats['std'],
                    'percent': percent,
                    'calls': stats['total_calls']
                })
            
            results.sort(key=lambda x: x['mean'], reverse=True)
            
            # Print header
            print(f"\n{'='*80}")
            print(f"MPI AGGREGATED TIMING SUMMARY (across {size} processes)")
            print(f"{'='*80}")
            print(f"{'Component':<30} {'Mean(s)':>10} {'Min(s)':>10} {'Max(s)':>10} {'%':>8}")
            print(f"{'-'*80}")
            
            # Print results
            for item in results:
                print(f"{item['name']:<30} {item['mean']:>10.4f} "
                      f"{item['min']:>10.4f} {item['max']:>10.4f} "
                      f"{item['percent']:>7.2f}%")
            
            print(f"{'-'*80}")
            print(f"{'TOTAL (mean)':<30} {total_time:>10.4f} {'':>10} {'':>10} {total_percet:>7.2f}%")
            print(f"{'='*80}\n")
            
            # Print load imbalance info
            # print(f"LOAD IMBALANCE ANALYSIS:")
            # print(f"{'-'*80}")
            # for item in sorted(results, key=lambda x: x['std'], reverse=True)[:5]:
            #     if item['mean'] > 0:
            #         imbalance = (item['max'] - item['min']) / item['mean'] * 100
            #         print(f"{item['name']:<30} Imbalance: {imbalance:>6.2f}% "
            #               f"(max/min ratio: {item['max']/max(item['min'],1e-10):>6.2f})")
            # print(f"{'='*80}\n")


# Global timer instance
_timer = SimpleMPITimer()


def timed(name: str = None):
    """Simple timing decorator"""
    def decorator(func):
        timing_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            _timer.start(timing_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _timer.stop(timing_name)
        return wrapper
    return decorator


# Convenience functions
def reset_timer():
    """Reset timer"""
    _timer.reset()


def print_summary(rank=None):
    """Print local summary"""
    _timer.print_local_summary(rank)


def print_mpi_summary(comm):
    """Print MPI aggregated summary"""
    _timer.print_mpi_summary(comm)


def enable_timing(enabled=True):
    """Enable/disable timing"""
    _timer.enabled = enabled


def get_timer():
    """Get timer instance"""
    return _timer


# Context manager
class Timer:
    """Simple context manager for timing"""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        _timer.start(self.name)
        return self
    
    def __exit__(self, *args):
        _timer.stop(self.name)