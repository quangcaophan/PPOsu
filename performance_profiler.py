import time
import psutil
import threading
from collections import deque, defaultdict
from contextlib import contextmanager

class PerformanceProfiler:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.timings = defaultdict(lambda: deque(maxlen=buffer_size))
        self.counters = defaultdict(int)
        self.cpu_usage = deque(maxlen=buffer_size)
        self.memory_usage = deque(maxlen=buffer_size)
        self.gpu_usage = deque(maxlen=buffer_size)
        self.monitoring = False
        self.monitor_thread = None
        
    @contextmanager
    def time_it(self, operation_name):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to ms
            self.timings[operation_name].append(duration)
    
    def start_system_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
    def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_system(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu)
                
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                gpu = psutil.cpu_percent(interval=0.1)
                self.gpu_usage.append(gpu)
                
                time.sleep(0.5)
            except Exception:
                break
    
    def increment_counter(self, counter_name):
        """Increment counter"""
        self.counters[counter_name] += 1
    
    def get_stats(self, operation_name=None):
        """Get performance statistics"""
        if operation_name:
            if operation_name in self.timings:
                import numpy as np
                times = list(self.timings[operation_name])
                return {
                    'operation': operation_name,
                    'count': len(times),
                    'avg_ms': np.mean(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'p95_ms': np.percentile(times, 95) if times else 0
                }
        else:
            stats = {}
            for op_name in self.timings.keys():
                stats[op_name] = self.get_stats(op_name)
            return stats
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*60)
        print("🔍 PERFORMANCE REPORT")
        print("="*60)
        
        # System resources
        if self.cpu_usage:
            import numpy as np
            print(f"\n📊 SYSTEM:")
            print(f"  CPU: Avg {np.mean(self.cpu_usage):.1f}% | Max {np.max(self.cpu_usage):.1f}%")
            print(f"  RAM: Avg {np.mean(self.memory_usage):.1f}% | Max {np.max(self.memory_usage):.1f}%")
            print(f"  GPU: Avg {np.mean(self.gpu_usage):.1f}% | Max {np.max(self.gpu_usage):.1f}%")
        
        # Timing stats
        stats = self.get_stats()
        if stats:
            print(f"\n⏱️ TIMING:")
            sorted_ops = sorted(stats.items(), key=lambda x: x[1]['avg_ms'], reverse=True)
            
            print(f"{'Operation':<20} {'Count':<8} {'Avg(ms)':<10} {'Max(ms)':<10}")
            print("-" * 55)
            
            for op_name, stat in sorted_ops:
                print(f"{op_name:<20} {stat['count']:<8} {stat['avg_ms']:<10.2f} {stat['max_ms']:<10.2f}")
        
        # Counters
        if self.counters:
            print(f"\n📈 COUNTERS:")
            for name, count in self.counters.items():
                print(f"  {name}: {count}")
        
        print("="*60)

# Global profiler instance
profiler = PerformanceProfiler()

def start_profiling():
    profiler.start_system_monitoring()
    print("🔍 Profiling started")

def stop_profiling():
    profiler.stop_system_monitoring()
    profiler.print_report()

def time_operation(operation_name):
    return profiler.time_it(operation_name)

def profile_environment_step(env_step_func):
    """Decorator để profile environment step function"""
    def wrapper(*args, **kwargs):
        with profiler.time_it('step_total'):
            # Profile individual components
            with profiler.time_it('action_execution'):
                pass  # This will be filled in by the actual implementation
            
            result = env_step_func(*args, **kwargs)
            
            profiler.increment_counter('total_steps')
            return result
    return wrapper

if __name__ == "__main__":
    # Demo usage
    print("Performance Profiler Demo")
    
    start_profiling()
    
    # Simulate some operations
    for i in range(10):
        with time_operation('demo_operation'):
            time.sleep(0.01)  # Simulate 10ms operation
        
        with time_operation('fast_operation'):
            time.sleep(0.001)  # Simulate 1ms operation
    
    time.sleep(2)  # Let system monitoring collect some data
    stop_profiling()