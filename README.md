# PPO Osu! Mania AI Agent - Refactored Version

A clean, modular, and scalable reinforcement learning agent for playing osu!mania using the PPO algorithm.

## 🏗️ **New Architecture**

### **Clean Separation of Concerns**
```
src/
├── core/           # Core utilities (config, logging)
│   ├── config_manager.py
│   └── logger.py
├── environment/    # Environment modules
│   ├── mania_env.py
│   ├── frame_processor.py
│   ├── reward_calculator.py
│   └── constants.py
├── training/       # Training modules
│   ├── trainer.py
│   └── callbacks.py
└── utils/          # Utility modules
    ├── visualization.py
    └── performance.py
```

## 🚀 **Quick Start**

### **1. Migration (if upgrading from old version)**
```bash
python migrate_to_refactored.py
```

### **2. Training**
```bash
# Train with 4K configuration
python main.py --config mania_4k

# Train with custom timesteps
python main.py --config mania_4k --timesteps 200000

# Show evaluation window
python main.py --config mania_4k --show-eval
```

### **3. Playing**
```bash
# Play with trained agent
python play_agent_refactored.py --config mania_4k

# Use specific model
python play_agent_refactored.py --model models/mania/4k/best_model/best_model.zip
```

## 📋 **Key Improvements**

### **1. Configuration Management**
- **Type-safe**: Full dataclass-based configuration
- **Validation**: Automatic config validation
- **Flexible**: Easy to extend and modify

```python
from src.core.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("mania_4k")
```

### **2. Modular Environment**
- **Frame Processing**: Separate `FrameProcessor` class
- **Reward Calculation**: Dedicated `RewardCalculator` class
- **Clean Interface**: Well-defined environment API

### **3. Advanced Training System**
- **Comprehensive Callbacks**: Multiple monitoring callbacks
- **Error Handling**: Graceful error recovery
- **Logging**: Structured logging throughout

### **4. Better Performance**
- **Optimized Frame Processing**: Background thread with queue
- **Memory Management**: Efficient state tracking
- **Performance Monitoring**: Real-time metrics

## 🔧 **Configuration**

### **Creating New Configurations**
```python
from src.core.config_manager import ConfigManager, AgentConfig

# Create default config
config_manager = ConfigManager()
config = config_manager.create_default_config("mania", 4)

# Customize parameters
config.reward_params.hit_300_reward = 3.0
config.training_params.ppo_params.learning_rate = 0.0001

# Save configuration
config_manager.save_config(config, "my_custom_config")
```

### **Configuration Structure**
```python
@dataclass
class AgentConfig:
    mode: str = "mania"
    num_keys: int = 4
    play_area: PlayArea
    max_steps: int = 15000
    reward_params: RewardParams
    training_params: TrainingParams
    timestamp: str = ""
```

## 📊 **Monitoring & Logging**

### **Structured Logging**
- **Multiple Levels**: DEBUG, INFO, WARNING, ERROR
- **Colored Output**: Easy to read console output
- **File Logging**: Automatic log rotation
- **Performance Metrics**: Real-time monitoring

### **TensorBoard Integration**
```bash
# View training progress
tensorboard --logdir tensorboard_logs
```

### **Performance Monitoring**
- **FPS Tracking**: Real-time frame rate monitoring
- **Memory Usage**: System resource monitoring
- **Reward Statistics**: Training progress tracking

## 🎯 **Advanced Features**

### **1. Curriculum Learning**
Automatically increases difficulty over time:
```python
# Gradually increase idle penalty
curriculum_callback = CurriculumCallback(env, initial_timesteps=20000)
```

### **2. Behavior Monitoring**
Track agent action patterns:
```python
# Monitor action distribution
behavior_callback = BehaviorMonitorCallback(check_freq=2000)
```

### **3. Learning Rate Scheduling**
Dynamic learning rate adjustment:
```python
# Schedule learning rate decay
lr_scheduler = LearningRateScheduler(initial_lr=0.0003, final_lr=0.00001)
```

### **4. Performance Monitoring**
Comprehensive performance tracking:
```python
# Monitor system and training performance
perf_monitor = PerformanceMonitorCallback(check_freq=1000)
```

## 🛠️ **Development**

### **Adding New Features**

#### **1. New Environment Component**
```python
# src/environment/my_component.py
class MyComponent:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        # Your logic here
        return processed_data
```

#### **2. New Training Callback**
```python
# src/training/my_callback.py
class MyCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Your callback logic
        return True
```

#### **3. New Utility**
```python
# src/utils/my_utility.py
class MyUtility:
    def __init__(self):
        self.logger = get_logger("my_utility")
    
    def do_something(self):
        # Your utility logic
        pass
```

### **Testing**
```bash
# Run specific tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## 📈 **Performance Optimization**

### **Frame Processing**
- **Background Thread**: Non-blocking frame capture
- **Queue Management**: Efficient frame buffering
- **Memory Optimization**: Minimal memory footprint

### **Training Efficiency**
- **Batch Processing**: Optimized batch sizes
- **Memory Management**: Efficient state tracking
- **Checkpointing**: Regular model saving

### **System Resources**
- **CPU Usage**: Optimized processing
- **Memory Usage**: Efficient data structures
- **GPU Utilization**: CUDA optimization when available

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Configuration Not Found**
```bash
# List available configs
python -c "from src.core.config_manager import ConfigManager; print(ConfigManager().list_configs())"
```

#### **2. Model Loading Error**
```bash
# Check model compatibility
python -c "from stable_baselines3 import PPO; model = PPO.load('path/to/model.zip')"
```

#### **3. Environment Issues**
```bash
# Test environment
python -c "from src.environment.mania_env import OsuManiaEnv; env = OsuManiaEnv(config)"
```

### **Debug Mode**
```bash
# Enable debug logging
python main.py --config mania_4k --log-level DEBUG
```

## 📚 **API Reference**

### **Core Classes**

#### **ConfigManager**
```python
config_manager = ConfigManager(config_dir="config")
config = config_manager.load_config("mania_4k")
config_manager.save_config(config, "new_config")
```

#### **PPOTrainer**
```python
trainer = PPOTrainer("mania_4k", show_eval_window=True)
trainer.run_training(total_timesteps=100000)
```

#### **OsuManiaEnv**
```python
env = OsuManiaEnv(config, show_window=True)
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch**
3. **Follow the coding standards**
4. **Add tests for new features**
5. **Submit a pull request**

### **Coding Standards**
- **Type Hints**: All functions must have type hints
- **Docstrings**: Comprehensive documentation
- **Logging**: Use structured logging
- **Error Handling**: Comprehensive error handling

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **Stable Baselines3**: PPO implementation
- **OpenAI Gym**: Environment framework
- **OpenCV**: Computer vision processing
- **osu!**: The amazing rhythm game

---

**Happy Training! 🎵🤖**
