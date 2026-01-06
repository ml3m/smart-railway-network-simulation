"""
Smart Railway Network Simulation - Configuration Management

This module provides configuration management for the simulation,
including preset configurations and the ability to load/save settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
import os


@dataclass
class SimulationSettings:
    """
    Main configuration settings for the railway simulation.
    
    Attributes:
        width: Grid width in cells
        height: Grid height in cells
        num_trains: Initial number of trains
        num_stations: Number of stations
        network_complexity: 'simple', 'medium', or 'complex'
        enable_weather: Enable weather effects
        enable_failures: Enable random track failures
        enable_passengers: Enable passenger simulation
        weather_change_probability: Probability of weather change per step
        failure_probability: Probability of track failure per step
        repair_probability: Probability of track repair per step
        max_trains: Maximum number of trains allowed
        simulation_speed: Steps per second for visualization
    """
    # Grid settings
    width: int = 50
    height: int = 50
    
    # Agent counts
    num_trains: int = 10
    num_stations: int = 5
    max_trains: int = 50
    
    # Network settings
    network_complexity: str = "medium"
    
    # Feature toggles
    enable_weather: bool = True
    enable_failures: bool = True
    enable_passengers: bool = True
    
    # Probabilities
    weather_change_probability: float = 0.02
    failure_probability: float = 0.0001
    repair_probability: float = 0.3
    
    # Visualization settings
    simulation_speed: int = 10  # Steps per second
    grid_cell_size: int = 12    # Pixels per cell
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.width < 20 or self.width > 200:
            raise ValueError(f"Width must be between 20 and 200, got {self.width}")
        if self.height < 20 or self.height > 200:
            raise ValueError(f"Height must be between 20 and 200, got {self.height}")
        if self.num_trains < 1 or self.num_trains > self.max_trains:
            raise ValueError(f"num_trains must be between 1 and {self.max_trains}")
        if self.num_stations < 2 or self.num_stations > 20:
            raise ValueError(f"num_stations must be between 2 and 20")
        if self.network_complexity not in ['simple', 'medium', 'complex']:
            raise ValueError(f"network_complexity must be 'simple', 'medium', or 'complex'")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationSettings':
        """Create settings from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def save(self, filepath: str):
        """Save settings to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationSettings':
        """Load settings from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Preset configurations for different scenarios
PRESETS: Dict[str, SimulationSettings] = {
    'default': SimulationSettings(),
    
    'small_network': SimulationSettings(
        width=30,
        height=30,
        num_trains=5,
        num_stations=3,
        network_complexity='simple',
        enable_failures=False
    ),
    
    'large_network': SimulationSettings(
        width=80,
        height=80,
        num_trains=25,
        num_stations=10,
        network_complexity='complex',
        max_trains=100
    ),
    
    'stress_test': SimulationSettings(
        width=100,
        height=100,
        num_trains=50,
        num_stations=15,
        network_complexity='complex',
        enable_weather=True,
        enable_failures=True,
        max_trains=150
    ),
    
    'ideal_conditions': SimulationSettings(
        num_trains=15,
        num_stations=6,
        network_complexity='medium',
        enable_weather=False,
        enable_failures=False
    ),
    
    'demo': SimulationSettings(
        width=50,
        height=50,
        num_trains=8,
        num_stations=5,
        network_complexity='medium',
        enable_weather=True,
        enable_failures=False,
        simulation_speed=5
    ),
    
    'high_traffic': SimulationSettings(
        width=60,
        height=60,
        num_trains=30,
        num_stations=8,
        network_complexity='complex',
        max_trains=80
    ),
}


def get_preset(name: str) -> SimulationSettings:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> Dict[str, str]:
    """List available presets with descriptions."""
    return {
        'default': 'Standard 50x50 grid with 10 trains, all features enabled',
        'small_network': 'Small 30x30 network for quick testing',
        'large_network': 'Large 80x80 complex network with many trains',
        'stress_test': 'Maximum load test configuration',
        'ideal_conditions': 'Medium network with no weather/failures',
        'demo': 'Balanced settings for demonstrations',
        'high_traffic': 'High train density scenario'
    }


# Environment variable based configuration
def load_from_env() -> SimulationSettings:
    """Load configuration from environment variables."""
    settings = SimulationSettings()
    
    env_mappings = {
        'RAILWAY_WIDTH': ('width', int),
        'RAILWAY_HEIGHT': ('height', int),
        'RAILWAY_NUM_TRAINS': ('num_trains', int),
        'RAILWAY_NUM_STATIONS': ('num_stations', int),
        'RAILWAY_COMPLEXITY': ('network_complexity', str),
        'RAILWAY_WEATHER': ('enable_weather', lambda x: x.lower() == 'true'),
        'RAILWAY_FAILURES': ('enable_failures', lambda x: x.lower() == 'true'),
        'RAILWAY_PASSENGERS': ('enable_passengers', lambda x: x.lower() == 'true'),
    }
    
    for env_var, (attr, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                setattr(settings, attr, converter(value))
            except (ValueError, TypeError):
                pass  # Ignore invalid values
    
    return settings

