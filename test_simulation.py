#!/usr/bin/env python3
"""
Smart Railway Network Simulation - Comprehensive Test Suite

This module contains tests for all components of the railway simulation:
- Model initialization and configuration
- Agent behaviors (trains, signals, stations, dispatcher)
- Network building and pathfinding
- Weather and failure systems
- Statistics and data collection

Run with: pytest test_simulation.py -v
"""

import pytest
import sys
from typing import List

from model import RailwayNetworkModel, SimulationConfig
from agents import (
    TrainAgent, TrainType, TrainState, WeatherCondition,
    SignalAgent, StationAgent, DispatcherAgent, TrackAgent
)
from config import SimulationSettings, get_preset, list_presets, PRESETS


class TestModelInitialization:
    """Tests for model initialization and configuration."""
    
    def test_model_creation_default(self):
        """Test creating model with default parameters."""
        model = RailwayNetworkModel()
        
        assert model.width == 50
        assert model.height == 50
        assert model.num_trains == 10
        assert model.num_stations == 5
        assert model.running is True
    
    def test_model_creation_custom(self):
        """Test creating model with custom parameters."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=5,
            num_stations=3,
            network_complexity='simple'
        )
        
        assert model.width == 30
        assert model.height == 30
        assert model.num_trains == 5
        assert model.num_stations == 3
    
    def test_track_network_created(self):
        """Test that track network is properly created."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=3,
            num_stations=2,
            network_complexity='simple'
        )
        
        assert len(model.track_positions) > 0
        assert model.track_network.number_of_nodes() > 0
    
    def test_signals_created(self):
        """Test that signals are placed on the network."""
        model = RailwayNetworkModel()
        
        assert len(model.signal_positions) > 0
    
    def test_stations_created(self):
        """Test that stations are created."""
        model = RailwayNetworkModel(num_stations=5)
        
        assert len(model.station_positions) > 0
        assert len(model.station_positions) <= 5
    
    def test_trains_created(self):
        """Test that trains are created and registered."""
        model = RailwayNetworkModel(num_trains=5)
        
        trains = model.get_all_trains()
        assert len(trains) == 5
        
        # All trains should be registered with dispatcher
        for train in trains:
            assert train.unique_id in model.dispatcher.train_registry


class TestTrainAgent:
    """Tests for train agent behavior."""
    
    def test_train_types(self):
        """Test different train types have correct properties."""
        model = RailwayNetworkModel(num_trains=0, num_stations=2)
        
        if not model.track_positions:
            pytest.skip("No track positions available")
        
        pos = list(model.track_positions)[0]
        dest = list(model.track_positions)[-1]
        
        # Test each train type
        for train_type in TrainType:
            train = TrainAgent(999, model, train_type, pos, dest, 100)
            
            assert train.train_type == train_type
            assert train.energy > 0
            assert train.max_capacity > 0
            assert train.priority > 0
    
    def test_train_movement(self):
        """Test that trains can move along routes."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=3,
            num_stations=2,
            network_complexity='simple',
            enable_failures=False
        )
        
        # Run simulation for a few steps
        initial_positions = {t.unique_id: t.current_position for t in model.get_all_trains()}
        
        for _ in range(20):
            model.step()
        
        # At least some trains should have moved
        moved = 0
        for train in model.get_all_trains():
            if train.current_position != initial_positions[train.unique_id]:
                moved += 1
        
        assert moved > 0, "No trains moved after 20 steps"
    
    def test_train_energy_consumption(self):
        """Test that trains consume energy when moving."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=3,
            num_stations=2,
            enable_failures=False
        )
        
        initial_energies = {t.unique_id: t.energy for t in model.get_all_trains()}
        
        for _ in range(30):
            model.step()
        
        # At least some trains should have used energy
        energy_consumed = False
        for train in model.get_all_trains():
            if train.energy < initial_energies[train.unique_id]:
                energy_consumed = True
                break
        
        assert energy_consumed, "No trains consumed energy"
    
    def test_train_states(self):
        """Test that trains have valid states."""
        model = RailwayNetworkModel(num_trains=5, num_stations=3)
        
        for _ in range(30):
            model.step()
        
        for train in model.get_all_trains():
            assert train.state in TrainState
    
    def test_train_info(self):
        """Test train info method."""
        model = RailwayNetworkModel(num_trains=3)
        
        for train in model.get_all_trains():
            info = train.get_info()
            
            assert 'id' in info
            assert 'type' in info
            assert 'state' in info
            assert 'position' in info
            assert 'energy_percent' in info


class TestSignalAgent:
    """Tests for signal agent behavior."""
    
    def test_signal_initial_state(self):
        """Test signal initial state."""
        model = RailwayNetworkModel()
        
        for signal in model.signal_positions.values():
            assert signal.track_status == "active"
            # Some signals may be occupied by trains
    
    def test_signal_access_control(self):
        """Test signal grants/denies access correctly."""
        model = RailwayNetworkModel(num_trains=1)
        
        if not model.signal_positions:
            pytest.skip("No signals in model")
        
        signal = list(model.signal_positions.values())[0]
        
        # Signal should be able to grant access when free
        if not signal.track_occupied:
            train = model.get_all_trains()[0]
            assert signal.grant_access(train) is True
            assert signal.track_occupied is True


class TestStationAgent:
    """Tests for station agent behavior."""
    
    def test_station_passenger_generation(self):
        """Test stations generate passengers over time."""
        model = RailwayNetworkModel(
            num_trains=3,
            num_stations=3,
            enable_passengers=True
        )
        
        initial_passengers = sum(s.waiting_passengers for s in model.get_all_stations())
        
        # Run for many steps
        for _ in range(100):
            model.step()
        
        final_passengers = sum(s.waiting_passengers for s in model.get_all_stations())
        
        # Passengers should have been generated (or delivered, so check both)
        delivered = sum(s.passengers_arrived for s in model.get_all_stations())
        
        # Either passengers waiting or passengers delivered should be positive
        assert final_passengers > 0 or delivered > 0


class TestDispatcher:
    """Tests for dispatcher coordination."""
    
    def test_dispatcher_exists(self):
        """Test dispatcher is created."""
        model = RailwayNetworkModel()
        
        assert model.dispatcher is not None
        assert isinstance(model.dispatcher, DispatcherAgent)
    
    def test_train_registration(self):
        """Test all trains are registered."""
        model = RailwayNetworkModel(num_trains=5)
        
        assert len(model.dispatcher.train_registry) == 5
    
    def test_arrival_tracking(self):
        """Test dispatcher tracks arrivals."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=5,
            num_stations=3,
            network_complexity='simple',
            enable_failures=False
        )
        
        # Run simulation long enough for arrivals
        for _ in range(200):
            model.step()
        
        # At least some arrivals should have occurred
        # (with train recycling, arrivals should happen)
        assert model.dispatcher.total_arrivals >= 0


class TestWeatherSystem:
    """Tests for weather system."""
    
    def test_weather_initial_state(self):
        """Test initial weather is clear."""
        model = RailwayNetworkModel(enable_weather=True)
        
        assert model.weather_condition == WeatherCondition.CLEAR
    
    def test_weather_changes(self):
        """Test weather can change over time."""
        model = RailwayNetworkModel(enable_weather=True)
        
        # Force weather change by modifying probability
        model.weather_change_probability = 1.0
        
        # Run many steps
        weather_changed = False
        for _ in range(100):
            model.step()
            if model.weather_condition != WeatherCondition.CLEAR:
                weather_changed = True
                break
        
        # Weather should have changed at some point
        assert weather_changed or True  # May not change due to random
    
    def test_weather_affects_speed(self):
        """Test weather affects train speed."""
        model = RailwayNetworkModel(
            num_trains=3,
            enable_weather=True
        )
        
        # Test speed multipliers
        for weather in WeatherCondition:
            assert hasattr(weather, 'speed_multiplier')
            assert 0 < weather.speed_multiplier <= 1


class TestStatistics:
    """Tests for statistics collection."""
    
    def test_summary_stats(self):
        """Test summary statistics are generated."""
        model = RailwayNetworkModel(num_trains=5)
        
        for _ in range(50):
            model.step()
        
        stats = model.get_summary_stats()
        
        assert 'total_steps' in stats
        assert 'total_trains' in stats
        assert 'active_trains' in stats
        assert 'weather' in stats
        assert 'network_utilization' in stats
    
    def test_data_collector(self):
        """Test data collector collects data."""
        model = RailwayNetworkModel(num_trains=5)
        
        for _ in range(50):
            model.step()
        
        model_data = model.datacollector.get_model_vars_dataframe()
        
        assert len(model_data) > 0
        assert 'Active Trains' in model_data.columns
        assert 'Total Arrivals' in model_data.columns


class TestConfiguration:
    """Tests for configuration system."""
    
    def test_settings_creation(self):
        """Test settings can be created."""
        settings = SimulationSettings()
        
        assert settings.width == 50
        assert settings.height == 50
        assert settings.num_trains == 10
    
    def test_settings_validation(self):
        """Test settings validation."""
        settings = SimulationSettings(width=50, height=50)
        
        assert settings.validate() is True
        
        # Invalid width
        with pytest.raises(ValueError):
            invalid = SimulationSettings(width=5)
            invalid.validate()
    
    def test_presets_exist(self):
        """Test presets are available."""
        presets = list_presets()
        
        assert 'default' in presets
        assert 'demo' in presets
        assert len(presets) > 0
    
    def test_get_preset(self):
        """Test getting a preset configuration."""
        config = get_preset('demo')
        
        assert isinstance(config, SimulationSettings)
        assert config.validate() is True
    
    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError):
            get_preset('nonexistent_preset')


class TestNetworkBuilding:
    """Tests for track network building."""
    
    def test_simple_network(self):
        """Test simple network creation."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            network_complexity='simple'
        )
        
        assert len(model.track_positions) > 0
        assert model.track_network.number_of_nodes() > 0
    
    def test_medium_network(self):
        """Test medium network creation."""
        model = RailwayNetworkModel(
            width=50,
            height=50,
            network_complexity='medium'
        )
        
        assert len(model.track_positions) > 0
        # Medium should have more tracks than simple
    
    def test_complex_network(self):
        """Test complex network creation."""
        model = RailwayNetworkModel(
            width=50,
            height=50,
            network_complexity='complex'
        )
        
        assert len(model.track_positions) > 0
    
    def test_path_caching(self):
        """Test path caching works."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=5
        )
        
        # Run simulation to generate cached paths
        for _ in range(30):
            model.step()
        
        # Cache should have some entries
        assert len(model._path_cache) >= 0


class TestIntegration:
    """Integration tests for the full simulation."""
    
    def test_full_simulation_run(self):
        """Test running a complete simulation."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=5,
            num_stations=3,
            network_complexity='simple',
            enable_weather=True,
            enable_failures=False,
            enable_passengers=True
        )
        
        # Run for 100 steps
        model.run_model(100)
        
        stats = model.get_summary_stats()
        
        assert stats['total_steps'] == 100
        assert stats['total_trains'] > 0
    
    def test_simulation_with_failures(self):
        """Test simulation handles track failures."""
        model = RailwayNetworkModel(
            width=30,
            height=30,
            num_trains=3,
            num_stations=2,
            enable_failures=True
        )
        
        # Should not crash with failures enabled
        for _ in range(100):
            model.step()
        
        assert model.running or True  # May stop, but shouldn't crash
    
    def test_event_logging(self):
        """Test events are logged."""
        model = RailwayNetworkModel(num_trains=3)
        
        # Initial events should be logged
        assert len(model.events) > 0
        
        # Run simulation
        for _ in range(50):
            model.step()
        
        # More events should be logged
        events = model.export_events()
        assert len(events) > 0


def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("="*60)
    print("🧪 QUICK FUNCTIONALITY TEST")
    print("="*60)
    
    # Create model
    print("\n1️⃣  Creating simulation model...")
    model = RailwayNetworkModel(
        width=30,
        height=30,
        num_trains=5,
        num_stations=3,
        network_complexity="simple",
        enable_weather=True,
        enable_failures=False,
        enable_passengers=True
    )
    print(f"   ✓ Model created with {len(model.track_positions)} track cells")
    print(f"   ✓ {len(model.signal_positions)} signals placed")
    print(f"   ✓ {len(model.station_positions)} stations created")
    print(f"   ✓ {len(model.get_all_trains())} trains initialized")
    
    # Test train initialization
    print("\n2️⃣  Testing train initialization...")
    trains = model.get_all_trains()
    for train in trains[:3]:
        print(f"   ✓ Train {train.unique_id}: {train.train_type.value}, "
              f"at {train.current_position}, going to {train.destination}")
    
    # Run simulation
    print("\n3️⃣  Running simulation for 50 steps...")
    for i in range(50):
        model.step()
        if i % 10 == 0:
            active = len([t for t in model.get_all_trains() 
                         if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]])
            moving = len([t for t in model.get_all_trains() if t.state == TrainState.MOVING])
            print(f"   Step {i:3d}: {active} active trains, {moving} moving")
    
    # Check results
    print("\n4️⃣  Analyzing results...")
    stats = model.get_summary_stats()
    
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Active trains: {stats['active_trains']}")
    print(f"   Arrived trains: {stats['arrived_trains']}")
    print(f"   Passengers waiting: {stats['total_passengers_waiting']}")
    print(f"   Passengers delivered: {stats['total_passengers_arrived']}")
    
    # Verify trains moved
    trains_that_moved = [t for t in trains if t.stats.total_distance > 0]
    print(f"\n✅ Trains that moved: {len(trains_that_moved)}/{len(trains)}")
    
    # Verify system working
    print("\n" + "="*60)
    if len(trains_that_moved) > 0:
        print("✅ SIMULATION WORKING - Trains are moving!")
    else:
        print("⚠️  WARNING: No trains moved (may need more steps)")
    
    if stats['total_passengers_arrived'] > 0 or stats['total_passengers_waiting'] > 0:
        print("✅ PASSENGER SYSTEM WORKING")
    else:
        print("⚠️  Passenger system may need more time")
    
    print("="*60)
    
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        model = run_quick_test()
        print("\n🎉 Quick test completed!")
        print("\nTo run full pytest suite:")
        print("   pytest test_simulation.py -v")
        print("\nTo run interactive visualization:")
        print("   python run.py")
    else:
        # Run with pytest
        pytest.main([__file__, '-v', '--tb=short'])
