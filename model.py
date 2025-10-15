"""
Smart Railway Network Simulation - Main Model
Railway network simulation model with comprehensive features.
"""

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agents import (
    TrackAgent, TrainAgent, SignalAgent, DispatcherAgent, StationAgent, PassengerAgent,
    TrainType, TrainState, WeatherCondition
)
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx


class RailwayNetworkModel(Model):
    """
    Main model for the smart railway network simulation.
    
    Features:
    - Multiple train types with different behaviors
    - Signal-based track access control
    - Central dispatcher for coordination
    - Dynamic weather conditions
    - Track failures and maintenance
    - Passenger simulation
    - Energy management
    - Comprehensive statistics collection
    """
    
    def __init__(self, width: int = 50, height: int = 50, 
                 num_trains: int = 10,
                 num_stations: int = 5,
                 network_complexity: str = "medium",
                 enable_weather: bool = True,
                 enable_failures: bool = True,
                 enable_passengers: bool = True):
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_trains = num_trains
        self.num_stations = num_stations
        self.network_complexity = network_complexity
        self.enable_weather = enable_weather
        self.enable_failures = enable_failures
        self.enable_passengers = enable_passengers
        
        # Initialize grid and scheduler
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        
        # Track network structure
        self.track_network = nx.Graph()
        self.track_positions: set = set()
        self.signal_positions: Dict[Tuple[int, int], SignalAgent] = {}
        self.station_positions: Dict[Tuple[int, int], StationAgent] = {}
        
        # Environment conditions
        self.weather_condition = WeatherCondition.CLEAR
        self.weather_change_probability = 0.02
        
        # Agent counters
        self.next_train_id = 0
        self.next_signal_id = 10000
        self.next_station_id = 20000
        self.next_passenger_id = 30000
        self.next_track_id = 40000
        
        # Create dispatcher (only one)
        self.dispatcher = DispatcherAgent(0, self)
        self.schedule.add(self.dispatcher)
        
        # Build railway network
        self._build_track_network()
        
        # Create track visualization agents
        self._create_track_agents()
        
        # Create signals
        self._create_signals()
        
        # Create stations
        self._create_stations()
        
        # Create initial trains
        self._create_initial_trains()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Active Trains": lambda m: sum(1 for a in m.schedule.agents 
                                              if isinstance(a, TrainAgent) and 
                                              a.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]),
                "Delayed Trains": lambda m: sum(1 for a in m.schedule.agents 
                                               if isinstance(a, TrainAgent) and 
                                               a.state == TrainState.DELAYED),
                "Total Arrivals": lambda m: m.dispatcher.total_arrivals,
                "Average Delay": lambda m: m.dispatcher.get_average_delay(),
                "Track Failures": lambda m: len(m.dispatcher.track_failures),
                "Weather": lambda m: m.weather_condition.value,
                "Total Energy": lambda m: sum(a.energy for a in m.schedule.agents 
                                             if isinstance(a, TrainAgent)),
                "Waiting Passengers": lambda m: sum(s.waiting_passengers for s in m.schedule.agents 
                                                   if isinstance(s, StationAgent)),
                "Priority Conflicts": lambda m: len(m.dispatcher.priority_conflicts),
            },
            agent_reporters={
                "State": lambda a: a.state.value if isinstance(a, TrainAgent) else None,
                "Energy": lambda a: a.energy if isinstance(a, TrainAgent) else None,
                "Delay": lambda a: a.delay_time if isinstance(a, TrainAgent) else None,
                "Passengers": lambda a: a.passengers if isinstance(a, TrainAgent) else None,
            }
        )
        
        self.running = True
    
    def _build_track_network(self):
        """Build the railway track network based on complexity level."""
        if self.network_complexity == "simple":
            self._build_simple_network()
        elif self.network_complexity == "medium":
            self._build_medium_network()
        else:
            self._build_complex_network()
    
    def _build_simple_network(self):
        """Build a simple linear track network."""
        # Main horizontal line
        for x in range(5, self.width - 5):
            y = self.height // 2
            self.track_positions.add((x, y))
            self.track_network.add_node((x, y))
            if x > 5:
                self.track_network.add_edge((x-1, y), (x, y))
        
        # Add some vertical branches
        for branch_x in [15, 25, 35]:
            for y in range(self.height // 2 - 10, self.height // 2 + 10):
                self.track_positions.add((branch_x, y))
                self.track_network.add_node((branch_x, y))
                if y > self.height // 2 - 10:
                    self.track_network.add_edge((branch_x, y-1), (branch_x, y))
    
    def _build_medium_network(self):
        """Build a medium complexity railway network with multiple lines - realistic metro-style network."""
        # Create a realistic railway network inspired by real metro systems
        # Features: Main ring line + radial lines + cross-city lines
        
        center_x, center_y = self.width // 2, self.height // 2
        
        # Helper function to ensure positions are within bounds
        def is_valid_pos(x, y):
            return 0 <= x < self.width and 0 <= y < self.height
        
        # 1. BUILD OUTER RING LINE (Circle Line) - properly connected
        ring_radius = 18
        ring_points = []
        num_ring_points = 72  # More points for smoother circle
        
        for i in range(num_ring_points):
            angle = (i * 360 / num_ring_points)
            rad = np.radians(angle)
            x = int(center_x + ring_radius * np.cos(rad))
            y = int(center_y + ring_radius * np.sin(rad))
            
            if is_valid_pos(x, y):
                ring_points.append((x, y))
        
        # Add ring line to network
        for i, point in enumerate(ring_points):
            self.track_positions.add(point)
            self.track_network.add_node(point)
            if i > 0:
                self.track_network.add_edge(ring_points[i-1], point)
        # Close the ring
        if len(ring_points) > 1:
            self.track_network.add_edge(ring_points[-1], ring_points[0])
        
        # 2. BUILD INNER RING LINE (smaller circle)
        inner_radius = 8
        inner_ring_points = []
        for i in range(36):
            angle = (i * 360 / 36)
            rad = np.radians(angle)
            x = int(center_x + inner_radius * np.cos(rad))
            y = int(center_y + inner_radius * np.sin(rad))
            if is_valid_pos(x, y):
                inner_ring_points.append((x, y))
        
        for i, point in enumerate(inner_ring_points):
            self.track_positions.add(point)
            self.track_network.add_node(point)
            if i > 0:
                self.track_network.add_edge(inner_ring_points[i-1], point)
        if len(inner_ring_points) > 1:
            self.track_network.add_edge(inner_ring_points[-1], inner_ring_points[0])
        
        # 3. BUILD RADIAL LINES (spoke pattern from center to outer ring)
        num_radials = 8
        for i in range(num_radials):
            angle = i * 360 / num_radials
            rad = np.radians(angle)
            
            # Create radial line from inner to outer ring
            for r in range(inner_radius + 1, ring_radius + 3):
                x = int(center_x + r * np.cos(rad))
                y = int(center_y + r * np.sin(rad))
                
                if is_valid_pos(x, y):
                    self.track_positions.add((x, y))
                    self.track_network.add_node((x, y))
                    
                    # Connect to previous point on this radial
                    prev_x = int(center_x + (r-1) * np.cos(rad))
                    prev_y = int(center_y + (r-1) * np.sin(rad))
                    if (prev_x, prev_y) in self.track_positions:
                        self.track_network.add_edge((prev_x, prev_y), (x, y))
        
        # 4. HORIZONTAL CROSS-CITY LINES
        for y_offset in [-12, -6, 0, 6, 12]:
            y = center_y + y_offset
            if is_valid_pos(0, y):
                for x in range(8, self.width - 8):
                    if is_valid_pos(x, y):
                        self.track_positions.add((x, y))
                        self.track_network.add_node((x, y))
                        if x > 8 and (x-1, y) in self.track_positions:
                            self.track_network.add_edge((x-1, y), (x, y))
        
        # 5. VERTICAL CROSS-CITY LINES  
        for x_offset in [-12, -6, 0, 6, 12]:
            x = center_x + x_offset
            if is_valid_pos(x, 0):
                for y in range(8, self.height - 8):
                    if is_valid_pos(x, y):
                        self.track_positions.add((x, y))
                        self.track_network.add_node((x, y))
                        if y > 8 and (x, y-1) in self.track_positions:
                            self.track_network.add_edge((x, y-1), (x, y))
        
        # 6. CONNECT INTERSECTIONS - ensure all crossing points are connected
        # This is critical for proper pathfinding
        for pos in list(self.track_positions):
            x, y = pos
            # Check all 4 directions and connect if track exists
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor in self.track_positions and is_valid_pos(neighbor[0], neighbor[1]):
                    # Add edge if not already present
                    if not self.track_network.has_edge(pos, neighbor):
                        self.track_network.add_edge(pos, neighbor)
    
    def _build_complex_network(self):
        """Build a complex railway network with multiple intersecting lines."""
        # Create a mesh network
        track_spacing = 5
        
        # Horizontal lines
        for y in range(10, self.height - 10, track_spacing):
            for x in range(5, self.width - 5):
                self.track_positions.add((x, y))
                self.track_network.add_node((x, y))
                if x > 5:
                    self.track_network.add_edge((x-1, y), (x, y))
        
        # Vertical lines
        for x in range(10, self.width - 10, track_spacing):
            for y in range(5, self.height - 5):
                self.track_positions.add((x, y))
                self.track_network.add_node((x, y))
                if y > 5:
                    self.track_network.add_edge((x, y-1), (x, y))
        
        # Diagonal connections for more interesting routes
        for x in range(10, self.width - 10, track_spacing):
            for y in range(10, self.height - 10, track_spacing):
                # Add diagonal tracks
                if (x + track_spacing, y + track_spacing) in self.track_positions:
                    for i in range(1, track_spacing):
                        diag_x, diag_y = x + i, y + i
                        if 0 <= diag_x < self.width and 0 <= diag_y < self.height:
                            self.track_positions.add((diag_x, diag_y))
                            self.track_network.add_node((diag_x, diag_y))
    
    def _create_track_agents(self):
        """Create track agents for visualization."""
        for pos in self.track_positions:
            track = TrackAgent(self.next_track_id, self, pos)
            self.next_track_id += 1
            # Don't add tracks to scheduler (they don't step)
            # Just place them on grid for visualization
            self.grid.place_agent(track, pos)
    
    def _create_signals(self):
        """Create signal agents at key track positions."""
        # Place signals at regular intervals and junctions
        signal_positions = []
        
        # Find junction points (nodes with degree > 2)
        for node in self.track_network.nodes():
            if self.track_network.degree(node) > 2:
                signal_positions.append(node)
        
        # Add signals at regular intervals along tracks
        track_list = list(self.track_positions)
        num_additional_signals = max(20, len(track_list) // 20)
        
        if track_list:
            sample_size = min(num_additional_signals, len(track_list))
            signal_positions.extend(random.sample(track_list, sample_size))
        
        # Create signal agents
        for pos in signal_positions:
            signal = SignalAgent(self.next_signal_id, self, pos)
            self.next_signal_id += 1
            self.signal_positions[pos] = signal
            self.schedule.add(signal)
            self.grid.place_agent(signal, pos)
    
    def _create_stations(self):
        """Create station agents at strategic locations."""
        # Find good station locations (track positions with good connectivity)
        track_list = list(self.track_positions)
        
        if len(track_list) < self.num_stations:
            station_positions = track_list
        else:
            # Select positions that are well-distributed
            station_positions = []
            step = len(track_list) // self.num_stations
            for i in range(self.num_stations):
                idx = min(i * step, len(track_list) - 1)
                station_positions.append(track_list[idx])
        
        # Create station agents
        station_names = ["Central", "North", "South", "East", "West", "Downtown", 
                        "Airport", "Harbor", "University", "Industrial"]
        
        for i, pos in enumerate(station_positions):
            name = station_names[i % len(station_names)]
            station = StationAgent(self.next_station_id, self, pos, name)
            self.next_station_id += 1
            self.station_positions[pos] = station
            self.schedule.add(station)
            self.grid.place_agent(station, pos)
    
    def _create_initial_trains(self):
        """Create initial train schedule - SPAWN TRAINS IMMEDIATELY for continuous movement."""
        # NEW APPROACH: Spawn initial trains immediately so they're always moving
        # Then schedule future departures for continuous operation
        
        if not self.track_positions or not self.station_positions:
            return
        
        station_list = list(self.station_positions.keys())
        track_list = list(self.track_positions)
        
        # Train type distribution for the schedule
        train_types = [
            TrainType.PASSENGER,
            TrainType.PASSENGER,
            TrainType.PASSENGER,
            TrainType.CARGO,
            TrainType.CARGO,
            TrainType.EXPRESS,
            TrainType.EXPRESS,
            TrainType.EMERGENCY,
        ]
        
        # SPAWN INITIAL TRAINS IMMEDIATELY (for immediate movement from start)
        for i in range(self.num_trains):
            train_type = train_types[i % len(train_types)]
            
            # Select route based on train type
            if train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
                if len(station_list) >= 2:
                    origin = random.choice(station_list)
                    destination = random.choice([s for s in station_list if s != origin])
                else:
                    origin = random.choice(track_list)
                    destination = random.choice(track_list)
            else:
                # Cargo and emergency can use any track
                origin = random.choice(track_list)
                destination = random.choice([t for t in track_list if t != origin])
            
            # Calculate expected arrival
            distance = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
            estimated_travel_time = max(distance // 2, 30)
            scheduled_arrival = estimated_travel_time + 20
            
            # Create and spawn train immediately
            train = TrainAgent(
                self.next_train_id,
                self,
                train_type,
                origin,
                destination,
                scheduled_arrival
            )
            self.next_train_id += 1
            
            self.dispatcher.register_train(train)
            self.schedule.add(train)
            self.grid.place_agent(train, origin)
            train.calculate_route()
        
        # Also schedule future departures for continuous operation
        for i in range(self.num_trains // 2):  # Schedule some future trains
            train_type = train_types[i % len(train_types)]
            departure_time = (i + 1) * 50 + random.randint(10, 30)
            
            if train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
                if len(station_list) >= 2:
                    origin = random.choice(station_list)
                    destination = random.choice([s for s in station_list if s != origin])
                else:
                    origin = random.choice(track_list)
                    destination = random.choice(track_list)
            else:
                origin = random.choice(track_list)
                destination = random.choice(track_list)
            
            self.dispatcher.schedule_train_departure(
                train_type=train_type,
                origin=origin,
                destination=destination,
                departure_time=departure_time
            )
    
    def is_track(self, pos: Tuple[int, int]) -> bool:
        """Check if a position has track."""
        return pos in self.track_positions
    
    def get_signal_at(self, pos: Tuple[int, int]) -> Optional[SignalAgent]:
        """Get signal agent at a position, if any."""
        return self.signal_positions.get(pos)
    
    def get_station_at(self, pos: Tuple[int, int]) -> Optional[StationAgent]:
        """Get station agent at a position, if any."""
        return self.station_positions.get(pos)
    
    def update_weather(self):
        """Update weather conditions randomly."""
        if not self.enable_weather:
            self.weather_condition = WeatherCondition.CLEAR
            return
        
        if random.random() < self.weather_change_probability:
            self.weather_condition = random.choice(list(WeatherCondition))
    
    def spawn_new_train(self, train_type: TrainType = None):
        """Spawn a new train during simulation."""
        if not self.track_positions:
            return
        
        track_list = list(self.track_positions)
        station_list = list(self.station_positions.keys())
        
        if train_type is None:
            train_type = random.choice(list(TrainType))
        
        # CRITICAL: Passenger and Express trains must go between stations!
        if train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
            if len(station_list) >= 2:
                start_pos = random.choice(station_list)
                destination = random.choice(station_list)
                attempts = 0
                while start_pos == destination and attempts < 10:
                    destination = random.choice(station_list)
                    attempts += 1
            else:
                start_pos = random.choice(track_list)
                destination = random.choice(track_list)
        else:
            start_pos = random.choice(track_list)
            destination = random.choice(track_list)
        
        scheduled_arrival = self.schedule.steps + random.randint(50, 150)
        
        train = TrainAgent(self.next_train_id, self, train_type, start_pos, 
                         destination, scheduled_arrival)
        self.next_train_id += 1
        
        self.dispatcher.register_train(train)
        self.schedule.add(train)
        self.grid.place_agent(train, start_pos)
        train.calculate_route()
    
    def get_trains_by_state(self, state: TrainState) -> List[TrainAgent]:
        """Get all trains in a specific state."""
        return [a for a in self.schedule.agents 
                if isinstance(a, TrainAgent) and a.state == state]
    
    def get_all_trains(self) -> List[TrainAgent]:
        """Get all train agents."""
        return [a for a in self.schedule.agents if isinstance(a, TrainAgent)]
    
    def get_all_stations(self) -> List[StationAgent]:
        """Get all station agents."""
        return [a for a in self.schedule.agents if isinstance(a, StationAgent)]
    
    def get_network_utilization(self) -> float:
        """Calculate percentage of tracks currently occupied."""
        occupied = sum(1 for signal in self.signal_positions.values() 
                      if signal.track_occupied)
        total = len(self.signal_positions)
        return (occupied / total * 100) if total > 0 else 0
    
    def step(self):
        """Execute one step of the simulation."""
        # Update weather
        self.update_weather()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Step all agents (dispatcher will spawn trains based on schedule)
        self.schedule.step()
        
        # Simulation runs continuously - trains are recycled automatically
        # No need to check for stopping condition in normal operations
    
    def run_model(self, steps: int = 500):
        """Run the model for a specified number of steps."""
        for i in range(steps):
            if not self.running:
                break
            self.step()
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics."""
        trains = self.get_all_trains()
        stations = self.get_all_stations()
        
        return {
            'total_steps': self.schedule.steps,
            'total_trains': len(trains),
            'active_trains': len([t for t in trains if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]]),
            'arrived_trains': len([t for t in trains if t.state == TrainState.ARRIVED]),
            'delayed_trains': len([t for t in trains if t.delay_time > 0]),
            'average_delay': self.dispatcher.get_average_delay(),
            'total_energy': sum(t.energy for t in trains),
            'average_energy': sum(t.energy for t in trains) / len(trains) if trains else 0,
            'total_passengers_waiting': sum(s.waiting_passengers for s in stations),
            'total_passengers_arrived': sum(s.passengers_arrived for s in stations),
            'track_failures': len(self.dispatcher.track_failures),
            'priority_conflicts': len(self.dispatcher.priority_conflicts),
            'network_utilization': self.get_network_utilization(),
            'weather': self.weather_condition.value,
        }

