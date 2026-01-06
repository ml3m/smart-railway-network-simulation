"""
Smart Railway Network Simulation - Main Model

A sophisticated multi-agent railway network simulation featuring:
- Multiple train types (Passenger, Cargo, Express, Emergency)
- Signal-based track access control
- Central dispatcher for coordination and conflict resolution
- Dynamic weather conditions affecting operations
- Track failures and maintenance simulation
- Passenger boarding/alighting at stations
- Energy management for trains
- Real-time statistics and analytics

Built with Mesa framework for agent-based modeling.
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
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
import logging
import json

# Configure module logger with custom formatting
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class SimulationEvent:
    """Record of a significant simulation event for analytics."""
    step: int
    event_type: str
    description: str
    data: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class SimulationConfig:
    """Configuration settings for the simulation."""
    width: int = 50
    height: int = 50
    num_trains: int = 10
    num_stations: int = 5
    network_complexity: str = "medium"
    enable_weather: bool = True
    enable_failures: bool = True
    enable_passengers: bool = True
    weather_change_probability: float = 0.02
    failure_probability: float = 0.0001
    repair_probability: float = 0.3
    passenger_generation_rate: int = 5
    max_trains: int = 50
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'num_trains': self.num_trains,
            'num_stations': self.num_stations,
            'network_complexity': self.network_complexity,
            'enable_weather': self.enable_weather,
            'enable_failures': self.enable_failures,
            'enable_passengers': self.enable_passengers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RailwayNetworkModel(Model):
    """
    Main model for the smart railway network simulation.
    
    This model simulates a realistic railway network with multiple train types,
    signal-based track access control, passenger management, and dynamic
    environmental conditions.
    
    Features:
    - Multiple train types with different behaviors and priorities
    - Signal-based track access control with queue management
    - Central dispatcher for coordination and conflict resolution
    - Dynamic weather conditions affecting train speeds
    - Track failures and automatic repair simulation
    - Passenger boarding/alighting at stations
    - Energy management and refueling
    - Comprehensive real-time statistics collection
    - Event logging for analytics
    
    Attributes:
        width: Grid width in cells
        height: Grid height in cells
        num_trains: Initial number of trains to spawn
        num_stations: Number of stations to create
        network_complexity: Complexity level ('simple', 'medium', 'complex')
        enable_weather: Whether weather affects operations
        enable_failures: Whether random track failures occur
        enable_passengers: Whether passenger simulation is active
        config: SimulationConfig object with all settings
        events: List of simulation events for analytics
    """
    
    def __init__(self, width: int = 50, height: int = 50, 
                 num_trains: int = 10,
                 num_stations: int = 5,
                 network_complexity: str = "medium",
                 enable_weather: bool = True,
                 enable_failures: bool = True,
                 enable_passengers: bool = True,
                 config: SimulationConfig = None):
        super().__init__()
        
        # Store configuration
        if config:
            self.config = config
            self.width = config.width
            self.height = config.height
            self.num_trains = config.num_trains
            self.num_stations = config.num_stations
            self.network_complexity = config.network_complexity
            self.enable_weather = config.enable_weather
            self.enable_failures = config.enable_failures
            self.enable_passengers = config.enable_passengers
        else:
            self.config = SimulationConfig(
                width=width, height=height, num_trains=num_trains,
                num_stations=num_stations, network_complexity=network_complexity,
                enable_weather=enable_weather, enable_failures=enable_failures,
                enable_passengers=enable_passengers
            )
            self.width = width
            self.height = height
            self.num_trains = num_trains
            self.num_stations = num_stations
            self.network_complexity = network_complexity
            self.enable_weather = enable_weather
            self.enable_failures = enable_failures
            self.enable_passengers = enable_passengers
        
        # Initialize path cache for route caching
        self._path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[int, int]]] = {}
        
        # Event logging for analytics
        self.events: List[SimulationEvent] = []
        
        # Initialize grid and scheduler FIRST (needed for event logging)
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        
        # Now we can log events
        self._log_event("simulation_start", "Simulation initialized", {
            'width': self.width, 'height': self.height,
            'num_trains': self.num_trains, 'num_stations': self.num_stations,
            'complexity': self.network_complexity
        })
        
        logger.info(f"RailwayNetworkModel initialized: {self.width}x{self.height} grid, "
                   f"{self.num_trains} trains, {self.num_stations} stations")
        
        # Track network structure using NetworkX graph
        self.track_network: nx.Graph = nx.Graph()
        self.track_positions: Set[Tuple[int, int]] = set()
        self.signal_positions: Dict[Tuple[int, int], SignalAgent] = {}
        self.station_positions: Dict[Tuple[int, int], StationAgent] = {}
        
        # Environment conditions
        self.weather_condition = WeatherCondition.CLEAR
        self.weather_change_probability = self.config.weather_change_probability if config else 0.02
        self._weather_duration = 0  # How long current weather has lasted
        
        # Agent counters (using different ranges to avoid ID conflicts)
        self.next_train_id = 0
        self.next_signal_id = 10000
        self.next_station_id = 20000
        self.next_passenger_id = 30000
        self.next_track_id = 40000
        
        # Performance metrics
        self._step_times: List[float] = []
        self._peak_active_trains = 0
        
        # Create dispatcher (central coordination agent)
        self.dispatcher = DispatcherAgent(0, self)
        self.schedule.add(self.dispatcher)
        
        # Build railway network infrastructure
        self._build_track_network()
        self._create_track_agents()
        self._create_signals()
        self._create_stations()
        self._create_initial_trains()
        
        # Initialize data collection for analytics
        self.datacollector = DataCollector(
            model_reporters={
                "Active Trains": lambda m: sum(1 for a in m.schedule.agents 
                                              if isinstance(a, TrainAgent) and 
                                              a.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]),
                "Delayed Trains": lambda m: sum(1 for a in m.schedule.agents 
                                               if isinstance(a, TrainAgent) and 
                                               a.state == TrainState.DELAYED),
                "Moving Trains": lambda m: sum(1 for a in m.schedule.agents 
                                              if isinstance(a, TrainAgent) and 
                                              a.state == TrainState.MOVING),
                "Waiting Trains": lambda m: sum(1 for a in m.schedule.agents 
                                               if isinstance(a, TrainAgent) and 
                                               a.state == TrainState.WAITING),
                "Total Arrivals": lambda m: m.dispatcher.total_arrivals,
                "Average Delay": lambda m: m.dispatcher.get_average_delay(),
                "Track Failures": lambda m: len(m.dispatcher.track_failures),
                "Weather": lambda m: m.weather_condition.value,
                "Total Energy": lambda m: sum(a.energy for a in m.schedule.agents 
                                             if isinstance(a, TrainAgent)),
                "Average Energy": lambda m: np.mean([a.energy for a in m.schedule.agents 
                                                    if isinstance(a, TrainAgent)] or [0]),
                "Waiting Passengers": lambda m: sum(s.waiting_passengers for s in m.schedule.agents 
                                                   if isinstance(s, StationAgent)),
                "Passengers Delivered": lambda m: sum(s.passengers_arrived for s in m.schedule.agents 
                                                     if isinstance(s, StationAgent)),
                "Priority Conflicts": lambda m: len(m.dispatcher.priority_conflicts),
                "Network Utilization": lambda m: m.get_network_utilization(),
            },
            agent_reporters={
                "State": lambda a: a.state.value if isinstance(a, TrainAgent) else None,
                "Energy": lambda a: a.energy if isinstance(a, TrainAgent) else None,
                "Delay": lambda a: a.delay_time if isinstance(a, TrainAgent) else None,
                "Passengers": lambda a: a.passengers if isinstance(a, TrainAgent) else None,
                "Position": lambda a: a.current_position if isinstance(a, TrainAgent) else None,
                "TrainType": lambda a: a.train_type.value if isinstance(a, TrainAgent) else None,
            }
        )
        
        self.running = True
        self._log_event("infrastructure_ready", "Railway network infrastructure complete", {
            'track_cells': len(self.track_positions),
            'signals': len(self.signal_positions),
            'stations': len(self.station_positions)
        })
    
    def _log_event(self, event_type: str, description: str, data: Dict = None):
        """Log a simulation event for analytics."""
        event = SimulationEvent(
            step=self.schedule.steps if hasattr(self, 'schedule') else 0,
            event_type=event_type,
            description=description,
            data=data or {}
        )
        self.events.append(event)
        logger.debug(f"Event: {event_type} - {description}")
    
    def _build_track_network(self):
        """
        Build the railway track network based on complexity level.
        """
        if self.network_complexity == "simple":
            self._build_simple_network()
        elif self.network_complexity == "medium":
            self._build_medium_network()
        else:
            self._build_complex_network()
    
    def _build_simple_network(self):
        """Build a simple linear track network that scales with grid size."""
        margin = 5
        center_y = self.height // 2
        
        # Main horizontal line
        for x in range(margin, self.width - margin):
            self.track_positions.add((x, center_y))
            self.track_network.add_node((x, center_y))
            if x > margin:
                self.track_network.add_edge((x-1, center_y), (x, center_y))
        
        # Add vertical branches - scale based on grid width
        # Use proportional positions (1/4, 1/2, 3/4 of the grid)
        branch_positions = [
            int(self.width * 0.25),
            int(self.width * 0.5),
            int(self.width * 0.75)
        ]
        
        # Filter to valid positions and ensure uniqueness
        branch_positions = [x for x in branch_positions if margin < x < self.width - margin]
        branch_positions = list(set(branch_positions))
        
        branch_height = min(10, (self.height - 2 * margin) // 3)
        
        for branch_x in branch_positions:
            for y in range(center_y - branch_height, center_y + branch_height + 1):
                if 0 <= y < self.height:
                    self.track_positions.add((branch_x, y))
                    self.track_network.add_node((branch_x, y))
                    if y > center_y - branch_height and (branch_x, y-1) in self.track_positions:
                        self.track_network.add_edge((branch_x, y-1), (branch_x, y))
    
    def _build_medium_network(self):
        """Build a medium complexity railway network with multiple lines - realistic metro-style network."""
        # Create a realistic railway network inspired by real metro systems
        # Features: Main ring line + radial lines + cross-city lines
        # All dimensions scale based on grid size
        
        center_x, center_y = self.width // 2, self.height // 2
        min_dim = min(self.width, self.height)
        margin = max(5, min_dim // 10)
        
        # Helper function to ensure positions are within bounds
        def is_valid_pos(x, y):
            return margin <= x < self.width - margin and margin <= y < self.height - margin
        
        # Scale radii based on grid size
        ring_radius = min(18, (min_dim - 2 * margin) // 3)
        inner_radius = max(4, ring_radius // 2)
        
        # 1. BUILD OUTER RING LINE (Circle Line) - properly connected
        ring_points = []
        num_ring_points = max(36, ring_radius * 4)  # More points for smoother circle
        
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
        inner_ring_points = []
        for i in range(24):
            angle = (i * 360 / 24)
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
        
        # 4. HORIZONTAL CROSS-CITY LINES - scale offsets based on grid
        line_spacing = max(4, min_dim // 8)
        y_offsets = [-2 * line_spacing, -line_spacing, 0, line_spacing, 2 * line_spacing]
        
        for y_offset in y_offsets:
            y = center_y + y_offset
            if 0 <= y < self.height:
                for x in range(margin, self.width - margin):
                    if is_valid_pos(x, y):
                        self.track_positions.add((x, y))
                        self.track_network.add_node((x, y))
                        if x > margin and (x-1, y) in self.track_positions:
                            self.track_network.add_edge((x-1, y), (x, y))
        
        # 5. VERTICAL CROSS-CITY LINES  
        x_offsets = [-2 * line_spacing, -line_spacing, 0, line_spacing, 2 * line_spacing]
        
        for x_offset in x_offsets:
            x = center_x + x_offset
            if 0 <= x < self.width:
                for y in range(margin, self.height - margin):
                    if is_valid_pos(x, y):
                        self.track_positions.add((x, y))
                        self.track_network.add_node((x, y))
                        if y > margin and (x, y-1) in self.track_positions:
                            self.track_network.add_edge((x, y-1), (x, y))
        
        # 6. CONNECT INTERSECTIONS - ensure all crossing points are connected
        # This is critical for proper pathfinding
        for pos in list(self.track_positions):
            x, y = pos
            # Check all 4 directions and connect if track exists
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor in self.track_positions and 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    # Add edge if not already present
                    if not self.track_network.has_edge(pos, neighbor):
                        self.track_network.add_edge(pos, neighbor)
    
    def _build_complex_network(self):
        """Build a complex railway network with multiple intersecting lines."""
        # Create a mesh network - scale spacing based on grid size
        min_dim = min(self.width, self.height)
        track_spacing = max(4, min_dim // 10)
        margin = max(5, min_dim // 10)
        
        # Horizontal lines
        for y in range(margin, self.height - margin, track_spacing):
            for x in range(margin, self.width - margin):
                self.track_positions.add((x, y))
                self.track_network.add_node((x, y))
                if x > margin:
                    self.track_network.add_edge((x-1, y), (x, y))
        
        # Vertical lines
        for x in range(margin, self.width - margin, track_spacing):
            for y in range(margin, self.height - margin):
                self.track_positions.add((x, y))
                self.track_network.add_node((x, y))
                if y > margin:
                    self.track_network.add_edge((x, y-1), (x, y))
        
        # Diagonal connections for more interesting routes
        for x in range(margin, self.width - margin, track_spacing):
            for y in range(margin, self.height - margin, track_spacing):
                # Add diagonal tracks
                if (x + track_spacing, y + track_spacing) in self.track_positions:
                    for i in range(1, track_spacing):
                        diag_x, diag_y = x + i, y + i
                        if margin <= diag_x < self.width - margin and margin <= diag_y < self.height - margin:
                            self.track_positions.add((diag_x, diag_y))
                            self.track_network.add_node((diag_x, diag_y))
        
        # Connect all adjacent tracks for proper pathfinding
        for pos in list(self.track_positions):
            x, y = pos
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor in self.track_positions:
                    if not self.track_network.has_edge(pos, neighbor):
                        self.track_network.add_edge(pos, neighbor)
    
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
    
    def get_cached_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get cached path or compute new one using networkx shortest path.
        Caches the result for future calls to improve performance.
        """
        key = (start, goal)
        if key in self._path_cache:
            logger.debug(f"Cache hit for path {start} -> {goal}")
            return self._path_cache[key]
        try:
            path = nx.shortest_path(self.track_network, start, goal)
            logger.debug(f"Computed new path {start} -> {goal}: {path}")
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {start} and {goal}")
            path = []
        self._path_cache[key] = path
        return path
    
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
        """
        Execute one simulation step.
        
        This method:
        1. Updates weather conditions
        2. Collects data for analytics
        3. Steps all agents (dispatcher coordinates trains)
        4. Updates performance metrics
        """
        import time
        start_time = time.time()
        
        # Update weather conditions
        self.update_weather()
        
        # Collect data for analytics
        self.datacollector.collect(self)
        
        # Step all agents (dispatcher will spawn trains based on schedule)
        self.schedule.step()
        
        # Track performance metrics
        step_duration = time.time() - start_time
        self._step_times.append(step_duration)
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        
        # Track peak active trains
        active_count = len([t for t in self.get_all_trains() 
                          if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]])
        self._peak_active_trains = max(self._peak_active_trains, active_count)
        
        # Log milestone events
        if self.schedule.steps % 100 == 0 and self.schedule.steps > 0:
            self._log_event("milestone", f"Reached step {self.schedule.steps}", {
                'arrivals': self.dispatcher.total_arrivals,
                'active_trains': active_count
            })
    
    def run_model(self, steps: int = 500):
        """Run the model for a specified number of steps."""
        self._log_event("batch_run_start", f"Starting batch run for {steps} steps")
        for i in range(steps):
            if not self.running:
                break
            self.step()
        self._log_event("batch_run_complete", f"Batch run completed at step {self.schedule.steps}")

    def run_steps(self, steps: int):
        """Execute a specific number of simulation steps and return summary statistics."""
        for _ in range(steps):
            if not self.running:
                break
            self.step()
        return self.get_summary_stats()

    def get_statistics(self) -> Dict:
        """Alias for get_summary_stats to provide a concise API."""
        return self.get_summary_stats()
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics."""
        trains = self.get_all_trains()
        stations = self.get_all_stations()
        
        # Calculate train counts by state
        state_counts = {}
        for state in TrainState:
            state_counts[state.value] = len([t for t in trains if t.state == state])
        
        # Calculate train counts by type
        type_counts = {}
        for train_type in TrainType:
            type_counts[train_type.value] = len([t for t in trains if t.train_type == train_type])
        
        return {
            'total_steps': self.schedule.steps,
            'total_trains': len(trains),
            'active_trains': len([t for t in trains if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]]),
            'moving_trains': state_counts.get('moving', 0),
            'waiting_trains': state_counts.get('waiting', 0),
            'arrived_trains': self.dispatcher.total_arrivals,  # Use dispatcher's arrival counter
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
            'trains_by_state': state_counts,
            'trains_by_type': type_counts,
            'peak_active_trains': self._peak_active_trains,
            'scheduled_departures': len(self.dispatcher.scheduled_departures),
            'completed_trips': len(self.dispatcher.completed_trips),
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the simulation."""
        avg_step_time = np.mean(self._step_times) if self._step_times else 0
        return {
            'average_step_time_ms': avg_step_time * 1000,
            'steps_per_second': 1 / avg_step_time if avg_step_time > 0 else 0,
            'total_agents': len(self.schedule.agents),
            'cache_size': len(self._path_cache),
            'events_logged': len(self.events),
        }
    
    def export_events(self, filepath: str = None) -> List[Dict]:
        """Export simulation events as list of dictionaries."""
        events_data = [
            {
                'step': e.step,
                'type': e.event_type,
                'description': e.description,
                'data': e.data,
                'timestamp': e.timestamp
            }
            for e in self.events
        ]
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2)
        
        return events_data
    
    def reset(self):
        """Reset the simulation to initial state."""
        # Clear caches
        self._path_cache.clear()
        self.events.clear()
        self._step_times.clear()
        self._peak_active_trains = 0
        
        # Reset weather
        self.weather_condition = WeatherCondition.CLEAR
        self._weather_duration = 0
        
        # Note: Full reset requires creating new model instance
        self._log_event("reset", "Simulation reset requested")

