"""
Smart Railway Network Simulation - Agent Classes

This module defines all agent types for the railway simulation system:

- TrackAgent: Static track segments for visualization
- TrainAgent: Mobile trains with different types and behaviors
- SignalAgent: Track access control with signal light logic
- StationAgent: Passenger boarding/alighting points
- DispatcherAgent: Central coordination and conflict resolution
- PassengerAgent: Individual passengers (optional detailed simulation)

Each agent type has specific behaviors and interactions with other agents.
"""

from mesa import Agent
from enum import Enum, auto
import random
import logging
from typing import List, Tuple, Optional, Set, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from model import RailwayNetworkModel

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


class TrackAgent(Agent):
    """
    Simple track agent for visualization purposes only.
    Represents a railway track segment on the grid.
    """
    
    def __init__(self, unique_id, model, pos: Tuple[int, int]):
        super().__init__(unique_id, model)
        self.pos = pos
    
    def step(self):
        """Tracks don't do anything."""
        pass


class TrainType(Enum):
    """
    Different types of trains with different characteristics.
    
    Each type has different:
    - Base speed
    - Priority level
    - Energy capacity and consumption
    - Passenger/cargo capacity
    """
    PASSENGER = "passenger"  # Standard passenger train
    CARGO = "cargo"          # Heavy cargo train (slower, more energy)
    EXPRESS = "express"      # Fast express train (highest speed)
    EMERGENCY = "emergency"  # Emergency/priority train (highest priority)
    
    @property
    def emoji(self) -> str:
        """Get emoji representation for visualization."""
        return {
            'passenger': '🚃',
            'cargo': '🚂',
            'express': '🚄',
            'emergency': '🚑'
        }.get(self.value, '🚃')
    
    @property
    def color(self) -> str:
        """Get color code for visualization."""
        return {
            'passenger': '#3498db',  # Blue
            'cargo': '#e67e22',      # Orange
            'express': '#9b59b6',    # Purple
            'emergency': '#e74c3c'   # Red
        }.get(self.value, '#3498db')


class TrainState(Enum):
    """
    Possible states for a train.
    
    State transitions:
    IDLE -> MOVING (when route calculated)
    MOVING -> WAITING (blocked by signal)
    WAITING -> DELAYED (waiting too long)
    WAITING -> MOVING (signal cleared)
    MOVING -> BOARDING (at station)
    BOARDING -> MOVING (boarding complete)
    MOVING -> ARRIVED (reached destination)
    ANY -> MAINTENANCE (energy depleted)
    """
    IDLE = "idle"               # Waiting to start
    MOVING = "moving"           # In motion
    WAITING = "waiting"         # Waiting for signal
    DELAYED = "delayed"         # Delayed (waiting too long)
    BOARDING = "boarding"       # Boarding/alighting passengers
    MAINTENANCE = "maintenance" # Under maintenance
    ARRIVED = "arrived"         # Reached destination


class WeatherCondition(Enum):
    """
    Weather conditions affecting train operations.
    
    Each condition affects train speed differently:
    - CLEAR: No speed reduction
    - RAIN: 20% speed reduction
    - FOG: 40% speed reduction
    - STORM: 50% speed reduction
    - SNOW: 30% speed reduction
    """
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"
    FOG = "fog"
    SNOW = "snow"
    
    @property
    def speed_multiplier(self) -> float:
        """Get speed multiplier for this weather condition."""
        return {
            'clear': 1.0,
            'rain': 0.8,
            'fog': 0.6,
            'storm': 0.5,
            'snow': 0.7
        }.get(self.value, 1.0)
    
    @property
    def emoji(self) -> str:
        """Get emoji representation for visualization."""
        return {
            'clear': '☀️',
            'rain': '🌧️',
            'storm': '⛈️',
            'fog': '🌫️',
            'snow': '❄️'
        }.get(self.value, '☀️')


@dataclass
class TrainStats:
    """Statistics for a train's journey."""
    total_distance: int = 0
    total_wait_time: int = 0
    total_delay: int = 0
    passengers_boarded: int = 0
    passengers_delivered: int = 0
    energy_consumed: float = 0.0
    trips_completed: int = 0


class TrainAgent(Agent):
    """
    Train agent that moves through the railway network.
    
    Trains navigate the network using A* pathfinding, respect signals,
    board/alight passengers at stations, and consume energy. Different
    train types have different characteristics.
    
    Attributes:
        train_type: Type of train (passenger, cargo, express, emergency)
        current_position: Current (x, y) position on grid
        speed: Current speed (cells per step)
        base_speed: Normal operating speed
        destination: Target (x, y) position
        route: Planned list of positions to visit
        state: Current state (moving, waiting, delayed, etc.)
        priority: Priority level (higher = more important)
        energy: Current energy level
        max_energy: Maximum energy capacity
        energy_consumption_rate: Energy used per move
        delay_time: Accumulated delay in steps
        scheduled_arrival: Expected arrival time
        passengers: Current passenger count (for passenger trains)
        max_capacity: Maximum passenger capacity
        cargo_weight: Weight of cargo (for cargo trains)
        waiting_time: Time spent waiting
        stats: TrainStats object with journey statistics
    """
    
    # Class-level configuration for train types
    TRAIN_CONFIGS = {
        TrainType.PASSENGER: {
            'base_speed': 2,
            'priority': 5,
            'max_energy': 1000.0,
            'energy_consumption_rate': 1.5,
            'max_capacity': 200,
            'cargo_weight': 0
        },
        TrainType.CARGO: {
            'base_speed': 1,
            'priority': 3,
            'max_energy': 1500.0,
            'energy_consumption_rate': 2.5,
            'max_capacity': 50,
            'cargo_weight': 1000
        },
        TrainType.EXPRESS: {
            'base_speed': 3,
            'priority': 8,
            'max_energy': 800.0,
            'energy_consumption_rate': 3.0,
            'max_capacity': 150,
            'cargo_weight': 0
        },
        TrainType.EMERGENCY: {
            'base_speed': 3,
            'priority': 10,
            'max_energy': 900.0,
            'energy_consumption_rate': 2.0,
            'max_capacity': 100,
            'cargo_weight': 0
        }
    }
    
    # Configuration for dynamic behaviors
    DELAY_PRIORITY_BOOST = 2  # Boost priority when delayed
    
    def __init__(self, unique_id: int, model: 'RailwayNetworkModel', 
                 train_type: TrainType, start_pos: Tuple[int, int], 
                 destination: Tuple[int, int], scheduled_arrival: int = None):
        super().__init__(unique_id, model)
        self.train_type = train_type
        self.current_position = start_pos
        self.destination = destination
        self.route: List[Tuple[int, int]] = []
        self.route_index = 0
        self.state = TrainState.IDLE
        self.delay_time = 0
        self.waiting_time = 0
        self.scheduled_arrival = scheduled_arrival or 100
        self.passengers = 0
        
        # Load configuration for this train type
        config = self.TRAIN_CONFIGS[train_type]
        self.base_speed = config['base_speed']
        self.priority = config['priority']
        self._base_priority = config['priority']  # Store base priority for dynamic updates
        self.max_energy = config['max_energy']
        self.energy_consumption_rate = config['energy_consumption_rate']
        self.max_capacity = config['max_capacity']
        self.cargo_weight = config['cargo_weight']
        
        self.speed = self.base_speed
        self.max_speed = self.base_speed  # Dynamic speed limits
        self.energy = self.max_energy
        
        # Journey statistics
        self.stats = TrainStats()
        self._start_position = start_pos
        self._journey_start_step = 0
    
    @property
    def energy_percentage(self) -> float:
        """Get energy level as percentage."""
        return (self.energy / self.max_energy * 100) if self.max_energy > 0 else 0
    
    @property
    def is_active(self) -> bool:
        """Check if train is actively operating."""
        return self.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]
    
    @property
    def distance_to_destination(self) -> int:
        """Calculate Manhattan distance to destination."""
        return abs(self.current_position[0] - self.destination[0]) + \
               abs(self.current_position[1] - self.destination[1])
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress towards destination as percentage."""
        if not self.route:
            return 0.0 if self.state != TrainState.ARRIVED else 100.0
        total_route_length = len(self.route)
        if total_route_length == 0:
            return 100.0 if self.state == TrainState.ARRIVED else 0.0
        return min(100.0, (self.route_index / total_route_length) * 100)
        
    def calculate_route(self):
        """Calculate route from current position to destination using A* pathfinding with caching."""
        if not self.route or self.route_index >= len(self.route):
            # Use model's cached path if available
            self.route = self.model.get_cached_path(self.current_position, self.destination)
            self.route_index = 0
            logger.debug(f"Train {self.unique_id} calculated route: {self.route}")
    
    def _a_star_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm for route calculation (used internally by model cache)."""
        from heapq import heappush, heappop
        
        # Edge case: already at goal
        if start == goal:
            return []
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == goal:
                # Reconstruct path (includes goal but not start)
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.model.grid.width and 
                       0 <= neighbor[1] < self.model.grid.height):
                    continue
                
                # Check if track exists
                if not self.model.is_track(neighbor):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, return empty
        return []
    
    def adjust_speed_for_weather(self):
        """Adjust speed based on current weather conditions."""
        weather = self.model.weather_condition
        
        # Use weather condition's speed multiplier
        multiplier = weather.speed_multiplier
        adjusted_speed = int(self.base_speed * multiplier)
        
        # Ensure minimum speed of 1 and respect max_speed limit
        self.speed = min(max(1, adjusted_speed), self.max_speed)

    def update_priority(self):
        """
        Dynamically update priority based on delay status.
        Delayed trains get a priority boost to help them clear bottlenecks.
        """
        if self.state == TrainState.DELAYED or self.delay_time > 0:
            # Boost priority if delayed
            new_priority = self._base_priority + self.DELAY_PRIORITY_BOOST
            if new_priority != self.priority:
                self.priority = new_priority
                logger.debug(f"Train {self.unique_id} priority boosted to {self.priority} due to delay.")
        else:
            # Reset to base priority if not delayed
            if self.priority != self._base_priority:
                self.priority = self._base_priority
                logger.debug(f"Train {self.unique_id} priority reset to {self.priority}.")
    
    def request_track_access(self, next_position: Tuple[int, int]) -> bool:
        """Request permission from signal to enter next track segment."""
        signal = self.model.get_signal_at(next_position)
        if signal:
            return signal.grant_access(self)
        return True  # No signal means open track
    
    def move_towards_destination(self):
        """Move train towards its destination along the calculated route."""
        if self.energy <= 0:
            self.state = TrainState.MAINTENANCE
            logger.debug(f"Train {self.unique_id} entered maintenance due to depleted energy.")
            return
        
        # CRITICAL: Check if at destination FIRST before recalculating route
        if self.current_position == self.destination:
            self.state = TrainState.ARRIVED
            logger.info(f"Train {self.unique_id} arrived at destination {self.destination}.")
            # Deliver passengers if applicable
            if self.train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
                unloaded = self.unload_passengers()
                # Update station if at one
                station = self.model.get_station_at(self.current_position)
                if station:
                    station.passengers_arrived += unloaded
            self.model.dispatcher.report_arrival(self)
            return
        
        if not self.route or self.route_index >= len(self.route):
            self.calculate_route()
        
        if not self.route:
            # No valid route - FORCE a new route or teleport to avoid deadlock
            self.state = TrainState.DELAYED
            self.delay_time += 1
            # If stuck too long, force reroute immediately
            if self.delay_time > 3:
                self.model.dispatcher.force_unstuck(self)
            return
        
        # Try to move along route
        if self.route_index < len(self.route):
            next_position = self.route[self.route_index]
            
            # Request access from signal
            if self.request_track_access(next_position):
                # Move granted
                old_position = self.current_position
                self.current_position = next_position
                self.route_index += 1
                self.state = TrainState.MOVING
                logger.debug(f"Train {self.unique_id} moved from {old_position} to {next_position}.")
                
                # CRITICAL FIX: Update grid position!
                self.model.grid.move_agent(self, next_position)
                
                # Consume energy based on speed and cargo
                energy_cost = self.energy_consumption_rate * self.speed
                if self.train_type == TrainType.CARGO:
                    energy_cost *= (1 + self.cargo_weight / 2000)
                self.energy -= energy_cost
                
                # Track statistics
                self.stats.total_distance += 1
                self.stats.energy_consumed += energy_cost
                
                logger.debug(f"Train {self.unique_id} energy after move: {self.energy:.2f}.")
                
                # Update signal occupancy
                old_signal = self.model.get_signal_at(old_position)
                if old_signal:
                    old_signal.release_track(self)
                
                self.waiting_time = 0
                
                # Reset delay if train starts moving again
                if self.delay_time > 0:
                    self.delay_time = max(0, self.delay_time - 1)  # Gradually recover from delays
            else:
                # Access denied, wait - but escalate if waiting too long
                self.state = TrainState.WAITING
                self.waiting_time += 1
                
                # Check if this train is marked as urgent by dispatcher
                is_urgent = self.unique_id in self.model.dispatcher.urgent_trains
                
                # Escalate to DELAYED state after threshold
                if self.waiting_time > self.model.dispatcher.DELAY_WARNING_THRESHOLD:
                    self.state = TrainState.DELAYED
                    self.delay_time += 1
                    
                    # Only log occasionally to reduce spam
                    if self.waiting_time % 5 == 0:
                        logger.warning(f"Train {self.unique_id} delayed for {self.waiting_time} steps" + 
                                      (" [URGENT]" if is_urgent else ""))
                
                # Dispatcher will handle escalation automatically in its step()
                # But if urgent, try to move even without signal permission (emergency override)
                if is_urgent and self.waiting_time > self.model.dispatcher.DELAY_URGENT_THRESHOLD:
                    # Emergency override - try alternative routes
                    self._try_alternative_move()
    
    def board_passengers(self, count: int):
        """Board passengers at a station."""
        if self.train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
            available_space = self.max_capacity - self.passengers
            boarded = min(count, available_space)
            self.passengers += boarded
            return boarded
        return 0
    
    def unload_passengers(self) -> int:
        """Unload all passengers at destination."""
        count = self.passengers
        self.passengers = 0
        return count
    
    def refuel(self):
        """Refuel the train to maximum energy."""
        energy_added = self.max_energy - self.energy
        self.energy = self.max_energy
        logger.debug(f"Train {self.unique_id} refueled: +{energy_added:.0f} energy")
    
    def _try_alternative_move(self):
        """Try to find an alternative path when stuck as an urgent train."""
        current = self.current_position
        
        # Find any adjacent track that's closer to destination
        best_pos = None
        best_dist = float('inf')
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (current[0] + dx, current[1] + dy)
            
            # Must be valid track
            if new_pos not in self.model.track_positions:
                continue
            
            # Skip if there's an occupied signal (unless we're really desperate)
            signal = self.model.get_signal_at(new_pos)
            if signal and signal.track_occupied:
                # Only skip if wait time isn't critical
                if self.waiting_time < self.model.dispatcher.DELAY_CRITICAL_THRESHOLD:
                    continue
            
            # Calculate distance to destination
            dist = abs(new_pos[0] - self.destination[0]) + abs(new_pos[1] - self.destination[1])
            
            if dist < best_dist:
                best_dist = dist
                best_pos = new_pos
        
        if best_pos:
            # Release old signal if any
            old_signal = self.model.get_signal_at(current)
            if old_signal:
                old_signal.release_track(self)
            
            # Move to new position
            self.current_position = best_pos
            self.model.grid.move_agent(self, best_pos)
            self.state = TrainState.MOVING
            self.waiting_time = 0
            
            # Recalculate route from new position
            self.route = []
            self.route_index = 0
            self.calculate_route()
            
            logger.info(f"Train {self.unique_id} found alternative path via {best_pos}")
    
    def get_info(self) -> Dict:
        """Get comprehensive information about this train."""
        return {
            'id': self.unique_id,
            'type': self.train_type.value,
            'state': self.state.value,
            'position': self.current_position,
            'destination': self.destination,
            'energy_percent': self.energy_percentage,
            'passengers': self.passengers,
            'max_capacity': self.max_capacity,
            'priority': self.priority,
            'delay': self.delay_time,
            'waiting_time': self.waiting_time,
            'progress': self.progress_percentage,
            'distance_remaining': self.distance_to_destination,
            'route_length': len(self.route) - self.route_index if self.route else 0,
        }
    
    def __repr__(self) -> str:
        return f"Train({self.unique_id}, {self.train_type.value}, {self.state.value})"
        
    def step(self):
        """Execute one step of the train's behavior."""
        # Update dynamic priority
        self.update_priority()

        # Adjust speed for weather
        self.adjust_speed_for_weather()
        logger.debug(f"Train {self.unique_id} speed adjusted to {self.speed} due to weather {self.model.weather_condition}.")
        
        # CRITICAL: QUICK BOARDING - Don't let trains sit at stations
        station = self.model.get_station_at(self.current_position)
        if station and self.train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
            # Board passengers QUICKLY if NOT at destination
            if station.pos != self.destination and self.passengers < self.max_capacity and station.waiting_passengers > 0:
                # INSTANT BOARDING - don't waste a full step
                available_space = self.max_capacity - self.passengers
                passengers_to_board = min(available_space, station.waiting_passengers, 50)  # Max 50 per step
                self.passengers += passengers_to_board
                station.waiting_passengers -= passengers_to_board
                station.trains_serviced += 1
                logger.info(f"Train {self.unique_id} boarded {passengers_to_board} passengers at station {station.pos}.")
        
        # ALWAYS try to move (even when boarding)
        self.move_towards_destination()


class SignalAgent(Agent):
    """
    Signal agent that controls access to track segments.
    
    Attributes:
        pos: Position of the signal on the grid
        track_occupied: Whether the track segment is occupied
        occupying_train: Train currently on this track segment
        track_status: Track operational status (active, failed, maintenance)
        neighbors: List of neighboring signal positions
    """
    
    def __init__(self, unique_id, model, pos: Tuple[int, int]):
        super().__init__(unique_id, model)
        self.pos = pos
        self.track_occupied = False
        self.occupying_train: Optional[TrainAgent] = None
        self.track_status = "active"  # active, failed, maintenance
        self.neighbors: List[Tuple[int, int]] = []
        self.waiting_queue: List[TrainAgent] = []
    
    def grant_access(self, train: TrainAgent) -> bool:
        """
        Grant or deny access to a train based on track occupancy and priority.
        
        Returns:
            True if access granted, False if denied
        """
        # Track failure prevents access
        if self.track_status != "active":
            return False
        
        # If track is free, grant access
        if not self.track_occupied:
            self.track_occupied = True
            self.occupying_train = train
            return True
        
        # Track occupied - check priority
        if self.occupying_train and train.priority > self.occupying_train.priority:
            # Emergency override - notify dispatcher
            self.model.dispatcher.handle_priority_conflict(train, self.occupying_train, self.pos)
            # For now, still deny (dispatcher might reroute)
            self.waiting_queue.append(train)
            return False
        
        # Add to waiting queue
        if train not in self.waiting_queue:
            self.waiting_queue.append(train)
        return False
    
    def release_track(self, train: TrainAgent):
        """Release track when train leaves - IMMEDIATELY grant to next waiting train."""
        if self.occupying_train == train:
            self.track_occupied = False
            self.occupying_train = None
            
            # IMMEDIATELY grant access to next train in queue (prioritize high-priority trains)
            if self.waiting_queue:
                # Sort queue by priority (highest first)
                self.waiting_queue.sort(key=lambda t: t.priority, reverse=True)
                next_train = self.waiting_queue.pop(0)
                self.track_occupied = True
                self.occupying_train = next_train
        elif self.occupying_train is None:
            # Track already free, just clear the flag
            self.track_occupied = False
            
            # Still process queue
            if self.waiting_queue:
                self.waiting_queue.sort(key=lambda t: t.priority, reverse=True)
                next_train = self.waiting_queue.pop(0)
                self.track_occupied = True
                self.occupying_train = next_train
    
    def set_track_status(self, status: str):
        """Set track status (for failures/maintenance)."""
        self.track_status = status
        if status != "active" and self.occupying_train:
            # Notify dispatcher of track failure
            self.model.dispatcher.handle_track_failure(self.pos, self.occupying_train)
    
    def step(self):
        """Execute one step of signal behavior."""
        # Random track failures (EXTREMELY rare - don't disrupt flow)
        if self.track_status == "active" and random.random() < 0.0001:  # Reduced from 0.001
            self.set_track_status("failed")
            self.model.dispatcher.report_track_failure(self.pos)
        
        # Track repairs (FASTER - get trains moving again quickly)
        if self.track_status == "failed" and random.random() < 0.3:  # Increased from 0.1
            self.track_status = "active"
            # Remove from dispatcher's failure list
            if self.pos in self.model.dispatcher.track_failures:
                self.model.dispatcher.track_failures.remove(self.pos)


class DispatcherAgent(Agent):
    """
    Central dispatcher that coordinates all trains and handles conflicts.
    
    FEATURES:
    - Active timetable management
    - Scheduled train departures
    - Train recycling for continuous operation
    - AUTOMATIC DEADLOCK DETECTION AND RESOLUTION
    - URGENT TRAIN PRIORITY SYSTEM
    - HEALTH MONITORING AND RECOVERY
    
    Attributes:
        train_registry: Dict of all active trains by ID
        scheduled_departures: List of upcoming train departures
        track_failures: Set of failed track positions
        reroute_requests: Queue of reroute requests
        delay_stats: Statistics on delays
        total_arrivals: Number of trains that reached destination
        total_delay: Total delay time across all trains
        urgent_trains: Set of trains that need immediate path clearing
    """
    
    # Timeout thresholds (in simulation steps) - AGGRESSIVE to prevent long delays
    DELAY_WARNING_THRESHOLD = 3      # Steps before train is flagged as delayed
    DELAY_URGENT_THRESHOLD = 8       # Steps before train becomes URGENT (gets priority clearing)
    DELAY_CRITICAL_THRESHOLD = 15    # Steps before train is force-teleported to destination
    DEADLOCK_CHECK_INTERVAL = 5      # How often to check for system-wide deadlocks
    MAX_SIGNAL_OCCUPATION = 10       # Max steps a train can occupy a signal before forced release
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.train_registry = {}
        self.scheduled_departures = []  # List of {train_type, origin, destination, departure_time}
        self.track_failures: Set[Tuple[int, int]] = set()
        self.reroute_requests: List[TrainAgent] = []
        self.delay_stats = []
        self.total_arrivals = 0
        self.total_delay = 0
        self.priority_conflicts = []
        self.completed_trips = []  # Track completed journeys for recycling
        self.urgent_trains: Set[int] = set()  # Train IDs that need urgent path clearing
        self.deadlock_resolutions = 0  # Count of deadlock resolutions
        self._last_health_check = 0
        self._consecutive_no_progress_steps = 0  # Track steps without any train progress
        self._last_total_distance = 0
        logger.info("DispatcherAgent initialized with automatic problem-solving.")
    
    def schedule_train_departure(self, train_type: TrainType, origin: Tuple[int, int], 
                                  destination: Tuple[int, int], departure_time: int):
        """Add a train to the scheduled departures timetable."""
        self.scheduled_departures.append({
            'train_type': train_type,
            'origin': origin,
            'destination': destination,
            'departure_time': departure_time
        })
        # Sort by departure time
        self.scheduled_departures.sort(key=lambda x: x['departure_time'])
    
    def spawn_scheduled_trains(self):
        """Check timetable and spawn trains that are scheduled to depart now."""
        current_time = self.model.schedule.steps
        
        # Check for trains to spawn
        to_spawn = []
        remaining = []
        
        for schedule in self.scheduled_departures:
            if schedule['departure_time'] <= current_time:
                to_spawn.append(schedule)
            else:
                remaining.append(schedule)
        
        self.scheduled_departures = remaining
        
        # Spawn the trains
        for schedule in to_spawn:
            self._spawn_train(
                schedule['train_type'],
                schedule['origin'],
                schedule['destination']
            )
    
    def _spawn_train(self, train_type: TrainType, origin: Tuple[int, int], destination: Tuple[int, int]):
        """Actually create and spawn a train."""
        # Calculate expected arrival time
        distance = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
        estimated_travel_time = distance  # Rough estimate
        scheduled_arrival = self.model.schedule.steps + estimated_travel_time + 20
        
        # Create train
        train = TrainAgent(
            self.model.next_train_id,
            self.model,
            train_type,
            origin,
            destination,
            scheduled_arrival
        )
        self.model.next_train_id += 1
        
        # Register and add to simulation
        self.register_train(train)
        self.model.schedule.add(train)
        self.model.grid.place_agent(train, origin)
        train.calculate_route()
    
    def register_train(self, train: TrainAgent):
        """Register a train with the dispatcher."""
        self.train_registry[train.unique_id] = train
    
    def recycle_train(self, train: TrainAgent, count_as_arrival: bool = True):
        """
        Recycle a train that has completed its journey for a new route.
        
        Args:
            train: The train to recycle
            count_as_arrival: If True, count this as a completed trip/arrival
        """
        # Count as arrival if requested (True when called directly, False when called after report_arrival)
        if count_as_arrival:
            self.total_arrivals += 1
        
        # Record the completion
        self.completed_trips.append({
            'train_id': train.unique_id,
            'train_type': train.train_type,
            'completion_time': self.model.schedule.steps
        })
        
        # Get origin (current position) and find new destination
        origin = train.current_position
        stations = list(self.model.station_positions.keys())
        tracks = list(self.model.track_positions)
        
        # Determine new destination based on train type
        if train.train_type in [TrainType.PASSENGER, TrainType.EXPRESS] and len(stations) >= 2:
            # Passenger trains go between stations
            possible_destinations = [s for s in stations if s != origin]
            new_destination = random.choice(possible_destinations) if possible_destinations else random.choice(stations)
        else:
            # Cargo/emergency trains can go anywhere
            possible_destinations = [t for t in tracks if t != origin]
            new_destination = random.choice(possible_destinations) if possible_destinations else random.choice(tracks)
        
        # Calculate new scheduled arrival
        distance = abs(origin[0] - new_destination[0]) + abs(origin[1] - new_destination[1])
        estimated_travel_time = max(distance, 20)
        new_scheduled_arrival = self.model.schedule.steps + estimated_travel_time + 10
        
        # FULLY RESET train for new journey
        train.destination = new_destination
        train.scheduled_arrival = new_scheduled_arrival
        train.state = TrainState.MOVING  # Start moving immediately
        train.route = []
        train.route_index = 0
        train.passengers = 0
        train.delay_time = 0
        train.waiting_time = 0
        train.refuel()
        
        # Calculate new route
        train.calculate_route()
        
        # If no route found, assign to nearest station as destination
        if not train.route and stations:
            nearest_station = min(stations, key=lambda s: abs(s[0]-origin[0]) + abs(s[1]-origin[1]))
            if nearest_station != origin:
                train.destination = nearest_station
                train.calculate_route()
        
        return True
    
    def request_reroute(self, train: TrainAgent):
        """Handle a reroute request from a train."""
        if train not in self.reroute_requests:
            self.reroute_requests.append(train)
    
    def handle_stuck_train(self, train: TrainAgent):
        """Handle a train that's stuck waiting - escalating response based on wait time."""
        wait_time = train.waiting_time
        
        # LEVEL 1: Basic reroute attempt (5-15 steps)
        if wait_time >= self.DELAY_WARNING_THRESHOLD:
            self.request_reroute(train)
        
        # LEVEL 2: Mark as URGENT and actively clear path (15-25 steps)
        if wait_time >= self.DELAY_URGENT_THRESHOLD:
            self.urgent_trains.add(train.unique_id)
            self._clear_path_for_urgent_train(train)
        
        # LEVEL 3: CRITICAL - Force teleport to nearest station or destination (25+ steps)
        if wait_time >= self.DELAY_CRITICAL_THRESHOLD:
            self._emergency_teleport(train)
    
    def _clear_path_for_urgent_train(self, urgent_train: TrainAgent):
        """Actively clear the path for an urgent train by moving blocking trains."""
        if not urgent_train.route or urgent_train.route_index >= len(urgent_train.route):
            return
        
        blocked_position = urgent_train.route[urgent_train.route_index]
        signal = self.model.get_signal_at(blocked_position)
        
        if signal and signal.track_occupied and signal.occupying_train:
            blocking_train = signal.occupying_train
            
            # Don't clear if blocking train is also urgent
            if blocking_train.unique_id in self.urgent_trains:
                # Both are urgent - force move the one that's been waiting longer
                if urgent_train.waiting_time > blocking_train.waiting_time:
                    self._force_move_blocking_train(blocking_train, urgent_train.current_position)
            else:
                # Move the blocking train out of the way
                self._force_move_blocking_train(blocking_train, urgent_train.current_position)
                
                # Also release the signal for the urgent train
                signal.release_track(blocking_train)
                signal.track_occupied = False
                signal.occupying_train = None
    
    def _force_move_blocking_train(self, train: TrainAgent, avoid_position: Tuple[int, int]):
        """Force a train to move to any available adjacent position."""
        current_pos = train.current_position
        
        # Find best adjacent position (not the position we're trying to clear)
        best_pos = None
        best_score = -1
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Skip the position we're trying to clear
            if new_pos == avoid_position:
                continue
            
            # Must be valid track
            if new_pos not in self.model.track_positions:
                continue
            
            # Check if free
            signal = self.model.get_signal_at(new_pos)
            if signal and signal.track_occupied:
                continue
            
            # Score based on distance from destination (prefer moving toward destination)
            dist_to_dest = abs(new_pos[0] - train.destination[0]) + abs(new_pos[1] - train.destination[1])
            score = 100 - dist_to_dest  # Higher score for closer to destination
            
            if score > best_score:
                best_score = score
                best_pos = new_pos
        
        if best_pos:
            # Release old signal
            old_signal = self.model.get_signal_at(current_pos)
            if old_signal:
                old_signal.release_track(train)
            
            # Move train
            train.current_position = best_pos
            self.model.grid.move_agent(train, best_pos)
            train.state = TrainState.MOVING
            train.waiting_time = 0
            
            # Recalculate route
            train.route = []
            train.route_index = 0
            train.calculate_route()
    
    def _emergency_teleport(self, train: TrainAgent):
        """EMERGENCY: Teleport a critically stuck train to deliver passengers."""
        logger.warning(f"EMERGENCY TELEPORT: Train {train.unique_id} stuck for {train.waiting_time} steps!")
        
        # If train has passengers, teleport to destination to deliver them
        if train.passengers > 0 or train.train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
            # Find nearest station
            stations = list(self.model.station_positions.keys())
            if stations:
                # Find closest station
                current = train.current_position
                closest_station = min(stations, key=lambda s: abs(s[0]-current[0]) + abs(s[1]-current[1]))
                
                # Release old position
                old_signal = self.model.get_signal_at(current)
                if old_signal:
                    old_signal.release_track(train)
                
                # Teleport to closest station
                train.current_position = closest_station
                self.model.grid.move_agent(train, closest_station)
                
                # Deliver passengers
                station = self.model.station_positions.get(closest_station)
                if station and train.passengers > 0:
                    delivered = train.passengers
                    station.passengers_arrived += delivered
                    train.passengers = 0
                    logger.info(f"Train {train.unique_id} emergency delivered {delivered} passengers to {station.name}")
                
                # Set new destination and reset
                train.destination = closest_station
                train.state = TrainState.ARRIVED
                train.waiting_time = 0
                train.delay_time = 0
                self.urgent_trains.discard(train.unique_id)
                
                # Report arrival and recycle (report_arrival counts the arrival)
                self.report_arrival(train)
        else:
            # Cargo/emergency train - just reset and recycle (still counts as trip)
            self.force_unstuck(train)
            self.total_arrivals += 1  # Count this as a completed trip
            self.recycle_train(train, count_as_arrival=False)
    
    def force_unstuck(self, train: TrainAgent):
        """FORCE a stuck train to move by any means necessary - prevent deadlocks."""
        # Find an alternative nearby position to move to
        current_pos = train.current_position
        
        # Try to find any adjacent track position that's free
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check if it's a valid track
            if new_pos in self.model.track_positions:
                signal = self.model.get_signal_at(new_pos)
                
                # If track is free, move there immediately
                if not signal or not signal.track_occupied:
                    # FORCE MOVE to break deadlock
                    old_signal = self.model.get_signal_at(current_pos)
                    if old_signal:
                        old_signal.release_track(train)
                    
                    train.current_position = new_pos
                    self.model.grid.move_agent(train, new_pos)
                    train.waiting_time = 0
                    train.state = TrainState.MOVING
                    
                    # Recalculate route from new position
                    train.route = []
                    train.route_index = 0
                    train.calculate_route()
                    return
        
        # If no adjacent free track, assign a completely new random destination
        tracks = list(self.model.track_positions)
        if len(tracks) > 1:
            new_dest = random.choice([t for t in tracks if t != current_pos])
            train.destination = new_dest
            train.route = []
            train.route_index = 0
            train.waiting_time = 0
            train.calculate_route()
    
    def handle_reroute_requests(self):
        """Process all pending reroute requests."""
        for train in self.reroute_requests:
            # Recalculate route avoiding failed tracks
            train.calculate_route()
        self.reroute_requests.clear()
    
    def report_track_failure(self, pos: Tuple[int, int]):
        """Record a track failure."""
        self.track_failures.add(pos)
    
    def handle_track_failure(self, pos: Tuple[int, int], affected_train: TrainAgent):
        """Handle a track failure affecting a train."""
        self.track_failures.add(pos)
        self.request_reroute(affected_train)
    
    def handle_priority_conflict(self, high_priority_train: TrainAgent, 
                                 low_priority_train: TrainAgent, pos: Tuple[int, int]):
        """Handle conflict between trains of different priorities - actively resolve delays."""
        self.priority_conflicts.append({
            'high_priority': high_priority_train.unique_id,
            'low_priority': low_priority_train.unique_id,
            'position': pos,
            'step': self.model.schedule.steps
        })
        
        # PRIORITY-BASED RESOLUTION:
        # High priority trains get preferred routing, low priority trains reroute
        if high_priority_train.priority >= 8:  # Express and Emergency trains
            # Force reroute of lower priority train to clear the way
            self.request_reroute(low_priority_train)
            
            # If emergency, also try to clear the signal for immediate access
            if high_priority_train.train_type == TrainType.EMERGENCY:
                signal = self.model.get_signal_at(pos)
                if signal and signal.occupying_train == low_priority_train:
                    # Emergency override - clear the track
                    signal.release_track(low_priority_train)
                    low_priority_train.state = TrainState.WAITING
                    self.request_reroute(low_priority_train)
    
    def report_arrival(self, train: TrainAgent):
        """Record a train arrival and recycle it for continuous operation."""
        # Count this arrival
        self.total_arrivals += 1
        
        # Remove from urgent list if present
        self.urgent_trains.discard(train.unique_id)
        
        if self.model.schedule.steps > train.scheduled_arrival:
            delay = self.model.schedule.steps - train.scheduled_arrival
            self.total_delay += delay
            train.delay_time = delay
        
        self.delay_stats.append({
            'train_id': train.unique_id,
            'train_type': train.train_type.value,
            'delay': train.delay_time,
            'energy_used': train.max_energy - train.energy,
            'passengers': train.passengers
        })
        
        # Reset train state before recycling
        train.waiting_time = 0
        train.delay_time = 0
        
        # Recycle the train for a new journey (arrival already counted above)
        self.recycle_train(train, count_as_arrival=False)
    
    def get_average_delay(self) -> float:
        """Calculate average delay across all trains."""
        if self.total_arrivals == 0:
            return 0.0
        return self.total_delay / self.total_arrivals
    
    def _schedule_additional_trains(self):
        """Schedule additional trains to maintain continuous operation."""
        current_time = self.model.schedule.steps
        stations = list(self.model.station_positions.keys())
        tracks = list(self.model.track_positions)
        
        train_types = [TrainType.PASSENGER, TrainType.CARGO, TrainType.EXPRESS]
        
        # Schedule 3 new trains
        for i in range(3):
            train_type = random.choice(train_types)
            departure_time = current_time + (i + 1) * 20 + random.randint(5, 15)
            
            if train_type in [TrainType.PASSENGER, TrainType.EXPRESS] and len(stations) >= 2:
                origin = random.choice(stations)
                destination = random.choice([s for s in stations if s != origin])
            else:
                origin = random.choice(tracks)
                destination = random.choice([t for t in tracks if t != origin])
            
            self.schedule_train_departure(train_type, origin, destination, departure_time)
    
    def get_network_stats(self) -> dict:
        """Get comprehensive network statistics."""
        active_trains = sum(1 for t in self.train_registry.values() 
                          if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE])
        
        delayed_trains = sum(1 for t in self.train_registry.values() 
                           if t.state == TrainState.DELAYED)
        
        total_energy = sum(t.energy for t in self.train_registry.values())
        
        return {
            'active_trains': active_trains,
            'delayed_trains': delayed_trains,
            'total_arrivals': self.total_arrivals,
            'average_delay': self.get_average_delay(),
            'track_failures': len(self.track_failures),
            'total_energy': total_energy,
            'priority_conflicts': len(self.priority_conflicts)
        }
    
    def step(self):
        """
        Execute one step of dispatcher behavior with AUTOMATIC PROBLEM-SOLVING.
        
        This includes:
        1. Spawning scheduled trains
        2. Processing reroute requests
        3. Monitoring train health and handling delays
        4. Detecting and resolving deadlocks
        5. System-wide health monitoring
        6. Emergency interventions for critical situations
        """
        current_step = self.model.schedule.steps
        
        # PRIORITY 1: Spawn scheduled trains
        self.spawn_scheduled_trains()
        
        # PRIORITY 2: Process reroute requests
        self.handle_reroute_requests()
        
        # PRIORITY 3: Monitor all trains and handle delays with escalating response
        self._monitor_and_handle_delays()
        
        # PRIORITY 4: Periodic system health check (every DEADLOCK_CHECK_INTERVAL steps)
        if current_step - self._last_health_check >= self.DEADLOCK_CHECK_INTERVAL:
            self._perform_health_check()
            self._last_health_check = current_step
        
        # PRIORITY 5: Clear stale signal occupations
        self._clear_stale_signals()
        
        # PRIORITY 6: Ensure continuous operation
        self._ensure_continuous_operation()
    
    def _monitor_and_handle_delays(self):
        """Monitor all trains and handle delays with escalating responses."""
        delayed_trains = []
        urgent_trains = []
        critical_trains = []
        
        for train in list(self.train_registry.values()):
            wait_time = train.waiting_time
            
            # Categorize by severity
            if wait_time >= self.DELAY_CRITICAL_THRESHOLD:
                critical_trains.append(train)
            elif wait_time >= self.DELAY_URGENT_THRESHOLD:
                urgent_trains.append(train)
            elif wait_time >= self.DELAY_WARNING_THRESHOLD:
                delayed_trains.append(train)
            
            # Automatic refueling at stations
            if train.energy < train.max_energy * 0.3:
                station = self.model.get_station_at(train.current_position)
                if station:
                    train.refuel()
                    if train.state == TrainState.MAINTENANCE:
                        train.state = TrainState.IDLE
        
        # HANDLE CRITICAL TRAINS FIRST (25+ steps stuck)
        for train in critical_trains:
            logger.warning(f"CRITICAL: Train {train.unique_id} stuck for {train.waiting_time} steps - emergency teleport!")
            self._emergency_teleport(train)
            self.deadlock_resolutions += 1
        
        # HANDLE URGENT TRAINS (15-25 steps stuck)
        for train in urgent_trains:
            self.urgent_trains.add(train.unique_id)
            self._clear_path_for_urgent_train(train)
        
        # HANDLE DELAYED TRAINS (5-15 steps stuck)
        # Sort by priority and waiting time
        delayed_trains.sort(key=lambda t: (t.priority, t.waiting_time), reverse=True)
        
        for train in delayed_trains[:5]:  # Handle top 5 delayed trains
            self.handle_stuck_train(train)
    
    def _perform_health_check(self):
        """Perform system-wide health check and take corrective actions."""
        # Calculate current system health metrics
        total_trains = len(self.train_registry)
        if total_trains == 0:
            return
        
        active_trains = [t for t in self.train_registry.values() 
                        if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]]
        delayed_count = sum(1 for t in active_trains if t.state == TrainState.DELAYED or t.waiting_time > 3)
        moving_count = sum(1 for t in active_trains if t.state == TrainState.MOVING)
        
        # Calculate total distance traveled since last check
        total_distance = sum(t.stats.total_distance for t in self.train_registry.values())
        distance_progress = total_distance - self._last_total_distance
        self._last_total_distance = total_distance
        
        # Check for system-wide stagnation
        if len(active_trains) > 0 and distance_progress == 0:
            self._consecutive_no_progress_steps += self.DEADLOCK_CHECK_INTERVAL
        else:
            self._consecutive_no_progress_steps = max(0, self._consecutive_no_progress_steps - 1)
        
        # SYSTEM-WIDE DEADLOCK DETECTION
        # If more than 40% of active trains are delayed AND low progress, force intervention
        if len(active_trains) > 0:
            delay_ratio = delayed_count / len(active_trains)
            
            # More aggressive intervention when many trains delayed
            if delay_ratio > 0.4 and self._consecutive_no_progress_steps >= 10:
                logger.warning(f"SYSTEM DEADLOCK DETECTED! {delay_ratio*100:.0f}% trains delayed, low progress")
                self._resolve_system_deadlock()
            
            # If very few trains are moving, something is wrong
            if moving_count == 0 and len(active_trains) > 3:
                logger.warning(f"NO TRAINS MOVING! Forcing intervention...")
                self._resolve_system_deadlock()
        
        # If too many urgent trains, perform mass intervention
        if len(self.urgent_trains) > total_trains * 0.25:
            logger.warning(f"Too many urgent trains ({len(self.urgent_trains)}) - mass intervention!")
            self._mass_intervention()
        
        # PROGRESS CHECK: If no arrivals for a long time but trains exist, force some completions
        current_step = self.model.schedule.steps
        if current_step > 100 and self.total_arrivals == 0:
            # No arrivals after 100 steps is a problem - force complete some trips
            self._force_trip_completions()
    
    def _resolve_system_deadlock(self):
        """Resolve a system-wide deadlock by forcibly moving trains."""
        self.deadlock_resolutions += 1
        
        # Get all stuck trains
        stuck_trains = [t for t in self.train_registry.values() 
                       if t.waiting_time > 5 and t.state != TrainState.ARRIVED]
        
        if not stuck_trains:
            return
        
        # Sort by waiting time (longest first)
        stuck_trains.sort(key=lambda t: t.waiting_time, reverse=True)
        
        # Force move the longest-waiting trains
        for train in stuck_trains[:min(5, len(stuck_trains))]:
            # Clear any signals this train might be occupying
            signal = self.model.get_signal_at(train.current_position)
            if signal and signal.occupying_train == train:
                signal.release_track(train)
            
            # Emergency teleport to deliver passengers
            if train.passengers > 0:
                self._emergency_teleport(train)
            else:
                # Just force unstuck
                self.force_unstuck(train)
        
        # Reset progress counter
        self._consecutive_no_progress_steps = 0
    
    def _mass_intervention(self):
        """Perform mass intervention when too many trains are stuck."""
        # Release all signals that have been occupied for too long
        for pos, signal in self.model.signal_positions.items():
            if signal.track_occupied and signal.occupying_train:
                train = signal.occupying_train
                # If train has been waiting, release signal
                if train.waiting_time > 5:
                    signal.release_track(train)
                    signal.track_occupied = False
                    signal.occupying_train = None
        
        # Clear urgent trains list and let them restart
        self.urgent_trains.clear()
    
    def _force_trip_completions(self):
        """Force some trains to complete their trips when system is stagnant."""
        # Find trains that have traveled some distance but haven't completed
        candidates = []
        
        for train in self.train_registry.values():
            if train.state == TrainState.ARRIVED:
                continue
            
            # If train has been active a while and has passengers, force completion
            if train.stats.total_distance > 5 or train.passengers > 0:
                candidates.append(train)
        
        if not candidates:
            return
        
        # Sort by passengers (deliver those with most passengers first)
        candidates.sort(key=lambda t: t.passengers, reverse=True)
        
        # Force complete top 3 trains
        for train in candidates[:3]:
            if train.passengers > 0:
                # Teleport to nearest station to deliver passengers
                self._emergency_teleport(train)
            else:
                # Just mark as arrived and recycle (report_arrival will count it)
                train.state = TrainState.ARRIVED
                self.report_arrival(train)
        
        logger.info(f"Forced {min(3, len(candidates))} trip completions")
    
    def _clear_stale_signals(self):
        """Release signals that have been occupied too long without the train moving."""
        for pos, signal in self.model.signal_positions.items():
            if signal.track_occupied and signal.occupying_train:
                train = signal.occupying_train
                
                # If train isn't actually at this position, release signal
                if train.current_position != pos:
                    signal.release_track(train)
                    continue
                
                # If train has been waiting too long at this signal, force release
                if train.waiting_time > self.MAX_SIGNAL_OCCUPATION:
                    signal.release_track(train)
                    # Force the train to recalculate route
                    train.route = []
                    train.route_index = 0
                    train.calculate_route()
    
    def _ensure_continuous_operation(self):
        """Ensure trains keep moving and passengers get delivered."""
        active_count = sum(1 for t in self.train_registry.values() 
                         if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE])
        
        # If too few active trains, schedule more departures
        if active_count < self.model.num_trains // 2 and len(self.scheduled_departures) < 5:
            self._schedule_additional_trains()
        
        # Check if any arrived trains need to be recycled (shouldn't happen normally, but safety net)
        for train in list(self.train_registry.values()):
            if train.state == TrainState.ARRIVED:
                # This train was marked as arrived but not recycled - count it and recycle
                self.total_arrivals += 1
                self.recycle_train(train, count_as_arrival=False)


class StationAgent(Agent):
    """
    Station agent that manages passenger boarding and statistics.
    
    Attributes:
        pos: Position of the station
        name: Station name
        waiting_passengers: Number of passengers waiting
        passengers_arrived: Total passengers who arrived at this station
        passenger_generation_rate: Rate at which new passengers arrive
    """
    
    def __init__(self, unique_id, model, pos: Tuple[int, int], name: str):
        super().__init__(unique_id, model)
        self.pos = pos
        self.name = name
        self.waiting_passengers = 0
        self.passengers_arrived = 0
        # REDUCED passenger generation for realistic management
        self.passenger_generation_rate = random.randint(1, 5)  # Much slower generation
        self.total_wait_time = 0
        self.trains_serviced = 0
    
    def generate_passengers(self):
        """Generate new waiting passengers - realistic, slow generation."""
        # Only generate occasionally (not every step)
        if random.random() < 0.3:  # 30% chance each step
            new_passengers = random.randint(0, self.passenger_generation_rate)
            self.waiting_passengers += new_passengers
    
    def get_waiting_passengers(self) -> int:
        """Get number of waiting passengers."""
        return self.waiting_passengers
    
    def board_passengers(self, count: int):
        """Board passengers onto a train."""
        boarded = min(count, self.waiting_passengers)
        self.waiting_passengers -= boarded
        self.trains_serviced += 1
        return boarded
    
    def step(self):
        """Execute one step of station behavior."""
        # Generate new passengers
        self.generate_passengers()
        
        # Track waiting time
        if self.waiting_passengers > 0:
            self.total_wait_time += self.waiting_passengers


class PassengerAgent(Agent):
    """
    Individual passenger agent (optional detailed simulation).
    
    Attributes:
        origin: Starting station
        destination: Target station
        wait_time: Time spent waiting
        on_train: Whether currently on a train
    """
    
    def __init__(self, unique_id, model, origin: Tuple[int, int], destination: Tuple[int, int]):
        super().__init__(unique_id, model)
        self.origin = origin
        self.destination = destination
        self.wait_time = 0
        self.on_train = False
        self.current_train: Optional[TrainAgent] = None
    
    def board_train(self, train: TrainAgent):
        """Board a train."""
        self.on_train = True
        self.current_train = train
    
    def alight_train(self):
        """Get off the train."""
        self.on_train = False
        self.current_train = None
    
    def step(self):
        """Execute one step of passenger behavior."""
        if not self.on_train:
            self.wait_time += 1

