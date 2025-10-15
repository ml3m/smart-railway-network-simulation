"""
Smart Railway Network Simulation - Agent Classes
All agent types for the railway simulation system.
"""

from mesa import Agent
from enum import Enum
import random
from typing import List, Tuple, Optional, Set


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
    """Different types of trains with different characteristics."""
    PASSENGER = "passenger"
    CARGO = "cargo"
    EXPRESS = "express"
    EMERGENCY = "emergency"


class TrainState(Enum):
    """Possible states for a train."""
    IDLE = "idle"
    MOVING = "moving"
    WAITING = "waiting"
    DELAYED = "delayed"
    BOARDING = "boarding"
    MAINTENANCE = "maintenance"
    ARRIVED = "arrived"


class WeatherCondition(Enum):
    """Weather conditions affecting train operations."""
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"
    FOG = "fog"
    SNOW = "snow"


class TrainAgent(Agent):
    """
    Train agent that moves through the railway network.
    
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
    """
    
    def __init__(self, unique_id, model, train_type: TrainType, start_pos: Tuple[int, int], 
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
        self.energy = 1000.0
        
        # Set train-type-specific attributes
        if train_type == TrainType.PASSENGER:
            self.base_speed = 2
            self.priority = 5
            self.max_energy = 1000.0
            self.energy_consumption_rate = 1.5
            self.max_capacity = 200
            self.cargo_weight = 0
        elif train_type == TrainType.CARGO:
            self.base_speed = 1
            self.priority = 3
            self.max_energy = 1500.0
            self.energy_consumption_rate = 2.5
            self.max_capacity = 50
            self.cargo_weight = 1000
        elif train_type == TrainType.EXPRESS:
            self.base_speed = 3
            self.priority = 8
            self.max_energy = 800.0
            self.energy_consumption_rate = 3.0
            self.max_capacity = 150
            self.cargo_weight = 0
        elif train_type == TrainType.EMERGENCY:
            self.base_speed = 3
            self.priority = 10
            self.max_energy = 900.0
            self.energy_consumption_rate = 2.0
            self.max_capacity = 100
            self.cargo_weight = 0
        
        self.speed = self.base_speed
        self.energy = self.max_energy
        
    def calculate_route(self):
        """Calculate route from current position to destination using A* pathfinding."""
        if not self.route or self.route_index >= len(self.route):
            # Simple pathfinding - move towards destination
            self.route = self._a_star_path(self.current_position, self.destination)
            self.route_index = 0
    
    def _a_star_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm for route calculation."""
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
        
        if weather == WeatherCondition.CLEAR:
            self.speed = self.base_speed
        elif weather == WeatherCondition.RAIN:
            self.speed = max(1, int(self.base_speed * 0.8))
        elif weather == WeatherCondition.FOG:
            self.speed = max(1, int(self.base_speed * 0.6))
        elif weather == WeatherCondition.STORM:
            self.speed = max(1, int(self.base_speed * 0.5))
        elif weather == WeatherCondition.SNOW:
            self.speed = max(1, int(self.base_speed * 0.7))
    
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
            return
        
        # CRITICAL: Check if at destination FIRST before recalculating route
        if self.current_position == self.destination:
            self.state = TrainState.ARRIVED
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
                
                # CRITICAL FIX: Update grid position!
                self.model.grid.move_agent(self, next_position)
                
                # Consume energy based on speed and cargo
                energy_cost = self.energy_consumption_rate * self.speed
                if self.train_type == TrainType.CARGO:
                    energy_cost *= (1 + self.cargo_weight / 2000)
                self.energy -= energy_cost
                
                # Update signal occupancy
                old_signal = self.model.get_signal_at(old_position)
                if old_signal:
                    old_signal.release_track(self)
                
                self.waiting_time = 0
                
                # Reset delay if train starts moving again
                if self.delay_time > 0:
                    self.delay_time = max(0, self.delay_time - 1)  # Gradually recover from delays
            else:
                # Access denied, wait - but not for long!
                self.state = TrainState.WAITING
                self.waiting_time += 1
                
                # AGGRESSIVE DEADLOCK DETECTION
                if self.waiting_time > 5:  # After just 5 steps of waiting
                    self.state = TrainState.DELAYED
                    self.delay_time += 1
                    # Immediately request help from dispatcher
                    self.model.dispatcher.handle_stuck_train(self)
                    
                # FORCE UNSTUCK after 10 steps
                if self.waiting_time > 10:
                    self.model.dispatcher.force_unstuck(self)
    
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
        self.energy = self.max_energy
        
    def step(self):
        """Execute one step of the train's behavior."""
        # DON'T skip if arrived - let dispatcher handle recycling
        # The train will be recycled automatically after reporting arrival
            
        # Adjust speed for weather
        self.adjust_speed_for_weather()
        
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
                # DON'T RETURN - continue moving in the same step!
        
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
    
    NOW INCLUDES:
    - Active timetable management
    - Scheduled train departures
    - Train recycling for continuous operation
    - Better coordination and monitoring
    
    Attributes:
        train_registry: Dict of all active trains by ID
        scheduled_departures: List of upcoming train departures
        track_failures: Set of failed track positions
        reroute_requests: Queue of reroute requests
        delay_stats: Statistics on delays
        total_arrivals: Number of trains that reached destination
        total_delay: Total delay time across all trains
    """
    
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
    
    def recycle_train(self, train: TrainAgent):
        """Recycle a train that has completed its journey for a new route."""
        # Record the completion
        self.completed_trips.append({
            'train_id': train.unique_id,
            'train_type': train.train_type,
            'completion_time': self.model.schedule.steps
        })
        
        # Schedule a new trip for this train - ALWAYS recycle for continuous operation
        if train.train_type in [TrainType.PASSENGER, TrainType.EXPRESS]:
            # Passenger trains get new station-to-station routes
            stations = list(self.model.station_positions.keys())
            if len(stations) >= 2:
                # Current position is the origin for next trip
                origin = train.current_position
                
                # Pick a different destination (not the current position)
                possible_destinations = [s for s in stations if s != origin]
                if possible_destinations:
                    new_destination = random.choice(possible_destinations)
                else:
                    # Fallback: pick any station
                    new_destination = random.choice(stations)
                
                # Calculate new scheduled arrival time
                distance = abs(origin[0] - new_destination[0]) + abs(origin[1] - new_destination[1])
                estimated_travel_time = max(distance // 2, 30)  # Rough estimate
                new_scheduled_arrival = self.model.schedule.steps + estimated_travel_time + 20
                
                # Reset train for new journey
                train.destination = new_destination
                train.scheduled_arrival = new_scheduled_arrival
                train.state = TrainState.IDLE
                train.route = []
                train.route_index = 0
                train.passengers = 0  # Passengers already delivered
                train.delay_time = 0  # Reset delay
                train.waiting_time = 0  # Reset waiting
                train.refuel()  # Refuel at station
                
                # Calculate new route
                train.calculate_route()
                
                return True
            else:
                # Not enough stations, use track positions
                tracks = list(self.model.track_positions)
                if len(tracks) >= 2:
                    origin = train.current_position
                    possible_destinations = [t for t in tracks if t != origin]
                    if possible_destinations:
                        new_destination = random.choice(possible_destinations)
                        
                        distance = abs(origin[0] - new_destination[0]) + abs(origin[1] - new_destination[1])
                        estimated_travel_time = max(distance // 2, 30)
                        new_scheduled_arrival = self.model.schedule.steps + estimated_travel_time + 20
                        
                        train.destination = new_destination
                        train.scheduled_arrival = new_scheduled_arrival
                        train.state = TrainState.IDLE
                        train.route = []
                        train.route_index = 0
                        train.passengers = 0
                        train.delay_time = 0
                        train.waiting_time = 0
                        train.refuel()
                        train.calculate_route()
                        
                        return True
        else:
            # Cargo and emergency trains can go anywhere on the track network
            tracks = list(self.model.track_positions)
            if len(tracks) >= 2:
                origin = train.current_position
                possible_destinations = [t for t in tracks if t != origin]
                if possible_destinations:
                    new_destination = random.choice(possible_destinations)
                else:
                    new_destination = random.choice(tracks)
                
                distance = abs(origin[0] - new_destination[0]) + abs(origin[1] - new_destination[1])
                estimated_travel_time = max(distance // 2, 30)
                new_scheduled_arrival = self.model.schedule.steps + estimated_travel_time + 20
                
                train.destination = new_destination
                train.scheduled_arrival = new_scheduled_arrival
                train.state = TrainState.IDLE
                train.route = []
                train.route_index = 0
                train.delay_time = 0
                train.waiting_time = 0
                train.refuel()
                train.calculate_route()
                
                return True
        
        return False
    
    def request_reroute(self, train: TrainAgent):
        """Handle a reroute request from a train."""
        if train not in self.reroute_requests:
            self.reroute_requests.append(train)
    
    def handle_stuck_train(self, train: TrainAgent):
        """Handle a train that's stuck waiting - clear the way aggressively."""
        # First, try to clear the blocking track
        if train.route and train.route_index < len(train.route):
            blocked_position = train.route[train.route_index]
            signal = self.model.get_signal_at(blocked_position)
            
            if signal and signal.track_occupied and signal.occupying_train:
                blocking_train = signal.occupying_train
                
                # If blocking train is lower priority, force it to move or reroute
                if train.priority > blocking_train.priority:
                    # High priority train - clear the way immediately
                    self.request_reroute(blocking_train)
                    # Also try to release the signal
                    if blocking_train.state == TrainState.WAITING:
                        signal.release_track(blocking_train)
        
        # Also request reroute for the stuck train as backup
        self.request_reroute(train)
    
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
        self.total_arrivals += 1
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
        
        # Recycle the train for a new journey after arrival (continuous operation)
        self.recycle_train(train)
    
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
        """Execute one step of dispatcher behavior - AGGRESSIVE deadlock prevention."""
        # PRIORITY: Spawn scheduled trains (realistic timetable management)
        self.spawn_scheduled_trains()
        
        # Process reroute requests FIRST
        self.handle_reroute_requests()
        
        # AGGRESSIVE DEADLOCK DETECTION AND PREVENTION
        stuck_trains = []
        waiting_trains = []
        
        for train in list(self.train_registry.values()):
            # Detect stuck trains (waiting too long)
            if train.waiting_time > 5:  # Very aggressive threshold
                stuck_trains.append(train)
                
            # Track all waiting trains
            if train.state == TrainState.WAITING:
                waiting_trains.append(train)
            
            # If train low on energy at a station, refuel it INSTANTLY
            if train.energy < train.max_energy * 0.3:
                station = self.model.get_station_at(train.current_position)
                if station:
                    train.refuel()
                    train.state = TrainState.IDLE if train.state == TrainState.MAINTENANCE else train.state
        
        # RESOLVE DEADLOCKS: If multiple trains are stuck, break the deadlock
        if len(stuck_trains) > 1:
            # Potential deadlock situation - prioritize by train priority
            stuck_trains.sort(key=lambda t: (t.priority, -t.waiting_time), reverse=True)
            
            # Force unstuck the highest priority trains first
            for train in stuck_trains[:3]:  # Handle top 3 stuck trains
                if train.waiting_time > 8:
                    self.force_unstuck(train)
                else:
                    self.handle_stuck_train(train)
        
        # Even if just one train stuck for a while, help it
        elif len(stuck_trains) == 1:
            train = stuck_trains[0]
            if train.waiting_time > 8:
                self.force_unstuck(train)
        
        # ENSURE CONTINUOUS OPERATION: Schedule more trains if needed
        active_count = sum(1 for t in self.train_registry.values() 
                          if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE])
        
        # If too few active trains, schedule more departures
        if active_count < self.model.num_trains // 2 and len(self.scheduled_departures) < 5:
            self._schedule_additional_trains()


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

