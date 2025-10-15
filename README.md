# 🚄 Smart Railway Network Simulator

A comprehensive multi-agent railway network simulation built with MESA (Multi-agent Ecosystem for Simulation and Analysis). This project simulates a realistic railway system with intelligent trains, signals, dispatchers, stations, and passengers.

## ✨ Features

### 🚂 **Multiple Train Types**
- **Passenger Trains**: Optimized for speed and passenger capacity
- **Cargo Trains**: Heavy, slower, high energy consumption
- **Express Trains**: High priority, fast, premium service
- **Emergency Trains**: Highest priority, can override other traffic

### 🎯 **Core Simulation Components**
- **TrainAgent**: Autonomous trains with pathfinding, energy management, and state tracking
- **SignalAgent**: Track access control with priority management
- **DispatcherAgent**: Central coordination, rerouting, and conflict resolution
- **StationAgent**: Passenger management and boarding operations
- **PassengerAgent**: Individual passenger simulation (optional detailed mode)

### 🌍 **Dynamic Environment**
- **Weather System**: Clear, Rain, Storm, Fog, Snow (affects train speed)
- **Track Failures**: Random failures requiring rerouting
- **Maintenance**: Energy management and refueling
- **Real-time Adaptation**: Trains adapt to changing conditions

### 📊 **Advanced Features**
- **Energy Management**: Trains consume energy based on speed, cargo, and acceleration
- **Timetables & Scheduling**: Scheduled arrivals with delay tracking
- **Priority System**: Emergency vehicles can override normal traffic
- **Passenger Simulation**: Realistic boarding, waiting times, and arrival tracking
- **Network Topology**: Simple, Medium, or Complex network configurations
- **Comprehensive Statistics**: Real-time monitoring and analytics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (already created in your directory)

### Setup

1. Activate your virtual environment:
```bash
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Interactive Visualization (Default)

Launch the web-based visualization server:

```bash
python run.py
```

Then open your browser to `http://127.0.0.1:8521`

**Visualization Legend:**
- 🔵 **Blue** = Passenger Train
- 🟠 **Orange** = Cargo Train  
- 🟣 **Purple** = Express Train
- 🔴 **Red** = Emergency Train

**Signal Colors:**
- 🟢 **Green** = Track Free
- 🟡 **Yellow** = Track Occupied
- 🔴 **Red** = Track Failed

**Stations:** 🚉 Teal squares

### Batch Simulation Mode

Run simulation without visualization and export results:

```bash
# Basic batch run
python run.py --batch

# Custom configuration
python run.py --batch --steps 1000 --trains 20 --stations 8 --complexity complex

# Disable specific features
python run.py --batch --no-weather --no-failures

# Custom output prefix
python run.py --batch --output my_simulation
```

**Output files:**
- `*_model_data.csv`: Time-series data
- `*_agent_data.csv`: Individual agent data
- `*_plots.png`: Comprehensive visualizations
- `*_summary.csv`: Summary statistics

### Comparative Analysis

Compare different configurations:

```bash
python run.py --compare
```

This runs 4 predefined scenarios and generates comparison charts.

### Command-Line Options

```
--batch              Run in batch mode without visualization
--compare            Run comparative analysis
--steps N            Number of simulation steps (default: 500)
--trains N           Number of trains (default: 10)
--stations N         Number of stations (default: 5)
--complexity LEVEL   Network complexity: simple, medium, complex
--no-weather         Disable weather effects
--no-failures        Disable track failures
--no-passengers      Disable passenger simulation
--host HOST          Server host for interactive mode
--port PORT          Server port for interactive mode
--output PREFIX      Output file prefix for batch mode
```

## 📈 Metrics & Analytics

The simulation tracks:

### Train Metrics
- Active trains, delayed trains, arrivals
- Average delay times
- Energy consumption and efficiency
- Waiting times
- State transitions

### Network Metrics
- Track utilization
- Failed track segments
- Priority conflicts
- Signal occupancy

### Passenger Metrics
- Waiting passengers per station
- Total passengers transported
- Average wait times
- Boarding efficiency

## 🏗️ Architecture

### File Structure
```
smart-railway-network-simulation/
├── agents.py           # All agent classes
├── model.py            # Main simulation model
├── visualization.py    # MESA visualization and plotting
├── run.py             # Entry point and CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

### Agent Hierarchy

```
Model (RailwayNetworkModel)
├── DispatcherAgent (singleton)
├── TrainAgent (multiple)
│   ├── PassengerAgent subtype
│   ├── CargoAgent subtype
│   ├── ExpressAgent subtype
│   └── EmergencyAgent subtype
├── SignalAgent (at junctions and intervals)
├── StationAgent (at strategic locations)
└── PassengerAgent (optional, individual passengers)
```

### Key Algorithms

**A* Pathfinding**: Trains use A* algorithm to find optimal routes through the network

**Priority Management**: Higher priority trains can request rerouting of lower priority traffic

**Conflict Resolution**: Dispatcher manages conflicts at track junctions

**Dynamic Rerouting**: Trains automatically reroute when encountering failures

**Energy Optimization**: Trains balance speed vs energy consumption

## 🎓 Educational Use

This project demonstrates:

- **Multi-agent systems**: Autonomous agents with local decision-making
- **Emergent behavior**: Complex patterns from simple rules
- **Resource management**: Track allocation, energy management
- **Priority scheduling**: Handling competing objectives
- **Real-time adaptation**: Dynamic response to failures and changes
- **Data visualization**: Real-time monitoring and post-analysis
- **Scalability**: Configurable complexity levels

## 🔬 Experimental Ideas

### Easy Extensions
- Add more train types (maintenance, luxury)
- Implement rush hour traffic patterns
- Add train coupling/decoupling
- Implement multi-car trains
- Add track switches and sidings

### Advanced Extensions
- **Reinforcement Learning Dispatcher**: Train an RL agent to optimize routing
- **Predictive Maintenance**: Predict failures before they occur
- **Dynamic Pricing**: Adjust fares based on demand
- **Multi-line Transfers**: Passengers change trains
- **Autonomous Coordination**: Trains negotiate without dispatcher
- **LLM-based Planning**: Use language models for high-level planning

## 📊 Example Results

After running a simulation, you'll see:

```
🚄 SMART RAILWAY NETWORK SIMULATION - SUMMARY REPORT
============================================================

📊 GENERAL STATISTICS
  Total Simulation Steps: 500
  Final Weather Condition: RAIN
  Network Utilization: 45.2%

🚂 TRAIN STATISTICS
  Total Trains: 15
  Active Trains: 8
  Arrived Trains: 7
  Delayed Trains: 3
  Average Delay: 12.5 steps

⚡ ENERGY STATISTICS
  Total Fleet Energy: 8547.3
  Average Train Energy: 569.8

👥 PASSENGER STATISTICS
  Currently Waiting: 45
  Total Arrived: 432

⚠️  OPERATIONAL ISSUES
  Track Failures: 2
  Priority Conflicts: 5
```

## 🐛 Troubleshooting

**Port already in use:**
```bash
python run.py --port 8522
```

**Dependencies not installing (macOS Silicon):**
```bash
# Make sure you're using the right Python
which python
python --version  # Should be 3.8+

# Upgrade pip
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt
```

**Simulation running slowly:**
- Reduce number of trains: `--trains 5`
- Use simpler network: `--complexity simple`
- Disable features: `--no-weather --no-passengers`

## 📝 License

This is an educational project. Feel free to use, modify, and extend it.

## 🤝 Contributing

This is a learning project, but suggestions are welcome! Areas for improvement:

- Performance optimization
- More realistic train physics
- Better visualization options
- Additional agent types
- Machine learning integration

## 🙏 Acknowledgments

Built with:
- **MESA**: Multi-agent simulation framework
- **NetworkX**: Graph algorithms for routing
- **Matplotlib**: Data visualization
- **Pandas**: Data analysis
- **NumPy**: Numerical computing

---

**Happy Simulating! 🚄✨**

# smart-railway-network-simulation
