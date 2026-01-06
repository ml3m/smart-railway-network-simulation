"""
Smart Railway Network Simulation - Modern Solara Web Visualization

A beautiful, modern web interface for the railway simulation using Solara.
This provides an alternative to the Mesa visualization with a more polished UI.

Run with: solara run solara_app.py
"""

import solara
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors
from io import BytesIO
import base64
from typing import Optional

from model import RailwayNetworkModel
from agents import TrainType, TrainState, WeatherCondition
from config import SimulationSettings, PRESETS, list_presets


# Global state
model_state = solara.reactive(None)
running_state = solara.reactive(False)
step_count = solara.reactive(0)
config_state = solara.reactive(SimulationSettings())


def create_model(config: SimulationSettings) -> RailwayNetworkModel:
    """Create a new simulation model."""
    return RailwayNetworkModel(
        width=config.width,
        height=config.height,
        num_trains=config.num_trains,
        num_stations=config.num_stations,
        network_complexity=config.network_complexity,
        enable_weather=config.enable_weather,
        enable_failures=config.enable_failures,
        enable_passengers=config.enable_passengers
    )


def render_grid(model: RailwayNetworkModel) -> str:
    """Render the simulation grid as a base64 image."""
    if model is None:
        return ""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)
    
    # Set background color (grass green)
    ax.set_facecolor('#7cb342')
    
    # Draw tracks
    for pos in model.track_positions:
        rect = Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, 
                         facecolor='#2d5016', edgecolor='none')
        ax.add_patch(rect)
    
    # Draw stations
    for pos, station in model.station_positions.items():
        rect = Rectangle((pos[0] - 0.4, pos[1] - 0.4), 0.8, 0.8,
                         facecolor='#16a085', edgecolor='#0e6655', linewidth=2)
        ax.add_patch(rect)
    
    # Draw signals
    for pos, signal in model.signal_positions.items():
        if signal.track_status == "failed":
            color = '#c0392b'
        elif signal.track_occupied:
            color = '#f39c12'
        else:
            color = '#2ecc71'
        
        circle = Circle((pos[0], pos[1]), 0.3, facecolor=color, 
                        edgecolor='white', linewidth=1)
        ax.add_patch(circle)
    
    # Draw trains
    type_colors = {
        TrainType.PASSENGER: '#3498db',
        TrainType.CARGO: '#e67e22',
        TrainType.EXPRESS: '#9b59b6',
        TrainType.EMERGENCY: '#e74c3c'
    }
    
    for train in model.get_all_trains():
        color = type_colors.get(train.train_type, '#3498db')
        
        # Border based on state
        if train.state == TrainState.DELAYED:
            edge_color = '#c0392b'
            linewidth = 3
        elif train.state == TrainState.WAITING:
            edge_color = '#f39c12'
            linewidth = 2
        elif train.state == TrainState.ARRIVED:
            edge_color = '#27ae60'
            linewidth = 2
        else:
            edge_color = '#2c3e50'
            linewidth = 1
        
        rect = Rectangle(
            (train.current_position[0] - 0.35, train.current_position[1] - 0.35),
            0.7, 0.7,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=linewidth
        )
        ax.add_patch(rect)
    
    ax.set_xlim(-1, model.width + 1)
    ax.set_ylim(-1, model.height + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#7cb342', dpi=80)
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


@solara.component
def ConfigPanel():
    """Configuration panel component."""
    preset_names = list(PRESETS.keys())
    selected_preset = solara.use_reactive("default")
    
    with solara.Card("⚙️ Configuration", margin=0):
        solara.Select(
            label="Preset",
            value=selected_preset.value,
            values=preset_names,
            on_value=selected_preset.set
        )
        
        config = config_state.value
        
        solara.SliderInt(
            label="Trains",
            value=config.num_trains,
            min=2,
            max=30,
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'num_trains': v})
            )
        )
        
        solara.SliderInt(
            label="Stations",
            value=config.num_stations,
            min=2,
            max=10,
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'num_stations': v})
            )
        )
        
        solara.Select(
            label="Network Complexity",
            value=config.network_complexity,
            values=["simple", "medium", "complex"],
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'network_complexity': v})
            )
        )
        
        solara.Checkbox(
            label="Enable Weather",
            value=config.enable_weather,
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'enable_weather': v})
            )
        )
        
        solara.Checkbox(
            label="Enable Failures",
            value=config.enable_failures,
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'enable_failures': v})
            )
        )
        
        solara.Checkbox(
            label="Enable Passengers",
            value=config.enable_passengers,
            on_value=lambda v: config_state.set(
                SimulationSettings(**{**config.__dict__, 'enable_passengers': v})
            )
        )


@solara.component
def ControlPanel():
    """Simulation control panel."""
    
    def reset():
        model = create_model(config_state.value)
        model_state.set(model)
        step_count.set(0)
        running_state.set(False)
    
    def step():
        if model_state.value:
            model_state.value.step()
            step_count.set(step_count.value + 1)
    
    def toggle_run():
        running_state.set(not running_state.value)
    
    with solara.Card("🎮 Controls", margin=0):
        with solara.Row():
            solara.Button(
                "Reset",
                on_click=reset,
                color="warning"
            )
            solara.Button(
                "Step",
                on_click=step,
                disabled=running_state.value
            )
            solara.Button(
                "Stop" if running_state.value else "Start",
                on_click=toggle_run,
                color="success" if not running_state.value else "error"
            )
        
        solara.Text(f"Step: {step_count.value}")


@solara.component
def StatsPanel():
    """Statistics display panel."""
    model = model_state.value
    
    if model is None:
        with solara.Card("📊 Statistics", margin=0):
            solara.Text("No simulation running")
        return
    
    stats = model.get_summary_stats()
    
    # Calculate efficiency
    total_passengers = stats['total_passengers_arrived'] + stats['total_passengers_waiting']
    efficiency = (stats['total_passengers_arrived'] / max(1, total_passengers)) * 100
    
    with solara.Card("📊 Statistics", margin=0):
        with solara.Columns([1, 1]):
            with solara.Column():
                solara.Text(f"🚂 Active Trains: {stats['active_trains']}")
                solara.Text(f"✅ Trips Completed: {stats['arrived_trains']}")
                solara.Text(f"⚠️ Currently Delayed: {stats['delayed_trains']}")
                solara.Text(f"⏱️ Avg Delay: {stats['average_delay']:.1f} steps")
            
            with solara.Column():
                solara.Text(f"👥 Passengers Waiting: {stats['total_passengers_waiting']}")
                solara.Text(f"🎯 Passengers Delivered: {stats['total_passengers_arrived']}")
                solara.Text(f"🌤️ Weather: {stats['weather']}")
                solara.Text(f"📈 Efficiency: {efficiency:.0f}%")
        
        # Show system health indicators
        solara.Markdown("---")
        if stats['delayed_trains'] > stats['active_trains'] * 0.5:
            solara.Warning(f"⚠️ High delays detected - automatic resolution in progress")
        elif efficiency < 50:
            solara.Warning(f"⚠️ Low efficiency - system may need attention")
        else:
            solara.Success(f"✅ System operating normally ({stats['network_utilization']:.0f}% network utilization)")


@solara.component
def LegendPanel():
    """Legend panel showing train types and states."""
    with solara.Card("📍 Legend", margin=0):
        solara.Markdown("""
**Trains:**
- 🔵 Blue = Passenger
- 🟠 Orange = Cargo
- 🟣 Purple = Express
- 🔴 Red = Emergency

**Signals:**
- 🟢 Green = Free
- 🟡 Yellow = Occupied
- 🔴 Red = Failed

**Stations:**
- 🟦 Teal squares
        """)


@solara.component
def GridView():
    """Main grid visualization."""
    model = model_state.value
    
    if model is None:
        with solara.Card("🗺️ Railway Network", margin=0):
            solara.Info("Click 'Reset' to initialize the simulation")
        return
    
    image_data = render_grid(model)
    
    with solara.Card("🗺️ Railway Network", margin=0):
        solara.Image(image_data)


@solara.component
def TrainTable():
    """Table showing train details."""
    model = model_state.value
    
    if model is None:
        return
    
    trains = model.get_all_trains()
    
    with solara.Card("🚂 Active Trains", margin=0):
        headers = ["ID", "Type", "State", "Energy", "Passengers"]
        
        rows = []
        for train in trains[:15]:  # Show first 15
            rows.append([
                f"T{train.unique_id}",
                train.train_type.value.title(),
                train.state.value.upper(),
                f"{train.energy_percentage:.0f}%",
                str(train.passengers) if train.train_type in [TrainType.PASSENGER, TrainType.EXPRESS] else "-"
            ])
        
        if rows:
            solara.DataFrame(
                {"ID": [r[0] for r in rows],
                 "Type": [r[1] for r in rows],
                 "State": [r[2] for r in rows],
                 "Energy": [r[3] for r in rows],
                 "Passengers": [r[4] for r in rows]}
            )


@solara.component
def Page():
    """Main application page."""
    # Auto-step when running
    def auto_step():
        if running_state.value and model_state.value:
            model_state.value.step()
            step_count.set(step_count.value + 1)
    
    solara.use_effect(lambda: None, [step_count.value])  # Force re-render
    
    if running_state.value:
        import asyncio
        solara.use_thread(auto_step, [running_state.value])
    
    with solara.AppBarTitle():
        solara.Text("🚄 Smart Railway Network Simulation")
    
    with solara.Sidebar():
        ConfigPanel()
        solara.Div(style={"height": "20px"})
        ControlPanel()
        solara.Div(style={"height": "20px"})
        LegendPanel()
    
    with solara.Column(style={"padding": "20px"}):
        with solara.Row():
            with solara.Column(style={"flex": "2"}):
                GridView()
            with solara.Column(style={"flex": "1"}):
                StatsPanel()
                solara.Div(style={"height": "20px"})
                TrainTable()


# For solara run
app = Page

