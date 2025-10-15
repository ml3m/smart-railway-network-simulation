"""
Smart Railway Network Simulation - Enhanced Visualization
Beautiful, realistic visualization with animated trains and environmental graphics.
"""

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider, Choice
from model import RailwayNetworkModel
from agents import TrackAgent, TrainAgent, SignalAgent, StationAgent, TrainType, TrainState, WeatherCondition
import matplotlib.pyplot as plt
import pandas as pd


class NetworkStatsElement(TextElement):
    """Display real-time network statistics with enhanced formatting."""
    
    def __init__(self):
        pass
    
    def render(self, model):
        stats = model.get_summary_stats()
        
        # Determine status color
        if stats['delayed_trains'] > stats['active_trains'] * 0.3:
            status_color = "#e74c3c"
            status = "⚠️ HIGH DELAYS"
        elif stats['track_failures'] > 3:
            status_color = "#e67e22"
            status = "⚠️ TRACK ISSUES"
        else:
            status_color = "#27ae60"
            status = "✓ OPERATIONAL"
        
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h2 style="margin-top: 0; text-align: center; font-size: 24px;">🚄 RAILWAY CONTROL CENTER</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 15px;">
                <strong style="font-size: 18px; color: {status_color};">{status}</strong>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{stats['total_steps']}</div>
                    <div style="font-size: 12px; opacity: 0.9;">SIMULATION STEP</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{stats['weather'].upper()}</div>
                    <div style="font-size: 12px; opacity: 0.9;">WEATHER</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 32px; font-weight: bold;">{stats['network_utilization']:.0f}%</div>
                    <div style="font-size: 12px; opacity: 0.9;">UTILIZATION</div>
                </div>
            </div>
            
            <h3 style="margin-top: 20px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 5px;">🚂 TRAIN OPERATIONS</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 8px;">
                <div style="background: rgba(52, 152, 219, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 22px; font-weight: bold;">{stats['active_trains']}</div>
                    <div style="font-size: 10px;">ACTIVE</div>
                </div>
                <div style="background: rgba(46, 204, 113, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 22px; font-weight: bold;">{stats['arrived_trains']}</div>
                    <div style="font-size: 10px;">TRIPS DONE</div>
                </div>
                <div style="background: rgba(231, 76, 60, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 22px; font-weight: bold;">{stats['delayed_trains']}</div>
                    <div style="font-size: 10px;">DELAYED</div>
                </div>
                <div style="background: rgba(241, 196, 15, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 22px; font-weight: bold;">{stats['average_delay']:.1f}</div>
                    <div style="font-size: 10px;">AVG DELAY</div>
                </div>
                <div style="background: rgba(155, 89, 182, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 22px; font-weight: bold;">{model.dispatcher.scheduled_departures.__len__()}</div>
                    <div style="font-size: 10px;">SCHEDULED</div>
                </div>
            </div>
            
            <h3 style="margin-top: 20px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 5px;">👥 PASSENGER SERVICES</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(155, 89, 182, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{stats['total_passengers_waiting']}</div>
                    <div style="font-size: 11px;">WAITING</div>
                </div>
                <div style="background: rgba(26, 188, 156, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{stats['total_passengers_arrived']}</div>
                    <div style="font-size: 11px;">DELIVERED</div>
                </div>
            </div>
            
            <h3 style="margin-top: 20px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 5px;">⚡ SYSTEM STATUS</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                <div style="background: rgba(230, 126, 34, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{stats['average_energy']:.0f}</div>
                    <div style="font-size: 11px;">AVG ENERGY</div>
                </div>
                <div style="background: rgba(192, 57, 43, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{stats['track_failures']}</div>
                    <div style="font-size: 11px;">FAILURES</div>
                </div>
                <div style="background: rgba(142, 68, 173, 0.3); padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 24px; font-weight: bold;">{stats['priority_conflicts']}</div>
                    <div style="font-size: 11px;">CONFLICTS</div>
                </div>
            </div>
        </div>
        """
        return html


class LegendElement(TextElement):
    """Display comprehensive legend for all map elements."""
    
    def __init__(self):
        pass
    
    def render(self, model):
        html = """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
            <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">📍 MAP LEGEND</h3>
            
            <div style="margin-bottom: 20px;">
                <h4 style="color: #34495e; margin-bottom: 10px;">🚂 Trains (Colored Squares)</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #3498db; border: 2px solid #2c3e50; margin-right: 10px;"></div>
                        <span>Passenger Train</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #e67e22; border: 2px solid #2c3e50; margin-right: 10px;"></div>
                        <span>Cargo Train</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #9b59b6; border: 2px solid #2c3e50; margin-right: 10px;"></div>
                        <span>Express Train</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #e74c3c; border: 2px solid #2c3e50; margin-right: 10px;"></div>
                        <span>Emergency Train</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h4 style="color: #34495e; margin-bottom: 10px;">🚦 Signals (Colored Circles)</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #2ecc71; border-radius: 50%; margin-right: 10px; border: 1px solid #27ae60;"></div>
                        <span>Track Free</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #f39c12; border-radius: 50%; margin-right: 10px; border: 1px solid #e67e22;"></div>
                        <span>Occupied</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #c0392b; border-radius: 50%; margin-right: 10px; border: 1px solid #a93226;"></div>
                        <span>Failed</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h4 style="color: #34495e; margin-bottom: 10px;">📍 Infrastructure</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #16a085; border: 2px solid #0e6655; margin-right: 10px;"></div>
                        <span>Station</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #2d5016; margin-right: 10px;"></div>
                        <span>Railway Track</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 24px; height: 24px; background: #7cb342; margin-right: 10px;"></div>
                        <span>Grass/Field</span>
                    </div>
                </div>
            </div>
            
            <div>
                <h4 style="color: #34495e; margin-bottom: 10px;">🚂 Train States (Border Colors)</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #3498db; border: 2px solid #2c3e50; margin-right: 8px;"></div>
                        <span>Moving (Normal)</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #3498db; border: 3px solid #f39c12; margin-right: 8px;"></div>
                        <span>Waiting (Orange)</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #3498db; border: 4px solid #c0392b; margin-right: 8px;"></div>
                        <span>Delayed (Red)</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background: #3498db; border: 3px solid #27ae60; margin-right: 8px;"></div>
                        <span>Arrived (Green)</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #e8f5e9; border-radius: 5px; border-left: 4px solid #66bb6a;">
                <strong>💡 Nature Theme:</strong> Grass green background with dark green tracks. Trains are colored squares, signals are circles!
            </div>
        </div>
        """
        return html


class ScheduleElement(TextElement):
    """Display real-time schedule showing upcoming departures and active trains."""

    def __init__(self):
        pass

    def render(self, model):
        departures = model.dispatcher.scheduled_departures
        current_time = model.schedule.steps
        
        # Get all active trains
        active_trains = [t for t in model.get_all_trains() if t.state not in [TrainState.ARRIVED, TrainState.MAINTENANCE]]
        
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; height: 450px; overflow: hidden;">
            <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">🕒 REAL-TIME SCHEDULE (Step: {current_time})</h3>
            
            <h4 style="color: #34495e; margin-top: 15px; margin-bottom: 10px;">🚂 Active Trains ({len(active_trains)})</h4>
            <div style="height: 180px; overflow: hidden; margin-bottom: 20px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed;">
                    <thead>
                        <tr style="background: #34495e; color: white;">
                            <th style="padding: 6px; text-align: left; width: 12%;">ID</th>
                            <th style="padding: 6px; text-align: left; width: 15%;">Type</th>
                            <th style="padding: 6px; text-align: left; width: 15%;">State</th>
                            <th style="padding: 6px; text-align: left; width: 20%;">Position</th>
                            <th style="padding: 6px; text-align: left; width: 23%;">Destination</th>
                            <th style="padding: 6px; text-align: right; width: 15%;">Energy</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Display active trains
        for train in sorted(active_trains, key=lambda t: t.unique_id)[:15]:  # Show up to 15 trains
            # Color-code by type
            type_colors = {
                'passenger': '#3498db',
                'cargo': '#e67e22',
                'express': '#9b59b6',
                'emergency': '#e74c3c'
            }
            type_color = type_colors.get(train.train_type.value, '#95a5a6')
            
            # Color-code by state
            state_colors = {
                'moving': '#27ae60',
                'waiting': '#f39c12',
                'delayed': '#e74c3c',
                'boarding': '#3498db',
                'idle': '#95a5a6'
            }
            state_color = state_colors.get(train.state.value, '#95a5a6')
            
            energy_percent = (train.energy / train.max_energy * 100) if train.max_energy > 0 else 0
            energy_color = '#27ae60' if energy_percent > 50 else ('#f39c12' if energy_percent > 20 else '#e74c3c')
            
            html += f"""
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px; color: {type_color}; font-weight: bold;">T{train.unique_id}</td>
                <td style="padding: 6px;">{train.train_type.value.title()[:4]}</td>
                <td style="padding: 6px; color: {state_color}; font-weight: bold;">{train.state.value.upper()[:4]}</td>
                <td style="padding: 6px; font-size: 10px;">({train.current_position[0]},{train.current_position[1]})</td>
                <td style="padding: 6px; font-size: 10px;">({train.destination[0]},{train.destination[1]})</td>
                <td style="padding: 6px; text-align: right; color: {energy_color};">{energy_percent:.0f}%</td>
            </tr>
            """
        
        if not active_trains:
            html += '<tr><td colspan="6" style="padding: 8px; text-align: center; color: #6c757d;">No active trains currently.</td></tr>'
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <h4 style="color: #34495e; margin-top: 15px; margin-bottom: 10px;">📅 Upcoming Departures</h4>
            <div style="overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed;">
                    <thead>
                        <tr style="background: #e9ecef; color: #495057;">
                            <th style="padding: 6px; text-align: left; width: 35%;">Time</th>
                            <th style="padding: 6px; text-align: left; width: 25%;">Type</th>
                            <th style="padding: 6px; text-align: left; width: 40%;">Route</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Display next 5 departures
        for d in departures[:5]:
            time_until = d['departure_time'] - current_time
            time_color = '#e74c3c' if time_until <= 5 else ('#f39c12' if time_until <= 15 else '#27ae60')
            
            html += f"""
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px; color: {time_color}; font-weight: bold;">{d['departure_time']} (T-{time_until})</td>
                <td style="padding: 6px;">{d['train_type'].value.title()}</td>
                <td style="padding: 6px; font-size: 10px;">{d['origin']} → {d['destination']}</td>
            </tr>
            """
        
        if not departures:
            html += '<tr><td colspan="3" style="padding: 8px; text-align: center; color: #6c757d;">No new departures scheduled.</td></tr>'

        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        return html


def agent_portrayal(agent):
    """
    Enhanced agent portrayal with beautiful, realistic graphics.
    Nature-themed grass green background with darker green tracks.
    """
    if isinstance(agent, TrackAgent):
        # Railway track visualization - darker green for contrast with grass background
        return {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Layer": 0,
            "Color": "#2d5016",  # Dark forest green for tracks
            "text": "",  # No emoji, cleaner look
            "text_color": "#1a3409"
        }
    
    elif isinstance(agent, TrainAgent):
        # Enhanced train visualization - consistent colored squares
        portrayal = {
            "Shape": "rect",
            "w": 0.9,
            "h": 0.9,
            "Filled": "true",
            "Layer": 3,
            "stroke_color": "#2c3e50",
            "stroke": 2,
            "text": "",  # No text by default for cleaner look
            "text_color": "white"
        }
        
        # Color by train type with vibrant colors (CONSISTENT with legend)
        if agent.train_type == TrainType.PASSENGER:
            portrayal["Color"] = "#3498db"  # Bright Blue
        elif agent.train_type == TrainType.CARGO:
            portrayal["Color"] = "#e67e22"  # Orange
        elif agent.train_type == TrainType.EXPRESS:
            portrayal["Color"] = "#9b59b6"  # Purple
        elif agent.train_type == TrainType.EMERGENCY:
            portrayal["Color"] = "#e74c3c"  # Red
        
        # Visual state indicators - only change color for special states
        if agent.state == TrainState.WAITING:
            # Keep type color but add waiting indicator with border
            portrayal["stroke_color"] = "#f39c12"  # Orange border for waiting
            portrayal["stroke"] = 3
        elif agent.state == TrainState.DELAYED:
            # Pulsing red border for delayed
            portrayal["stroke_color"] = "#c0392b"  # Dark Red border
            portrayal["stroke"] = 4
        elif agent.state == TrainState.ARRIVED:
            # Green border when arrived
            portrayal["stroke_color"] = "#27ae60"
            portrayal["stroke"] = 3
        elif agent.state == TrainState.BOARDING:
            # Yellow border when boarding
            portrayal["stroke_color"] = "#f1c40f"
            portrayal["stroke"] = 3
        elif agent.state == TrainState.MAINTENANCE:
            # Gray out when in maintenance
            portrayal["Color"] = "#7f8c8d"
            portrayal["stroke_color"] = "#34495e"
            portrayal["stroke"] = 2
        
        # Add a glow effect for high-priority trains
        if hasattr(agent, 'priority') and agent.priority >= 8:
            portrayal["stroke_color"] = "#f39c12"
            portrayal["stroke"] = 3
        
        # Low energy warning
        if hasattr(agent, 'energy') and hasattr(agent, 'max_energy'):
            if agent.energy < agent.max_energy * 0.2:
                portrayal["stroke_color"] = "#e74c3c"
                portrayal["stroke"] = 3
        
        return portrayal
    
    elif isinstance(agent, SignalAgent):
        # Beautiful signal lights
        portrayal = {
            "Shape": "circle",
            "r": 0.4,
            "Filled": "true",
            "Layer": 2,
        }
        
        # Traffic light colors
        if agent.track_status == "failed":
            portrayal["Color"] = "#c0392b"  # Dark Red
            portrayal["text"] = "✖"
            portrayal["text_color"] = "white"
        elif agent.track_occupied:
            portrayal["Color"] = "#f39c12"  # Amber/Yellow
            portrayal["text"] = "●"
            portrayal["text_color"] = "white"
        else:
            portrayal["Color"] = "#2ecc71"  # Bright Green
            portrayal["text"] = "●"
            portrayal["text_color"] = "white"
        
        # Add subtle glow
        portrayal["stroke_color"] = "#ecf0f1"
        portrayal["stroke"] = 1
        
        return portrayal
    
    elif isinstance(agent, StationAgent):
        # Beautiful station representation - CONSISTENT colored squares
        portrayal = {
            "Shape": "rect",
            "w": 0.95,
            "h": 0.95,
            "Filled": "true",
            "Layer": 1,
            "Color": "#16a085",  # Teal (matches legend)
            "stroke_color": "#0e6655",
            "stroke": 2,
            "text": "",  # No emoji for consistency
            "text_color": "white"
        }
        
        # Highlight stations with many waiting passengers
        if hasattr(agent, 'waiting_passengers') and agent.waiting_passengers > 50:
            portrayal["stroke_color"] = "#d35400"  # Orange border for crowded
            portrayal["stroke"] = 4
        
        return portrayal
    
    # Empty cells - GRASS GREEN BACKGROUND (nature theme)
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "Color": "#7cb342",  # Grass green background
    }


def get_track_portrayal():
    """Background track visualization."""
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "Color": "#95a5a6"  # Gray for tracks
    }


def grid_portrayal(agent):
    """
    Grid portrayal that shows tracks, signals, stations, and trains with emojis.
    """
    if agent is None:
        return None
    
    return agent_portrayal(agent)


def create_server(model_params=None):
    """
    Create enhanced MESA visualization server with beautiful UI.
    """
    if model_params is None:
        model_params = {
            "width": 50,
            "height": 50,
            "num_trains": Slider(
                "Number of Trains",
                10,
                2,
                30,
                1
            ),
            "num_stations": Slider(
                "Number of Stations",
                5,
                3,
                10,
                1
            ),
            "network_complexity": Choice(
                "Network Complexity",
                value="medium",
                choices=["simple", "medium", "complex"]
            ),
            "enable_weather": Choice(
                "Enable Weather Effects",
                value=True,
                choices=[True, False]
            ),
            "enable_failures": Choice(
                "Enable Track Failures",
                value=True,
                choices=[True, False]
            ),
            "enable_passengers": Choice(
                "Enable Passenger Simulation",
                value=True,
                choices=[True, False]
            ),
        }
    
    # Create large, detailed grid visualization
    grid = CanvasGrid(
        agent_portrayal,
        50, 50,
        600, 600  # Larger canvas for better visibility
    )
    
    # Create statistics and legend
    stats_element = NetworkStatsElement()
    legend_element = LegendElement()
    schedule_element = ScheduleElement()
    
    # Enhanced charts with better styling
    train_chart = ChartModule(
        [
            {"Label": "Active Trains", "Color": "#3498db"},
            {"Label": "Delayed Trains", "Color": "#e74c3c"},
            {"Label": "Total Arrivals", "Color": "#2ecc71"},
        ],
        data_collector_name="datacollector"
    )
    
    delay_chart = ChartModule(
        [
            {"Label": "Average Delay", "Color": "#e67e22"},
        ],
        data_collector_name="datacollector"
    )
    
    energy_chart = ChartModule(
        [
            {"Label": "Total Energy", "Color": "#f39c12"},
        ],
        data_collector_name="datacollector"
    )
    
    passenger_chart = ChartModule(
        [
            {"Label": "Waiting Passengers", "Color": "#9b59b6"},
        ],
        data_collector_name="datacollector"
    )
    
    # Create server with all elements
    server = ModularServer(
        RailwayNetworkModel,
        [grid, stats_element, schedule_element, legend_element, train_chart, delay_chart, energy_chart, passenger_chart],
        "🚄 Smart Railway Network Simulation - Control Center",
        model_params
    )
    
    return server


def plot_simulation_results(model):
    """
    Generate beautiful, comprehensive plots from simulation results.
    """
    # Get data from datacollector
    model_data = model.datacollector.get_model_vars_dataframe()
    
    # Create professional figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Train Activity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(model_data.index, model_data['Active Trains'], label='Active', color='#3498db', linewidth=2.5, marker='o', markersize=4, markevery=10)
    ax1.plot(model_data.index, model_data['Delayed Trains'], label='Delayed', color='#e74c3c', linewidth=2.5, marker='s', markersize=4, markevery=10)
    ax1.plot(model_data.index, model_data['Total Arrivals'], label='Arrivals', color='#2ecc71', linewidth=2.5, marker='^', markersize=4, markevery=10)
    ax1.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Trains', fontsize=11, fontweight='bold')
    ax1.set_title('🚂 Train Activity Over Time', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#ffffff')
    
    # Plot 2: Average Delay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(model_data.index, model_data['Average Delay'], color='#e67e22', linewidth=2.5)
    ax2.fill_between(model_data.index, model_data['Average Delay'], alpha=0.3, color='#e67e22')
    ax2.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Delay (steps)', fontsize=11, fontweight='bold')
    ax2.set_title('⏱️ Average Train Delay', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#ffffff')
    
    # Plot 3: Track Failures
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(model_data.index, model_data['Track Failures'], color='#c0392b', linewidth=2.5)
    ax3.fill_between(model_data.index, model_data['Track Failures'], alpha=0.3, color='#c0392b')
    ax3.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Failures', fontsize=11, fontweight='bold')
    ax3.set_title('⚠️ Track Failures Over Time', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#ffffff')
    
    # Plot 4: Energy Consumption
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(model_data.index, model_data['Total Energy'], color='#f39c12', linewidth=2.5)
    ax4.fill_between(model_data.index, model_data['Total Energy'], alpha=0.3, color='#f39c12')
    ax4.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Energy', fontsize=11, fontweight='bold')
    ax4.set_title('⚡ Fleet Energy Levels', fontsize=13, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor('#ffffff')
    
    # Plot 5: Passenger Statistics
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(model_data.index, model_data['Waiting Passengers'], color='#9b59b6', linewidth=2.5)
    ax5.fill_between(model_data.index, model_data['Waiting Passengers'], alpha=0.3, color='#9b59b6')
    ax5.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Waiting Passengers', fontsize=11, fontweight='bold')
    ax5.set_title('👥 Passenger Wait Queue', fontsize=13, fontweight='bold', pad=15)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_facecolor('#ffffff')
    
    # Plot 6: Priority Conflicts
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(model_data.index, model_data['Priority Conflicts'], color='#8e44ad', linewidth=2.5)
    ax6.fill_between(model_data.index, model_data['Priority Conflicts'], alpha=0.3, color='#8e44ad')
    ax6.set_xlabel('Simulation Step', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Conflicts', fontsize=11, fontweight='bold')
    ax6.set_title('⚔️ Priority Conflicts', fontsize=13, fontweight='bold', pad=15)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_facecolor('#ffffff')
    
    # Add overall title
    fig.suptitle('🚄 Smart Railway Network Simulation - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def export_results(model, filename_prefix='railway_simulation'):
    """
    Export simulation results to CSV and generate plots.
    """
    # Export model data
    model_data = model.datacollector.get_model_vars_dataframe()
    model_data.to_csv(f'{filename_prefix}_model_data.csv')
    
    # Export agent data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    if not agent_data.empty:
        agent_data.to_csv(f'{filename_prefix}_agent_data.csv')
    
    # Generate and save plots
    fig = plot_simulation_results(model)
    fig.savefig(f'{filename_prefix}_plots.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    
    # Export summary statistics
    summary = model.get_summary_stats()
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{filename_prefix}_summary.csv', index=False)
    
    print(f"\n✅ Results exported:")
    print(f"   📊 {filename_prefix}_model_data.csv")
    print(f"   📊 {filename_prefix}_agent_data.csv")
    print(f"   📈 {filename_prefix}_plots.png")
    print(f"   📋 {filename_prefix}_summary.csv")
    
    return model_data, agent_data, summary


def print_summary_report(model):
    """
    Print a beautiful formatted summary report of the simulation.
    """
    stats = model.get_summary_stats()
    
    print("\n" + "="*70)
    print("🚄 SMART RAILWAY NETWORK SIMULATION - FINAL REPORT")
    print("="*70)
    
    print("\n📊 GENERAL STATISTICS")
    print(f"  ⏱️  Total Simulation Steps: {stats['total_steps']}")
    print(f"  🌦️  Final Weather Condition: {stats['weather'].upper()}")
    print(f"  📈 Network Utilization: {stats['network_utilization']:.1f}%")
    
    print("\n🚂 TRAIN STATISTICS")
    print(f"  🚄 Total Trains: {stats['total_trains']}")
    print(f"  ▶️  Active Trains: {stats['active_trains']}")
    print(f"  ✅ Arrived Trains: {stats['arrived_trains']}")
    print(f"  ⚠️  Delayed Trains: {stats['delayed_trains']}")
    print(f"  ⏰ Average Delay: {stats['average_delay']:.2f} steps")
    
    print("\n⚡ ENERGY STATISTICS")
    print(f"  🔋 Total Fleet Energy: {stats['total_energy']:.1f}")
    print(f"  📊 Average Train Energy: {stats['average_energy']:.1f}")
    
    print("\n👥 PASSENGER STATISTICS")
    print(f"  ⏳ Currently Waiting: {stats['total_passengers_waiting']}")
    print(f"  ✅ Total Delivered: {stats['total_passengers_arrived']}")
    
    print("\n⚠️  OPERATIONAL ISSUES")
    print(f"  🔴 Track Failures: {stats['track_failures']}")
    print(f"  ⚔️  Priority Conflicts: {stats['priority_conflicts']}")
    
    # Calculate efficiency metrics
    if stats['total_trains'] > 0:
        arrival_rate = (stats['arrived_trains'] / stats['total_trains']) * 100
        print("\n✨ EFFICIENCY METRICS")
        print(f"  📈 Arrival Success Rate: {arrival_rate:.1f}%")
        if stats['arrived_trains'] > 0:
            print(f"  ⏱️  Avg Delay (Arrived Trains): {stats['average_delay']:.2f} steps")
    
    print("\n" + "="*70 + "\n")
