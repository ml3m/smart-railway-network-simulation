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


class StyleInjectorElement(TextElement):
    """Inject global CSS for modern dark theme design."""
    
    def __init__(self):
        pass
    
    def render(self, model):
        return '''
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* ===== GLOBAL DARK THEME ===== */
            body {
                background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                color: #e4e4e7 !important;
                min-height: 100vh;
            }
            
            .container, .container-fluid {
                background: transparent !important;
            }
            
            /* ===== NAVBAR STYLING ===== */
            .navbar, nav.navbar {
                background: rgba(13, 17, 23, 0.95) !important;
                backdrop-filter: blur(20px) !important;
                -webkit-backdrop-filter: blur(20px) !important;
                border-bottom: 1px solid rgba(99, 102, 241, 0.3) !important;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
                padding: 0.75rem 1.5rem !important;
            }
            
            .navbar-brand {
                color: #fff !important;
                font-weight: 700 !important;
                font-size: 1.35rem !important;
                text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
            }
            
            .navbar .btn, .navbar button, .navbar a.nav-link {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                padding: 0.5rem 1.25rem !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
                margin: 0 0.25rem !important;
            }
            
            .navbar .btn:hover, .navbar button:hover, .navbar a.nav-link:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6) !important;
            }
            
            /* ===== SIDEBAR STYLING ===== */
            .col-md-3, .col-sm-3, [class*="sidebar"] {
                background: rgba(13, 17, 23, 0.9) !important;
                backdrop-filter: blur(15px) !important;
                -webkit-backdrop-filter: blur(15px) !important;
                border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
                padding: 1.5rem !important;
            }
            
            /* Parameter labels */
            .badge, label, .form-label {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
                color: white !important;
                font-weight: 500 !important;
                padding: 0.5rem 1rem !important;
                border-radius: 6px !important;
                display: inline-block !important;
                margin-bottom: 0.5rem !important;
                font-size: 0.85rem !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
            }
            
            /* Custom sliders */
            input[type="range"] {
                -webkit-appearance: none !important;
                width: 100% !important;
                height: 8px !important;
                border-radius: 4px !important;
                background: rgba(99, 102, 241, 0.25) !important;
                outline: none !important;
                margin: 0.75rem 0 !important;
            }
            
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none !important;
                width: 20px !important;
                height: 20px !important;
                border-radius: 50% !important;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
                cursor: pointer !important;
                box-shadow: 0 2px 10px rgba(99, 102, 241, 0.6) !important;
                transition: all 0.2s ease !important;
            }
            
            input[type="range"]::-webkit-slider-thumb:hover {
                transform: scale(1.1) !important;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.8) !important;
            }
            
            /* Custom dropdowns */
            select, .form-select {
                background: rgba(13, 17, 23, 0.9) !important;
                border: 1px solid rgba(99, 102, 241, 0.35) !important;
                color: #e4e4e7 !important;
                padding: 0.6rem 1rem !important;
                border-radius: 8px !important;
                font-size: 0.9rem !important;
                width: 100% !important;
                margin-bottom: 1rem !important;
                cursor: pointer !important;
                transition: all 0.2s ease !important;
            }
            
            select:hover, select:focus {
                border-color: rgba(99, 102, 241, 0.7) !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
                outline: none !important;
            }
            
            select option {
                background: #0d1117 !important;
                color: #e4e4e7 !important;
            }
            
            /* ===== MAIN CONTENT AREA ===== */
            .col-md-9, .col-sm-9 {
                background: transparent !important;
                padding: 1.5rem !important;
            }
            
            /* FPS Slider area */
            .fps-slider, [class*="fps"] {
                background: rgba(13, 17, 23, 0.8) !important;
                border-radius: 10px !important;
                padding: 1rem !important;
                margin-bottom: 1rem !important;
                border: 1px solid rgba(99, 102, 241, 0.15) !important;
            }
            
            /* ===== CANVAS GRID STYLING - BORDER ONLY ===== */
            canvas {
                border-radius: 12px !important;
                box-shadow: 
                    0 0 0 2px rgba(99, 102, 241, 0.4),
                    0 0 30px rgba(99, 102, 241, 0.15),
                    0 8px 32px rgba(0, 0, 0, 0.6) !important;
            }
            
            /* Canvas container wrapper - only for border effect */
            .world-grid, [class*="world"], [class*="grid-container"] {
                border-radius: 14px !important;
                padding: 4px !important;
                border: 1px solid rgba(99, 102, 241, 0.3) !important;
            }
            
            /* ===== CHART STYLING ===== */
            .chart-container, [class*="chart"] {
                background: rgba(13, 17, 23, 0.9) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 12px !important;
                padding: 1rem !important;
                margin-top: 1rem !important;
                border: 1px solid rgba(99, 102, 241, 0.2) !important;
            }
            
            /* ===== TEXT COLORS ===== */
            p, span, div, td, th, li {
                color: #c9d1d9 !important;
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: #fff !important;
            }
            
            /* ===== SCROLLBAR ===== */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(13, 17, 23, 0.5);
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
            }
            
            /* ===== ANIMATIONS ===== */
            @keyframes pulse-glow {
                0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
                50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); }
            }
            
            /* ===== RESPONSIVE FIXES ===== */
            .row {
                margin: 0 !important;
            }
        </style>
        '''


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
        <div style="font-family: 'Inter', -apple-system, sans-serif; background: rgba(26, 26, 46, 0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); padding: 24px; border-radius: 16px; color: #e4e4e7; box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 1px solid rgba(99, 102, 241, 0.2); margin-bottom: 20px;">
            <h2 style="margin-top: 0; text-align: center; font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">🚄 RAILWAY CONTROL CENTER</h2>
            <div style="background: rgba(99, 102, 241, 0.15); padding: 12px; border-radius: 10px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <strong style="font-size: 16px; color: {status_color}; text-shadow: 0 0 10px {status_color}40;">{status}</strong>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px;">
                <div style="background: rgba(99, 102, 241, 0.12); padding: 16px; border-radius: 12px; text-align: center; border: 1px solid rgba(99, 102, 241, 0.15);">
                    <div style="font-size: 28px; font-weight: 700; color: #a78bfa;">{stats['total_steps']}</div>
                    <div style="font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Simulation Step</div>
                </div>
                <div style="background: rgba(99, 102, 241, 0.12); padding: 16px; border-radius: 12px; text-align: center; border: 1px solid rgba(99, 102, 241, 0.15);">
                    <div style="font-size: 28px; font-weight: 700; color: #60a5fa;">{stats['weather'].upper()}</div>
                    <div style="font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Weather</div>
                </div>
                <div style="background: rgba(99, 102, 241, 0.12); padding: 16px; border-radius: 12px; text-align: center; border: 1px solid rgba(99, 102, 241, 0.15);">
                    <div style="font-size: 28px; font-weight: 700; color: #34d399;">{stats['network_utilization']:.0f}%</div>
                    <div style="font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Utilization</div>
                </div>
            </div>
            
            <h3 style="margin-top: 24px; border-bottom: 1px solid rgba(99, 102, 241, 0.3); padding-bottom: 8px; font-size: 14px; font-weight: 600; color: #a78bfa;">🚂 TRAIN OPERATIONS</h3>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 12px;">
                <div style="background: rgba(59, 130, 246, 0.15); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <div style="font-size: 20px; font-weight: 700; color: #60a5fa;">{stats['active_trains']}</div>
                    <div style="font-size: 9px; opacity: 0.7; margin-top: 2px;">ACTIVE</div>
                </div>
                <div style="background: rgba(34, 197, 94, 0.15); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(34, 197, 94, 0.2);">
                    <div style="font-size: 20px; font-weight: 700; color: #4ade80;">{stats['arrived_trains']}</div>
                    <div style="font-size: 9px; opacity: 0.7; margin-top: 2px;">ARRIVED</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.15); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(239, 68, 68, 0.2);">
                    <div style="font-size: 20px; font-weight: 700; color: #f87171;">{stats['delayed_trains']}</div>
                    <div style="font-size: 9px; opacity: 0.7; margin-top: 2px;">DELAYED</div>
                </div>
                <div style="background: rgba(251, 191, 36, 0.15); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(251, 191, 36, 0.2);">
                    <div style="font-size: 20px; font-weight: 700; color: #fbbf24;">{stats['average_delay']:.1f}</div>
                    <div style="font-size: 9px; opacity: 0.7; margin-top: 2px;">AVG DELAY</div>
                </div>
                <div style="background: rgba(168, 85, 247, 0.15); padding: 12px 8px; border-radius: 10px; text-align: center; border: 1px solid rgba(168, 85, 247, 0.2);">
                    <div style="font-size: 20px; font-weight: 700; color: #c084fc;">{model.dispatcher.scheduled_departures.__len__()}</div>
                    <div style="font-size: 9px; opacity: 0.7; margin-top: 2px;">SCHEDULED</div>
                </div>
            </div>
            
            <h3 style="margin-top: 24px; border-bottom: 1px solid rgba(99, 102, 241, 0.3); padding-bottom: 8px; font-size: 14px; font-weight: 600; color: #a78bfa;">👥 PASSENGER SERVICES</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;">
                <div style="background: rgba(168, 85, 247, 0.15); padding: 14px; border-radius: 10px; text-align: center; border: 1px solid rgba(168, 85, 247, 0.2);">
                    <div style="font-size: 22px; font-weight: 700; color: #c084fc;">{stats['total_passengers_waiting']}</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">WAITING</div>
                </div>
                <div style="background: rgba(20, 184, 166, 0.15); padding: 14px; border-radius: 10px; text-align: center; border: 1px solid rgba(20, 184, 166, 0.2);">
                    <div style="font-size: 22px; font-weight: 700; color: #2dd4bf;">{stats['total_passengers_arrived']}</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">DELIVERED</div>
                </div>
            </div>
            
            <h3 style="margin-top: 24px; border-bottom: 1px solid rgba(99, 102, 241, 0.3); padding-bottom: 8px; font-size: 14px; font-weight: 600; color: #a78bfa;">⚡ SYSTEM STATUS</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 12px;">
                <div style="background: rgba(249, 115, 22, 0.15); padding: 14px; border-radius: 10px; text-align: center; border: 1px solid rgba(249, 115, 22, 0.2);">
                    <div style="font-size: 22px; font-weight: 700; color: #fb923c;">{stats['average_energy']:.0f}</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">AVG ENERGY</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.15); padding: 14px; border-radius: 10px; text-align: center; border: 1px solid rgba(239, 68, 68, 0.2);">
                    <div style="font-size: 22px; font-weight: 700; color: #f87171;">{stats['track_failures']}</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">FAILURES</div>
                </div>
                <div style="background: rgba(168, 85, 247, 0.15); padding: 14px; border-radius: 10px; text-align: center; border: 1px solid rgba(168, 85, 247, 0.2);">
                    <div style="font-size: 22px; font-weight: 700; color: #c084fc;">{stats['priority_conflicts']}</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">CONFLICTS</div>
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
        <div style="font-family: 'Inter', -apple-system, sans-serif; background: rgba(26, 26, 46, 0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); padding: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin-top: 16px; border: 1px solid rgba(99, 102, 241, 0.2);">
            <h3 style="margin-top: 0; color: #a78bfa; border-bottom: 1px solid rgba(99, 102, 241, 0.3); padding-bottom: 10px; font-size: 15px; font-weight: 600;">📍 MAP LEGEND</h3>
            
            <div style="margin-bottom: 16px;">
                <h4 style="color: #c4b5fd; margin-bottom: 10px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">🚂 Train Types</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(0, 212, 255, 0.15); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #00d4ff; border: 2px solid #ffffff; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 12px; color: #e4e4e7;">Passenger</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(255, 149, 0, 0.15); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #ff9500; border: 2px solid #ffffff; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 12px; color: #e4e4e7;">Cargo</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(191, 90, 242, 0.15); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #bf5af2; border: 2px solid #ffffff; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 12px; color: #e4e4e7;">Express</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(255, 69, 58, 0.15); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #ff453a; border: 2px solid #ffffff; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 12px; color: #e4e4e7;">Emergency</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-bottom: 16px;">
                <h4 style="color: #c4b5fd; margin-bottom: 10px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">🚦 Signals</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(48, 209, 88, 0.15); border-radius: 6px;">
                        <div style="width: 14px; height: 14px; background: #30d158; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px #30d15890;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Free</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(255, 214, 10, 0.15); border-radius: 6px;">
                        <div style="width: 14px; height: 14px; background: #ffd60a; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px #ffd60a90;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Busy</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(255, 69, 58, 0.15); border-radius: 6px;">
                        <div style="width: 14px; height: 14px; background: #ff453a; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px #ff453a90;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Failed</span>
                    </div>
                </div>
            </div>
            
            <div>
                <h4 style="color: #c4b5fd; margin-bottom: 10px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">📍 Infrastructure</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #ffffff; border: 2px solid #00d4ff; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Station</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(61, 79, 95, 0.25); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #3d4f5f; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Track</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 6px; background: rgba(13, 17, 23, 0.5); border-radius: 6px;">
                        <div style="width: 18px; height: 18px; background: #0d1117; border: 1px solid #30363d; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 11px; color: #e4e4e7;">Empty</span>
                    </div>
                </div>
            </div>
        </div>
        """
        return html


class CustomCanvasGrid(CanvasGrid):
    """Custom CanvasGrid that allows overriding the grid line color."""
    
    def __init__(self, portrayal_method, grid_width, grid_height, 
                 canvas_width, canvas_height, grid_color="#333333"):
        super().__init__(portrayal_method, grid_width, grid_height, 
                        canvas_width, canvas_height)
        self.grid_color = grid_color
        
        # Inject JavaScript to override the grid line color
        # We need to patch after the CanvasModule is initialized
        self.js_code += f"""
        // Override grid line color after initialization
        (function() {{
            var gridColor = "{self.grid_color}";
            var gridWidth = {grid_width};
            var gridHeight = {grid_height};
            var canvasWidth = {canvas_width};
            var canvasHeight = {canvas_height};
            
            // Wait for elements to be created, then patch
            var patchInterval = setInterval(function() {{
                // Find all CanvasModule instances in elements array
                if (typeof elements !== 'undefined' && elements.length > 0) {{
                    for (var i = 0; i < elements.length; i++) {{
                        var el = elements[i];
                        if (el && el.gridViz && el.gridViz.drawGridLines && !el._gridColorPatched) {{
                            el._gridColorPatched = true;
                            
                            // Get the canvas context
                            var context = el.context;
                            var cellWidth = Math.floor(canvasWidth / gridWidth);
                            var cellHeight = Math.floor(canvasHeight / gridHeight);
                            
                            // Override drawGridLines on this specific instance
                            el.gridViz.drawGridLines = function() {{
                                context.beginPath();
                                context.strokeStyle = gridColor;
                                var maxX = cellWidth * gridWidth;
                                var maxY = cellHeight * gridHeight;
                                
                                for (var y = 0; y <= maxY; y += cellHeight) {{
                                    context.moveTo(0, y + 0.5);
                                    context.lineTo(maxX, y + 0.5);
                                }}
                                for (var x = 0; x <= maxX; x += cellWidth) {{
                                    context.moveTo(x + 0.5, 0);
                                    context.lineTo(x + 0.5, maxY);
                                }}
                                context.stroke();
                            }};
                            
                            clearInterval(patchInterval);
                            console.log("Grid color patched to: " + gridColor);
                        }}
                    }}
                }}
            }}, 100);
            
            // Clear interval after 10 seconds to prevent infinite loop
            setTimeout(function() {{ clearInterval(patchInterval); }}, 10000);
        }})();
        """
        
    def render(self, model):
        return super().render(model)


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
        <div style="font-family: 'Inter', -apple-system, sans-serif; background: rgba(26, 26, 46, 0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); padding: 20px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin-top: 16px; border: 1px solid rgba(99, 102, 241, 0.2); max-height: 400px; overflow: hidden;">
            <h3 style="margin-top: 0; color: #a78bfa; border-bottom: 1px solid rgba(99, 102, 241, 0.3); padding-bottom: 10px; font-size: 15px; font-weight: 600;">🕒 REAL-TIME SCHEDULE (Step: {current_time})</h3>
            
            <h4 style="color: #c4b5fd; margin-top: 12px; margin-bottom: 8px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">🚂 Active Trains ({len(active_trains)})</h4>
            <div style="max-height: 160px; overflow-y: auto; margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 11px; table-layout: fixed;">
                    <thead>
                        <tr style="background: rgba(99, 102, 241, 0.2); color: #e4e4e7;">
                            <th style="padding: 8px 6px; text-align: left; width: 12%; font-weight: 600;">ID</th>
                            <th style="padding: 8px 6px; text-align: left; width: 15%; font-weight: 600;">Type</th>
                            <th style="padding: 8px 6px; text-align: left; width: 15%; font-weight: 600;">State</th>
                            <th style="padding: 8px 6px; text-align: left; width: 20%; font-weight: 600;">Position</th>
                            <th style="padding: 8px 6px; text-align: left; width: 23%; font-weight: 600;">Dest</th>
                            <th style="padding: 8px 6px; text-align: right; width: 15%; font-weight: 600;">Energy</th>
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
            <tr style="border-bottom: 1px solid rgba(99, 102, 241, 0.1); transition: background 0.2s;" onmouseover="this.style.background='rgba(99, 102, 241, 0.1)'" onmouseout="this.style.background='transparent'">
                <td style="padding: 6px; color: {type_color}; font-weight: 600;">{train.unique_id}</td>
                <td style="padding: 6px; color: #a5b4fc;">{train.train_type.value.title()[:4]}</td>
                <td style="padding: 6px; color: {state_color}; font-weight: 600;">{train.state.value.upper()[:4]}</td>
                <td style="padding: 6px; font-size: 10px; color: #94a3b8;">({train.current_position[0]},{train.current_position[1]})</td>
                <td style="padding: 6px; font-size: 10px; color: #94a3b8;">({train.destination[0]},{train.destination[1]})</td>
                <td style="padding: 6px; text-align: right; color: {energy_color}; font-weight: 600;">{energy_percent:.0f}%</td>
            </tr>
            """
        
        if not active_trains:
            html += '<tr><td colspan="6" style="padding: 12px; text-align: center; color: #64748b; font-style: italic;">No active trains currently.</td></tr>'
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <h4 style="color: #c4b5fd; margin-top: 12px; margin-bottom: 8px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">📅 Upcoming Departures</h4>
            <div style="overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 11px; table-layout: fixed;">
                    <thead>
                        <tr style="background: rgba(99, 102, 241, 0.15); color: #e4e4e7;">
                            <th style="padding: 8px 6px; text-align: left; width: 35%; font-weight: 600;">Time</th>
                            <th style="padding: 8px 6px; text-align: left; width: 25%; font-weight: 600;">Type</th>
                            <th style="padding: 8px 6px; text-align: left; width: 40%; font-weight: 600;">Route</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Display next 5 departures
        for d in departures[:5]:
            time_until = d['departure_time'] - current_time
            time_color = '#e74c3c' if time_until <= 5 else ('#f39c12' if time_until <= 15 else '#27ae60')
            
            html += f"""
            <tr style="border-bottom: 1px solid rgba(99, 102, 241, 0.1);">
                <td style="padding: 6px; color: {time_color}; font-weight: 600;">{d['departure_time']} (T-{time_until})</td>
                <td style="padding: 6px; color: #a5b4fc;">{d['train_type'].value.title()}</td>
                <td style="padding: 6px; font-size: 10px; color: #94a3b8;">{d['origin']} → {d['destination']}</td>
            </tr>
            """
        
        if not departures:
            html += '<tr><td colspan="3" style="padding: 12px; text-align: center; color: #64748b; font-style: italic;">No departures scheduled.</td></tr>'

        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        return html


def agent_portrayal(agent):
    """
    Enhanced agent portrayal with high-contrast colors.
    Vibrant, saturated colors for maximum visibility on dark background.
    Color-blind friendly palette with good contrast ratios.
    """
    if isinstance(agent, TrackAgent):
        # Railway track - GREEN for high visibility and clear railway paths
        return {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Layer": 0,
            "Color": "#228B22",  # Forest Green - clear railway path visibility
            "stroke_color": "#000000",  # Black border for definition
            "stroke": 1,
            "text": "",
            "text_color": "#ffffff"
        }
    
    elif isinstance(agent, TrainAgent):
        # High-contrast train visualization with vibrant colors
        portrayal = {
            "Shape": "rect",
            "w": 0.85,
            "h": 0.85,
            "Filled": "true",
            "Layer": 3,
            "stroke_color": "#ffffff",  # White border for visibility
            "stroke": 2,
            "text": "",
            "text_color": "white"
        }
        
        # VIBRANT colors for high contrast on dark background
        if agent.train_type == TrainType.PASSENGER:
            portrayal["Color"] = "#00d4ff"  # Bright Cyan
        elif agent.train_type == TrainType.CARGO:
            portrayal["Color"] = "#ff9500"  # Bright Orange
        elif agent.train_type == TrainType.EXPRESS:
            portrayal["Color"] = "#bf5af2"  # Bright Magenta/Purple
        elif agent.train_type == TrainType.EMERGENCY:
            portrayal["Color"] = "#ff453a"  # Bright Red
        
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
        # High-visibility signal lights with glow effect
        portrayal = {
            "Shape": "circle",
            "r": 0.45,
            "Filled": "true",
            "Layer": 2,
        }
        
        # Bright, saturated traffic light colors
        if agent.track_status == "failed":
            portrayal["Color"] = "#ff453a"  # Bright Red
            portrayal["text"] = "✖"
            portrayal["text_color"] = "white"
        elif agent.track_occupied:
            portrayal["Color"] = "#ffd60a"  # Bright Yellow
            portrayal["text"] = "●"
            portrayal["text_color"] = "#1a1a1a"
        else:
            portrayal["Color"] = "#30d158"  # Bright Green
            portrayal["text"] = "●"
            portrayal["text_color"] = "white"
        
        # White border for glow effect
        portrayal["stroke_color"] = "#ffffff"
        portrayal["stroke"] = 1
        
        return portrayal
    
    elif isinstance(agent, StationAgent):
        # HIGH-VISIBILITY station - bright white/cyan for maximum contrast
        portrayal = {
            "Shape": "rect",
            "w": 0.92,
            "h": 0.92,
            "Filled": "true",
            "Layer": 1,
            "Color": "#ffffff",  # Bright white for maximum visibility
            "stroke_color": "#00d4ff",  # Cyan glow border
            "stroke": 3,
            "text": "",
            "text_color": "#0d1117"
        }
        
        # Highlight stations with many waiting passengers
        if hasattr(agent, 'waiting_passengers') and agent.waiting_passengers > 50:
            portrayal["Color"] = "#ffd60a"  # Yellow when crowded
            portrayal["stroke_color"] = "#ff9500"  # Orange border
            portrayal["stroke"] = 4
        
        return portrayal
    
    # Empty cells - Dark background to match the dark theme
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "Color": "#1a1a2e",  # Dark blue-gray - matches dark theme
        "stroke_color": "#2a2a3e",  # Subtle dark gray grid lines
        "stroke": 1,
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
    grid = CustomCanvasGrid(
        agent_portrayal,
        50, 50,
        600, 600  # Larger canvas for better visibility
    )
    
    # Create style injector for global CSS (must be first!)
    style_injector = StyleInjectorElement()
    
    # Create statistics and legend
    stats_element = NetworkStatsElement()
    legend_element = LegendElement()
    schedule_element = ScheduleElement()
    
    # Enhanced charts with better styling
    train_chart = ChartModule(
        [
            {"Label": "Active Trains", "Color": "#60a5fa"},
            {"Label": "Delayed Trains", "Color": "#f87171"},
            {"Label": "Total Arrivals", "Color": "#4ade80"},
        ],
        data_collector_name="datacollector"
    )
    
    delay_chart = ChartModule(
        [
            {"Label": "Average Delay", "Color": "#fb923c"},
        ],
        data_collector_name="datacollector"
    )
    
    energy_chart = ChartModule(
        [
            {"Label": "Total Energy", "Color": "#fbbf24"},
        ],
        data_collector_name="datacollector"
    )
    
    passenger_chart = ChartModule(
        [
            {"Label": "Waiting Passengers", "Color": "#c084fc"},
        ],
        data_collector_name="datacollector"
    )
    
    # Create server with all elements - style injector FIRST for CSS override
    server = ModularServer(
        RailwayNetworkModel,
        [style_injector, grid, stats_element, schedule_element, legend_element, train_chart, delay_chart, energy_chart, passenger_chart],
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
