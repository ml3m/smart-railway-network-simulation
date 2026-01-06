#!/usr/bin/env python3
"""
Smart Railway Network Simulation - Main Entry Point

This script provides multiple modes for running the railway simulation:
1. Interactive web visualization (default)
2. Batch simulation with data export
3. Comparative analysis across different configurations
4. Quick demo mode

Usage:
    python run.py                    # Interactive visualization
    python run.py --batch            # Batch simulation
    python run.py --demo             # Quick demo mode
    python run.py --compare          # Comparative analysis
    python run.py --preset demo      # Use preset configuration
"""

import argparse
import sys
import time
from pathlib import Path

# Check for optional dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from model import RailwayNetworkModel
from agents import TrainState, TrainType
from config import SimulationSettings, get_preset, list_presets, PRESETS


# Initialize console for rich output
console = Console() if RICH_AVAILABLE else None


def print_header():
    """Print beautiful header."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]🚄 SMART RAILWAY NETWORK SIMULATION[/bold cyan]\n"
            "[dim]A sophisticated multi-agent railway network simulator[/dim]",
            border_style="cyan",
            padding=(1, 2)
        ))
    else:
        print("\n" + "="*60)
        print("🚄 SMART RAILWAY NETWORK SIMULATION")
        print("="*60)


def print_config(config: SimulationSettings):
    """Print configuration details."""
    if RICH_AVAILABLE:
        table = Table(title="Configuration", box=box.ROUNDED, show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Grid Size", f"{config.width} × {config.height}")
        table.add_row("Trains", str(config.num_trains))
        table.add_row("Stations", str(config.num_stations))
        table.add_row("Network", config.network_complexity.title())
        table.add_row("Weather", "✓ Enabled" if config.enable_weather else "✗ Disabled")
        table.add_row("Failures", "✓ Enabled" if config.enable_failures else "✗ Disabled")
        table.add_row("Passengers", "✓ Enabled" if config.enable_passengers else "✗ Disabled")
        
        console.print(table)
    else:
        print(f"\n⚙️  Configuration:")
        print(f"   Grid: {config.width}×{config.height}")
        print(f"   Trains: {config.num_trains}")
        print(f"   Stations: {config.num_stations}")
        print(f"   Network: {config.network_complexity}")


def print_results(stats: dict):
    """Print simulation results."""
    if RICH_AVAILABLE:
        # Train statistics table
        train_table = Table(title="🚂 Train Statistics", box=box.ROUNDED)
        train_table.add_column("Metric", style="cyan")
        train_table.add_column("Value", style="green", justify="right")
        
        train_table.add_row("Total Trains", str(stats['total_trains']))
        train_table.add_row("Active", str(stats['active_trains']))
        train_table.add_row("Moving", str(stats.get('moving_trains', 'N/A')))
        train_table.add_row("Arrived", str(stats['arrived_trains']))
        train_table.add_row("Delayed", str(stats['delayed_trains']))
        train_table.add_row("Average Delay", f"{stats['average_delay']:.2f} steps")
        
        # Passenger statistics table
        passenger_table = Table(title="👥 Passenger Statistics", box=box.ROUNDED)
        passenger_table.add_column("Metric", style="cyan")
        passenger_table.add_column("Value", style="green", justify="right")
        
        passenger_table.add_row("Waiting", str(stats['total_passengers_waiting']))
        passenger_table.add_row("Delivered", str(stats['total_passengers_arrived']))
        
        # System statistics table
        system_table = Table(title="⚡ System Statistics", box=box.ROUNDED)
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green", justify="right")
        
        system_table.add_row("Total Steps", str(stats['total_steps']))
        system_table.add_row("Weather", stats['weather'].upper())
        system_table.add_row("Network Utilization", f"{stats['network_utilization']:.1f}%")
        system_table.add_row("Track Failures", str(stats['track_failures']))
        system_table.add_row("Priority Conflicts", str(stats['priority_conflicts']))
        system_table.add_row("Average Energy", f"{stats['average_energy']:.0f}")
        
        console.print()
        console.print(train_table)
        console.print()
        console.print(passenger_table)
        console.print()
        console.print(system_table)
    else:
        print("\n📊 SIMULATION RESULTS")
        print("="*40)
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Total Trains: {stats['total_trains']}")
        print(f"  Active: {stats['active_trains']}")
        print(f"  Arrived: {stats['arrived_trains']}")
        print(f"  Delayed: {stats['delayed_trains']}")
        print(f"  Average Delay: {stats['average_delay']:.2f}")
        print(f"  Passengers Waiting: {stats['total_passengers_waiting']}")
        print(f"  Passengers Delivered: {stats['total_passengers_arrived']}")


def run_interactive_visualization(config: SimulationSettings, host='127.0.0.1', port=8521):
    """Launch the interactive MESA visualization server."""
    print_header()
    
    if RICH_AVAILABLE:
        console.print("\n[bold green]🌐 Starting interactive visualization server...[/bold green]")
        console.print(f"   [dim]URL:[/dim] [link=http://{host}:{port}]http://{host}:{port}[/link]")
        console.print()
        
        instruction_panel = Panel(
            "[bold]Instructions:[/bold]\n"
            "• Open the URL in your web browser\n"
            "• Click 'Start' to begin simulation\n"
            "• Use 'Step' for manual control\n"
            "• Click 'Reset' to restart\n"
            "• Press Ctrl+C to stop server",
            title="📖 How to Use",
            border_style="blue"
        )
        console.print(instruction_panel)
        
        legend_panel = Panel(
            "[bold]Trains:[/bold]\n"
            "  [blue]●[/blue] Blue = Passenger    [yellow]●[/yellow] Orange = Cargo\n"
            "  [magenta]●[/magenta] Purple = Express   [red]●[/red] Red = Emergency\n\n"
            "[bold]Signals:[/bold]\n"
            "  [green]●[/green] Green = Free    [yellow]●[/yellow] Yellow = Occupied    [red]●[/red] Red = Failed\n\n"
            "[bold]Stations:[/bold]\n"
            "  [cyan]■[/cyan] Teal squares",
            title="🎨 Legend",
            border_style="cyan"
        )
        console.print(legend_panel)
    else:
        print(f"\n🌐 Starting interactive visualization server...")
        print(f"   URL: http://{host}:{port}")
        print("\n📖 Instructions:")
        print("   - Open the URL in your web browser")
        print("   - Click 'Start' to begin simulation")
        print("   - Press Ctrl+C to stop the server")
    
    try:
        from visualization import create_server
        server = create_server()
        server.port = port
        server.launch()
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n\n[bold yellow]👋 Server stopped. Goodbye![/bold yellow]")
        else:
            print("\n\n👋 Server stopped. Goodbye!")
        sys.exit(0)


def run_batch_simulation(config: SimulationSettings, steps: int = 500, 
                        export_prefix: str = 'railway_sim', show_progress: bool = True):
    """Run a batch simulation without visualization."""
    print_header()
    print_config(config)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]🚀 Running batch simulation for {steps} steps...[/bold]\n")
    else:
        print(f"\n🚀 Running batch simulation for {steps} steps...\n")
    
    # Create model
    model = RailwayNetworkModel(
        width=config.width,
        height=config.height,
        num_trains=config.num_trains,
        num_stations=config.num_stations,
        network_complexity=config.network_complexity,
        enable_weather=config.enable_weather,
        enable_failures=config.enable_failures,
        enable_passengers=config.enable_passengers
    )
    
    # Run simulation with progress bar
    if RICH_AVAILABLE and show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Simulating...", total=steps)
            
            for i in range(steps):
                model.step()
                progress.update(task, advance=1)
                
                if not model.running:
                    break
    else:
        for i in range(steps):
            if i % 50 == 0:
                print(f"   Step {i}/{steps}...")
            model.step()
            if not model.running:
                break
    
    # Get and display results
    stats = model.get_summary_stats()
    print_results(stats)
    
    # Export results
    if RICH_AVAILABLE:
        console.print(f"\n[bold]📊 Exporting results...[/bold]")
    else:
        print("\n📊 Exporting results...")
    
    try:
        from visualization import export_results, plot_simulation_results
        import matplotlib.pyplot as plt
        
        export_results(model, export_prefix)
        
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Results exported to {export_prefix}_*.csv")
        else:
            print(f"✓ Results exported to {export_prefix}_*.csv")
        
        # Show plots
        fig = plot_simulation_results(model)
        plt.show()
        
    except ImportError as e:
        if RICH_AVAILABLE:
            console.print(f"[yellow]⚠️ Could not export visualizations: {e}[/yellow]")
        else:
            print(f"⚠️ Could not export visualizations: {e}")
    
    return model


def run_demo(steps: int = 100):
    """Run a quick demo simulation."""
    print_header()
    
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]🎮 DEMO MODE[/bold cyan]")
        console.print("[dim]Running a quick demonstration of the simulation[/dim]\n")
    else:
        print("\n🎮 DEMO MODE")
        print("Running a quick demonstration...\n")
    
    # Use demo preset
    config = get_preset('demo')
    
    # Create model
    model = RailwayNetworkModel(
        width=config.width,
        height=config.height,
        num_trains=config.num_trains,
        num_stations=config.num_stations,
        network_complexity=config.network_complexity,
        enable_weather=config.enable_weather,
        enable_failures=config.enable_failures,
        enable_passengers=config.enable_passengers
    )
    
    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] Created model with {len(model.track_positions)} track cells")
        console.print(f"[green]✓[/green] {len(model.signal_positions)} signals placed")
        console.print(f"[green]✓[/green] {len(model.station_positions)} stations created")
        console.print(f"[green]✓[/green] {len(model.get_all_trains())} trains initialized")
        console.print()
    
    # Run simulation with live updates
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Running demo...", total=steps)
            
            for i in range(steps):
                model.step()
                progress.update(task, advance=1)
                
                # Print milestones
                if i == steps // 4:
                    progress.console.print("[dim]   Quarter way through...[/dim]")
                elif i == steps // 2:
                    progress.console.print("[dim]   Halfway there...[/dim]")
                elif i == 3 * steps // 4:
                    progress.console.print("[dim]   Almost done...[/dim]")
    else:
        for i in range(steps):
            model.step()
            if i % 20 == 0:
                trains = model.get_all_trains()
                moving = len([t for t in trains if t.state == TrainState.MOVING])
                print(f"   Step {i}: {moving} trains moving")
    
    # Show results
    stats = model.get_summary_stats()
    print_results(stats)
    
    if RICH_AVAILABLE:
        console.print("\n[bold green]✅ Demo complete![/bold green]")
        console.print("\n[dim]To run the full interactive visualization:[/dim]")
        console.print("   [cyan]python run.py[/cyan]")
    else:
        print("\n✅ Demo complete!")
        print("\nTo run the full interactive visualization:")
        print("   python run.py")
    
    return model


def run_comparative_analysis(configurations: list, steps: int = 500):
    """Run multiple simulations and compare results."""
    print_header()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]🔬 COMPARATIVE ANALYSIS[/bold cyan]")
        console.print(f"[dim]Running {len(configurations)} configurations for comparison[/dim]\n")
    else:
        print(f"\n🔬 COMPARATIVE ANALYSIS")
        print(f"Running {len(configurations)} configurations...\n")
    
    results = []
    
    for i, config in enumerate(configurations):
        config_name = config.pop('name', f'config_{i+1}')
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold]Configuration {i+1}/{len(configurations)}: {config_name}[/bold]")
        else:
            print(f"\nConfiguration {i+1}/{len(configurations)}: {config_name}")
        
        # Create model
        model = RailwayNetworkModel(width=50, height=50, **config)
        
        # Run simulation
        model.run_model(steps)
        
        # Collect results
        stats = model.get_summary_stats()
        stats['config_name'] = config_name
        results.append(stats)
    
    # Display comparison
    if RICH_AVAILABLE:
        table = Table(title="📊 Comparison Results", box=box.DOUBLE)
        table.add_column("Configuration", style="cyan")
        table.add_column("Arrivals", justify="right")
        table.add_column("Avg Delay", justify="right")
        table.add_column("Avg Energy", justify="right")
        table.add_column("Failures", justify="right")
        
        for result in results:
            table.add_row(
                result['config_name'],
                str(result['arrived_trains']),
                f"{result['average_delay']:.2f}",
                f"{result['average_energy']:.0f}",
                str(result['track_failures'])
            )
        
        console.print()
        console.print(table)
    else:
        print("\n" + "="*64)
        print("📊 COMPARISON RESULTS")
        print("="*64)
        print(f"{'Configuration':<20} {'Arrivals':<10} {'Avg Delay':<12} {'Energy':<12} {'Failures':<10}")
        print("-" * 64)
        for result in results:
            print(f"{result['config_name']:<20} "
                  f"{result['arrived_trains']:<10} "
                  f"{result['average_delay']:<12.2f} "
                  f"{result['average_energy']:<12.0f} "
                  f"{result['track_failures']:<10}")
    
    return results


def list_available_presets():
    """List available configuration presets."""
    if RICH_AVAILABLE:
        table = Table(title="📋 Available Presets", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="dim")
        
        for name, description in list_presets().items():
            table.add_row(name, description)
        
        console.print(table)
    else:
        print("\n📋 Available Presets:")
        print("-" * 50)
        for name, description in list_presets().items():
            print(f"  {name}: {description}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Smart Railway Network Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Interactive visualization
  python run.py --batch             # Batch simulation
  python run.py --demo              # Quick demo
  python run.py --preset large_network --batch
  python run.py --compare           # Compare configurations
  python run.py --list-presets      # List available presets
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode without visualization')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo simulation')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparative analysis')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available configuration presets')
    
    # Configuration
    parser.add_argument('--preset', type=str, default=None,
                       help='Use a preset configuration')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of simulation steps (default: 500)')
    parser.add_argument('--trains', type=int, default=None,
                       help='Number of trains')
    parser.add_argument('--stations', type=int, default=None,
                       help='Number of stations')
    parser.add_argument('--complexity', choices=['simple', 'medium', 'complex'],
                       default=None, help='Network complexity')
    
    # Feature toggles
    parser.add_argument('--no-weather', action='store_true',
                       help='Disable weather effects')
    parser.add_argument('--no-failures', action='store_true',
                       help='Disable track failures')
    parser.add_argument('--no-passengers', action='store_true',
                       help='Disable passenger simulation')
    
    # Server parameters
    parser.add_argument('--host', default='127.0.0.1',
                       help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8521,
                       help='Server port (default: 8521)')
    
    # Output
    parser.add_argument('--output', default='railway_sim',
                       help='Output file prefix (default: railway_sim)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        list_available_presets()
        return
    
    # Load configuration
    if args.preset:
        try:
            config = get_preset(args.preset)
        except ValueError as e:
            print(f"Error: {e}")
            list_available_presets()
            sys.exit(1)
    else:
        config = SimulationSettings()
    
    # Override with command line arguments
    if args.trains is not None:
        config.num_trains = args.trains
    if args.stations is not None:
        config.num_stations = args.stations
    if args.complexity is not None:
        config.network_complexity = args.complexity
    if args.no_weather:
        config.enable_weather = False
    if args.no_failures:
        config.enable_failures = False
    if args.no_passengers:
        config.enable_passengers = False
    
    # Run selected mode
    if args.demo:
        run_demo(args.steps)
    elif args.compare:
        configurations = [
            {'name': 'low_traffic', 'num_trains': 5, 'num_stations': 5,
             'network_complexity': 'simple', 'enable_weather': True,
             'enable_failures': True, 'enable_passengers': True},
            {'name': 'medium_traffic', 'num_trains': 10, 'num_stations': 5,
             'network_complexity': 'medium', 'enable_weather': True,
             'enable_failures': True, 'enable_passengers': True},
            {'name': 'high_traffic', 'num_trains': 20, 'num_stations': 8,
             'network_complexity': 'complex', 'enable_weather': True,
             'enable_failures': True, 'enable_passengers': True},
            {'name': 'ideal_conditions', 'num_trains': 15, 'num_stations': 6,
             'network_complexity': 'medium', 'enable_weather': False,
             'enable_failures': False, 'enable_passengers': True},
        ]
        run_comparative_analysis(configurations, args.steps)
    elif args.batch:
        run_batch_simulation(config, args.steps, args.output, not args.quiet)
    else:
        run_interactive_visualization(config, args.host, args.port)


if __name__ == '__main__':
    main()
