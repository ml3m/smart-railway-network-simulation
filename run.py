#!/usr/bin/env python3
"""
Smart Railway Network Simulation - Main Entry Point

This script provides multiple modes for running the railway simulation:
1. Interactive web visualization (default)
2. Batch simulation with data export
3. Comparative analysis across different configurations
"""

import argparse
import sys
from model import RailwayNetworkModel
from visualization import (
    create_server, 
    export_results, 
    print_summary_report,
    plot_simulation_results
)
import matplotlib.pyplot as plt


def run_interactive_visualization(host='127.0.0.1', port=8521):
    """
    Launch the interactive MESA visualization server.
    
    Args:
        host: Server host address
        port: Server port number
    """
    print("\n" + "="*60)
    print("🚄 SMART RAILWAY NETWORK SIMULATION")
    print("="*60)
    print("\n🌐 Starting interactive visualization server...")
    print(f"   URL: http://{host}:{port}")
    print("\n📖 Instructions:")
    print("   - Open the URL in your web browser")
    print("   - Click 'Start' to begin simulation")
    print("   - Use 'Step' for manual control")
    print("   - Click 'Reset' to restart with new parameters")
    print("\n🎨 Visualization Legend:")
    print("   Trains:")
    print("     🔵 Blue = Passenger    🟠 Orange = Cargo")
    print("     🟣 Purple = Express    🔴 Red = Emergency")
    print("   Signals:")
    print("     🟢 Green = Free       🟡 Yellow = Occupied")
    print("     🔴 Red = Failed")
    print("   Stations:")
    print("     🚉 Teal squares with station icon")
    print("\n   Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        server = create_server()
        server.port = port
        server.launch()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        sys.exit(0)


def run_batch_simulation(steps=500, network_complexity="medium", 
                        num_trains=10, num_stations=5,
                        enable_weather=True, enable_failures=True, 
                        enable_passengers=True, export_prefix='railway_sim'):
    """
    Run a batch simulation without visualization and export results.
    
    Args:
        steps: Number of simulation steps
        network_complexity: Network complexity level
        num_trains: Number of trains
        num_stations: Number of stations
        enable_weather: Enable weather effects
        enable_failures: Enable random track failures
        enable_passengers: Enable passenger simulation
        export_prefix: Prefix for exported files
    """
    print("\n" + "="*60)
    print("🚄 SMART RAILWAY NETWORK SIMULATION - BATCH MODE")
    print("="*60)
    print(f"\n⚙️  Configuration:")
    print(f"   Network Complexity: {network_complexity}")
    print(f"   Number of Trains: {num_trains}")
    print(f"   Number of Stations: {num_stations}")
    print(f"   Simulation Steps: {steps}")
    print(f"   Weather Effects: {'Enabled' if enable_weather else 'Disabled'}")
    print(f"   Track Failures: {'Enabled' if enable_failures else 'Disabled'}")
    print(f"   Passenger Simulation: {'Enabled' if enable_passengers else 'Disabled'}")
    print("\n🚀 Starting simulation...")
    
    # Create and run model
    model = RailwayNetworkModel(
        width=50,
        height=50,
        num_trains=num_trains,
        num_stations=num_stations,
        network_complexity=network_complexity,
        enable_weather=enable_weather,
        enable_failures=enable_failures,
        enable_passengers=enable_passengers
    )
    
    # Run simulation with progress indicator
    for i in range(steps):
        if i % 50 == 0:
            print(f"   Step {i}/{steps}...")
        model.step()
        if not model.running:
            print(f"   Simulation completed at step {i}")
            break
    
    print("\n✅ Simulation complete!")
    
    # Print summary
    print_summary_report(model)
    
    # Export results
    print("📊 Exporting results...")
    export_results(model, export_prefix)
    
    # Show plots
    print("\n📈 Generating visualizations...")
    fig = plot_simulation_results(model)
    plt.show()
    
    return model


def run_comparative_analysis(configurations, steps=500):
    """
    Run multiple simulations with different configurations and compare results.
    
    Args:
        configurations: List of configuration dictionaries
        steps: Number of steps for each simulation
    """
    print("\n" + "="*60)
    print("🚄 SMART RAILWAY NETWORK - COMPARATIVE ANALYSIS")
    print("="*60)
    print(f"\n🔬 Running {len(configurations)} different configurations...")
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configurations)}: {config.get('name', f'Config {i+1}')}")
        print(f"{'='*60}")
        
        # Remove name from config if present
        config_copy = config.copy()
        config_name = config_copy.pop('name', f'config_{i+1}')
        
        # Create model
        model = RailwayNetworkModel(
            width=50,
            height=50,
            **config_copy
        )
        
        # Run simulation
        print(f"Running simulation...")
        model.run_model(steps)
        
        # Collect results
        stats = model.get_summary_stats()
        stats['config_name'] = config_name
        results.append(stats)
        
        # Export
        export_results(model, f'comparison_{config_name}')
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 COMPARATIVE RESULTS")
    print("="*60)
    
    print(f"\n{'Configuration':<20} {'Arrivals':<10} {'Avg Delay':<12} {'Energy':<12} {'Failures':<10}")
    print("-" * 64)
    
    for result in results:
        print(f"{result['config_name']:<20} "
              f"{result['arrived_trains']:<10} "
              f"{result['average_delay']:<12.2f} "
              f"{result['average_energy']:<12.1f} "
              f"{result['track_failures']:<10}")
    
    print("\n" + "="*60)
    
    # Create comparison plot
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Configuration Comparison', fontsize=16, fontweight='bold')
    
    # Plot comparisons
    df.plot(x='config_name', y='arrived_trains', kind='bar', ax=axes[0, 0], 
            color='#2ecc71', legend=False)
    axes[0, 0].set_title('Arrived Trains')
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    df.plot(x='config_name', y='average_delay', kind='bar', ax=axes[0, 1], 
            color='#e67e22', legend=False)
    axes[0, 1].set_title('Average Delay')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    df.plot(x='config_name', y='average_energy', kind='bar', ax=axes[1, 0], 
            color='#f39c12', legend=False)
    axes[1, 0].set_title('Average Energy')
    axes[1, 0].set_xlabel('')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    df.plot(x='config_name', y='track_failures', kind='bar', ax=axes[1, 1], 
            color='#c0392b', legend=False)
    axes[1, 1].set_title('Track Failures')
    axes[1, 1].set_xlabel('')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Smart Railway Network Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive visualization (default)
  python run.py
  
  # Run batch simulation with 1000 steps
  python run.py --batch --steps 1000
  
  # Run with specific configuration
  python run.py --batch --trains 20 --stations 8 --complexity complex
  
  # Run comparative analysis
  python run.py --compare
  
  # Disable certain features
  python run.py --batch --no-weather --no-failures
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode without visualization')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparative analysis with multiple configurations')
    
    # Simulation parameters
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of simulation steps (default: 500)')
    parser.add_argument('--trains', type=int, default=10,
                       help='Number of trains (default: 10)')
    parser.add_argument('--stations', type=int, default=5,
                       help='Number of stations (default: 5)')
    parser.add_argument('--complexity', choices=['simple', 'medium', 'complex'],
                       default='medium',
                       help='Network complexity (default: medium)')
    
    # Feature toggles
    parser.add_argument('--no-weather', action='store_true',
                       help='Disable weather effects')
    parser.add_argument('--no-failures', action='store_true',
                       help='Disable track failures')
    parser.add_argument('--no-passengers', action='store_true',
                       help='Disable passenger simulation')
    
    # Server parameters
    parser.add_argument('--host', default='127.0.0.1',
                       help='Server host for interactive mode (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8521,
                       help='Server port for interactive mode (default: 8521)')
    
    # Output
    parser.add_argument('--output', default='railway_sim',
                       help='Output file prefix for batch mode (default: railway_sim)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparative analysis
        configurations = [
            {
                'name': 'low_traffic',
                'num_trains': 5,
                'num_stations': 5,
                'network_complexity': 'simple',
                'enable_weather': True,
                'enable_failures': True,
                'enable_passengers': True
            },
            {
                'name': 'medium_traffic',
                'num_trains': 10,
                'num_stations': 5,
                'network_complexity': 'medium',
                'enable_weather': True,
                'enable_failures': True,
                'enable_passengers': True
            },
            {
                'name': 'high_traffic',
                'num_trains': 20,
                'num_stations': 8,
                'network_complexity': 'complex',
                'enable_weather': True,
                'enable_failures': True,
                'enable_passengers': True
            },
            {
                'name': 'ideal_conditions',
                'num_trains': 15,
                'num_stations': 6,
                'network_complexity': 'medium',
                'enable_weather': False,
                'enable_failures': False,
                'enable_passengers': True
            },
        ]
        run_comparative_analysis(configurations, args.steps)
        
    elif args.batch:
        # Run batch simulation
        run_batch_simulation(
            steps=args.steps,
            network_complexity=args.complexity,
            num_trains=args.trains,
            num_stations=args.stations,
            enable_weather=not args.no_weather,
            enable_failures=not args.no_failures,
            enable_passengers=not args.no_passengers,
            export_prefix=args.output
        )
    else:
        # Run interactive visualization
        run_interactive_visualization(args.host, args.port)


if __name__ == '__main__':
    main()

