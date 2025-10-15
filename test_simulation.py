#!/usr/bin/env python3
"""
Quick test script to verify simulation functionality.
"""

from model import RailwayNetworkModel
from agents import TrainState, TrainType
import sys


def test_simulation():
    """Run a quick test simulation to verify all features work."""
    
    print("="*60)
    print("🧪 TESTING RAILWAY SIMULATION")
    print("="*60)
    
    # Create model
    print("\n1️⃣  Creating simulation model...")
    model = RailwayNetworkModel(
        width=30,
        height=30,
        num_trains=5,
        num_stations=3,
        network_complexity="simple",
        enable_weather=True,
        enable_failures=False,  # Disable for testing
        enable_passengers=True
    )
    print(f"   ✓ Model created with {len(model.track_positions)} track cells")
    print(f"   ✓ {len(model.signal_positions)} signals placed")
    print(f"   ✓ {len(model.station_positions)} stations created")
    print(f"   ✓ {len(model.get_all_trains())} trains initialized")
    
    # Test train initialization
    print("\n2️⃣  Testing train initialization...")
    trains = model.get_all_trains()
    for train in trains[:3]:  # Check first 3
        print(f"   ✓ Train {train.unique_id}: {train.train_type.value}, "
              f"at {train.current_position}, going to {train.destination}")
    
    # Test stations
    print("\n3️⃣  Testing station passenger generation...")
    stations = model.get_all_stations()
    initial_passengers = sum(s.waiting_passengers for s in stations)
    print(f"   Initial passengers waiting: {initial_passengers}")
    
    # Run simulation for 50 steps
    print("\n4️⃣  Running simulation for 50 steps...")
    for i in range(50):
        model.step()
        if i % 10 == 0:
            active = len([t for t in model.get_all_trains() if t.state not in [TrainState.ARRIVED]])
            moved = len([t for t in model.get_all_trains() if t.state == TrainState.MOVING])
            print(f"   Step {i:3d}: {active} active trains, {moved} moving")
    
    # Check results
    print("\n5️⃣  Analyzing results...")
    stats = model.get_summary_stats()
    
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Active trains: {stats['active_trains']}")
    print(f"   Arrived trains: {stats['arrived_trains']}")
    print(f"   Total passengers waiting: {stats['total_passengers_waiting']}")
    print(f"   Total passengers delivered: {stats['total_passengers_arrived']}")
    
    # Detailed train status
    print("\n6️⃣  Train status after simulation:")
    for train in trains:
        print(f"   Train {train.unique_id} ({train.train_type.value}): "
              f"State={train.state.value}, Energy={train.energy:.0f}, "
              f"Passengers={train.passengers}")
    
    # Check if trains moved
    trains_that_moved = [t for t in trains if t.route_index > 0]
    print(f"\n✅ Trains that moved: {len(trains_that_moved)}/{len(trains)}")
    
    # Check if any trains arrived
    arrived_trains = [t for t in trains if t.state == TrainState.ARRIVED]
    print(f"✅ Trains that arrived: {len(arrived_trains)}/{len(trains)}")
    
    # Check passenger boarding
    trains_with_passengers = [t for t in trains 
                             if t.train_type in [TrainType.PASSENGER, TrainType.EXPRESS] 
                             and t.passengers > 0]
    print(f"✅ Trains with passengers: {len(trains_with_passengers)}")
    
    # Overall assessment
    print("\n" + "="*60)
    if len(trains_that_moved) > 0:
        print("✅ TRAINS ARE MOVING - Movement system works!")
    else:
        print("❌ WARNING: No trains moved")
    
    if stats['total_passengers_arrived'] > 0:
        print("✅ PASSENGERS DELIVERED - Passenger system works!")
    else:
        print("⚠️  No passengers delivered yet (may need more steps)")
    
    if len(arrived_trains) > 0:
        print("✅ TRAINS ARRIVING - Navigation system works!")
    else:
        print("⚠️  No trains arrived yet (may need more steps or longer distances)")
    
    print("="*60)
    
    return model


if __name__ == "__main__":
    try:
        model = test_simulation()
        print("\n🎉 Test completed! Ready to run full simulation.")
        print("\nTo run the full interactive visualization:")
        print("   python run.py")
        print("\nTo run a batch simulation:")
        print("   python run.py --batch --steps 500")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

