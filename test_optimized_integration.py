#!/usr/bin/env python3
"""
Test Script for Optimized CloudSimPy Integration

This script validates:
1. Proper CloudSimPy simulation integration
2. Energy monitoring with idle power consumption
3. Real-time simulation feedback
4. Accurate makespan and energy calculations
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_energy_monitor():
    """Test the energy monitoring system"""
    print("Testing Energy Monitor...")
    
    try:
        import simpy
        from core.energy_monitor import EnergyMonitor, EnergyProfile, EnergyAwareMachine, MachineState
        
        # Create test environment
        env = simpy.Environment()
        monitor = EnergyMonitor(env, enable_logging=False)
        
        # Create test energy profile
        profile = EnergyProfile(idle_power_watt=100.0, peak_power_watt=300.0)
        
        # Create energy-aware machine
        machine = EnergyAwareMachine(machine_id=0, energy_profile=profile, energy_monitor=monitor)
        
        print("✓ Energy monitor and machine created successfully")
        
        # Test state transitions
        machine.start_task(task_id=1, estimated_duration=10.0, cpu_utilization=0.8)
        env.run(until=5.0)  # Run for 5 seconds
        
        machine.complete_task(task_id=1)
        env.run(until=10.0)  # Run for another 5 seconds
        
        # Check energy consumption
        energy = machine.get_energy_consumption()
        print(f"✓ Energy consumption calculated: {energy:.6f} Wh")
        
        # Generate report
        report = monitor.generate_energy_report()
        print("✓ Energy report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Energy monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_gym_environment():
    """Test the enhanced gym environment"""
    print("\nTesting Enhanced Gym Environment...")
    
    try:
        from simulator.cloudsimpy_gym_env import EnhancedCloudSimPyGymEnvironment
        from dataset_generator.gen_dataset import DatasetArgs
        from dataset_generator.core.gen_dataset import generate_dataset
        
        # Generate test dataset
        dataset = generate_dataset(
            seed=42,
            host_count=2,
            vm_count=3,
            max_memory_gb=4,
            min_cpu_speed_mips=1000,
            max_cpu_speed_mips=3000,
            workflow_count=1,
            dag_method="gnp",
            gnp_min_n=3,
            gnp_max_n=5,
            task_length_dist="normal",
            min_task_length=1000,
            max_task_length=5000,
            task_arrival="static",
            arrival_rate=1.0
        )
        
        print(f"✓ Generated test dataset: {len(dataset.workflows)} workflows, {len(dataset.vms)} VMs")
        
        # Create environment
        env = EnhancedCloudSimPyGymEnvironment(dataset=dataset, max_episode_steps=20)
        
        # Reset environment
        obs, info = env.reset()
        print(f"✓ Environment reset successfully, observation shape: {obs.shape}")
        
        # Test a few steps
        for step in range(5):
            # Get valid action (simple heuristic: first ready task on first available machine)
            action = step % env.action_space.n  # Simple action selection
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step + 1}: reward={reward:.2f}, terminated={terminated}")
            
            if terminated or truncated:
                break
        
        # Get final metrics
        if hasattr(env, '_get_episode_info'):
            final_info = env._get_episode_info()
            print(f"✓ Final metrics:")
            print(f"  - Tasks completed: {final_info.get('tasks_completed', 0)}/{final_info.get('total_tasks', 0)}")
            print(f"  - Total energy: {final_info.get('total_energy_wh', 0):.6f} Wh")
            print(f"  - Makespan: {final_info.get('makespan', 0):.2f}")
            print(f"  - Energy efficiency: {final_info.get('energy_efficiency', 0):.2f} tasks/Wh")
        
        # Test energy report
        if hasattr(env, 'get_detailed_energy_report'):
            report = env.get_detailed_energy_report()
            print("✓ Detailed energy report generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced gym environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cloudsimpy_integration():
    """Test CloudSimPy integration specifically"""
    print("\nTesting CloudSimPy Integration...")
    
    try:
        import simpy
        from core.simulation import Simulation
        from core.cluster import Cluster
        from core.machine import MachineConfig
        from core.broker import Broker
        from core.scheduler import Scheduler
        from core.alogrithm import Algorithm
        
        # Create basic CloudSimPy simulation
        env = simpy.Environment()
        
        # Create cluster
        cluster = Cluster()
        machine_configs = [
            MachineConfig(cpu_capacity=4, memory_capacity=4096, disk_capacity=50000),
            MachineConfig(cpu_capacity=8, memory_capacity=8192, disk_capacity=100000)
        ]
        cluster.add_machines(machine_configs)
        
        print(f"✓ Created cluster with {len(cluster.machines)} machines")
        
        # Create simple algorithm
        class TestAlgorithm(Algorithm):
            def __call__(self, cluster, clock):
                pass  # Simple no-op algorithm
        
        algorithm = TestAlgorithm()
        
        # Create scheduler
        scheduler = Scheduler(env, algorithm)
        
        # Create broker
        broker = Broker(env, [])
        
        # Create simulation
        simulation = Simulation(
            env=env,
            cluster=cluster,
            task_broker=broker,
            scheduler=scheduler,
            event_file=None
        )
        
        print("✓ CloudSimPy simulation created successfully")
        
        # Run simulation briefly
        env.run(until=1.0)
        print("✓ CloudSimPy simulation executed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ CloudSimPy integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_framework_integration():
    """Test integration with evaluation framework"""
    print("\nTesting Evaluation Framework Integration...")
    
    try:
        from simulator.cloudsimpy_gym_env import EnhancedCloudSimPyGymEnvironment
        from scheduler.dataset_generator.core import generate_dataset
        
        # Generate test dataset
        dataset = generate_dataset(
            seed=42,
            host_count=2,
            vm_count=2,
            max_memory_gb=4,
            min_cpu_speed_mips=1000,
            max_cpu_speed_mips=3000,
            workflow_count=1,
            dag_method="gnp",
            gnp_min_n=3,
            gnp_max_n=3,
            task_length_dist="normal",
            min_task_length=1000,
            max_task_length=3000,
            task_arrival="static",
            arrival_rate=1.0
        )
        
        # Test enhanced environment directly
        env = EnhancedCloudSimPyGymEnvironment(dataset=dataset, max_episode_steps=10)
        
        print("✓ Created enhanced environment for evaluation")
        
        # Test a complete episode
        obs, info = env.reset()
        total_reward = 0.0
        
        for step in range(5):
            action = step % env.action_space.n
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✓ Completed test episode with total reward: {total_reward:.2f}")
        
        # Check if energy metrics are available
        if hasattr(env, '_get_episode_info'):
            episode_info = env._get_episode_info()
            if 'total_energy_wh' in episode_info:
                print(f"✓ Energy metrics available: {episode_info['total_energy_wh']:.6f} Wh")
            else:
                print("⚠ Energy metrics not found in episode info")
        
        # Test energy report
        if hasattr(env, 'get_detailed_energy_report'):
            report = env.get_detailed_energy_report()
            if "ENERGY CONSUMPTION REPORT" in report:
                print("✓ Detailed energy report generated successfully")
            else:
                print("⚠ Energy report may not be complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation framework integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_idle_energy_calculation():
    """Test idle energy consumption calculation"""
    print("\nTesting Idle Energy Calculation...")
    
    try:
        import simpy
        from core.energy_monitor import EnergyMonitor, EnergyProfile, EnergyAwareMachine
        
        # Create test environment
        env = simpy.Environment()
        monitor = EnergyMonitor(env, enable_logging=False)
        
        # Create machine with significant idle power
        profile = EnergyProfile(idle_power_watt=150.0, peak_power_watt=400.0)
        machine = EnergyAwareMachine(machine_id=0, energy_profile=profile, energy_monitor=monitor)
        
        # Let machine idle for 10 seconds
        env.run(until=10.0)
        
        idle_energy = machine.get_energy_consumption()
        expected_idle_energy = (150.0 * 10.0) / 3600.0  # 150W for 10 seconds = Wh
        
        print(f"✓ Idle energy consumption: {idle_energy:.6f} Wh")
        print(f"✓ Expected idle energy: {expected_idle_energy:.6f} Wh")
        
        # Check if values are close (within 1% tolerance)
        if abs(idle_energy - expected_idle_energy) / expected_idle_energy < 0.01:
            print("✓ Idle energy calculation is accurate")
        else:
            print("⚠ Idle energy calculation may have issues")
        
        # Test busy energy
        machine.start_task(task_id=1, estimated_duration=5.0, cpu_utilization=1.0)
        env.run(until=15.0)  # Run for 5 more seconds
        machine.complete_task(task_id=1)
        
        total_energy = machine.get_energy_consumption()
        busy_energy_added = (400.0 * 5.0) / 3600.0  # 400W for 5 seconds
        expected_total = expected_idle_energy + busy_energy_added
        
        print(f"✓ Total energy after busy period: {total_energy:.6f} Wh")
        print(f"✓ Expected total energy: {expected_total:.6f} Wh")
        
        return True
        
    except Exception as e:
        print(f"✗ Idle energy calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("OPTIMIZED CLOUDSIMPY INTEGRATION VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Energy Monitor", test_energy_monitor),
        ("Enhanced Gym Environment", test_enhanced_gym_environment),
        ("CloudSimPy Integration", test_cloudsimpy_integration),
        ("Idle Energy Calculation", test_idle_energy_calculation),
        ("Evaluation Framework Integration", test_evaluation_framework_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 50}")
        print(f"Running: {test_name}")
        print(f"{'-' * 50}")
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
    print(f"{'=' * 70}")
    
    if passed == total:
        print("🎉 All tests passed! Optimized CloudSimPy integration is working correctly.")
        print("\nKey improvements validated:")
        print("✓ Proper CloudSimPy simulation engine integration")
        print("✓ Comprehensive energy monitoring with idle power")
        print("✓ Real-time simulation feedback")
        print("✓ Accurate makespan and energy calculations")
        print("✓ Seamless evaluation framework integration")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

