#!/usr/bin/env python3
"""
Basic functionality test for separated environments
"""

import sys
import traceback
from environments import (
    create_training_environment, 
    create_testing_environment,
    EnvironmentBuilder,
    EnvironmentFactory,
    EnvironmentType
)

def test_basic_functionality():
    """Test basic functionality of separated environments"""
    print("Testing Separated Environments")
    print("=" * 50)
    
    # Test dataset arguments
    dataset_args = {
        "num_workflows": 3,
        "num_tasks_per_workflow": 5,
        "num_vms": 4,
        "num_hosts": 2,
        "seed": 42
    }
    
    try:
        # Test 1: Training environment creation
        print("Test 1: Creating training environment...")
        train_env = create_training_environment(
            dataset_args=dataset_args,
            performance_mode="speed"
        )
        print(f"✓ Training environment created")
        print(f"  Action space: {train_env.action_space}")
        print(f"  Observation space shape: {train_env.observation_space.shape}")
        
        # Test basic reset
        obs, info = train_env.reset(seed=42)
        print(f"✓ Training environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Test basic step
        action = train_env.action_space.sample()
        obs, reward, terminated, truncated, info = train_env.step(action)
        print(f"✓ Training environment step successful")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}")
        
        train_env.close()
        print("✓ Training environment closed")
        
    except Exception as e:
        print(f"✗ Training environment test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test 2: Builder pattern
        print("\nTest 2: Testing builder pattern...")
        builder_env = (EnvironmentBuilder()
                      .training()
                      .with_dataset_args(**dataset_args)
                      .with_performance_mode("speed")
                      .with_reward_weights(makespan=0.7, energy=0.3)
                      .with_seed(42)
                      .build())
        
        print("✓ Builder pattern environment created")
        obs, info = builder_env.reset()
        print("✓ Builder environment reset successful")
        builder_env.close()
        print("✓ Builder environment closed")
        
    except Exception as e:
        print(f"✗ Builder pattern test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test 3: Environment factory
        print("\nTest 3: Testing environment factory...")
        available_types = EnvironmentFactory.list_available()
        print(f"✓ Available environment types: {[t.value for t in available_types]}")
        
        if EnvironmentType.TRAINING in available_types:
            print("✓ Training environment type available")
        if EnvironmentType.TESTING in available_types:
            print("✓ Testing environment type available")
        
    except Exception as e:
        print(f"✗ Environment factory test failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✓ All basic functionality tests passed!")
    print("The separated environments are working correctly.")
    return True

def test_performance_comparison():
    """Test performance comparison between environments"""
    print("\nPerformance Comparison Test")
    print("=" * 30)
    
    dataset_args = {
        "num_workflows": 2,
        "num_tasks_per_workflow": 4,
        "num_vms": 3,
        "num_hosts": 2,
        "seed": 42
    }
    
    try:
        import time
        
        # Test training environment speed
        print("Testing training environment speed...")
        train_env = create_training_environment(dataset_args=dataset_args)
        
        start_time = time.time()
        for i in range(10):
            obs, _ = train_env.reset(seed=42 + i)
            for _ in range(5):
                action = train_env.action_space.sample()
                obs, reward, terminated, truncated, info = train_env.step(action)
                if terminated or truncated:
                    break
        
        training_time = time.time() - start_time
        train_env.close()
        
        print(f"✓ Training environment: {training_time:.3f} seconds for 10 episodes")
        print(f"  Average time per episode: {training_time/10:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("CloudSimPy Separated Environments - Basic Test")
    print("=" * 60)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("Basic functionality tests failed!")
        return 1
    
    # Test performance
    if not test_performance_comparison():
        print("Performance tests failed!")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("The separated environments implementation is working correctly.")
    print("\nNext steps:")
    print("1. Try the complete example: python examples/separated_training_example.py")
    print("2. Migrate existing code: python migrate_to_separated_envs.py --all")
    print("3. Read the documentation: README_SEPARATED_ENVIRONMENTS.md")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

