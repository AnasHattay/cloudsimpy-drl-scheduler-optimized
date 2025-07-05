"""
Test script for the comprehensive evaluation framework

This script tests all components of the evaluation framework to ensure
they work correctly together.
"""

import sys
import os
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_scenario_generator():
    """Test the scenario generator"""
    print("Testing Scenario Generator...")
    
    try:
        from evaluation.scenario_generator import create_evaluation_scenarios, DAGType, ResourceCondition
        
        generator = create_evaluation_scenarios()
        scenarios = generator.get_all_scenarios()
        
        print(f"✓ Created {len(scenarios)} scenarios")
        
        # Test scenario generation
        test_scenario = scenarios[0]
        dataset = generator.generate_scenario_dataset(test_scenario)
        
        print(f"✓ Generated dataset: {len(dataset.workflows)} workflows, {len(dataset.vms)} VMs, {len(dataset.hosts)} hosts")
        
        # Test different scenario types
        linear_scenarios = generator.get_scenarios_by_type(DAGType.LINEAR)
        parallel_scenarios = generator.get_scenarios_by_type(DAGType.PARALLEL)
        
        print(f"✓ Found {len(linear_scenarios)} linear scenarios, {len(parallel_scenarios)} parallel scenarios")
        
        return True
        
    except Exception as e:
        print(f"✗ Scenario Generator test failed: {e}")
        traceback.print_exc()
        return False


def test_heuristic_algorithms():
    """Test heuristic algorithms"""
    print("\nTesting Heuristic Algorithms...")
    
    try:
        from evaluation.heuristic_algorithms import get_all_heuristic_algorithms
        from evaluation.scenario_generator import create_evaluation_scenarios
        
        # Get algorithms and test scenario
        algorithms = get_all_heuristic_algorithms()
        generator = create_evaluation_scenarios()
        scenario = generator.get_all_scenarios()[0]
        dataset = generator.generate_scenario_dataset(scenario)
        
        print(f"✓ Found {len(algorithms)} heuristic algorithms")
        
        # Test each algorithm
        for algorithm in algorithms[:3]:  # Test first 3 to save time
            try:
                decisions = algorithm.schedule(dataset)
                print(f"✓ {algorithm.name}: scheduled {len(decisions)} tasks")
            except Exception as e:
                print(f"✗ {algorithm.name} failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Heuristic Algorithms test failed: {e}")
        traceback.print_exc()
        return False


def test_drl_agent_evaluator():
    """Test DRL agent evaluator"""
    print("\nTesting DRL Agent Evaluator...")
    
    try:
        from evaluation.drl_agent_evaluator import DRLAgentEvaluator, RandomDRLAgent
        from evaluation.scenario_generator import create_evaluation_scenarios
        
        # Create test components
        evaluator = DRLAgentEvaluator(max_episode_steps=50, timeout_seconds=30.0)
        agent = RandomDRLAgent(seed=42)
        
        generator = create_evaluation_scenarios()
        scenario = generator.get_all_scenarios()[0]
        dataset = generator.generate_scenario_dataset(scenario)
        
        print("✓ Created DRL evaluator and random agent")
        
        # Test evaluation
        result = evaluator.evaluate_agent(agent, dataset, scenario.name)
        
        print(f"✓ Evaluation completed:")
        print(f"  - Makespan: {result.makespan:.2f}")
        print(f"  - Energy: {result.total_energy:.4f} Wh")
        print(f"  - Success Rate: {result.success_rate:.2%}")
        print(f"  - Episode Length: {result.episode_length}")
        
        return True
        
    except Exception as e:
        print(f"✗ DRL Agent Evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_evaluator():
    """Test comprehensive evaluator"""
    print("\nTesting Comprehensive Evaluator...")
    
    try:
        from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
        from evaluation.drl_agent_evaluator import RandomDRLAgent
        
        # Create evaluator with minimal settings for testing
        evaluator = ComprehensiveEvaluator(output_dir="test_evaluation_output", num_runs=2)
        
        # Add a test DRL agent
        evaluator.add_drl_agent(RandomDRLAgent(seed=42))
        
        print("✓ Created comprehensive evaluator")
        
        # Run evaluation on a subset of scenarios
        test_scenarios = ["linear_abundant"]  # Just one scenario for testing
        analysis = evaluator.run_comprehensive_evaluation(scenarios=test_scenarios)
        
        print("✓ Comprehensive evaluation completed")
        print(f"  - Total scenarios: {analysis.get('summary', {}).get('total_scenarios', 'N/A')}")
        print(f"  - Total algorithms: {analysis.get('summary', {}).get('total_algorithms', 'N/A')}")
        
        # Test report generation
        report = evaluator.generate_report()
        print("✓ Generated evaluation report")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive Evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration"""
    print("\nTesting Full Integration...")
    
    try:
        # Import all components
        from evaluation.scenario_generator import create_evaluation_scenarios
        from evaluation.heuristic_algorithms import get_all_heuristic_algorithms
        from evaluation.drl_agent_evaluator import RandomDRLAgent
        from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
        
        # Create minimal evaluation setup
        evaluator = ComprehensiveEvaluator(output_dir="integration_test", num_runs=1)
        
        # Add one DRL agent
        evaluator.add_drl_agent(RandomDRLAgent(seed=123))
        
        # Limit to one scenario and fewer algorithms for speed
        evaluator.heuristic_algorithms = evaluator.heuristic_algorithms[:2]  # Just 2 heuristics
        
        # Run evaluation
        analysis = evaluator.run_comprehensive_evaluation(scenarios=["linear_abundant"])
        
        # Check results
        if evaluator.raw_results and evaluator.aggregated_results:
            print("✓ Full integration test passed")
            print(f"  - Raw results: {len(evaluator.raw_results)} entries")
            print(f"  - Aggregated results: {len(evaluator.aggregated_results)} entries")
            return True
        else:
            print("✗ Integration test failed: No results generated")
            return False
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("EVALUATION FRAMEWORK VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Scenario Generator", test_scenario_generator),
        ("Heuristic Algorithms", test_heuristic_algorithms),
        ("DRL Agent Evaluator", test_drl_agent_evaluator),
        ("Comprehensive Evaluator", test_comprehensive_evaluator),
        ("Full Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests passed! The evaluation framework is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

