#!/usr/bin/env python3
"""
Example: Evaluating Your Trained DRL Model

This script shows how to evaluate your trained DRL (GNN) agent against
heuristic algorithms using the comprehensive evaluation framework.

Usage:
    python examples/trained_model_evaluation.py --model_path path/to/your/model.pt
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from evaluation.drl_agent_evaluator import RandomDRLAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DRL model vs heuristics")
    parser.add_argument("--model_path", type=str, 
                       help="Path to your trained model file (.pt, .pth, or .pkl)")
    parser.add_argument("--model_name", type=str, default="My-DRL-Agent",
                       help="Name for your DRL agent in results")
    parser.add_argument("--output_dir", type=str, default="drl_evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of evaluation runs per scenario")
    parser.add_argument("--scenarios", nargs="+", 
                       default=["linear_abundant", "parallel_bottleneck", "complex_heterogeneous"],
                       help="Scenarios to evaluate on")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with fewer runs and scenarios")
    
    args = parser.parse_args()
    
    # Quick evaluation settings
    if args.quick:
        args.num_runs = 2
        args.scenarios = ["linear_abundant"]
        print("🚀 Running quick evaluation...")
    else:
        print("🔬 Running comprehensive evaluation...")
    
    print(f"📊 Evaluation Settings:")
    print(f"   Model: {args.model_path if args.model_path else 'Random baseline only'}")
    print(f"   Scenarios: {args.scenarios}")
    print(f"   Runs per scenario: {args.num_runs}")
    print(f"   Output directory: {args.output_dir}")
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )
    
    # Add your trained model if provided
    if args.model_path:
        print(f"\n📥 Loading trained model: {args.model_path}")
        success = evaluator.add_drl_agent_from_path(
            model_path=args.model_path,
            model_type="auto",  # Auto-detect model type
            name=args.model_name
        )
        
        if success:
            print(f"✅ Successfully loaded model as '{args.model_name}'")
        else:
            print(f"❌ Failed to load model. Adding random baseline instead.")
            evaluator.add_drl_agent(RandomDRLAgent(seed=42))
    else:
        print("\n🎲 No model provided. Using random DRL agent for demonstration.")
        evaluator.add_drl_agent(RandomDRLAgent(seed=42))
    
    # Run evaluation
    print(f"\n🏃 Starting evaluation...")
    print(f"   This will compare your DRL agent against {len(evaluator.heuristic_algorithms)} heuristic algorithms")
    print(f"   across {len(args.scenarios)} scenarios with {args.num_runs} runs each.")
    print(f"   Total evaluations: {(len(evaluator.heuristic_algorithms) + 1) * len(args.scenarios) * args.num_runs}")
    
    try:
        analysis = evaluator.run_comprehensive_evaluation(scenarios=args.scenarios)
        
        # Generate report
        print(f"\n📝 Generating comprehensive report...")
        report = evaluator.generate_report()
        
        # Print summary
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"\n📊 Quick Summary:")
        
        summary = analysis.get('summary', {})
        print(f"   • Total scenarios evaluated: {summary.get('total_scenarios', 'N/A')}")
        print(f"   • Total algorithms compared: {summary.get('total_algorithms', 'N/A')}")
        print(f"   • Best makespan algorithm: {summary.get('overall_best_makespan', {}).get('algorithm', 'N/A')}")
        print(f"   • Best energy algorithm: {summary.get('overall_best_energy', {}).get('algorithm', 'N/A')}")
        print(f"   • Best efficiency algorithm: {summary.get('overall_best_efficiency', {}).get('algorithm', 'N/A')}")
        
        # DRL vs Heuristic comparison
        drl_vs_heuristic = analysis.get('performance_rankings', {}).get('drl_vs_heuristic', {})
        if drl_vs_heuristic:
            print(f"\n🤖 DRL vs Heuristic Comparison:")
            print(f"   • DRL avg makespan: {drl_vs_heuristic.get('drl_avg_makespan', 0):.2f}")
            print(f"   • Heuristic avg makespan: {drl_vs_heuristic.get('heuristic_avg_makespan', 0):.2f}")
            print(f"   • DRL avg energy: {drl_vs_heuristic.get('drl_avg_energy', 0):.4f} Wh")
            print(f"   • Heuristic avg energy: {drl_vs_heuristic.get('heuristic_avg_energy', 0):.4f} Wh")
        
        print(f"\n📋 Generated Files:")
        print(f"   • Raw results: {args.output_dir}/raw_results.csv")
        print(f"   • Aggregated results: {args.output_dir}/aggregated_results.csv")
        print(f"   • Analysis data: {args.output_dir}/analysis.json")
        print(f"   • Comprehensive report: {args.output_dir}/evaluation_report.md")
        print(f"   • Visualizations: {args.output_dir}/visualizations/")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review the comprehensive report: {args.output_dir}/evaluation_report.md")
        print(f"   2. Analyze visualizations in: {args.output_dir}/visualizations/")
        print(f"   3. Use raw data for custom analysis: {args.output_dir}/raw_results.csv")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        print(f"💡 Try running with --quick flag for faster debugging")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

