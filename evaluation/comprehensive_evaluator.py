"""
Comprehensive Evaluation and Analysis Framework

This module provides a complete evaluation framework that compares DRL agents
against heuristic algorithms across multiple scenarios and generates detailed
analysis reports with visualizations.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from evaluation.scenario_generator import ScenarioGenerator, ScenarioConfig, create_evaluation_scenarios
from evaluation.heuristic_algorithms import get_all_heuristic_algorithms, SchedulingAlgorithm, SchedulingDecision
from evaluation.drl_agent_evaluator import DRLAgent, DRLAgentEvaluator, EvaluationResult, load_trained_model


@dataclass
class ComparisonResult:
    """Results from comparing algorithms across scenarios"""
    scenario_name: str
    algorithm_name: str
    algorithm_type: str  # "heuristic" or "drl"
    makespan: float
    total_energy: float
    energy_efficiency: float
    success_rate: float
    execution_time: float
    additional_metrics: Dict[str, Any] = None


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs"""
    algorithm_name: str
    algorithm_type: str
    scenario_name: str
    makespan_mean: float
    makespan_std: float
    energy_mean: float
    energy_std: float
    efficiency_mean: float
    efficiency_std: float
    success_rate_mean: float
    success_rate_std: float
    execution_time_mean: float
    execution_time_std: float


class ComprehensiveEvaluator:
    """Main evaluation framework for comparing DRL agents vs heuristics"""
    
    def __init__(self, output_dir: str = "evaluation_results", num_runs: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.num_runs = num_runs
        
        # Initialize components
        self.scenario_generator = create_evaluation_scenarios()
        self.heuristic_algorithms = get_all_heuristic_algorithms()
        self.drl_evaluator = DRLAgentEvaluator(max_episode_steps=1000, timeout_seconds=300.0)
        
        # Results storage
        self.raw_results = []
        self.aggregated_results = []
        
    def add_drl_agent(self, agent: DRLAgent):
        """Add a DRL agent to the evaluation"""
        if not hasattr(self, 'drl_agents'):
            self.drl_agents = []
        self.drl_agents.append(agent)
    
    def add_drl_agent_from_path(self, model_path: str, model_type: str = "auto", name: Optional[str] = None):
        """Add a DRL agent from model file"""
        agent = load_trained_model(model_path, model_type)
        if agent:
            if name:
                agent.name = name
            self.add_drl_agent(agent)
            return True
        return False
    
    def evaluate_heuristic_on_scenario(self, algorithm: SchedulingAlgorithm, 
                                     scenario: ScenarioConfig) -> List[ComparisonResult]:
        """Evaluate a heuristic algorithm on a scenario"""
        results = []
        
        for run in range(self.num_runs):
            # Generate dataset with different seed for each run
            scenario_copy = ScenarioConfig(**asdict(scenario))
            scenario_copy.seed = scenario.seed + run
            dataset = self.scenario_generator.generate_scenario_dataset(scenario_copy)
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Run heuristic algorithm
                decisions = algorithm.schedule(dataset)
                execution_time = time.time() - start_time
                
                # Calculate metrics
                if decisions:
                    makespan = max(d.estimated_finish_time for d in decisions)
                    total_energy = sum(d.estimated_energy for d in decisions)
                    total_tasks = sum(len(w.tasks) for w in dataset.workflows)
                    success_rate = len(decisions) / total_tasks if total_tasks > 0 else 0.0
                    energy_efficiency = len(decisions) / total_energy if total_energy > 0 else 0.0
                else:
                    makespan = float('inf')
                    total_energy = 0.0
                    success_rate = 0.0
                    energy_efficiency = 0.0
                
                result = ComparisonResult(
                    scenario_name=f"{scenario.name}_run_{run + 1}",
                    algorithm_name=algorithm.name,
                    algorithm_type="heuristic",
                    makespan=makespan,
                    total_energy=total_energy,
                    energy_efficiency=energy_efficiency,
                    success_rate=success_rate,
                    execution_time=execution_time,
                    additional_metrics={
                        "scheduled_tasks": len(decisions),
                        "total_tasks": sum(len(w.tasks) for w in dataset.workflows)
                    }
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {algorithm.name} on {scenario.name} run {run + 1}: {e}")
                # Add failed result
                result = ComparisonResult(
                    scenario_name=f"{scenario.name}_run_{run + 1}",
                    algorithm_name=algorithm.name,
                    algorithm_type="heuristic",
                    makespan=float('inf'),
                    total_energy=0.0,
                    energy_efficiency=0.0,
                    success_rate=0.0,
                    execution_time=time.time() - start_time,
                    additional_metrics={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def evaluate_drl_agent_on_scenario(self, agent: DRLAgent, 
                                     scenario: ScenarioConfig) -> List[ComparisonResult]:
        """Evaluate a DRL agent on a scenario"""
        results = []
        
        for run in range(self.num_runs):
            # Generate dataset with different seed for each run
            scenario_copy = ScenarioConfig(**asdict(scenario))
            scenario_copy.seed = scenario.seed + run
            dataset = self.scenario_generator.generate_scenario_dataset(scenario_copy)
            
            try:
                # Evaluate DRL agent
                eval_result = self.drl_evaluator.evaluate_agent(
                    agent, dataset, f"{scenario.name}_run_{run + 1}"
                )
                
                result = ComparisonResult(
                    scenario_name=eval_result.scenario_name,
                    algorithm_name=eval_result.agent_name,
                    algorithm_type="drl",
                    makespan=eval_result.makespan,
                    total_energy=eval_result.total_energy,
                    energy_efficiency=eval_result.energy_efficiency,
                    success_rate=eval_result.success_rate,
                    execution_time=eval_result.execution_time,
                    additional_metrics={
                        "episode_length": eval_result.episode_length,
                        "total_reward": eval_result.total_reward,
                        "scheduled_tasks": len(eval_result.decisions)
                    }
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {agent.name} on {scenario.name} run {run + 1}: {e}")
                # Add failed result
                result = ComparisonResult(
                    scenario_name=f"{scenario.name}_run_{run + 1}",
                    algorithm_name=agent.name,
                    algorithm_type="drl",
                    makespan=float('inf'),
                    total_energy=0.0,
                    energy_efficiency=0.0,
                    success_rate=0.0,
                    execution_time=0.0,
                    additional_metrics={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def run_comprehensive_evaluation(self, scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across all scenarios and algorithms"""
        
        # Select scenarios to evaluate
        if scenarios is None:
            eval_scenarios = self.scenario_generator.get_all_scenarios()
        else:
            eval_scenarios = [s for s in self.scenario_generator.get_all_scenarios() 
                            if s.name in scenarios]
        
        print(f"Starting comprehensive evaluation on {len(eval_scenarios)} scenarios...")
        print(f"Heuristic algorithms: {len(self.heuristic_algorithms)}")
        print(f"DRL agents: {len(getattr(self, 'drl_agents', []))}")
        print(f"Runs per algorithm-scenario: {self.num_runs}")
        
        total_evaluations = (len(self.heuristic_algorithms) + len(getattr(self, 'drl_agents', []))) * len(eval_scenarios)
        current_evaluation = 0
        
        # Clear previous results
        self.raw_results = []
        
        # Evaluate heuristic algorithms
        for scenario in eval_scenarios:
            print(f"\n--- Evaluating Scenario: {scenario.name} ---")
            print(f"Description: {scenario.description}")
            
            for algorithm in self.heuristic_algorithms:
                current_evaluation += 1
                print(f"[{current_evaluation}/{total_evaluations}] Evaluating {algorithm.name}...")
                
                results = self.evaluate_heuristic_on_scenario(algorithm, scenario)
                self.raw_results.extend(results)
        
        # Evaluate DRL agents
        if hasattr(self, 'drl_agents'):
            for scenario in eval_scenarios:
                for agent in self.drl_agents:
                    current_evaluation += 1
                    print(f"[{current_evaluation}/{total_evaluations}] Evaluating {agent.name}...")
                    
                    results = self.evaluate_drl_agent_on_scenario(agent, scenario)
                    self.raw_results.extend(results)
        
        # Aggregate results
        self._aggregate_results()
        
        # Save results
        self._save_results()
        
        # Generate analysis
        analysis = self._generate_analysis()
        
        print(f"\nEvaluation completed! Results saved to {self.output_dir}")
        return analysis
    
    def _aggregate_results(self):
        """Aggregate results across multiple runs"""
        self.aggregated_results = []
        
        # Group results by algorithm and scenario (excluding run number)
        grouped = {}
        for result in self.raw_results:
            # Remove run number from scenario name
            base_scenario = result.scenario_name.split('_run_')[0]
            key = (result.algorithm_name, result.algorithm_type, base_scenario)
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Calculate aggregated statistics
        for (algorithm_name, algorithm_type, scenario_name), results in grouped.items():
            if not results:
                continue
            
            # Filter out infinite values for statistics
            valid_results = [r for r in results if r.makespan != float('inf')]
            
            if valid_results:
                makespans = [r.makespan for r in valid_results]
                energies = [r.total_energy for r in valid_results]
                efficiencies = [r.energy_efficiency for r in valid_results]
                success_rates = [r.success_rate for r in results]  # Include all for success rate
                exec_times = [r.execution_time for r in results]
                
                aggregated = AggregatedResults(
                    algorithm_name=algorithm_name,
                    algorithm_type=algorithm_type,
                    scenario_name=scenario_name,
                    makespan_mean=np.mean(makespans),
                    makespan_std=np.std(makespans),
                    energy_mean=np.mean(energies),
                    energy_std=np.std(energies),
                    efficiency_mean=np.mean(efficiencies),
                    efficiency_std=np.std(efficiencies),
                    success_rate_mean=np.mean(success_rates),
                    success_rate_std=np.std(success_rates),
                    execution_time_mean=np.mean(exec_times),
                    execution_time_std=np.std(exec_times)
                )
                self.aggregated_results.append(aggregated)
    
    def _save_results(self):
        """Save results to files"""
        
        # Save raw results
        raw_df = pd.DataFrame([asdict(r) for r in self.raw_results])
        raw_df.to_csv(self.output_dir / "raw_results.csv", index=False)
        
        # Save aggregated results
        agg_df = pd.DataFrame([asdict(r) for r in self.aggregated_results])
        agg_df.to_csv(self.output_dir / "aggregated_results.csv", index=False)
        
        # Save as JSON for programmatic access
        with open(self.output_dir / "raw_results.json", 'w') as f:
            json.dump([asdict(r) for r in self.raw_results], f, indent=2)
        
        with open(self.output_dir / "aggregated_results.json", 'w') as f:
            json.dump([asdict(r) for r in self.aggregated_results], f, indent=2)
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of results"""
        
        if not self.aggregated_results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame([asdict(r) for r in self.aggregated_results])
        
        analysis = {
            "summary": self._generate_summary_analysis(df),
            "scenario_analysis": self._generate_scenario_analysis(df),
            "algorithm_analysis": self._generate_algorithm_analysis(df),
            "performance_rankings": self._generate_performance_rankings(df)
        }
        
        # Save analysis
        with open(self.output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        return analysis
    
    def _generate_summary_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary analysis"""
        
        summary = {
            "total_scenarios": df['scenario_name'].nunique(),
            "total_algorithms": df['algorithm_name'].nunique(),
            "heuristic_algorithms": df[df['algorithm_type'] == 'heuristic']['algorithm_name'].nunique(),
            "drl_algorithms": df[df['algorithm_type'] == 'drl']['algorithm_name'].nunique(),
            "overall_best_makespan": {
                "algorithm": df.loc[df['makespan_mean'].idxmin(), 'algorithm_name'],
                "value": df['makespan_mean'].min()
            },
            "overall_best_energy": {
                "algorithm": df.loc[df['energy_mean'].idxmin(), 'algorithm_name'],
                "value": df['energy_mean'].min()
            },
            "overall_best_efficiency": {
                "algorithm": df.loc[df['efficiency_mean'].idxmax(), 'algorithm_name'],
                "value": df['efficiency_mean'].max()
            }
        }
        
        return summary
    
    def _generate_scenario_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate per-scenario analysis"""
        
        scenario_analysis = {}
        
        for scenario in df['scenario_name'].unique():
            scenario_df = df[df['scenario_name'] == scenario]
            
            scenario_analysis[scenario] = {
                "best_makespan": {
                    "algorithm": scenario_df.loc[scenario_df['makespan_mean'].idxmin(), 'algorithm_name'],
                    "value": scenario_df['makespan_mean'].min(),
                    "type": scenario_df.loc[scenario_df['makespan_mean'].idxmin(), 'algorithm_type']
                },
                "best_energy": {
                    "algorithm": scenario_df.loc[scenario_df['energy_mean'].idxmin(), 'algorithm_name'],
                    "value": scenario_df['energy_mean'].min(),
                    "type": scenario_df.loc[scenario_df['energy_mean'].idxmin(), 'algorithm_type']
                },
                "best_efficiency": {
                    "algorithm": scenario_df.loc[scenario_df['efficiency_mean'].idxmax(), 'algorithm_name'],
                    "value": scenario_df['efficiency_mean'].max(),
                    "type": scenario_df.loc[scenario_df['efficiency_mean'].idxmax(), 'algorithm_type']
                },
                "algorithm_count": len(scenario_df),
                "avg_success_rate": scenario_df['success_rate_mean'].mean()
            }
        
        return scenario_analysis
    
    def _generate_algorithm_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate per-algorithm analysis"""
        
        algorithm_analysis = {}
        
        for algorithm in df['algorithm_name'].unique():
            algo_df = df[df['algorithm_name'] == algorithm]
            
            algorithm_analysis[algorithm] = {
                "type": algo_df.iloc[0]['algorithm_type'],
                "scenarios_evaluated": len(algo_df),
                "avg_makespan": algo_df['makespan_mean'].mean(),
                "avg_energy": algo_df['energy_mean'].mean(),
                "avg_efficiency": algo_df['efficiency_mean'].mean(),
                "avg_success_rate": algo_df['success_rate_mean'].mean(),
                "avg_execution_time": algo_df['execution_time_mean'].mean(),
                "best_scenario": algo_df.loc[algo_df['makespan_mean'].idxmin(), 'scenario_name'],
                "worst_scenario": algo_df.loc[algo_df['makespan_mean'].idxmax(), 'scenario_name']
            }
        
        return algorithm_analysis
    
    def _generate_performance_rankings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance rankings"""
        
        # Overall rankings
        makespan_ranking = df.groupby('algorithm_name')['makespan_mean'].mean().sort_values().to_dict()
        energy_ranking = df.groupby('algorithm_name')['energy_mean'].mean().sort_values().to_dict()
        efficiency_ranking = df.groupby('algorithm_name')['efficiency_mean'].mean().sort_values(ascending=False).to_dict()
        
        # DRL vs Heuristic comparison
        drl_performance = df[df['algorithm_type'] == 'drl'].groupby('algorithm_name').agg({
            'makespan_mean': 'mean',
            'energy_mean': 'mean',
            'efficiency_mean': 'mean'
        }).mean()
        
        heuristic_performance = df[df['algorithm_type'] == 'heuristic'].groupby('algorithm_name').agg({
            'makespan_mean': 'mean',
            'energy_mean': 'mean',
            'efficiency_mean': 'mean'
        }).mean()
        
        rankings = {
            "makespan_ranking": makespan_ranking,
            "energy_ranking": energy_ranking,
            "efficiency_ranking": efficiency_ranking,
            "drl_vs_heuristic": {
                "drl_avg_makespan": drl_performance.get('makespan_mean', 0),
                "heuristic_avg_makespan": heuristic_performance.get('makespan_mean', 0),
                "drl_avg_energy": drl_performance.get('energy_mean', 0),
                "heuristic_avg_energy": heuristic_performance.get('energy_mean', 0),
                "drl_avg_efficiency": drl_performance.get('efficiency_mean', 0),
                "heuristic_avg_efficiency": heuristic_performance.get('efficiency_mean', 0)
            }
        }
        
        return rankings
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison across scenarios
        self._plot_performance_comparison(df, viz_dir)
        
        # 2. Algorithm performance heatmap
        self._plot_performance_heatmap(df, viz_dir)
        
        # 3. DRL vs Heuristic comparison
        self._plot_drl_vs_heuristic(df, viz_dir)
        
        # 4. Scenario difficulty analysis
        self._plot_scenario_difficulty(df, viz_dir)
        
        # 5. Energy efficiency analysis
        self._plot_energy_efficiency(df, viz_dir)
    
    def _plot_performance_comparison(self, df: pd.DataFrame, viz_dir: Path):
        """Plot performance comparison across scenarios"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison Across Scenarios', fontsize=16)
        
        # Makespan comparison
        makespan_pivot = df.pivot(index='scenario_name', columns='algorithm_name', values='makespan_mean')
        sns.heatmap(makespan_pivot, annot=True, fmt='.2f', ax=axes[0,0], cmap='YlOrRd')
        axes[0,0].set_title('Makespan (lower is better)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Energy comparison
        energy_pivot = df.pivot(index='scenario_name', columns='algorithm_name', values='energy_mean')
        sns.heatmap(energy_pivot, annot=True, fmt='.4f', ax=axes[0,1], cmap='YlOrRd')
        axes[0,1].set_title('Total Energy (lower is better)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Efficiency comparison
        efficiency_pivot = df.pivot(index='scenario_name', columns='algorithm_name', values='efficiency_mean')
        sns.heatmap(efficiency_pivot, annot=True, fmt='.2f', ax=axes[1,0], cmap='YlGnBu')
        axes[1,0].set_title('Energy Efficiency (higher is better)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_pivot = df.pivot(index='scenario_name', columns='algorithm_name', values='success_rate_mean')
        sns.heatmap(success_pivot, annot=True, fmt='.2f', ax=axes[1,1], cmap='YlGnBu')
        axes[1,1].set_title('Success Rate (higher is better)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, df: pd.DataFrame, viz_dir: Path):
        """Plot algorithm performance heatmap"""
        
        # Normalize metrics for comparison
        df_norm = df.copy()
        df_norm['makespan_norm'] = 1 / (df_norm['makespan_mean'] / df_norm['makespan_mean'].max())  # Invert for "higher is better"
        df_norm['energy_norm'] = 1 / (df_norm['energy_mean'] / df_norm['energy_mean'].max())  # Invert for "higher is better"
        df_norm['efficiency_norm'] = df_norm['efficiency_mean'] / df_norm['efficiency_mean'].max()
        df_norm['success_norm'] = df_norm['success_rate_mean']
        
        # Calculate overall score
        df_norm['overall_score'] = (df_norm['makespan_norm'] + df_norm['energy_norm'] + 
                                  df_norm['efficiency_norm'] + df_norm['success_norm']) / 4
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        score_pivot = df_norm.pivot(index='scenario_name', columns='algorithm_name', values='overall_score')
        sns.heatmap(score_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
        plt.title('Overall Performance Score (higher is better)')
        plt.xlabel('Algorithm')
        plt.ylabel('Scenario')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drl_vs_heuristic(self, df: pd.DataFrame, viz_dir: Path):
        """Plot DRL vs Heuristic comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DRL vs Heuristic Algorithms Comparison', fontsize=16)
        
        metrics = ['makespan_mean', 'energy_mean', 'efficiency_mean', 'success_rate_mean']
        titles = ['Makespan', 'Total Energy', 'Energy Efficiency', 'Success Rate']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Box plot comparing DRL vs Heuristic
            sns.boxplot(data=df, x='algorithm_type', y=metric, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Algorithm Type')
            
            # Add statistical annotation
            drl_values = df[df['algorithm_type'] == 'drl'][metric]
            heuristic_values = df[df['algorithm_type'] == 'heuristic'][metric]
            
            if len(drl_values) > 0 and len(heuristic_values) > 0:
                from scipy import stats
                statistic, p_value = stats.mannwhitneyu(drl_values, heuristic_values, alternative='two-sided')
                ax.text(0.5, 0.95, f'p-value: {p_value:.4f}', transform=ax.transAxes, 
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'drl_vs_heuristic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_difficulty(self, df: pd.DataFrame, viz_dir: Path):
        """Plot scenario difficulty analysis"""
        
        # Calculate scenario difficulty metrics
        scenario_stats = df.groupby('scenario_name').agg({
            'makespan_mean': ['mean', 'std'],
            'success_rate_mean': 'mean',
            'execution_time_mean': 'mean'
        }).round(3)
        
        scenario_stats.columns = ['avg_makespan', 'makespan_variance', 'avg_success_rate', 'avg_exec_time']
        scenario_stats = scenario_stats.reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Scenario Difficulty Analysis', fontsize=16)
        
        # Average makespan per scenario
        axes[0,0].bar(range(len(scenario_stats)), scenario_stats['avg_makespan'])
        axes[0,0].set_title('Average Makespan by Scenario')
        axes[0,0].set_xlabel('Scenario')
        axes[0,0].set_ylabel('Makespan')
        axes[0,0].set_xticks(range(len(scenario_stats)))
        axes[0,0].set_xticklabels(scenario_stats['scenario_name'], rotation=45)
        
        # Makespan variance (difficulty indicator)
        axes[0,1].bar(range(len(scenario_stats)), scenario_stats['makespan_variance'])
        axes[0,1].set_title('Makespan Variance by Scenario (Difficulty)')
        axes[0,1].set_xlabel('Scenario')
        axes[0,1].set_ylabel('Makespan Std Dev')
        axes[0,1].set_xticks(range(len(scenario_stats)))
        axes[0,1].set_xticklabels(scenario_stats['scenario_name'], rotation=45)
        
        # Success rate per scenario
        axes[1,0].bar(range(len(scenario_stats)), scenario_stats['avg_success_rate'])
        axes[1,0].set_title('Average Success Rate by Scenario')
        axes[1,0].set_xlabel('Scenario')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_xticks(range(len(scenario_stats)))
        axes[1,0].set_xticklabels(scenario_stats['scenario_name'], rotation=45)
        
        # Execution time per scenario
        axes[1,1].bar(range(len(scenario_stats)), scenario_stats['avg_exec_time'])
        axes[1,1].set_title('Average Execution Time by Scenario')
        axes[1,1].set_xlabel('Scenario')
        axes[1,1].set_ylabel('Execution Time (s)')
        axes[1,1].set_xticks(range(len(scenario_stats)))
        axes[1,1].set_xticklabels(scenario_stats['scenario_name'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'scenario_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_efficiency(self, df: pd.DataFrame, viz_dir: Path):
        """Plot energy efficiency analysis"""
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot: Energy vs Makespan
        for algo_type in df['algorithm_type'].unique():
            subset = df[df['algorithm_type'] == algo_type]
            plt.scatter(subset['makespan_mean'], subset['energy_mean'], 
                       label=f'{algo_type.upper()} algorithms', alpha=0.7, s=60)
        
        plt.xlabel('Makespan')
        plt.ylabel('Total Energy (Wh)')
        plt.title('Energy vs Makespan Trade-off Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add algorithm labels
        for _, row in df.iterrows():
            plt.annotate(row['algorithm_name'], 
                        (row['makespan_mean'], row['energy_mean']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'energy_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        
        if not self.aggregated_results:
            return "No results available. Run evaluation first."
        
        # Load analysis
        analysis_file = self.output_dir / "analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
        else:
            analysis = {"error": "Analysis not found"}
        
        # Generate report
        report = f"""
# Comprehensive DRL vs Heuristic Evaluation Report

## Executive Summary

This report presents a comprehensive comparison of Deep Reinforcement Learning (DRL) agents
against traditional heuristic algorithms for workflow scheduling in cloud environments.

### Key Findings

- **Total Scenarios Evaluated**: {analysis.get('summary', {}).get('total_scenarios', 'N/A')}
- **Total Algorithms Tested**: {analysis.get('summary', {}).get('total_algorithms', 'N/A')}
- **Heuristic Algorithms**: {analysis.get('summary', {}).get('heuristic_algorithms', 'N/A')}
- **DRL Agents**: {analysis.get('summary', {}).get('drl_algorithms', 'N/A')}

### Overall Best Performers

- **Best Makespan**: {analysis.get('summary', {}).get('overall_best_makespan', {}).get('algorithm', 'N/A')} 
  ({analysis.get('summary', {}).get('overall_best_makespan', {}).get('value', 'N/A'):.2f})
- **Best Energy Efficiency**: {analysis.get('summary', {}).get('overall_best_energy', {}).get('algorithm', 'N/A')} 
  ({analysis.get('summary', {}).get('overall_best_energy', {}).get('value', 'N/A'):.4f} Wh)
- **Best Overall Efficiency**: {analysis.get('summary', {}).get('overall_best_efficiency', {}).get('algorithm', 'N/A')} 
  ({analysis.get('summary', {}).get('overall_best_efficiency', {}).get('value', 'N/A'):.2f} tasks/Wh)

## Detailed Analysis

### Scenario Performance

The evaluation covered various scenario types including:
- Linear DAGs (high critical path dependency)
- Parallel DAGs (high parallelism potential)
- Diamond DAGs (fork-join patterns)
- Complex heterogeneous workloads
- Resource-constrained environments
- Power-efficiency focused scenarios

### Algorithm Comparison

#### DRL vs Heuristic Performance
"""
        
        # Add DRL vs Heuristic comparison if available
        drl_vs_heuristic = analysis.get('performance_rankings', {}).get('drl_vs_heuristic', {})
        if drl_vs_heuristic:
            report += f"""
- **Average Makespan**: DRL: {drl_vs_heuristic.get('drl_avg_makespan', 0):.2f}, Heuristic: {drl_vs_heuristic.get('heuristic_avg_makespan', 0):.2f}
- **Average Energy**: DRL: {drl_vs_heuristic.get('drl_avg_energy', 0):.4f} Wh, Heuristic: {drl_vs_heuristic.get('heuristic_avg_energy', 0):.4f} Wh
- **Average Efficiency**: DRL: {drl_vs_heuristic.get('drl_avg_efficiency', 0):.2f}, Heuristic: {drl_vs_heuristic.get('heuristic_avg_efficiency', 0):.2f}
"""
        
        report += f"""

## Conclusions and Recommendations

Based on the comprehensive evaluation across multiple scenarios and metrics:

1. **Algorithm Selection**: Choose algorithms based on specific optimization goals
   - For makespan optimization: Use the best-performing algorithm per scenario
   - For energy efficiency: Consider power-aware variants
   - For balanced performance: Evaluate trade-offs between metrics

2. **Scenario-Specific Insights**: Different algorithms excel in different scenarios
   - Linear DAGs benefit from critical path optimization
   - Parallel DAGs require effective load balancing
   - Resource-constrained scenarios need careful resource allocation

3. **DRL Agent Performance**: 
   - DRL agents show promise but require careful training and tuning
   - Performance varies significantly across different scenario types
   - Consider ensemble approaches combining DRL with heuristics

## Files Generated

- `raw_results.csv`: Detailed results for each run
- `aggregated_results.csv`: Statistical summary across runs
- `analysis.json`: Comprehensive analysis data
- `visualizations/`: Performance comparison plots

## Methodology

- **Evaluation Runs**: {self.num_runs} runs per algorithm-scenario combination
- **Metrics**: Makespan, Total Energy, Energy Efficiency, Success Rate
- **Statistical Analysis**: Mean and standard deviation across runs
- **Visualization**: Heatmaps, box plots, and scatter plots for analysis

---

*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open(self.output_dir / "evaluation_report.md", 'w') as f:
            f.write(report)
        
        return report


if __name__ == "__main__":
    # Example usage
    evaluator = ComprehensiveEvaluator(output_dir="test_evaluation", num_runs=2)
    
    # Add a random DRL agent for testing
    from evaluation.drl_agent_evaluator import RandomDRLAgent
    evaluator.add_drl_agent(RandomDRLAgent(seed=42))
    
    # Run evaluation on a subset of scenarios
    analysis = evaluator.run_comprehensive_evaluation(scenarios=["linear_abundant", "parallel_bottleneck"])
    
    # Generate report
    report = evaluator.generate_report()
    print("\nEvaluation completed!")
    print(f"Results saved to: {evaluator.output_dir}")

