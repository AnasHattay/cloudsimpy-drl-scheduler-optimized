"""
Energy Monitoring System for CloudSimPy

This module adds comprehensive energy consumption tracking to CloudSimPy,
including idle power consumption which is missing from the original implementation.
"""

import simpy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class MachineState(Enum):
    """Machine power states"""
    IDLE = "idle"
    BUSY = "busy"
    OFF = "off"


@dataclass
class EnergyProfile:
    """Energy consumption profile for a machine"""
    idle_power_watt: float  # Power consumption when idle
    peak_power_watt: float  # Power consumption at full utilization
    min_power_watt: float = None  # Minimum power (defaults to idle_power)
    
    def __post_init__(self):
        if self.min_power_watt is None:
            self.min_power_watt = self.idle_power_watt


@dataclass
class EnergyRecord:
    """Record of energy consumption over time"""
    timestamp: float
    machine_id: int
    state: MachineState
    power_watt: float
    energy_consumed_wh: float = 0.0  # Cumulative energy
    utilization: float = 0.0  # CPU utilization (0.0 to 1.0)


@dataclass
class MachineEnergyState:
    """Energy state tracking for a machine"""
    machine_id: int
    energy_profile: EnergyProfile
    current_state: MachineState = MachineState.IDLE
    current_power_watt: float = 0.0
    total_energy_consumed_wh: float = 0.0
    last_update_time: float = 0.0
    utilization: float = 0.0
    
    # Time tracking
    idle_time: float = 0.0
    busy_time: float = 0.0
    off_time: float = 0.0
    
    # Task tracking
    current_task_id: Optional[int] = None
    tasks_executed: int = 0
    
    def __post_init__(self):
        self.current_power_watt = self.energy_profile.idle_power_watt


class EnergyMonitor:
    """
    Comprehensive energy monitoring system for CloudSimPy
    
    Features:
    - Idle power consumption tracking
    - Dynamic power scaling based on utilization
    - Real-time energy consumption calculation
    - Detailed energy reporting and analytics
    """
    
    def __init__(self, env: simpy.Environment, enable_logging: bool = False):
        self.env = env
        self.machine_states: Dict[int, MachineEnergyState] = {}
        self.energy_records: List[EnergyRecord] = []
        self.enable_logging = enable_logging
        
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Start the energy monitoring process
        self.monitoring_process = env.process(self._energy_monitoring_loop())
    
    def register_machine(self, machine_id: int, energy_profile: EnergyProfile):
        """Register a machine for energy monitoring"""
        state = MachineEnergyState(
            machine_id=machine_id,
            energy_profile=energy_profile,
            last_update_time=self.env.now
        )
        self.machine_states[machine_id] = state
        
        if self.logger:
            self.logger.info(f"Registered machine {machine_id} for energy monitoring")
    
    def update_machine_state(self, machine_id: int, new_state: MachineState, 
                           utilization: float = 0.0, task_id: Optional[int] = None):
        """Update machine state and calculate energy consumption"""
        if machine_id not in self.machine_states:
            if self.logger:
                self.logger.warning(f"Machine {machine_id} not registered for energy monitoring")
            return
        
        machine_state = self.machine_states[machine_id]
        current_time = self.env.now
        time_delta = current_time - machine_state.last_update_time
        
        # Calculate energy consumed in previous state
        energy_delta = machine_state.current_power_watt * time_delta / 3600.0  # Convert to Wh
        machine_state.total_energy_consumed_wh += energy_delta
        
        # Update time tracking
        if machine_state.current_state == MachineState.IDLE:
            machine_state.idle_time += time_delta
        elif machine_state.current_state == MachineState.BUSY:
            machine_state.busy_time += time_delta
        elif machine_state.current_state == MachineState.OFF:
            machine_state.off_time += time_delta
        
        # Update state
        machine_state.current_state = new_state
        machine_state.utilization = max(0.0, min(1.0, utilization))
        machine_state.last_update_time = current_time
        
        # Calculate new power consumption based on state and utilization
        if new_state == MachineState.OFF:
            machine_state.current_power_watt = 0.0
        elif new_state == MachineState.IDLE:
            machine_state.current_power_watt = machine_state.energy_profile.idle_power_watt
        elif new_state == MachineState.BUSY:
            # Linear interpolation between idle and peak power based on utilization
            idle_power = machine_state.energy_profile.idle_power_watt
            peak_power = machine_state.energy_profile.peak_power_watt
            machine_state.current_power_watt = idle_power + (peak_power - idle_power) * utilization
        
        # Update task tracking
        if task_id is not None and machine_state.current_task_id != task_id:
            if task_id is not None:
                machine_state.tasks_executed += 1
            machine_state.current_task_id = task_id
        
        # Record energy data
        record = EnergyRecord(
            timestamp=current_time,
            machine_id=machine_id,
            state=new_state,
            power_watt=machine_state.current_power_watt,
            energy_consumed_wh=machine_state.total_energy_consumed_wh,
            utilization=utilization
        )
        self.energy_records.append(record)
        
        if self.logger:
            self.logger.debug(f"Machine {machine_id}: {new_state.value}, "
                            f"Power: {machine_state.current_power_watt:.2f}W, "
                            f"Utilization: {utilization:.2%}")
    
    def start_task_execution(self, machine_id: int, task_id: int, 
                           estimated_utilization: float = 1.0):
        """Mark the start of task execution on a machine"""
        self.update_machine_state(machine_id, MachineState.BUSY, 
                                estimated_utilization, task_id)
    
    def complete_task_execution(self, machine_id: int, task_id: int):
        """Mark the completion of task execution on a machine"""
        self.update_machine_state(machine_id, MachineState.IDLE, 0.0, None)
    
    def get_machine_energy_consumption(self, machine_id: int) -> float:
        """Get total energy consumption for a machine"""
        if machine_id in self.machine_states:
            # Update to current time first
            self._update_machine_energy(machine_id)
            return self.machine_states[machine_id].total_energy_consumed_wh
        return 0.0
    
    def get_total_energy_consumption(self) -> float:
        """Get total energy consumption across all machines"""
        total = 0.0
        for machine_id in self.machine_states:
            total += self.get_machine_energy_consumption(machine_id)
        return total
    
    def get_energy_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate energy efficiency metrics"""
        total_energy = self.get_total_energy_consumption()
        total_tasks = sum(state.tasks_executed for state in self.machine_states.values())
        total_time = self.env.now
        
        metrics = {
            "total_energy_wh": total_energy,
            "total_tasks": total_tasks,
            "energy_per_task_wh": total_energy / max(total_tasks, 1),
            "average_power_w": total_energy * 3600.0 / max(total_time, 1),
            "tasks_per_wh": total_tasks / max(total_energy, 0.001)
        }
        
        # Calculate idle vs busy energy breakdown
        total_idle_energy = 0.0
        total_busy_energy = 0.0
        
        for state in self.machine_states.values():
            idle_energy = (state.energy_profile.idle_power_watt * state.idle_time) / 3600.0
            # Approximate busy energy (this is simplified)
            busy_energy = state.total_energy_consumed_wh - idle_energy
            
            total_idle_energy += idle_energy
            total_busy_energy += max(0, busy_energy)
        
        metrics.update({
            "idle_energy_wh": total_idle_energy,
            "busy_energy_wh": total_busy_energy,
            "idle_energy_percentage": (total_idle_energy / max(total_energy, 0.001)) * 100,
            "busy_energy_percentage": (total_busy_energy / max(total_energy, 0.001)) * 100
        })
        
        return metrics
    
    def get_machine_utilization_stats(self) -> Dict[int, Dict[str, float]]:
        """Get utilization statistics for each machine"""
        stats = {}
        
        for machine_id, state in self.machine_states.items():
            total_time = state.idle_time + state.busy_time + state.off_time
            
            stats[machine_id] = {
                "idle_time": state.idle_time,
                "busy_time": state.busy_time,
                "off_time": state.off_time,
                "total_time": total_time,
                "utilization_percentage": (state.busy_time / max(total_time, 1)) * 100,
                "idle_percentage": (state.idle_time / max(total_time, 1)) * 100,
                "tasks_executed": state.tasks_executed,
                "total_energy_wh": state.total_energy_consumed_wh,
                "average_power_w": (state.total_energy_consumed_wh * 3600.0) / max(total_time, 1)
            }
        
        return stats
    
    def _update_machine_energy(self, machine_id: int):
        """Update energy consumption for a machine to current time"""
        if machine_id not in self.machine_states:
            return
        
        machine_state = self.machine_states[machine_id]
        current_time = self.env.now
        time_delta = current_time - machine_state.last_update_time
        
        if time_delta > 0:
            energy_delta = machine_state.current_power_watt * time_delta / 3600.0
            machine_state.total_energy_consumed_wh += energy_delta
            machine_state.last_update_time = current_time
            
            # Update time tracking
            if machine_state.current_state == MachineState.IDLE:
                machine_state.idle_time += time_delta
            elif machine_state.current_state == MachineState.BUSY:
                machine_state.busy_time += time_delta
            elif machine_state.current_state == MachineState.OFF:
                machine_state.off_time += time_delta
    
    def _energy_monitoring_loop(self):
        """Background process for continuous energy monitoring"""
        while True:
            # Update all machines to current time
            for machine_id in self.machine_states:
                self._update_machine_energy(machine_id)
            
            # Wait for next monitoring interval (1 second)
            yield self.env.timeout(1.0)
    
    def generate_energy_report(self) -> str:
        """Generate a comprehensive energy consumption report"""
        metrics = self.get_energy_efficiency_metrics()
        utilization_stats = self.get_machine_utilization_stats()
        
        report = []
        report.append("=" * 60)
        report.append("ENERGY CONSUMPTION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append(f"\nOVERALL METRICS:")
        report.append(f"  Total Energy Consumed: {metrics['total_energy_wh']:.4f} Wh")
        report.append(f"  Total Tasks Executed: {metrics['total_tasks']}")
        report.append(f"  Energy per Task: {metrics['energy_per_task_wh']:.4f} Wh/task")
        report.append(f"  Tasks per Wh: {metrics['tasks_per_wh']:.2f} tasks/Wh")
        report.append(f"  Average Power: {metrics['average_power_w']:.2f} W")
        
        # Energy breakdown
        report.append(f"\nENERGY BREAKDOWN:")
        report.append(f"  Idle Energy: {metrics['idle_energy_wh']:.4f} Wh ({metrics['idle_energy_percentage']:.1f}%)")
        report.append(f"  Busy Energy: {metrics['busy_energy_wh']:.4f} Wh ({metrics['busy_energy_percentage']:.1f}%)")
        
        # Per-machine statistics
        report.append(f"\nPER-MACHINE STATISTICS:")
        for machine_id, stats in utilization_stats.items():
            report.append(f"  Machine {machine_id}:")
            report.append(f"    Utilization: {stats['utilization_percentage']:.1f}%")
            report.append(f"    Energy Consumed: {stats['total_energy_wh']:.4f} Wh")
            report.append(f"    Tasks Executed: {stats['tasks_executed']}")
            report.append(f"    Average Power: {stats['average_power_w']:.2f} W")
        
        return "\n".join(report)


class EnergyAwareMachine:
    """
    Enhanced machine class with energy monitoring capabilities
    
    This extends CloudSimPy machines with comprehensive energy tracking
    including idle power consumption.
    """
    
    def __init__(self, machine_id: int, energy_profile: EnergyProfile, 
                 energy_monitor: EnergyMonitor):
        self.machine_id = machine_id
        self.energy_profile = energy_profile
        self.energy_monitor = energy_monitor
        
        # Register with energy monitor
        energy_monitor.register_machine(machine_id, energy_profile)
    
    def start_task(self, task_id: int, estimated_duration: float, 
                   cpu_utilization: float = 1.0):
        """Start executing a task"""
        self.energy_monitor.start_task_execution(
            self.machine_id, task_id, cpu_utilization
        )
    
    def complete_task(self, task_id: int):
        """Complete task execution"""
        self.energy_monitor.complete_task_execution(self.machine_id, task_id)
    
    def set_idle(self):
        """Set machine to idle state"""
        self.energy_monitor.update_machine_state(
            self.machine_id, MachineState.IDLE, 0.0
        )
    
    def shutdown(self):
        """Shutdown the machine"""
        self.energy_monitor.update_machine_state(
            self.machine_id, MachineState.OFF, 0.0
        )
    
    def get_energy_consumption(self) -> float:
        """Get total energy consumption"""
        return self.energy_monitor.get_machine_energy_consumption(self.machine_id)

