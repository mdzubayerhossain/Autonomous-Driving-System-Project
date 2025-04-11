 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .perception import PerceptionModule
from .prediction import PredictionModule
from .planning import PlanningModule
from .control import ControlModule
from .safety import SafetyModule

class AutonomousDrivingSystem(nn.Module):
    """
    Complete autonomous driving system integrating all modules
    """
    def __init__(self, config):
        super(AutonomousDrivingSystem, self).__init__()
        
        # Core modules
        self.perception = PerceptionModule(config)
        self.prediction = PredictionModule()
        self.planning = PlanningModule()
        self.control = ControlModule()
        self.safety = SafetyModule()
        
        # Feature integration
        self.integration_network = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),  # perception + prediction + planning features
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sensors, vehicle_state, route_info):
        """
        End-to-end autonomous driving pipeline
        
        Args:
            sensors: Dictionary containing camera, lidar, and radar data
            vehicle_state: Current state of the vehicle (position, velocity, etc.)
            route_info: High-level route information
            
        Returns:
            Dictionary with control commands and intermediate results
        """
        # Perception
        perception_results = self.perception(
            sensors['camera'],
            sensors['lidar'],
            sensors['radar']
        )
        
        # Prediction
        prediction_results = self.prediction(
            perception_results['objects']['cls_scores'],
            perception_results['objects']['attr_preds'],
            perception_results['features'],
            object_positions=None  # Would be extracted from perception results
        )
        
        # Planning
        planning_results = self.planning(
            perception_results['features'],
            prediction_results['trajectories'].mean(dim=2),  # Average across modes
            vehicle_state,
            route_info
        )
        
        # Control
        control_results = self.control(
            vehicle_state,
            planning_results['best_trajectory']
        )
        
        # Safety assessment and intervention
        safety_results = self.safety(
            perception_results['features'],
            prediction_results['trajectories'],
            planning_results['best_trajectory'],
            control_results['control'],
            vehicle_state
        )
        
        # Integrate features for system-level understanding
        integrated_features = self.integration_network(
            torch.cat([
                perception_results['features'].mean(dim=(2, 3)),
                prediction_results['trajectories'].mean(dim=(1, 2, 3)),
                planning_results['best_trajectory'].mean(dim=1) # Average across trajectory points              
            ], dim=1)
        )
        
        # Final control command
        final_control = control_results['control'] + safety_results['intervention']

        final_control = torch.clamp(final_control, min=-1.0, max=1.0)  # Ensure control limits
        
        return {
            'control': final_control,  # Final control command
            'safety': safety_results,   # Safety assessment results
            'planning': planning_results,  # Planning results
            'prediction': prediction_results,  # Prediction results
            'perception': perception_results,  # Perception results
            'integrated_features': integrated_features  # Integrated features for further processing
        }
        # Note: The actual implementation would involve more complex data handling and processing.