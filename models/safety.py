 
import torch
import torch.nn as nn
import torch.nn.functional as F

class SafetyModule(nn.Module):
    """
    Safety monitoring and intervention system
    """
    def __init__(self):
        super(SafetyModule, self).__init__()
        
        # Safety assessment network
        self.safety_network = nn.Sequential(
            nn.Linear(256 + 3 + 6, 128),  # perception features + control + vehicle state
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)  # collision risk, stability risk, compliance risk, overall risk
        )
        
        # Emergency intervention network
        self.intervention_network = nn.Sequential(
            nn.Linear(4 + 3, 64),  # risk assessment + proposed control
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # emergency control adjustment
        )
    
    def forward(self, perception_features, predicted_trajectories, planned_path, 
                proposed_control, vehicle_state):
        batch_size = perception_features.size(0)
        
        # Flatten perception features
        perception_flat = perception_features.mean(dim=(2, 3))
        
        # Assess safety risks
        safety_input = torch.cat([
            perception_flat,
            proposed_control,
            vehicle_state
        ], dim=1)
        
        risk_assessment = self.safety_network(safety_input)
        collision_risk = torch.sigmoid(risk_assessment[:, 0])
        stability_risk = torch.sigmoid(risk_assessment[:, 1])
        compliance_risk = torch.sigmoid(risk_assessment[:, 2])
        overall_risk = torch.sigmoid(risk_assessment[:, 3])
        
        # Determine if intervention is needed
        intervention_threshold = 0.7
        intervention_needed = overall_risk > intervention_threshold
        
        # Calculate intervention control adjustments
        intervention_input = torch.cat([
            risk_assessment,
            proposed_control
        ], dim=1)
        
        control_adjustment = self.intervention_network(intervention_input)
        
        # Apply intervention if needed
        final_control = torch.where(
            intervention_needed.unsqueeze(1).expand(-1, 3),
            control_adjustment,
            proposed_control
        )
        
        return {
            'final_control': final_control,
            'collision_risk': collision_risk,
            'stability_risk': stability_risk,
            'compliance_risk': compliance_risk,
            'overall_risk': overall_risk,
            'intervention_applied': intervention_needed
        }