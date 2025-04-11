import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.lstm import ControlRefinementLSTM

class ControlModule(nn.Module):
    """
    Converts planned path to vehicle control commands
    """
    def __init__(self, hidden_dim=256):
        super(ControlModule, self).__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 64),  # Current pose + velocity
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # Path encoder
        self.path_encoder = nn.Sequential(
            nn.Linear(30, 128),  # 10 future waypoints with x,y,heading
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        
        # Control decoder
        self.control_decoder = nn.Sequential(
            nn.Linear(128 + 128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3)  # steering, throttle, brake
        )
        
        # Control refinement
        self.control_refinement = ControlRefinementLSTM(
            input_dim=3 + 6,  # control + vehicle state
            hidden_dim=64,
            output_dim=3
        )
    
    def forward(self, current_state, planned_path):
        batch_size = current_state.size(0)
        
        # Encode current state
        state_features = self.state_encoder(current_state)
        
        # Encode planned path (next 10 steps)
        path_input = planned_path[:, :10].reshape(batch_size, -1)  # Flatten first 10 steps
        path_features = self.path_encoder(path_input)
        
        # Generate control
        control_input = torch.cat([state_features, path_features], dim=1)
        raw_control = self.control_decoder(control_input)
        
        # Refine control with temporal consistency
        refined_control = self.control_refinement(
            raw_control, 
            current_state
        )
        
        # Apply control constraints
        steering = torch.tanh(refined_control[:, 0]).unsqueeze(1)  # Between -1 and 1
        throttle = torch.sigmoid(refined_control[:, 1]).unsqueeze(1)  # Between 0 and 1
        brake = torch.sigmoid(refined_control[:, 2]).unsqueeze(1)  # Between 0 and 1
        
        # Combine into final control output
        control_output = torch.cat([steering, throttle, brake], dim=1)
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'control': control_output
        }