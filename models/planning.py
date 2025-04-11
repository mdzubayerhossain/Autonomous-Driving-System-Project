import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanningModule(nn.Module):
    """
    Plans optimal path based on perception and prediction results
    """
    def __init__(self, feature_dim=256, hidden_dim=512, planning_horizon=80):
        super(PlanningModule, self).__init__()
        
        self.planning_horizon = planning_horizon
        
        # Feature encoder for state representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + 64, hidden_dim),  # perception features + vehicle state
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Cost volume computation
        self.cost_encoder = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=3, padding=1),  # perception + prediction features
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Path optimizer using differentiable optimization
        self.path_optimizer = DifferentiablePathOptimizer(
            horizon=planning_horizon,
            latent_dim=hidden_dim
        )
        
        # Value network for evaluation
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, perception_features, prediction_features, vehicle_state, route_features):
        batch_size = perception_features.size(0)
        
        # Encode current state
        state_input = torch.cat([perception_features.mean(dim=(2, 3)), vehicle_state], dim=1)
        state_encoding = self.feature_encoder(state_input)
        
        # Compute cost volume
        cost_input = torch.cat([perception_features, prediction_features], dim=1)
        cost_volume = self.cost_encoder(cost_input)
        
        # Generate and optimize paths
        paths, path_features = self.path_optimizer(state_encoding, cost_volume, route_features)
        
        # Evaluate path quality
        path_values = self.value_network(path_features)
        
        # Select best path
        best_path_idx = torch.argmax(path_values, dim=1)
        best_paths = torch.gather(paths, 1, best_path_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.planning_horizon, 3))
        best_paths = best_paths.squeeze(1)  # [batch, horizon, 3] (x, y, heading)
        
        return {
            'trajectories': paths,          # All candidate trajectories
            'values': path_values,          # Value scores for each trajectory
            'best_trajectory': best_paths,  # Selected best trajectory
            'cost_volume': cost_volume      # For visualization
        }


class DifferentiablePathOptimizer(nn.Module):
    """Optimizes paths considering costs, constraints, and vehicle dynamics"""
    def __init__(self, horizon, latent_dim):
        super(DifferentiablePathOptimizer, self).__init__()
        
        self.horizon = horizon
        
        # Path generator network (produces control points for paths)
        self.path_generator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 10 * 3)  # 10 control points with x,y,heading
        )
        
        # Path optimizer layers
        # This would typically implement differentiable optimization or sampling-based planning
        self.optimizer_layers = nn.Sequential(
            nn.Linear(latent_dim + 10*3, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, horizon * 3)  # x,y,heading for each step
        )
    
    def forward(self, state_encoding, cost_volume, route_features):
        batch_size = state_encoding.size(0)
        
        # Generate base control points
        control_points = self.path_generator(state_encoding)
        control_points = control_points.view(batch_size, 10, 3)
        
        # Generate multiple candidate paths
        num_candidates = 6
        paths = []
        path_features = []
        
        for i in range(num_candidates):
            # Add structured noise to control points for diversity
            noise_scale = 0.1 * (i + 1)
            noisy_control_points = control_points + torch.randn_like(control_points) * noise_scale
            
            # Flatten and concatenate with context
            optimizer_input = torch.cat([
                state_encoding,
                noisy_control_points.view(batch_size, -1)
            ], dim=1)
            
            # Generate full path
            path = self.optimizer_layers(optimizer_input)
            path = path.view(batch_size, self.horizon, 3)
            
            # Evaluate path cost (sample from cost volume)
            path_cost = self.sample_cost_along_path(cost_volume, path[:, :, :2])
            
            # Compute path features for evaluation
            path_feat = torch.cat([
                state_encoding,
                path[:, 0].view(batch_size, -1),   # start state
                path[:, -1].view(batch_size, -1),  # end state
                path_cost.view(batch_size, -1)     # path cost
            ], dim=1)
            
            paths.append(path)
            path_features.append(path_feat)
        
        # Stack all candidates
        paths = torch.stack(paths, dim=1)  # [batch, candidates, horizon, 3]
        path_features = torch.stack(path_features, dim=1)  # [batch, candidates, feat_dim]
        
        return paths, path_features
    
    def sample_cost_along_path(self, cost_volume, path_xy):
        """Sample cost volume along the path (simplified placeholder)"""
        # This would use grid sampling to get costs at each path point
        batch_size = cost_volume.size(0)
        return torch.rand(batch_size, self.horizon)  # Placeholder