 
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionModule(nn.Module):
    """
    Processes multi-modal sensory inputs to detect objects, lanes, and traffic elements
    """
    def __init__(self, config):
        super(PerceptionModule, self).__init__()
        
        # Camera image processing backbone (ResNet variant)
        self.visual_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2)
        )
        
        # LiDAR point cloud processing network
        self.lidar_backbone = PointPillarsNetwork(
            in_channels=4,  # x, y, z, intensity
            out_channels=128
        )
        
        # Radar processing network
        self.radar_backbone = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # velocity + intensity
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Sensor fusion network
        self.fusion_network = MultiModalFusionNetwork(
            camera_channels=512,
            lidar_channels=128,
            radar_channels=64
        )
        
        # Object detection head
        self.object_detection = ObjectDetectionHead(
            in_channels=256, 
            num_classes=10  # car, pedestrian, cyclist, etc.
        )
        
        # Lane detection head
        self.lane_detection = LaneDetectionHead(
            in_channels=256
        )
        
        # Traffic sign/signal recognition
        self.traffic_recognition = TrafficElementHead(
            in_channels=256,
            num_elements=20  # stop signs, traffic lights, etc.
        )
    
    def forward(self, camera_data, lidar_data, radar_data):
        # Process individual sensor data
        camera_features = self.visual_backbone(camera_data)
        lidar_features = self.lidar_backbone(lidar_data)
        radar_features = self.radar_backbone(radar_data)
        
        # Fuse multi-modal features
        fused_features = self.fusion_network(camera_features, lidar_features, radar_features)
        
        # Generate perception outputs
        objects = self.object_detection(fused_features)
        lanes = self.lane_detection(fused_features)
        traffic_elements = self.traffic_recognition(fused_features)
        
        return {
            'objects': objects,
            'lanes': lanes,
            'traffic_elements': traffic_elements,
            'features': fused_features  # for downstream tasks
        }


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet architecture"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class PointPillarsNetwork(nn.Module):
    """Network for processing LiDAR point clouds using pillars representation"""
    def __init__(self, in_channels, out_channels):
        super(PointPillarsNetwork, self).__init__()
        # Simplified implementation - in practice would be more complex
        self.pillar_encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.scatter_bn = nn.BatchNorm2d(128)
        
        self.backbone = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, points):
        # In practice, this would include:
        # 1. Point cloud to pillars conversion
        # 2. Per-pillar feature extraction
        # 3. Scatter back to spatial grid
        # 4. Backbone processing
        
        # Simplified for demonstration
        pillar_features = self.pillar_encoder(points)
        pseudo_image = pillar_features.view(-1, 128, 200, 200)  # example dimensions
        pseudo_image = self.scatter_bn(pseudo_image)
        
        return self.backbone(pseudo_image)


class MultiModalFusionNetwork(nn.Module):
    """Fuses features from multiple sensor modalities"""
    def __init__(self, camera_channels, lidar_channels, radar_channels):
        super(MultiModalFusionNetwork, self).__init__()
        
        # Feature dimensionality alignment
        self.camera_projection = nn.Conv2d(camera_channels, 128, kernel_size=1)
        self.lidar_projection = nn.Conv2d(lidar_channels, 128, kernel_size=1)
        self.radar_projection = nn.Conv2d(radar_channels, 128, kernel_size=1)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(128)
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, camera_features, lidar_features, radar_features):
        # Align feature dimensions
        camera_proj = self.camera_projection(camera_features)
        lidar_proj = self.lidar_projection(lidar_features)
        radar_proj = self.radar_projection(radar_features)
        
        # Apply cross-modal attention
        attended_features = self.cross_attention(camera_proj, lidar_proj, radar_proj)
        
        # Concatenate and fuse
        concat_features = torch.cat([
            attended_features['camera'],
            attended_features['lidar'],
            attended_features['radar']
        ], dim=1)
        
        fused_features = self.fusion_conv(concat_features)
        
        return fused_features


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for feature fusion"""
    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # Query, key, value projections for each modality
        self.q_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
    
    def forward(self, camera_features, lidar_features, radar_features):
        batch_size = camera_features.size(0)
        
        # Process each modality with attention against others
        output = {}
        modalities = {
            'camera': camera_features,
            'lidar': lidar_features,
            'radar': radar_features
        }
        
        for target_name, target_feat in modalities.items():
            q = self.q_proj(target_feat).flatten(2).permute(0, 2, 1)
            
            # Aggregate attention from other modalities
            attended_feat = target_feat
            for source_name, source_feat in modalities.items():
                if source_name != target_name:
                    k = self.k_proj(source_feat).flatten(2).permute(0, 2, 1)
                    v = self.v_proj(source_feat).flatten(2).permute(0, 2, 1)
                    
                    # Compute attention
                    attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.feature_dim ** 0.5)
                    attn_weights = F.softmax(attn_weights, dim=2)
                    
                    # Apply attention
                    attn_output = torch.bmm(attn_weights, v)
                    attn_output = attn_output.permute(0, 2, 1).view(batch_size, self.feature_dim, 
                                                                   target_feat.size(2), target_feat.size(3))
                    
                    attended_feat = attended_feat + self.out_proj(attn_output)
            
            output[target_name] = attended_feat
        
        return output


class ObjectDetectionHead(nn.Module):
    """Object detection head for identifying vehicles, pedestrians, etc."""
    def __init__(self, in_channels, num_classes):
        super(ObjectDetectionHead, self).__init__()
        
        # Feature processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classification branch
        self.classification = nn.Conv2d(256, num_classes * 9, kernel_size=1)  # 9 anchors per location
        
        # Bounding box regression branch
        self.bbox_regression = nn.Conv2d(256, 4 * 9, kernel_size=1)  # x, y, w, h for 9 anchors
        
        # Object attributes (optional)
        self.attributes = nn.Conv2d(256, 8 * 9, kernel_size=1)  # velocity, orientation, etc.
    
    def forward(self, features):
        features = self.conv_layers(features)
        
        # Predict outputs
        cls_scores = self.classification(features)
        bbox_preds = self.bbox_regression(features)
        attr_preds = self.attributes(features)
        
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'attr_preds': attr_preds
        }


class LaneDetectionHead(nn.Module):
    """Lane detection head for identifying road markings and boundaries"""
    def __init__(self, in_channels):
        super(LaneDetectionHead, self).__init__()
        
        self.lane_network = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Lane segmentation
        self.lane_segmentation = nn.Conv2d(128, 3, kernel_size=1)  # background, lane, road edge
        
        # Lane embedding for instance separation
        self.lane_embedding = nn.Conv2d(128, 16, kernel_size=1)
    
    def forward(self, features):
        lane_features = self.lane_network(features)
        
        # Predict outputs
        segmentation = self.lane_segmentation(lane_features)
        embedding = self.lane_embedding(lane_features)
        
        return {
            'segmentation': segmentation,
            'embedding': embedding
        }


class TrafficElementHead(nn.Module):
    """Detects and classifies traffic signs, signals, and other road elements"""
    def __init__(self, in_channels, num_elements):
        super(TrafficElementHead, self).__init__()
        
        self.traffic_network = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Traffic element classification
        self.classification = nn.Conv2d(128, num_elements * 5, kernel_size=1)  # 5 anchors
        
        # Bounding box regression
        self.bbox_regression = nn.Conv2d(128, 4 * 5, kernel_size=1)  # x, y, w, h for 5 anchors
    
    def forward(self, features):
        traffic_features = self.traffic_network(features)
        
        # Predict outputs
        cls_scores = self.classification(traffic_features)
        bbox_preds = self.bbox_regression(traffic_features)
        
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }


class PredictionModule(nn.Module):
    """
    Predicts future trajectories of detected objects
    """
    def __init__(self, feature_dim=128, hidden_dim=256, future_steps=80):
        super(PredictionModule, self).__init__()
        
        self.future_steps = future_steps  # Prediction horizon (e.g., 8 seconds at 10Hz)
        
        # Object feature extraction
        self.object_encoder = nn.Sequential(
            nn.Linear(feature_dim + 8, hidden_dim),  # features + object attributes
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Map feature extraction (road network context)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Social interaction between objects
        self.interaction_gnn = SocialInteractionGNN(hidden_dim)
        
        # Trajectory decoder (multi-modal prediction)
        self.trajectory_decoder = MultiModalTrajectoryDecoder(
            hidden_dim=hidden_dim,
            future_steps=future_steps,
            num_modes=6  # Predict multiple possible trajectories
        )
    
    def forward(self, object_features, object_attributes, perception_features, object_positions):
        batch_size = object_features.size(0)
        num_objects = object_features.size(1)
        
        # Extract object features
        object_input = torch.cat([object_features, object_attributes], dim=-1)
        object_embeddings = self.object_encoder(object_input)
        
        # Extract map features for each object
        map_features = self.map_encoder(perception_features)
        sampled_map_features = self.sample_map_at_positions(map_features, object_positions)
        
        # Combine with object features
        object_context = torch.cat([object_embeddings, sampled_map_features], dim=-1)
        
        # Model social interactions
        interaction_features = self.interaction_gnn(object_context, object_positions)
        
        # Predict trajectories
        trajectories, confidences = self.trajectory_decoder(interaction_features)
        
        return {
            'trajectories': trajectories,  # Shape: [batch, num_objects, num_modes, future_steps, 2]
            'confidences': confidences     # Shape: [batch, num_objects, num_modes]
        }
    
    def sample_map_at_positions(self, map_features, positions):
        """Sample map features at object positions using grid sampling"""
        # Implementation would use grid_sample or similar to extract features
        # This is a simplified placeholder
        batch_size, num_objects = positions.shape[0], positions.shape[1]
        return torch.randn(batch_size, num_objects, 64)  # Placeholder


class SocialInteractionGNN(nn.Module):
    """Graph Neural Network for modeling interactions between objects"""
    def __init__(self, hidden_dim):
        super(SocialInteractionGNN, self).__init__()
        
        self.edge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # features of both objects + relative position
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.node_update = nn.GRUCell(hidden_dim * 2, hidden_dim)
    
    def forward(self, object_features, object_positions):
        batch_size, num_objects, feature_dim = object_features.shape
        updated_features = object_features.clone()
        
        # For each object, aggregate information from all other objects
        for i in range(num_objects):
            # Compute pairwise features
            ego_features = object_features[:, i:i+1].expand(-1, num_objects, -1)
            rel_positions = object_positions[:, i:i+1] - object_positions
            
            # Create edge inputs
            edge_inputs = torch.cat([
                ego_features,
                object_features,
                rel_positions
            ], dim=-1)
            
            # Compute edge features
            edge_features = self.edge_network(edge_inputs.view(-1, edge_inputs.size(-1)))
            edge_features = edge_features.view(batch_size, num_objects, -1)
            
            # Aggregate (excluding self)
            mask = torch.ones(num_objects, device=edge_features.device)
            mask[i] = 0
            mask = mask.view(1, -1, 1)
            
            agg_features = (edge_features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
            
            # Update node features
            combined = torch.cat([object_features[:, i], agg_features], dim=1)
            updated_features[:, i] = self.node_update(combined, object_features[:, i])
        
        return updated_features


class MultiModalTrajectoryDecoder(nn.Module):
    """Predicts multiple possible future trajectories with confidence scores"""
    def __init__(self, hidden_dim, future_steps, num_modes):
        super(MultiModalTrajectoryDecoder, self).__init__()
        
        self.future_steps = future_steps
        self.num_modes = num_modes
        
        # Mode selection network
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modes),
        )
        
        # Trajectory prediction networks (one per mode)
        self.trajectory_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, future_steps * 2)  # x,y for each future step
            ) for _ in range(num_modes)
        ])
    
    def forward(self, features):
        batch_size, num_objects, feature_dim = features.shape
        
        # Predict mode confidences
        mode_logits = self.mode_selector(features.view(-1, feature_dim))
        mode_probs = F.softmax(mode_logits, dim=-1)
        mode_probs = mode_probs.view(batch_size, num_objects, self.num_modes)
        
        # Predict trajectories for each mode
        trajectories = []
        for i, decoder in enumerate(self.trajectory_decoders):
            traj = decoder(features.view(-1, feature_dim))
            traj = traj.view(batch_size, num_objects, self.future_steps, 2)
            trajectories.append(traj)
        
        # Stack all mode predictions
        trajectories = torch.stack(trajectories, dim=2)  # [batch, objects, modes, steps, 2]
        
        return trajectories, mode_probs


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


class ControlRefinementLSTM(nn.Module):
    """Ensures smooth and consistent control over time"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ControlRefinementLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize hidden states
        self.hidden_state = None
        self.cell_state = None
    
    def forward(self, control_input, vehicle_state):
        batch_size = control_input.size(0)
        
        # Concatenate with vehicle state
        lstm_input = torch.cat([control_input, vehicle_state], dim=1)
        lstm_input = lstm_input.unsqueeze(1)  # Add time dimension
        
        # Initialize hidden states if needed
        if self.hidden_state is None or self.cell_state is None:
            self.hidden_state = torch.zeros(2, batch_size, self.lstm.hidden_size, device=control_input.device)
            self.cell_state = torch.zeros(2, batch_size, self.lstm.hidden_size, device=control_input.device)
        
        # Forward pass through LSTM
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            lstm_input, 
            (self.hidden_state.detach(), self.cell_state.detach())
        )
        
        # Generate refined control
        refined_control = self.output_layer(lstm_out.squeeze(1))
        
        return refined_control
    
    def reset_state(self):
        """Reset LSTM states between episodes"""
        self.hidden_state = None
        self.cell_state = None


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
                planning_results['best_trajectory'].mean(dim=1)
            ], dim=1)
        )
        
        return {
            'perception': perception_results,
            'prediction': prediction_results,
            'planning': planning_results,
            'control': control_results,
            'safety': safety_results,
            'final_control': safety_results['final_control'],
            'integrated_features': integrated_features
        }


# Configuration and utility functions
class AutoDrivingConfig:
    """Configuration parameters for the autonomous driving system"""
    def __init__(self):
        # Perception configuration
        self.camera_channels = 3
        self.lidar_points = 120000
        self.radar_dims = [400, 400]
        
        # Model dimensions
        self.feature_dim = 256
        self.hidden_dim = 512
        
        # Prediction configuration
        self.prediction_horizon = 80  # 8 seconds at 10Hz
        self.num_prediction_modes = 6
        
        # Planning configuration
        self.planning_horizon = 80  # 8 seconds
        self.num_planning_samples = 64
        
        # Safety thresholds
        self.collision_threshold = 0.7
        self.stability_threshold = 0.8
        self.compliance_threshold = 0.6


def create_autonomous_driving_model():
    """Factory function to create and initialize the model"""
    config = AutoDrivingConfig()
    model = AutonomousDrivingSystem(config)
    
    # Initialize with pretrained weights if available
    # model.load_state_dict(torch.load('pretrained_weights.pth'))
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_autonomous_driving_model()
    
    # Create dummy inputs
    batch_size = 1
    camera_data = torch.randn(batch_size, 3, 480, 640)
    lidar_data = torch.randn(batch_size, 120000, 4)
    radar_data = torch.randn(batch_size, 2, 400, 400)
    vehicle_state = torch.randn(batch_size, 6)  # position (3) + velocity (3)
    route_info = torch.randn(batch_size, 50, 2)  # 50 route points
    
    # Pack sensor data
    sensors = {
        'camera': camera_data,
        'lidar': lidar_data,
        'radar': radar_data
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(sensors, vehicle_state, route_info)
    
    # Extract control commands
    steering = outputs['final_control'][:, 0].item()
    throttle = outputs['final_control'][:, 1].item()
    brake = outputs['final_control'][:, 2].item()
    
    print(f"Steering: {steering:.4f}, Throttle: {throttle:.4f}, Brake: {brake:.4f}")
    print(f"Overall risk: {outputs['safety']['overall_risk'].item():.4f}")