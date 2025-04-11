 import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.residual import ResidualBlock
from ..networks.point_pillars import PointPillarsNetwork
from ..networks.attention import CrossModalAttention

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
