import time
import torch
import torch.nn.functional as F
from config import *
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Union
from torchvision.transforms import Compose, Normalize

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients.clear()
        self.activations.clear()

        x = x.requires_grad_(True)

        # model.model(x) returns a tuple of tensors where the first element is the output with bounding boxe and confidence scores and 
        # the rest are the list of feature maps at different scales.
        return self.model.model(x)[0]  

    def release(self):
        for handle in self.handles:
            handle.remove()    

class YOLO12GradCAM:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.gradients = dict()
        self.activations = dict()
        self.target_layers =  [self.model.model.model[i] for i in self.model.model.model[-1].f]
        self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers)
        self.outputs = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    
    def avg_pool_gradients(self,input_tensor,activations,grads):
        '''
        This function computes the global average pooling of the gradients across the spatial dimensions.
        '''
        if len(grads.shape) == 4:
            return torch.mean(grads, dim=(2, 3))
        else:
            raise ValueError("Invalid grads shape. Shape of grads should be 4 (2D image) ")


    def mul_gradients_activations(self,input_tensor,activations,grads):
        '''
        This function multiples the neuron importance (avg pool grads) with activations and sum across channel dimensions
        '''
        weights = self.avg_pool_gradients(input_tensor,activations,grads)
        weighted_activations = weights[:, :, None, None] * activations
        return weighted_activations.sum(dim=1)

    def compute_cam_per_layer(self,input_tensor,activations_and_grads):
        activations_list = [a for a in activations_and_grads.activations]
        grads_list = [g for g in activations_and_grads.gradients]

        if len(activations_list) != len(grads_list):
            raise ValueError("Number of activation layers and gradient layers must be the same.")

        target_size = input_tensor.shape[3], input_tensor.shape[2]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(min(len(activations_list), len(grads_list))):
            layer_activations = activations_list[i]
            layer_grads = grads_list[i]
            cam = self.mul_gradients_activations(input_tensor,layer_activations,layer_grads)   
            cam = F.relu(cam)
            scaled = self.scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(scaled)

        return cam_per_target_layer

    def aggregate_multi_layers(self,cam_per_target_layer):
        '''
        As we hooked multiple layers, we need to aggregate the saliency maps from different layers.
        Here we use max operation to aggregate the saliency maps from different layers preserving the maximum gradient information across multiple paths
        '''
        cam_per_target_layer = torch.stack(cam_per_target_layer, dim=0)
        result = torch.max(cam_per_target_layer,dim=0).values
        return result

    def scale_cam_image(self,feature_map, target_size):
        '''
        This function uses bilinear interpolation to scale the cam image to the target size. 
        '''
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # Perform bilinear interpolation
        resized = F.interpolate(
            feature_map,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Remove batch dimension if it was added
        if squeeze_batch:
            resized = resized.squeeze(0)
        
        return resized

    
    def forward(self, input_img,gt_idx):
        assert gt_idx is not None, "Please provide gt_idx for the target class"
        input_img = input_img.to(self.device)

        with torch.set_grad_enabled(True):
            raw_outputs = self.activations_and_grads(input_img)
            index = 4 + gt_idx
            subset_scores = raw_outputs[:,index,:]
            target_score = subset_scores[:,torch.argmax(subset_scores)]

            self.model.model.zero_grad()
            target_score.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_img,self.activations_and_grads)
        return self.aggregate_multi_layers(cam_per_layer).squeeze(0)



    def __call__(self,input_img,gt_idx):
        # store for final resize
        return self.forward(input_img,gt_idx)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            return True