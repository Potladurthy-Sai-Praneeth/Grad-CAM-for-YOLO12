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
        self.model.model.train()
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on input_img requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients.clear()
        self.activations.clear()

        x = x.requires_grad_(True)

        return self.model.model(x)

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

    
    def get_cam_weights(self,input_img: np.array,activations,grads) -> np.ndarray:
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        else:
            raise ValueError("Invalid grads shape. Shape of grads should be 4 (2D image) or 5 (3D image).")

    def get_cam_image(self,input_img: np.array,activations,grads):
        weights = self.get_cam_weights(input_img,activations,grads)
        weighted_activations = weights[:, :, None, None] * activations
        return weighted_activations.sum(axis=1)

    def compute_cam_per_layer(self,input_img: np.array,activations_and_grads: object) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in activations_and_grads.gradients]

        if len(activations_list) != len(grads_list):
            raise ValueError("Number of activation layers and gradient layers must be the same.")

        target_size = input_img.shape[3], input_img.shape[2]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(activations_list)):
            layer_activations = activations_list[i]
            layer_grads = grads_list[i]
            cam = self.get_cam_image(input_img,layer_activations,layer_grads)                
            cam = np.maximum(cam, 0)
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self,_input_img: np.array, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        W, H = _input_img.shape[3], _input_img.shape[2]
        return self.scale_cam_image(result, target_size=(W, H))

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    
    def forward(self, input_img):
        input_img = input_img.to(self.device)

        with torch.set_grad_enabled(True):
            raw_outputs = self.activations_and_grads(input_img)
            # This returns a list 3 tensors one for each spatial resolution

            raw_output = raw_outputs[-1]  
            # This -1 is necessary as feature maps are cascaded in reverse way, if we use 0 we only get one gradient for the largest feature map

            batch_size, num_channels, height, width = raw_output.shape
            preds = raw_output.permute(0, 2, 3, 1).reshape(batch_size, -1, num_channels - 4)
            
            box = preds[..., :4]
            cls = preds[..., 4:]

            max_scores, max_indices = torch.max(cls, dim=-1)
            best_pred_idx = torch.argmax(max_scores, dim=-1)

            target_score = cls[0, best_pred_idx[0], max_indices[0, best_pred_idx[0]]]

            self.model.model.zero_grad()
            target_score.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_img,self.activations_and_grads)
        return self.aggregate_multi_layers(input_img,cam_per_layer)



    def __call__(self,input_img):
        # store for final resize
        return self.forward(input_img)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True