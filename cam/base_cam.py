from matplotlib.pyplot import axis
import numpy as np
import torch
import cv2
from typing import List, Callable, Tuple
from torch import nn

from cam.activations_and_gradients import ActivationsAndGradients


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        # img = img - np.min(img)
        # img = img / (1e-7 + np.max(img))
        # if target_size is not None:
        #     img = cv2.resize(img, target_size)
        # result.append(img)
        img = img - torch.min(img)
        img = img / (1e-7 + torch.max(img))
        if target_size is not None:
            # kernel = nn.AdaptiveAvgPool2d(target_size)
            # img = kernel(torch.unsqueeze(img, 0))
            upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
            img = upsample(torch.unsqueeze(torch.unsqueeze(img, 0), 0))
            img = torch.squeeze(img)
        result.append(img)
    # result = np.float32(result)
    result = torch.stack(result)

    return result


class BaseCAM:
    def __init__(self,
                 model: nn.Module,
                 target_layers: List[nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
  
    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: nn.Module,
                      targets: List[nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> torch.Tensor:
        raise Exception("Not Implemented")

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[nn.Module],
                eigen_smooth: bool = False) -> torch.Tensor:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        model_out = self.activations_and_grads(input_tensor)

        if isinstance(model_out, tuple):
            outputs = model_out[0]
        else:
            outputs = model_out

        print(outputs.shape)
        if targets is None:
            # target_categories = np.argmax(outputs[..., 5:].cpu().data.numpy(), axis=-1) + 5
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = np.full_like(target_categories, 7)
            
            print(target_categories[..., :200])
            # target_categories = torch.argmax(outputs, axis=-1)
            print(target_categories.shape)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_height_width(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return height, width

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[nn.Module],
            eigen_smooth: bool) -> torch.Tensor:
        # activations_list = [a.cpu().data.numpy()
        #                     for a in self.activations_and_grads.activations]
        # grads_list = [g.cpu().data.numpy()
        #               for g in self.activations_and_grads.gradients]
        activations_list = self.activations_and_grads.activations
        grads_list = self.activations_and_grads.gradients
        
        target_size = self.get_target_height_width(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            # cam = np.maximum(cam, 0)
            cam = torch.maximum(cam, torch.zeros_like(cam))
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: List[torch.Tensor]) -> torch.Tensor:
        # cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # result = np.mean(cam_per_target_layer, axis=1)
        cam_per_target_layer = torch.cat(cam_per_target_layer, dim=1)
        cam_per_target_layer = torch.maximum(cam_per_target_layer, torch.zeros_like(cam_per_target_layer))
        result = torch.mean(cam_per_target_layer, dim=1)
        return scale_cam_image(result)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[nn.Module] = None,
                 eigen_smooth: bool = False) -> np.ndarray:

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
