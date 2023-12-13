import torch
import torch.nn as nn
import torchvision
import numpy as np
import pytorch_lightning as pl

class MobileNetV2(nn.Module):
    def __init__(self,
                 model_name='mobilenet_v2',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        super().__init__()
        self.model_name = model_name.lower()

        if 'mobilenet_v2' in model_name:
            self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

        self.model.features[18] = nn.Conv2d(320, 512, kernel_size=1, stride=1)
        self.model.features.add_module("19", nn.BatchNorm2d(512))
        self.model.features.add_module("20", nn.Upsample(size=(20, 20), mode='bilinear', align_corners=False))

        self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.model.features(x)
        return x


# 12.06  
# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np
# import pytorch_lightning as pl

# class MobileNetV2(nn.Module):
#     def __init__(self,
#                  model_name='mobilenet_v2',
#                  pretrained=True,
#                  layers_to_freeze=2,
#                  layers_to_crop=[],
#                  ):
#         super().__init__()
#         self.model_name = model_name.lower()

#         if 'mobilenet_v2' in model_name:
#             self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

#         self.model.features[18] = nn.Conv2d(320, 1024, kernel_size=1, stride=1)
#         self.model.features.add_module("19", nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
#         self.model.features.add_module("20", nn.Upsample(size=(20, 20), mode='bilinear', align_corners=False))

#         self.model.classifier = nn.Identity()

#     def forward(self, x):
#         x = self.model.features(x)
#         return x

# batch1024
# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np
# import pytorch_lightning as pl

# class MobileNetV2(nn.Module):
#     def __init__(self,
#                  model_name='mobilenet_v2',
#                  pretrained=True,
#                  layers_to_freeze=2,
#                  layers_to_crop=[],
#                  ):
#         super().__init__()
#         self.model_name = model_name.lower()

#         if 'mobilenet_v2' in model_name:
#             self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

#         self.model.features[18] = nn.Conv2d(320, 1024, kernel_size=1, stride=1)
#         self.model.features.add_module("19", nn.BatchNorm2d(1024))
#         self.model.features.add_module("20", nn.Upsample(size=(20, 20), mode='bilinear', align_corners=False))

#         self.model.classifier = nn.Identity()

#     def forward(self, x):
#         x = self.model.features(x)
#         return x
    
    

# add new layer
# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np
# import pytorch_lightning as pl

# class MobileNetV2(nn.Module):
#     def __init__(self,
#                  model_name='mobilenet_v2',
#                  pretrained=True,
#                  layers_to_freeze=2,
#                  layers_to_crop=[],
#                  ):
#         super().__init__()
#         self.model_name = model_name.lower()

#         if 'mobilenet_v2' in model_name:
#             self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)

#         new_layer = nn.Sequential(
#             nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#         )
        
#         self.model.features.add_module("19", new_layer)
#         self.model.features.add_module("20", nn.Upsample(size=(20, 20), mode='bilinear', align_corners=False))

#         self.model.classifier = nn.Identity()

#     def forward(self, x):
#         x = self.model.features(x)
#         return x