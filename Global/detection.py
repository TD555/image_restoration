# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import cv2
import sys
sys.path.append("Global")

from detection_models import networks
from detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)



class Detect_scratches():
    def __init__(self, test, full_size = "full_size"):
        self.test_image = Image.fromarray(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
        self.full_size = full_size
        
    def data_transforms(self, method=Image.BICUBIC):
        if self.full_size == "full_size":
            ow, oh = self.test_image.size
            h = int(round(oh / 16) * 16)
            w = int(round(ow / 16) * 16)
            if (h == oh) and (w == ow):
                return self.test_image
            return self.test_image.resize((w, h), method)
        
        elif self.full_size == "scale_256":
            ow, oh = self.test_image.size
            pw, ph = ow, oh
            if ow < oh:
                ow = 256
                oh = ph / pw * 256
            else:
                oh = 256
                ow = pw / ph * 256

            h = int(round(oh / 16) * 16)
            w = int(round(ow / 16) * 16)
            if (h == ph) and (w == pw):
                return self.test_image
            return self.test_image.resize((w, h), method)

    def scale_tensor(self, img_tensor, default_scale=256):
        _, _, w, h = img_tensor.shape
        if w < h:
            ow = default_scale
            oh = h / w * default_scale
        else:
            oh = default_scale
            ow = w / h * default_scale

        oh = int(round(oh / 16) * 16)
        ow = int(round(ow / 16) * 16)

        return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


    def blend_mask(img, mask):

        np_img = np.array(img).astype("float")

        return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


    def main(self):
        print("initializing the dataloader")

        model = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )

        ## load model
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/detection/FT_Epoch_latest.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

        model.eval()

        print("processing")

        transformed_image_PIL = self.data_transforms()
        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = self.scale_tensor(scratch_image)


        # scratch_image_scale = scratch_image_scale.to(0)

        with torch.no_grad():
            P = torch.sigmoid(model(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")

        tensor = (P >= 0.4).float()
        numpy_image = tensor.squeeze().cpu().detach().numpy()

        gc.collect()
        torch.cuda.empty_cache()

        return transformed_image_PIL, (numpy_image * 255).astype(np.uint8)
