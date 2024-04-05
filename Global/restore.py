import sys
sys.path.append("Global")
from PIL import Image, ImageFilter
import numpy as np
import cv2

class Remove_scratches():
    def __init__(self, input, mask, NL_use_mask = False):
        self.input = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
        self.mask = mask
        self.NL_use_mask = NL_use_mask
    
    def retouch_image(self, image_cv2):
        
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image_rgb)

        # Apply a filter to smooth the image (you can customize the filter)
        retouched_image = image.filter(ImageFilter.MedianFilter(1))

        print("without_scratches")
        
        return retouched_image

    def restore_scratched_parts(self, origin, mask):
        
        # print(origin.dtype, mask.dtype)
    
        ret , thresh = cv2.threshold(mask, 254, 255 , cv2.THRESH_BINARY)

        # lets make the lines thicker
        kernel = np.ones((1,1), np.uint8)
        mask = cv2.dilate(thresh , kernel , iterations=1)

        # lets restore the image
        restoredImage = cv2.inpaint(origin , mask , 3, cv2.INPAINT_TELEA)
        result = cv2.resize(restoredImage,(int(origin.shape[1] * 1.1), int(origin.shape[0] * 1.1)), interpolation=cv2.INTER_AREA)


        return result
    
    def irregular_hole_synthesize(self, img, mask):

        mask_np = img / 255
        img_new = mask * (1 - mask_np) + mask_np * 255

        hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

        return hole_img

    def main(self):

        restoredImage = self.restore_scratched_parts(self.input, self.mask)
        
        
        hsv_image = cv2.cvtColor(restoredImage, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)

        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
        
        v = clahe.apply(v)
    
        hsv_image = cv2.merge([h, s, v])
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return hsv_image
        return self.retouch_image(hsv_image)