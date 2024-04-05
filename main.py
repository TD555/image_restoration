from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import io
import os
import asyncio
import numpy as np
from Global.detection import Detect_scratches
from Global.restore import Remove_scratches
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import uvicorn
import gdown
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

  
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    import traceback
    error_traceback = traceback.format_exc()
    logger.info(error_traceback)
    return JSONResponse(content={"error": exc.detail, "traceback" : error_traceback}, status_code=exc.status_code)
   
@app.get('/')
def main():
    return "Davtashen's images restoration model"


models_paths = {r'gfpgan/weights/detection_Resnet50_Final.pth' : '1u6499of3lh9HoKnyHsyWKrCjwGmYp89W', r'gfpgan/weights/GFPGANv1.4.pth' : '1BN-bkknXqhPKt_FXVvAmCAkqcrQnx__a',
                r'gfpgan/weights/parsing_parsenet.pth' : '1pNM6ISaxIJIcAtlbpWOYXBMzBsG6IE1A', r'gfpgan/weights/realesr-general-x4v3.pth' : '1Osn6G9-ErdPRYmDyHW41ZmtIIIIE-P3k',
                r'Global/checkpoints/detection/FT_Epoch_latest.pt' : '11IcdV0_VI0YWmlTm71gaeCmS6ZkVQMHL'}

for model_path, file_id in models_paths.items():
    if not os.path.isfile(model_path):
        if model_path.endswith('.pth'):
            gdown.download(id=file_id, output=model_path, format='.pth')
        elif model_path.endswith('.pt'):
            gdown.download(id=file_id, output=model_path, format='.pt')


sr_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

realesrganer = RealESRGANer(scale=4, model_path='gfpgan/weights/realesr-general-x4v3.pth', model=sr_model, tile=0, tile_pad=10, pre_pad=0, half=False)

face_enhancer = GFPGANer(model_path='gfpgan/weights/GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=realesrganer)

# Function to enhance image with GFPGAN
async def enhance_faces(img):
    # Enhance faces with GFPGAN
    loop = asyncio.get_event_loop()

    _, _, img_enhanced = await loop.run_in_executor(None, face_enhancer.enhance, img, False, False, True)
    return img_enhanced


async def is_image(file: UploadFile):
    return file.content_type in ['image/png', 'image/jpeg']
 
async def get_bytes(img, format='.jpg'):
    print(img.shape)
    _, img_encoded = cv2.imencode(format, img)
    img_bytes = img_encoded.tobytes()
    return img_bytes


@app.post('/restore')
async def restore(file: UploadFile=File(None)):

    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not await is_image(file):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        logger.info(file, type(file))
        data = file.file.read()
        file.file.close()
        
        img_format = file.content_type.split('/')[-1]
        
        headers = {
            "Content-Type": file.content_type
        }

        nparr = np.frombuffer(data, np.uint8)

        original_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    except: raise HTTPException(status_code=422)

    try:
        detect_model = Detect_scratches(original_image)
        input, mask = detect_model.main()
    except: raise HTTPException(status_code=422, detail="Error during detecting scratches")
    
    try:
        scratch_model = Remove_scratches(input, mask)
        output_img = scratch_model.main()
    except: raise HTTPException(status_code=422, detail="Error during removing scratches")
    
    
    try:
        height, width, _ = output_img.shape
        if width > 1000:
            resized_img = cv2.resize(output_img, (0, 0), fx=500/width, fy=500/width)
        elif height > 1000:
            resized_img = cv2.resize(output_img, (0, 0), fx=500/height, fy=500/height)
        else: resized_img = cv2.resize(output_img, (0, 0), fx=0.5, fy=0.5)
        
        img = await asyncio.wait_for(enhance_faces(resized_img), timeout=30)
        if img_format == 'png':
            img_bytes = await get_bytes(img, '.png')
        else: img_bytes = await get_bytes(img)
    
    except Exception as e:
        print(str(e))
        if img_format == 'png':
            img_bytes = await get_bytes(output_img, '.png')
        else: img_bytes = await get_bytes(output_img)
        
        return  StreamingResponse(io.BytesIO(img_bytes), headers=headers)

    return StreamingResponse(io.BytesIO(img_bytes), headers=headers)


if __name__ == "__main__":
    uvicorn.run("main:app", workers=5, host='192.168.0.175')