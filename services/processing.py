from fastapi import UploadFile, APIRouter, Form, HTTPException,Request, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from skimage.transform import resize
from function_simulate.wifi_simulate.simulate_wifi import *
# from function_simulate.heat_model.main import wallExtraction_ai
from PIL import Image
import json
import jwt
from configs import *
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import aiofiles
import imutils
router = APIRouter(
    prefix="/wifi-simulation-api",
    tags=["2D"]
)

def check_token(request: Request) -> None:
    """
    Extract and validate the JWT token from the authorization header
    """
    try:
        token = request.headers["Authorization"].replace("Bearer ","")
        with open(base_dir.joinpath('private','jwtRS256','jwtRS256.key.pub'), "r") as fkey:
            key_public = fkey.read()
        token_decoded = jwt.decode(token, key_public, algorithms="RS256")
        if token_decoded["name"] != "pnc":
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as ex:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/transform_image")
async def transform_image(request: Request,file: UploadFile = File(...)):
    # check_token(request)
    # try:
    while True:
        async def process_image():
            content = await file.read()
            image_data = BytesIO(content)

            with Image.open(image_data) as image:
                image_np = np.array(image)
                # image_resized = image.resize((256, 256), resample=Image.LANCZOS)
                image =  imutils.resize(image_np, height=256)
                target, lines = wallExtraction(image)
                target =  np.array(target/target.max(),dtype=np.uint8) 
                # target_cut = roi_image(target, 256,256)
                # target_cut = target_cut*255
                # target_cut = (255 - target_cut*255).astype(np.uint8)
                
                buffered = BytesIO()
                target_cut = Image.fromarray(target*255)
                # target_cut.save(buffered, format="JPEG")
                encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return encoded_string, target_cut, lines.tolist()

        encoded_string, image_shape, lines = await process_image()
        img_path = f"{img_dir}/api_1.jpg"
        print("linh",np.array(image_shape).shape)
        cv2.imwrite(img_path, np.array(image_shape))
        # result = {'image': encoded_string,'image_shape': image_shape, 'lines': lines}      
        # return JSONResponse(content=result)
        return FileResponse(img_path)
    # except Exception as ex:
    #     raise HTTPException(
    #         status_code=400, detail="Invalid file or process image - {}".format(str(ex)))

@router.post("/decode_image")
async def test_api(request: Request, image: str = Form(...)):
    data = json.loads(image)
    decoded_image=base64.b64decode((data["image"]))
    with open(f'{img_dir}/image.jpeg', 'wb') as img_file:
        img_file.write(decoded_image)
    print("letrunglinh", data["image_shape"])
    return FileResponse(f'{img_dir}/image.jpeg')


@router.post("/simulate_wifi")
async def simulate_wifi(request: Request, file: UploadFile = File(...), data: str = Form(...)):
    # data = json.load(data)
    # shape = data["image_shape"]
    # print("letrunglinh",shape)
    # lines = data['lines']



    # image = np.zeros(shape[0],shape[1])
    # print("image",image.shape)
    # print("check")
    # return 1

    check_token(request)
    try:
        with open(f"{img_dir}/tempfile_2.jpg", "wb") as buffer:
            buffer.write(await file.read())
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, Image.open, f"{img_dir}/tempfile_2.jpg")
        img = np.array(img)
        labledLines_binary_temp = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)[1]
        labledLines_binary =  np.array(labledLines_binary_temp/labledLines_binary_temp.max(),dtype=np.uint8) 
    except Exception as ex:
        raise HTTPException(
            status_code=400, detail="Invalid file - {}".format(str(ex)))
    # try :
    # while True:
    
    
    data = json.loads(data)
    width = data["width"]
    height = data["length"]
    ghz = data["ghz"]
    TxLocation = data["TxLocation"]
    TxLocation = np.array(TxLocation)
    TxNum = len(TxLocation) 
    TxPower = data["TxPower"]
    wall_attenuation = data["wall_attenuation"]
    # except Exception as ex:
    #     raise HTTPException(
    #         status_code=400, detail="Invalid data - {}".format(str(ex)))
    # try :
    ################################    MAIN   ##############################################
    #########################################################################################
    # 2.4ghz 2.437 5ghz 5.507
    sim = multiWallModel() # creating a propagation object instance
    # (640,480) -> 1.5, 0.75
    sim.nodePerMeter = 1.5  # This identifies the resolution of the estimation
    # sim.pathLossExp = np.array([1.55, 1.55,1.55,1.55,1.4])  # path loss exponent of propagation! Keep is as a list for FSM sake! :)
    sim.propagationModel = "MW"       # Chose between 'FSPL' or 'MW' models. FSPL : free space path loss. MW: multi-wall model
    sim.propFreq = 2.4e9             # Propagation frequency in Hertz
    sim.d0 = 1                        # reference distance of 1 meters. This is usually 1 meters for indoor, but it may vary
    sim.wallsAtten = np.ones((1, 256))*wall_attenuation# 5 dB attenuation for all wall
    sim.wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS)
    sim.TxSuperposition = "Ind"  # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent

    # tính tỉ lệ giữa kích thước thực tế và kích thước hình vẽ trên ảnh 
    ratio, cut_image = sim.cal_ratio(labledLines_binary_temp,width,height)
    # labledLines_binary =  np.array(cut_image/cut_image.max(),dtype=np.uint8) 

    # sim.calibration nền trắng đường đen
    calUnit, gridX, gridY, gridXCul, gridYCul = sim.calibration(labledLines_binary,ratio, sim.nodePerMeter)

    sim.pathLossExp = np.full(TxNum, 1.55)

    # Calculating LoS from every Tx to Rx
    LoS = sim.lineOfSight(TxNum,TxLocation,gridX,gridY)
    print("Starting multi-wall model line of sight analysis.\nThis will take a while...")
    wallsOnLoS = sim.wallsOnLineOfSight(TxNum, TxLocation, gridXCul, gridYCul, labledLines_binary)

    RSS = np.zeros((gridXCul.size,TxNum)) 
    delaySpr = np.zeros((gridXCul.size,TxNum))

    print("Estimating (and optimizing) the propagation, this takes a while...")
    for i in range(TxNum):
        RSS[:, i:i + 1], delaySpr[:, i:i + 1], _, _,_= sim.mwModel(ghz,sim.pathLossExp[i], TxPower[i], LoS[i],wallsOnLoS[i],sim.wallsAtten)

    print("Almost done! Just preparing the propagation map :)")
    if sim.TxSuperposition.lower() == 'cw':
        RSS = np.sum((10**(RSS/10)) * (np.cos(2*np.pi*sim.propFreq*delaySpr) + np.sin(2*np.pi*sim.propFreq*delaySpr)*1j),1)
        RSS = 10* np.log10(np.abs(RSS))
    elif sim.TxSuperposition.lower() == 'ind':
        RSS = np.amax(RSS,1)
    delaySpr = np.amin(delaySpr,1)

    #  denormalization the RSS map as for some reason the imresize scale the image to an actual image
    alpha = np.min(np.min(RSS))
    beta  = np.max(np.max(RSS))
    gamma = beta - alpha

    # resizing to image size
    RSSImageResized = resize(np.reshape((RSS - alpha)/gamma,(gridX.shape)), labledLines_binary.shape, order=3)
    RSSImageResized = (RSSImageResized/np.max(np.max(RSSImageResized)) *gamma) + alpha

    # Converting to RGB image
    # dynamicRange = "linear" is the default, however "log" compresses the dynamic range twice!
    RSSRGBImage, _ = data2heatmap(RSSImageResized, dynamicRange = 'log') # "dynamicRange = "log" compresses the dynamic range twice

    # Overlay the structure image
    tempXY = np.where(labledLines_binary==1)
    RSSRGBImage[tempXY[0],tempXY[1],:] = [0,0,0]
    RSSRGBImage = np.array(RSSRGBImage*255,dtype=np.uint8)
    # RSSRGBImage = cv2.cvtColor(RSSRGBImage, cv2.COLOR_RGB2BGR)

    img_path = f"{img_dir}/wifi_simulate_image.jpg"
    print("trung", RSSRGBImage.shape)
    cv2.imwrite(img_path, RSSRGBImage)
    return FileResponse(img_path)
    # except Exception as ex:
    #     raise HTTPException(
    #         status_code=400, detail="Error when simulate the wifi signal propagation - {}".format(str(ex)))

