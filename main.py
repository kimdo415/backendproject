import os
import io
import time
import cv2
import numpy as np 
from cvu.detector.yolov5 import Yolov5 as Yolov5Onnx
from vidsz.opencv import Reader,Writer
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse
import base64
import face
import uvicorn
app = FastAPI()    
 
device='cpu'
weight='jordan.onnx'
class_names=['jordan11','jordan1','jordan4','jordan2','jordan3','jordan5','jordan6']
model=Yolov5Onnx(classes=class_names,
                 backend="onnx",
                 weight=weight,
                 device=device)
origins=["*"]
app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


img_dir='./images'
jimg_dir='./jordan_images'
wiki='https://ko.wikipedia.org/wiki/'


 
@app.post("/whoryou")
async def getface(file:UploadFile):
    s=time.time()
    try:
        content=await file.read()
        with open(os.path.join(img_dir,file.filename),"wb") as fp:
            fp.write(content)
        pre=face.predict(img_dir+"/"+file.filename,model_path='trained_knn_model.clf')
        e=time.time()
        if pre[0][0]!="unknown":
            return {"name":pre[0][0], "time": str(e-s), "link": wiki+pre[0][0], "drawpoint":pre[0][1]}
        else:
            return {"모른다"}
    except:
        return {"이미지 인식 오류 1. 사람의 얼굴의 일부분이 가려진 경우 또는 눈을 감은 상태라 보이지 않는 경우 2. 얼굴의 과도하게 확대된 경우 인식 불가능 3. 캐릭터 인식안됨 "}


@app.get("/lf")
async def lface():
    s=time.time()
    face.learn_face()
    e=time.time()
    return {"빌드완료: ": f"{str(e-s)} 초걸림"}

@app.post("/face2folder/{namu_name}")
async def faceolder(file:UploadFile, namu_name:str):
    try:
        makefile="./dyk_club/"+namu_name
        os.mkdir(makefile)
        content=await file.read()
        with open(os.path.join(makefile,file.filename),"wb") as fp:
            fp.write(content)
            return{"complete your request, Thx"}
    except:
        return{"already in our folder"}

@app.post("/collect/{name}")
async def collect(file:UploadFile, name:str):
    collectfile="./dyk_club/"+name
    content=await file.read()
    with open(os.path.join(collectfile,file.filename),"wb") as fp:
        fp.write(content)
        return{"thx after we will build this images"}

@app.get("/club_name")
async def callnames():
    club_list=os.listdir('./dyk_club')
    return club_list

@app.post("/jordan")
async def yoloj(file:UploadFile):
    content=await file.read()
    jcount={"Jordan1":0,"Jordan2":0,"Jordan3":0,"Jordan4":0,"Jordan5":0,"Jordan6":0,"Jordan11":0}
    image = cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    resized=cv2.resize(image,(640,640),interpolation=cv2.INTER_AREA)
    preds=model(resized)
    for i in preds:
        try:
            st=str(i).find("class=")+len("class=")
            end=str(i).find(";",st)
            cls_name=str(i)[st:end].strip()
            jcount[cls_name] +=1
        except:
            pass
        
    preds.draw(resized)
    is_pass,buffer=cv2.imencode(".jpg",resized) 
    img_base64=base64.b64encode(buffer).decode("utf-8")
    # try:
    #     with open("result.jpg","wb")as f:
    #         f.write(io_buf.getvalue())
    # finally:
    #     io_buf.close()
    response={"image":img_base64, "detected":jcount}
    
    return response    
    
    # output_image="result.jpg"
    # with open(os.path.join(jimg_dir,file.filename),"wb") as fp:
    #     fp.write(content)
    # j_path=jimg_dir+"/"+file.filename
    # jimg=cv2.imread(j_path)
    # resized=cv2.resize(jimg, (640,640),interpolation=cv2.INTER_AREA)
    # preds=model(resized)
    # preds.draw(resized)
    # cv2.imwrite(output_image, resized)
    #return FileResponse(output_image)
    
        
    
    
    
    
if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run("main:app", host="0.0.0.0", port= port, reload=True) 