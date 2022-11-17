#multi treading 
import asyncio
import nats
import os
import json
import numpy as np 
from PIL import Image
import cv2
import glob
from nanoid import generate
from multiprocessing import Process
import torch
from general import (check_requirements_pipeline)
import logging 
import threading
import gc
from track import run

path = "./Nats_output"

if os.path.exists(path) is False:
    os.mkdir(path)
    
# Multi-threading
count = 0
dict_frame = {}
frame = []
count_frame ={}
devicesUnique = []
processes = []

          
async def Video_creating(path, device_id):
    image_folder = path
    video_name = path+'/Nats_video'+str(device_id)+'.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc , 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()
    Process (target = run(source=video_name)).start()
    await asyncio.sleep(1)
    

async def batch_save(device_id,time_stamp,geo_location):
    BatchId = generate(size= 8)
    count_batch = 0
    for msg in dict_frame[device_id]:
        try :
            im = np.ndarray(
                    (512,
                    512),
                    buffer=np.array(msg),
                    dtype=np.uint8) 
            im = Image.fromarray(im)
            device_path = os.path.join(path,str(device_id))
            if os.path.exists(device_path) is False:
                os.mkdir(device_path)
            im.save(device_path+"/"+str(count_batch)+".jpeg")
            print("image save in batch count =",count_batch)
            await asyncio.sleep(1)
        except TypeError as e:
            print(TypeError," gstreamer error 64 >> ",e,"Device Id",device_id)
        count_batch += 1
    Process(target=await Video_creating(path=device_path, device_id=device_id)).start()
    await asyncio.sleep(1)
    if count_batch >= 10:
        pattern = device_path+"/**/*.jpeg"
        print(pattern,"patten line 65")
        print(pattern,"line 71 ", count_batch ,"device id ",device_id)
        for item in glob.iglob(pattern, recursive=True):
            print(item,"item 73 line")
            os.remove(item) 
    dict_frame[device_id].clear()
    count_frame[device_id] = 0 
    print(dict_frame[device_id],"entred the loop 237")
    gc.collect()
    torch.cuda.empty_cache()
                
async def stream_thread(device_id , frame_byte,timestamp, geo_location) :
    if len(dict_frame) == 0 :
        dict_frame[device_id] = list(frame_byte)
        count_frame[device_id] = 1 
    else:
        if device_id in list(dict_frame.keys()):
            dict_frame[device_id].append(list(frame_byte))
            count_frame[device_id] += 1
            if count_frame[device_id] % 10 == 0 :
                Process(target = await batch_save(device_id=device_id,time_stamp=timestamp,geo_location=geo_location)).start()
                await asyncio.sleep(1)
        else:
            dict_frame[device_id] = list(frame_byte)
            count_frame[device_id] = 1
    print(count_frame, "count frame ", threading.get_ident(),"Threading Id" ,device_id ,"Device id")
    await asyncio.sleep(1)


async def cb(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        frame_byte = np.ndarray(
                (512,
                512),
                buffer=np.array(bytes(frame_byte)),
                dtype=np.uint8)
        frame_byte = cv2.resize(frame_byte, (512 ,512))
        if device_id not in devicesUnique:
            t = Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location))
            t.start()
            processes.append(t)
            devicesUnique.append(device_id)
        else:
            ind = devicesUnique.index(device_id)
            t = processes[ind]
            Process(name = t.name, target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location))
    except TypeError as e:
        print(TypeError," gstreamer error 121 >> ", e)
        
    finally:
        print("done with work ")
        sem.release()
        
    logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
    logging.debug("Debug logging test...")
    logging.info("Program is working as expected")
    logging.warning("Warning, the program may not function properly")
    logging.error("The program encountered an error")
    logging.critical("The program crashed")
        
        

async def main():
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    await js.subscribe("stream.2.frame", cb=cb, stream="device_stream" , idle_heartbeat = 2)
    await js.subscribe("stream.2.frame", cb=cb ,stream="device_stream" ,idle_heartbeat = 2)
    
async def checkstatus():
    check_requirements_pipeline() 
    

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(checkstatus())
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
        
        
"""
Json Object For a Batch Video 

JsonObjectBatch= {ID , TimeStamp , {Data} } 
Data = {
    "person" : [ Device Id , [Re-Id] , [Frame TimeStamp] , [Lat , Lon], [Person_count] ,[Activity] ]
    "car":[ Device ID, [Lp Number] , [Frame TimeStamp] , [Lat , lon] ]
}  
Activity = [ "walking" , "standing" , "riding a bike" , "talking", "running", "climbing ladder"]

"""

"""
metapeople ={
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"face_id",
                    "activity":{"activities":activity_list , "boundaryCrossing":boundary}  
                    }
    
    metaVehicel = {
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"license_plate",
                    "activity":{"boundaryCrossing":boundary}
    }
    metaObj = {
                 "people":metapeople,
                 "vehicle":metaVehicel
               }
    
    metaBatch = {
        "Detect": "0: detection NO, 1: detection YES",
        "Count": {"people_count":str(avg_Batchcount),
                  "vehicle_count":str(avg_Batchcount)} ,
        "Object":metaObj
        
    }
    
    primary = { "deviceid":str(Device_id),
                "batchid":str(BatchId), 
                "timestamp":str(frame_timestamp), 
                "geo":str(Geo_location),
                "metaData": metaBatch}
    print(primary)
    
"""