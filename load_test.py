import threading, time, argparse, os, pickle, queue, numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import matplotlib as plt
import cv2
import logging

model_name = "yolov5n"
video_name = "/home/hbp/vdo.avi"
model = torch.hub.load("ultralytics/yolov5", model_name)

def data_processing():
    vid = cv2.VideoCapture(video_name)
    fps = vid.get(cv2.CAP_PROP_FPS)
    delay = 1/fps
    
    while vid.isOpened():
        _, frame = vid.read()
        #print(frame)
        #frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)

        if frame is None:
            #logging.warning("Empty Frame")
            break
        
        start_time = time.time()
        deadline = start_time+delay

        #Inference
        results = model(frame)

        if time.time() > deadline:
            print("Time over !!! calculate time : {}(s)".format(time.time()-start_time))
        else:
            time.sleep(deadline-time.time())
        print(results)

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='workload test')
    parser.add_argument('--data_path', default='/home/hbp/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--resolution', default=(640, 640), type=tuple, help='Image resolution')
    parser.add_argument('--num_cameras', default=4, type=int, help='Number of cameras')
    args = parser.parse_args()


    model_processes = [mp.Process(target=data_processing,args=()) for p in range(args.num_cameras)]

    for p in model_processes:
        p.start()
    
    for p in model_processes:
        p.join()