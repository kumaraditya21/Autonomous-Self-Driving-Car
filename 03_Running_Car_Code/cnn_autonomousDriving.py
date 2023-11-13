import cv2
import numpy as np
import cnn_utlis
import os

from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib_1 import Vilib
from time import sleep, time, strftime, localtime
import readchar

reset_mcu()
sleep(0.2)
px = Picarx()


if __name__=='__main__':
    
    try:
            
        speed = 1
        maxSteeringAngle=25
        steeringAngle=0 #Initial steering Angle

        Vilib.camera_start(vflip=False,hflip=False) #Start the camera
        sleep(2)
                
        px.set_camera_servo2_angle(-15) #Keep the camera angle same as the time of recording the data
        sleep(0.2)        

        while True:
            #STEP-1: Taking Image

            img=Vilib.img_array[0]
            
            
            #STEP-2 : PreProcessing the image
            processed_img = cnn_utlis.preprocess(img)
            


            #STEP-3 : Getting steering angle from CNN Modle
            output = cnn_utlis.Inference_Engine("/home/pi/CNN_Autonomous_Driving/model.tflite", processed_img)

            steeringAngle = max(-maxSteeringAngle, min(maxSteeringAngle,int(output[0][0]*25)))
            print("Model_strAngle=",steeringAngle)
            
            
            
            #STEP-4 : Controlling the Motor
            px.set_dir_servo_angle(steeringAngle)
            px.forward(speed)
            cv2.imshow('IMG',img)
            key=cv2.waitKey(5)
            if key == ord('q'):
                break


    finally:
        px.set_dir_servo_angle(0)
        px.stop()
        Vilib.camera_close()
        px.set_camera_servo2_angle(0)
        
    
