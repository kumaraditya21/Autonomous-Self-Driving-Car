import tflite_runtime.interpreter as tflite_interp
import cv2 
import numpy as np 
import pandas as pd
from IPython.display import clear_output

def Inference_Engine(model_path, image):

    interpreter = tflite_interp.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    #print("[+]expected input shape:", input_shape)
    input_type = input_details[0]['dtype']
    
    try:
        
        #print("[+]allocating tensor")
        interpreter.allocate_tensors()

        #internal processing
        input_tensor = np.array(image).astype(input_type)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        #inference 
        #print("[+]performing inference")
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        #print("[+]inference completed")
        return output
    
    except Exception as e:
        print("[-]caught exception: ", e)
        return None

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img = cv2.resize(img, (200, 66))
    img = img/255

    return img
