import os
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import cv2
import boto3
import json

def lambda_handler(event, context):

    
    s3 = boto3.client("s3")

   
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

    if event:

        
        bucket = event["bucket"]
        key = event["key"]
        file_object = s3.get_object(Bucket=bucket, Key=key)
        file_content = file_object["Body"].read()

        
        np_array = np.fromstring(file_content, np.uint8)

        
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

       
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

       
        for (x, y, w, h) in faces[:1]:
            face = image_np[y:y + h, x:x + w]
        
        
        resized_img = resize(face, (128*4, 64*4))

        
        fd = hog(
            resized_img, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=False, 
            multichannel=True
        )

        return {
            'status_code': 200,
            'body': json.dumps(fd)
        }