"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""
from cv2 import WINDOW_NORMAL

import cv2
from face_detect import find_faces
from image_commons import nparray_as_image, draw_with_alpha

# ----------------------
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
lola=1
font = cv2.FONT_HERSHEY_SIMPLEX
hit=0
#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Harsh', 'Unknown', 'A', 'B', 'C'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 480) # set video widht
cam.set(4, 320) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
#--------------------------

def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param emoticons: List of emotions images.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture("testing.mp4")
    
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    while True:

        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            prediction = model.predict(normalized_face)  # do prediction
            #if cv2.__version__ != '3.1.0':
            prediction = prediction[0]

            image_to_draw = emoticons[prediction]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))
            print(prediction)       #target variable
        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)
        img = webcam_image
        #----------------------
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        #lola=0    
        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                percent = 100 - confidence
                if (confidence < 60):
                    hit=1
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                    hit=0
                    global lola
                    if lola==2:
                        from twilio.rest import Client

                        account_sid = "AC22c8e1723b1046ac741bd563caa0995d"
            # # our Auth Token from twilio.com/console
                        auth_token  = "9422efc06f0c7ab5b637536569d2d818"

                        client = Client(account_sid, auth_token)

                        message = client.messages.create(
                            to="+919944333726", 
                            from_="+12407022806",
                            body="You are travelling with an unauthorized driver. Please check!")
                    lola=0
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                hit=0
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            def face(hit):
                return hit
            #print(percent)
        cv2.imshow(window_name,img) 





        #----------------------
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)

    # load model
    if cv2.__version__ == '3.1.0':
        fisher_face = cv2.face.createFisherFaceRecognizer()
    else:
        fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read('models/emotion_detection_model.xml')

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(fisher_face, emoticons, window_size=(1600, 1200), window_name=window_name, update_time=8)
