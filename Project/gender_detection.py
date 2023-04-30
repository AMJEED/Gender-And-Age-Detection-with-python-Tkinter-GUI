# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:43:14 2022

@author: um427
"""

# import filedialog module
from tkinter import *
from tkinter.ttk import *
import cv2
import numpy as np
FACE_PROTO = "deploy.prototxt.txt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

GENDER_MODEL = 'deploy_gender.prototxt'

GENDER_PROTO = 'gender_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

GENDER_LIST = ['Male', 'Female']
AGE_MODEL = 'deploy_age.prototxt'

AGE_PROTO = 'age_net.caffemodel'

AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
frame_width = 1280
frame_height = 720
# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)



def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def display_img(title, img):
    """Displays an image on screen and maintains the output until the user presses a key"""
    # Display Image on screen
    cv2.imshow(title, img)
    # Mantain output until user presses a key
    cv2.waitKey(0)
    # Destroy windows when user presses a key
    cv2.destroyAllWindows()
    
    
    
    # from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()
def predict_age_and_gender(input_path: str):
    """Predict the gender of the faces showing in the image"""
    #Initialize frame size
    #frame_width = 1280
    #frame_height = 720
    # Read Input Image
    img = cv2.imread(input_path)
    # resize the image, uncomment if you want to resize the image
    img = cv2.resize(img, (frame_width, frame_height))
    # Take a copy of the initial image and resize it
    frame = img.copy()
    #if frame.shape[1] > frame_width:
        #frame = image_resize(frame, width=frame_width)
    # predict the faces
    faces = get_faces(frame)
    # Loop over the faces detected
    # for idx, face in enumerate(faces):
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        age_preds = get_age_predictions(face_img)
        gender_preds = get_gender_predictions(face_img)
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]
        # Draw the box
        label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
        # label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
        print(label)
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        # Label processed image
        font_scale = 0.54
        cv2.putText(frame, label, (start_x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

        # Display processed image
    display_img("Gender Estimator", frame)
    # uncomment if you want to save the image
    cv2.imwrite("output.jpg", frame)
    # Cleanup
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    #webcam = cv2.VideoCapture(0)
    #check, frame = webcam.read()
    #cv2.imwrite(filename='C:\\Users\\um427\\Downloads\\saved_img.jpg', img=frame)
   
    import tkinter as tk
    from tkinter import filedialog

   
    
    def web_cam():
         webcam = cv2.VideoCapture(0)
         check, frame = webcam.read()
         img = cv2.imwrite(filename='saved_img.jpg', img=frame)   
         predict_age_and_gender('saved_img.jpg')   
         
         
         
    def browse():
         root = tk.Tk()
         root.withdraw()
         file_path = filedialog.askopenfilename()
         predict_age_and_gender(file_path)
         
  
    
    
   
    
    
    #webcam.release()
    #cv2.destroyAllWindows()
    root = Tk()
   
# Set Geometry(widthxheight)
    root.geometry('400x300')
    root.title("Gender Recongnition app")
# Create style Object
    style = Style()
 
    style.configure('TButton', font =
               ('calibri', 20, 'bold'),
                    borderwidth = '4')
 
# Changes will be reflected
# by the movement of mouse.
    style.map('TButton', foreground = [('active', '!disabled', 'green')],
                     background = [('active', 'black')])

    
# button 1
    btn1 = Button(root, text = 'Open webCam', command =web_cam)
    btn1.grid(row = 0, column = 3, padx = 100 ,pady = 20)
 
# button 2
    btn2 = Button(root, text = 'Browse image file', command = browse)
    btn2.grid(row = 1, column = 3, pady= 30, padx = 100)
    
    
      
   # Displaying the button
 
# Execute Tkinter
    root.mainloop()
    
    
    