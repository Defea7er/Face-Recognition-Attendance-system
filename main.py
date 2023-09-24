"""
This code snippet is responsible for encoding images, detecting faces in real-time using a webcam, and matching them with the existing images. It also records the attendance of the recognized faces in a CSV file.

Example:
Inputs:
- `path`: The path to the directory containing the images to be encoded.
- `images`: A list of images to be encoded.

Flow:
1. Load the images from the specified directory and convert them to RGB format.
2. Encode the images using the `face_recognition` library.
3. Initialize the webcam and start capturing frames.
4. Resize the captured frame and convert it to RGB format.
5. Detect faces in the resized frame using the `face_recognition` library.
6. Encode the detected faces.
7. Compare the encoded faces with the existing encoded images.
8. Find the best match based on the lowest face distance.
9. If a match is found, draw a rectangle around the face and display the name on the frame.
10. Record the attendance of the recognized face in a CSV file.
11. Display the frame with the recognized faces.
12. Repeat the process until the user presses 'q' to quit.

Outputs:
- The code snippet displays the real-time webcam feed with rectangles around recognized faces and their names.
- It records the attendance of the recognized faces in a CSV file.
"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Loading images using path
path = 'C:\\Users\\ASUS\\Downloads\\face-recog-attendance-master-1614261936-CKoushik[1]\\face-recog-attendance-master\\Images'
names = []
images=[]
List = os.listdir(path)
#print(List)
for name in List:
    img = cv2.imread(f'{path}/{name}')
    names.append(os.path.splitext(name)[0])
    # Process the image here instead of storing it in a list
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodeImg = face_recognition.face_encodings(img)[0]
    # Rest of the code remains the same

#print(names)

#Function to find encodings for all images in directory

def encode(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encode_list.append(encodeImg)
    return encode_list

def record_attendance(name):
    with open('C:\\Users\\ASUS\\Downloads\\face-recog-attendance-master-1614261936-CKoushik[1]\\face-recog-attendance-master\\attendancelist.csv', 'r+') as file:
        namelist = set()
        for line in file:
            record = line.split(',')
            namelist.add(record[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime("%I:%M:%S %p")
            file.writelines(f'\n{name},{dt}')

print("Encoding Images...")
encodeList = encode(images)
print("Encoding Completed.")

#Initializing webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #Reducing size of real-time image to 1/4th
    imgResize = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

    # Finding face in current frame
    face = face_recognition.face_locations(imgResize)
    # Encode detected face
    encodeImg = face_recognition.face_encodings(imgResize, face)

    #Finding matches with existing images
    for encodecurr, loc in zip(encodeImg, face):
        match = face_recognition.compare_faces(encodeList, encodecurr)
        faceDist = face_recognition.face_distance(encodeList, encodecurr)
        print(faceDist)
        #Lowest distance will be best match
        index_BestMatch = np.argmin(faceDist)

        if match[index_BestMatch]:
            name = names[index_BestMatch]
            y1,x2,y2,x1 = loc
            #Retaining original image size for rectangle location
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
            cv2.rectangle(img,(x1,y2-30),(x2,y2), (255,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+8, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255),2)
            record_attendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

