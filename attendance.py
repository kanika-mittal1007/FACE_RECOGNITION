import tkinter as tk
from tkinter import messagebox
import cv2
from cv2 import *
import os
from PIL import Image , ImageTk
import numpy as np

import face_recognition
from datetime import datetime

window = tk.Tk()
window.title("Face recognition system")


window.config(background="green")



# Position image

label2 = tk.Label(window, text = "PRESENTING ATTENDANCE MARK SYSTEM by FACE RECOGNITION TECHNIQUE!!!!", bd = 0, fg = "black", font = "ALGERIAN")
label2.pack()





image1 = Image.open("C:/Users/Ahana Gupta/Desktop/facial.jpg")
test = ImageTk.PhotoImage(image1)

label1 = tk.Label(window,image=test)
label1.image = test

# Position image
label1.place(x=570, y=210)


def attendance():
    path = 'images'
    images = []
    personNames = []
    myList = os.listdir(path)
    print(myList)
    for cu_img in myList:
        current_Img = cv2.imread(f'{path}/{cu_img}')
        images.append(current_Img)
        personNames.append(os.path.splitext(cu_img)[0])
    print(personNames)

    def faceEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def attendance(name):
        with open('attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                time_now = datetime.now()
                tStr = time_now.strftime('%H:%M:%S')
                dStr = time_now.strftime('%d/%m/%Y')
                f.writelines(f'\n{name},{tStr},{dStr}')

    encodeListKnown = faceEncodings(images)
    print('All Encodings Complete!!!')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(faces)
        encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                attendance(name)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


b1 = tk.Button(window, text="FACE RECOGNIZER", font=("ALGERIAN", 20), bg='white', fg='blue' ,command=attendance)
b1.place(relx=0.4,rely=0.45)

window.geometry("1000x1000")
window.mainloop()


