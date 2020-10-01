import cv2 as cv
import numpy as np
import os

#cac dia chi de luu tru du lieu
base_dir = '/home/le/PycharmProjects/pythonProject1/images'

#cac module de nhan dien khuon mat
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('data/haarcascade_fullbody.xml')
# car_cascale = cv.CascadeClassifier('data/haarcascade_russian_plate_number.xml')

cap = cv.VideoCapture(0)
_, frame1 = cap.read()
face1 = frame1
i = 50
while cap.isOpened():
    #img = cv.imread('8.jpg')
    _, frame1 = cap.read()
    #img = cv.resize(frame1,(512,512)   )
    img = frame1
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 13)
    # eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
        face1 = img[y:y+h, x:x+w]
        print(x,y,w,h)
        if (x,y,w,h) is not None:
            if i == 50:
                name = input('nhap ten: ')
                new_folder= os.path.join(base_dir, name)
                os.mkdir(new_folder)
                if name == "":
                    i = 30
                    break
                elif name == "exit":
                    break
                i = 0
            i = i + 1
            face1 = cv.resize(face1, (160, 160))
            text = r'/home/le/PycharmProjects/pythonProject1/images/' + str(name) + r'/anh' + str(i) + '.jpg'
            cv.imwrite(text, face1)
    cv.putText(img, "so anh : " + str(i + 1), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #cv.putText(img, "ten: " + name, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    if i < 50:
        cv.imshow('image', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    #elif cv.waitKey(1) & 0xFF == ord('c'):
     #   cv.imwrite(text, face1)
      #  print("save sucessfully")
cv.destroyAllWindows()
