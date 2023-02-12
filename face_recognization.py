import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file("face recognization/kalam sir.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgr = cv2.resize(img,(400,400))

img_compare = face_recognition.load_image_file("face recognization/BR sir.png")
img_compare = cv2.cvtColor(img_compare,cv2.COLOR_BGR2RGB)
img_compare_r = cv2.resize(img_compare,(400,400))

pos = face_recognition.face_locations(imgr)[0]
encode = face_recognition.face_encodings(imgr)[0]
cv2.rectangle(imgr,(pos[3],pos[0]),(pos[1],pos[2]),(255,0,0),3)

pos1 = face_recognition.face_locations(img_compare_r)[0]
encode1 = face_recognition.face_encodings(img_compare_r)[0]
cv2.rectangle(img_compare_r,(pos1[3],pos1[0]),(pos1[1],pos1[2]),(255,0,0),3)

result = face_recognition.compare_faces([encode],encode1)
print(result)
if (result == [True]):
    print("same person")
    text = "Match found "
else:
    print("Different person")
    text = "Match Not found "



variation = face_recognition.face_distance([encode],encode1)
variation = "%.2f" % variation
d =''.join(map(str,variation))


cv2.putText(img_compare_r,text,(45,60),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,0,255),2)
cv2.putText(img_compare_r,d,(45,90),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,0,255),2)





cv2.imshow("SIR APJ AK",imgr)
cv2.imshow("SIR APJ AK(compare)",img_compare_r)
cv2.waitKey(0)