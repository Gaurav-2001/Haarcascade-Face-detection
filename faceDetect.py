import cv2
cap= cv2.VideoCapture(0)
model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detect(photo):
    face=model.detectMultiScale(photo)
    if len(face) != 0:
        for nface in face:
            photo = cv2.rectangle(photo,
                               (nface[0],nface[1]),
                               (nface[0]+nface[2],nface[1]+nface[3]),
                               [0,255,0],
                               5)
        return photo
    else:
        return photo

while True:
    ret,photo=cap.read()
    cv2.imshow('Face',face_detect(photo))
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
