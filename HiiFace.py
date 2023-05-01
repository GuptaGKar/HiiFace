import cv2
import math
import argparse

def detect_face(net, frame, conf_threshold=0.7): 
    frame_open=frame.copy()     
    frame_height=frame_open.shape[0]
    frame_width=frame_open.shape[1]
    blob=cv2.dnn.blobFromImage(frame_open, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    capt=net.forward()
    face_box=[]
    for i in range(capt.shape[2]):
        confidence=capt[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(capt[0,0,i,3]*frame_width)
            y1=int(capt[0,0,i,4]*frame_height)
            x2=int(capt[0,0,i,5]*frame_width)
            y2=int(capt[0,0,i,6]*frame_height)
            face_box.append([x1,y1,x2,y2])
            cv2.rectangle(frame_open, (x1,y1), (x2,y2), (0,255,0), int(round(frame_height/150)), 8)
    return frame_open,face_box


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

face_protocol="opencv_face_detector.pbtxt"
face_modle="opencv_face_detector_uint8.pb"
age_protocol="age_deploy.prototxt"
age_modle="age_net.caffemodel"
gender_protocol="gender_deploy.prototxt"
gender_modle="gender_net.caffemodel"

mean_values=(78.4263377603, 87.7689143744, 114.895847746)
ages_list=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genders_list=['Male','Female']

faceNet=cv2.dnn.readNet(face_modle,face_protocol)
agenet=cv2.dnn.readNet(age_modle,age_protocol)
gendernet=cv2.dnn.readNet(gender_modle,gender_protocol)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,face_box=detect_face(faceNet,frame)
    if not face_box:
        print("No face detected")

    for faceBox in face_box:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), mean_values, swapRB=False)
        gendernet.setInput(blob)
        gender_find=gendernet.forward()
        gender=genders_list[gender_find[0].argmax()]
        print(f'Gender: {gender}')

        agenet.setInput(blob)
        age_find=agenet.forward()
        age=ages_list[age_find[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
