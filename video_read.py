import cv2 
import numpy as np

#create camera object
cam = cv2.VideoCapture(0)

#ask name 
fileName = input("Enter the name of person: ")
dataset_path ="./data/"
offset =20

#model
model = cv2.CascadeClassifier("haarcascade_frontal_alt.xml")

#list to save face data
faceData = []
skip =0

#read image from camera object 
while True :
    sucess, img = cam.read()
    if not sucess:
        print("Reading camera failed!")

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    

    faces = model.detectMultiScale(img,1.3,5)
    #sort face with largest bound area
    faces = sorted(faces,key= lambda f:f[2]*f[3])
    #pick largest face
    if len(faces)>0:
        f= faces[-1]

        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #crop and save the face
        cropped_face = img[ y - offset:y+h + offset, x - offset:x+w +offset]
        cropped_face = cv2.resize(cropped_face,(100,100))
        skip +=1
        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("saved so far"+ str(len(faceData)))


    cv2.imshow("Image window", img)
   # cv2.imshow("cropped face",cropped_face )

    key = cv2.waitKey(1) #pause here for 1 ms before you read next image
    if key == ord('q'):
        break

# write face data on disk 
faceData = np.asarray(faceData)
m= faceData.shape[0]
faceData= faceData.reshape((m,-1))

print(faceData.shape)

#save on the disk as np array
filepath = dataset_path + fileName +".npy"
np.save(filepath,faceData)
print("Data saved sucessfully"+ filepath)


#release cam
cam.release()
cv2.destroyAllWindows()

