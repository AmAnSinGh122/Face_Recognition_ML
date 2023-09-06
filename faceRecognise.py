import cv2
import numpy as np
import os

#data
dataset_path= "./data/"
faceData = []
labels = []
nameMap = {}
classId=0
offset =20

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f[:-4]
        #x-value
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        faceData.append(dataItem)


        #y-values
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)


XT = np.concatenate(faceData,axis=0)
yT = np.concatenate(labels,axis=0).reshape((-1,1))

print(XT.shape)
print(yT.shape)
print(nameMap)

#algorithm
def dist(p,q):
    return np.sqrt(np.sum((p - q)**2))

def knn(X,y,xt,k=5):

    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))
    
    dlist = sorted(dlist)
    dlist = np.array(dlist[:k], dtype=object)
    labels = dlist[:,1]

    labels, cnts = np.unique(labels,return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

#predictions

#create camera object
cam = cv2.VideoCapture(0)

#model
model = cv2.CascadeClassifier("haarcascade_frontal_alt.xml")

while True :
    sucess, img = cam.read()
    if not sucess:
        print("Reading camera failed!")    

    faces = model.detectMultiScale(img,1.3,5)

    #render box around each face and predicts its name 
    for f in faces:
        x,y,w,h = f 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  

        #crop and save the face
        cropped_face = img[ y - offset:y+h + offset, x - offset:x+w +offset]
        cropped_face = cv2.resize(cropped_face,(100,100))
    
        #predict name using knn
        classPredicted = knn(XT,yT,cropped_face.flatten())
        #name 
        namePredicted = nameMap[classPredicted]
        #Display the name 
        cv2.putText(img, namePredicted, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Prediction Window", img)

    key = cv2.waitKey(1) #pause here for 1 ms before you read next image
    if key == ord('q'):
        break

#release cam
cam.release()
cv2.destroyAllWindows()

       