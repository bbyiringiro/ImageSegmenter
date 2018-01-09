import numpy as np
import cv2,math
from matplotlib import pyplot as plt
#following are some but styled codes ;)
#by munyakabera jean claude
#identifies distinct objects in an image

#detecting objects by distance transform on edges
def edge_morphology_segments(imgg):
    global cam_h
    img=cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img=cv2.GaussianBlur(img,(5,5),0)
    thresh = cv2.Canny(img,100,200,L2gradient=True)
    sure_bg = cv2.dilate(thresh,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.absolute(sure_fg)
    sure_fg = np.uint8(sure_fg)
    sure_fg = cv2.GaussianBlur(sure_fg,(5,5),0)
    n,contours,hierarchy= cv2.findContours(sure_fg.copy(), cv2.RETR_LIST,\
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rand=np.random.randint(0,255)
        x,y,w,h=cv2.boundingRect(cnt)
        H,wi = imgg.shape[:2]
        yb=H-y
        dist,height=dist_h_calc(cam_h,H,yb,h)
        cv2.rectangle(imgg,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_PLAIN
        text='D:'+str(dist)+'\n'+'H:'+str(height)
        #cv2.putText(imgg,'D:'+str(dist),(x,y), font, 1,(0,255,0),1)
        #cv2.putText(imgg,'H:'+str(height),(x,y+12), font, 1,(0,255,0),1)
        #cv2.putText(imgg,'Y:'+str(yb)+'-'+str(yb-h),(x,y+24), font, 1,(0,255,0),1)
    return imgg,sure_fg
''' #haar cascade removed! shit :)
def haar_body_detector(img):
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in bodies:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img
'''
def rm_bg(img):    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    return fgmask

def dist_h_calc(cam_h,H,yb,h):
    pi=math.pi/(4*H)
    ys=yb-h
    H=H/2
    #choose case
    if yb>H and h>H: #first case
        theta=(yb-H)*pi
        _lambda=(H-ys)*pi
        dist=cam_h/math.tan(_lambda)
        o=(cam_h*math.tan(theta))/math.tan(_lambda)
        height=cam_h+o
        return round(dist,2),round(height,2)
    elif yb>H and h<=H: #SECOND CASE
        theta=(yb-H)*pi
        F=(ys-H)*pi
        phi=theta+F
        _lambda=pi
        R=(cam_h/math.tan(_lambda))*(math.tan(phi)-math.tan(F))
        dist=R/(math.tan(phi)-math.tan(F))
        height=cam_h+R
        return round(dist,2),round(height,2)
    else:
        return '',''

    
#cv2.namedWindow('rectangle overlay', cv2.WINDOW_NORMAL)
cam_h=2 #the camera height in meters 9GGSNVLSXJ3B

'''
img = cv2.imread('mlng.jpg')
img,edges=edge_morphology_segments(img)
#print 'done in',time.time()-now,'seconds'
plt.subplot(111),plt.imshow(img,cmap='Dark2')
plt.title('rectangle overlay'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('edge image'),plt.xticks([]),plt.yticks([])
plt.show()
'''
cap = cv2.VideoCapture('ww.webm')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    #detecting objects by morphologically transformin the foreground
    ret ,frame = cap.read()
    if ret == True:
        fgmask = fgbg.apply(frame)
        frame,edges=edge_morphology_segments(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(fgmask,(5,5),0)
        ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #otsu =cv2.dilate(otsu ,kernel,iterations = 2)
        laplacian = cv2.Laplacian(otsu,cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        #otsu = np.uint8(laplacian)
        otsu=cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        otsu=cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        otsu =cv2.dilate(otsu,kernel,iterations = 2)
        #ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #otsu=cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        #otsu=cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        otsu = cv2.GaussianBlur(otsu,(5,5),2)
        #fgmask =cv2.dilate(fgmask ,kernel,iterations = 3)
        #fgmask = cv2.GaussianBlur(fgmask,(5,5),0)
        n,contours,hierarchy= cv2.findContours(otsu.copy(), cv2.RETR_LIST,\
                                     cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rand=np.random.randint(0,255)
            x,y,w,h=cv2.boundingRect(cnt)
            H,wi = frame.shape[:2]
            yb=y+h
            ys=y
            dist,height=dist_h_calc(cam_h,H,yb,ys)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_PLAIN
            #cv2.putText(frame,'D:'+str(dist),(x,y), font, 1,(0,255,0),1)
            #cv2.putText(frame,'H:'+str(height),(x,y+12), font, 1,(0,255,0),1)
        cv2.imshow('foreground binary',otsu)
        cv2.imshow('rectangle overlay',frame)
        #frame,edges = edge_morphology_segments(frame)
        #cv2.imshow('generalized_morphology_segments',frame)
        #cv2.imshow('edges',edges) 
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()

