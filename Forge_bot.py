# Code Created by Jose Bendana, Carlos Salazar and Burzin Balsara 04/25/22
# Forgebot (Academic Purposes)

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.animation import FuncAnimation

def resize(img):
        return cv2.resize(img,(512,512)) # arg1- input image, arg- output_width, output_height
    

# cap=cv2.VideoCapture("ballmotion.m4v")
cap=cv2.VideoCapture("/Users/josebendana/Desktop/Forge_Bot/demo.mov")
# cap=cv2.VideoCapture("Videos/Carlos_Signature.MP4")
# cap=cv2.VideoCapture("Videos/Burzin_Signature.MOV")


x_coord=[]
y_coord=[]

while True:
    _, frame = cap.read()
    try:
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    except:
        break
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)   #change color from BGR to HSV
    #--------red color--------
    # lower_red = np.array([0,230,170])
    # upper_red = np.array([55,255,220])
    # lower_red = np.array([170,100,20])
    # upper_red = np.array([179,255,255])
    #--------blue color--------
    # lower_blue = np.array([100,100,20])
    # upper_blue = np.array([125,255,255])
    #--------green color--------
    lower_green = np.array([30,100,5])
    upper_green = np.array([90,255,255])
    
       
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area=cv2.contourArea(contour)
        if area>200: #Sets minumum area to filter noise of green spots
            M=cv2.moments(contour)
            if M["m00"]==0:
                M["m00"]==1
            x=int(M["m10"]/M["m00"])
            y=int(M["m01"]/M["m00"])
            
            x_coord.append(x)
            y_coord.append(-y)
            
            cv2.circle(frame,(x,y),7,(0, 255, 0), -1)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,"{},{}".format(x,y),(x+10,y),font,0.75,(0, 255, 0),1,cv2.LINE_AA )
            # newcont=cv2.convexHull(contour)
            # cv2.drawContours(frame, [newcont], 0, (0, 255, 0), 3)
        
    cv2.startWindowThread() 
    # cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
 
#--------Scale/Random--------
# sf=random.choice([0.1, 0.11, 0.12, 0.13, 0.14])
sf = 0.16  #Set scale accordingly

# --------Randomization--------
x_random=[]
y_random=[]
temp=0

for i in range(len(x_coord)):
    
    if i==round(len(x_coord)/15)*temp:
        choice=random.randint(1, 3)
        temp=temp+1

    if choice==1:
        off = random.randint(2, 3)
        x_random.append(x_coord[i]+off)
        y_random.append(y_coord[i]+off)
    if choice==2:
        off = random.randint(-3, -2)
        x_random.append(x_coord[i]+off)
        y_random.append(y_coord[i]+off)
    if choice==3:
        x_random.append(x_coord[i])
        y_random.append(y_coord[i])

    
#--------Randomization Spline Lines--------    
points_random = []
for xval,yval in zip(x_random,y_random):
    if (xval,yval) not in points_random:
        points_random.append((xval,yval))
        
coords_random = np.array(points_random)
randomx=coords_random[:,0]
randomy=coords_random[:,1]
tck,u = interpolate.splprep([randomx,randomy])
u=np.linspace(u.min(),u.max(),num=10000,endpoint=True)
out_random = interpolate.splev(u,tck)    
    
# --------Spline Lines Original Line--------    
points = []
for xval,yval in zip(x_coord,y_coord):
    if (xval,yval) not in points:
        points.append((xval,yval))
        
coords = np.array(points)
x=coords[:,0]
y=coords[:,1]
tck,u = interpolate.splprep([x,y])
u=np.linspace(u.min(),u.max(),num=10000,endpoint=True)
out = interpolate.splev(u,tck)

# --------Plot--------   
plt.figure()
# plt.plot(x, y, 'ro', out[0], out[1], 'b')
plt.plot(out[0], out[1], 'g')
plt.plot(out_random[0], out_random[1], 'k')
# plt.plot(randomx, randomy, 'go', out_random[0], out_random[1], 'k')
plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('B-Spline interpolation')
plt.show()

#--------Points--------
plt.figure()
plt.scatter(x_coord,y_coord)
plt.show()
#--------Lines--------
plt.figure()
plt.plot(x_coord,y_coord)
plt.show() 

#-------- Gcode generation Original Line--------
x_orig = list(out[0])
y_orig = list(out[1])
x = []
for val in x_orig:
    x.append(val*sf)
y = []
offset = (-1*min(y_orig)) + 10
for val in y_orig:
    y.append((val+offset)*sf)

#-------- Gcode generation Random Line--------
x_rand = list(out_random[0])
y_rand = list(out_random[1])
x_rdm = []
for val in x_rand:
    x_rdm.append(val*sf)
y_rdm= []
offset_rand = (-1*min(y_rand)) + 10
for val in y_rand:
    y_rdm.append((val+offset_rand)*sf)

def genGCode(x,y,filename):
    with open(filename, 'w') as f:
        f.write('G21 ; mm-mode\nG0 Z0; move to z-safe height\nG92 F1000 X{x0:.4f} Y{y0:.4f}\nM3S0\nG4 P0.5; Tool On\nG1 F300 Z-0.1000\n'.format(x0 = x[0], y0 = y[0]))
        x.pop(0)
        y.pop(0)
        for xval,yval in zip(x,y):
            f.write('G1 F1000 X{x_coord:.4f} Y{y_coord:.4f} Z-0.1000\n'.format(x_coord = xval, y_coord = yval))
        
genGCode(x,y,'demo_signature_original.gcode')
genGCode(x_rdm,y_rdm,'demo_signature_random.gcode')

    
    
cap.release()
cv2.destroyAllWindows()
