
# coding: utf-8

# In[167]:


import cv2
import math 
import numpy as np 
from skimage import measure
import imutils
from imutils import contours
from matplotlib import pyplot as plt



# In[203]:


img = cv2.imread("side2.jpg")
y1=int(100)
img=img[0:150, : ]
color = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(5,5),0)


# In[207]:


plt.imshow(img)
plt.title('cropped')
plt.show()


# In[199]:


ret,img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
plt.imshow(img)
plt.title('thresholded')
plt.show()


# In[184]:


img  = cv2.dilate(img, None, iterations=1)
plt.imshow(img)
plt.title('dilated')
plt.show()


# In[185]:


labels = measure.label(img, neighbors=8, background=0)
mask = np.zeros(img.shape, dtype="uint8")

for label in np.unique(labels):
    if(label==0):
        continue
        
    labelMask = np.zeros(img.shape, dtype="uint8")
    labelMask[labels==label] = 255
    numPixels = cv2.countNonZero(labelMask)
    
    if(numPixels>0):
        mask = cv2.add(mask, labelMask)


# In[186]:


plt.imshow(img)
plt.title('mask')
plt.show()


# In[187]:


cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]


# In[188]:


average_height = []
average_width = []
for cnt in cnts:
    sum1 = 0
    sum2 = 0
    for point in cnt:
        sum1=sum1+point[0][1]
        sum2 = sum2 + point[0][0]
    average_height.append(sum1/len(cnt))
    average_width.append(sum2/len(cnt))



right_most = []
for cnt in cnts:
    right = cnt[0][0][0]
    for point in cnt:
        if(point[0][0]<right):
            right = point[0][0]
    right_most.append(right)
    
rm = right_most.index(min(right_most))
print("right")
print(rm)
        


left_most = []
for cnt in cnts:
    left = cnt[0][0][0]
    for point in cnt:
        if(point[0][0]<left):
            left = point[0][0]
    left_most.append(left)
    
lm = left_most.index(min(left_most))
print("left")
print(lm)
        

    


# In[189]:


high = average_height.index(max(average_height))
low = average_height.index(min(average_height))
left = average_width.index(min(average_width))
i = [high, low]
print(low)


# In[193]:


x, y, w, h = cv2.boundingRect(cnts[low])
M = cv2.moments(cnts[low])
a1 = int(M["m10"] / M["m00"])
b1 = int(M["m01"] / M["m00"])
((cX, cY), radius) = cv2.minEnclosingCircle(cnts[low])
color = cv2.circle(color, (int(cX), int(cY)), int(radius),(255, 0, 0), 3)



(x, y, w, h) = cv2.boundingRect(cnts[lm])
M = cv2.moments(cnts[lm])
a2 = int(M["m10"] / M["m00"])
b2 = int(M["m01"] / M["m00"])
((cX, cY), radius) = cv2.minEnclosingCircle(cnts[left])
color = cv2.circle(color, (int(cX), int(cY)), int(radius),(255, 0, 0), 3)
ray1=cv2.line(color,(a1,b1), (a2,b2),(255, 0, 0), 1 )
ray2=cv2.line(color,(a1,b2), (a2,b2),(255, 0, 0), 1 )
angle=math.atan2(b1-2, a1-a2) * 180/math.pi
print(angle)
# In[194]:


plt.imshow(color)
plt.title('spotted')
plt.show()


