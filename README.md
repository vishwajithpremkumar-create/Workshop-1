# Workshop -1 Adding Sunglasses to Your Passport Photo Using OpenCV
# Name : Vishwajith P
# Reg No : 212225220122
## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceImage = cv2.imread('vishh.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

faceImage.shape
#resized_faceImage.shape
faceImage.shape

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(280,140))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[240:380,272:552]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[240:380,272:552]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")

# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[240:380,272:552]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```
## Output
<img width="382" height="557" alt="image" src="https://github.com/user-attachments/assets/c0b43a33-1f46-4ecd-838f-5cf56f262c5e" />
<img width="707" height="583" alt="image" src="https://github.com/user-attachments/assets/d14a51bd-5eba-429c-a6e8-c524af296a91" />
<img width="911" height="275" alt="image" src="https://github.com/user-attachments/assets/9af99a19-fa2a-4983-809e-10d838a9461f" />
<img width="445" height="555" alt="image" src="https://github.com/user-attachments/assets/bb7c3a67-dd17-432e-a316-95e4586205e8" />
<img width="920" height="260" alt="image" src="https://github.com/user-attachments/assets/c95bb746-e881-4904-afe7-7fd0a1ed650e" />
<img width="885" height="725" alt="image" src="https://github.com/user-attachments/assets/5794804d-7ebd-44da-929d-ce5868af74b0" />

## Result
Thus, adding sunglass to the passport photo using opencv is performed successfully.
