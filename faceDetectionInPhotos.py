import cv2
image_path = "amal.jpg"
image = cv2.imread(image_path)

#Resize the image for faster image processing
scale_percent = 30 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# convert to grayscale of each frames
grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# read the haarcascades to detect the faces in an image and detect faces in the input image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) > 0:
    for i, (x,y,w,h) in enumerate(faces):

        # To draw a rectangle in a face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = image[y:y + h, x:x + w]

         # Generate a unique filename for the cropped face image
        file_name = f'face{i}.jpg'
        cv2.imshow(f"Cropped Face {i}", face)
        cv2.imwrite("Image_detectedFaces/"+file_name, face)
        print(f"face{i}.jpg is saved")
else:
    print("No faces detected")

# display the image with detected faces
cv2.imshow("image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
