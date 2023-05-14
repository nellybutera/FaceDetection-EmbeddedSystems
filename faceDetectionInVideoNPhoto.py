import cv2
import uuid

# Create a face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    # putting a limit to avoid endless loop
    if len(faces) < 20:
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Generate a unique filename for the cropped face image
            file_name = f'face_{str(uuid.uuid4())}.jpg'

            # Crop the face from the frame
            face_crop = frame[y:y+h, x:x+w]

            # Save the cropped face image to disk
            cv2.imwrite("Video_detectedFaces/"+file_name, face_crop)
    else:
        print("No faces detected")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()