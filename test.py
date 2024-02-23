import cv2
import os

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ask the user for their name
name = input("Enter Your Name: ")

# Initialize a counter for the number of images saved
img_count = 0

# Create a folder for the dataset if it does not exist
dataset_folder = "D:/Mini_project/dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Check if a folder with the same name already exists
user_folder = os.path.join(dataset_folder, name)
while os.path.exists(user_folder):
    img_count += 1
    user_folder = os.path.join(dataset_folder, f"{name}{img_count}")

# Create the user's folder
os.makedirs(user_folder)

# Loop until we have taken 50 pictures
while img_count <= 50:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not captured properly, skip it
    if not ret:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the face image
        face_img = frame[y:y+h, x:x+w]
        img_name = f"{name}_{img_count}.jpg"
        cv2.imwrite(os.path.join(user_folder, img_name), face_img)

        # Print a message
        print(f"Saved image {img_name}")

        # Increment the image count
        img_count += 1

    # Display the frame with face rectangles
    cv2.imshow("Face Detection", frame)

    # Wait for a short time
    cv2.waitKey(1)

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()

print("Images saved successfully.")
