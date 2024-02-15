import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import pyttsx3
import time
from threading import Thread

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_text(text):
    """Function to speak text in a separate thread."""
    engine.say(text)
    engine.runAndWait()

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize and load your model here
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 36  # Update this to your number of classes
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model_path = 'd:\\kaggle\\model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

class_names_path = 'd:\\kaggle\\class.txt'
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Define preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))

last_prediction_time = time.time()
prediction_interval = 2  # seconds
last_prediction = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Perform inference at specified intervals
    if current_time - last_prediction_time > prediction_interval:
        # Convert the frame to a PIL Image and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_label = class_names[preds.item()]

        last_prediction = predicted_label
        last_prediction_time = current_time

        # Speak the prediction in a non-blocking manner
        Thread(target=speak_text, args=(last_prediction,)).start()

    # Display the prediction on the frame
    cv2.putText(frame, last_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the video file
    out.write(frame)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
