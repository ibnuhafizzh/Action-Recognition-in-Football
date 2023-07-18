import os
import cv2
import torch
from torchvision import transforms

from model import Model

model_path = "model.pth.tar.pth"
# Load your trained model

model = Model(input_size=512,
                  num_classes=17, window_size=20, 
                  vlad_cluster = 64,
                  framerate=2, pool="TCA").to(torch.device("cpu"))
checkpoint = torch.load(os.path.join("model.pth.tar"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
# model = torch.load(model_path)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# Start the video capture
cap = cv2.VideoCapture('1_224p.mp4')

while(cap.isOpened()):
    print("open")
    # Capture frame-by-frame
    ret, frame = cap.read()
    # image = transform(frame).unsqueeze(0)
    prediction = model(image)
    print(prediction)
    # if ret == True:
    #     print("masuk 2")
    #     # Preprocess the image and predict
    #     image = transform(frame).unsqueeze(0)
    #     # prediction = model(image)
    #     # print(prediction)

    #     # Display or use your prediction in some way...
        
    #     # For example, overlaying text on the video frame
    #     cv2.putText(frame, 
    #                 f'Prediction: null', 
    #                 (50, 50), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 
    #                 1, 
    #                 (0, 255, 0), 
    #                 2, 
    #                 cv2.LINE_AA)

    #     cv2.imshow('Frame', frame)

    #     # Press Q on keyboard to exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    # else: 
    #     break

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()