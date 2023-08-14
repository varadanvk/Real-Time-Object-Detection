import torch
import numpy as np
import cv2
import requests
import imutils

class WebcamObjectDetection:
    
   
    
    def __init__(self, output_file="output.avi"):
        self.model = self.load_model()
        self.model.conf = 0.4  
        self.model.iou = 0.3  
        self.classes = self.model.names
        self.out_file = output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord

    def class_to_label(self, class_index):
        return self.classes[int(class_index)]


    def findGeneralLocation(self, x, y):
        dirX = ""
        dirY = ""
        if( x < 4): dirX = "Left"
        elif(x<8): dirX = "MidLeft"
        elif(x==8): dirX = "Middle"
        elif(x<12): dirX = "MidRIght"
        elif(x<16): dirX = "Right"
        
        if(y<3): dirY = "Top"
        elif(y<6): dirY = "Middle"
        elif(y<9): dirY = "Bottom"
        
        cords = f"{dirX} {dirY}"
        return cords
            


    def plot_boxes(self, results, frame):
        
        # the webcam for this is in a 16:9 format
        # i have to divide it into that kind of grid
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
                xCord = (int)(((x1+x2)/2)/80)
                yCord = (int)(((y1+y2)/2)/80)
                cords = self.findGeneralLocation(xCord, yCord)
                
                print(f"{self.class_to_label(labels[i])} {cords} {xCord} {yCord}")


                

        return frame

   
            
    def run(self):
        
        cap = cv2.VideoCapture(0)  # Open the webcam for video capture
        assert cap.isOpened()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(60)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            cv2.imshow('Real-Time Object Detection', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or input == "quit":  # Press 'q' to quit
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    
    # def run(self):
    #     url = "http://10.53.28.151:8080/shot.jpg"

    #     while True:
    #         img_resp = requests.get(url)
    #         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    #         frame = cv2.imdecode(img_arr, -1)
    #         if frame is None:
    #             break

    #         results = self.score_frame(frame)
    #         frame = self.plot_boxes(results, frame)

    #         cv2.imshow('Real-Time Object Detection', frame)

    #         # Add video writing logic if you want to save the processed frames to a video file
    #         # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #         # out = cv2.VideoWriter(self.out_file, fourcc, 60, (1000, 1800))
    #         # out.write(frame)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     cv2.destroyAllWindows()



output_file = "output.avi"  # Specify the output file name
object_detection = WebcamObjectDetection(output_file)
object_detection.run()
