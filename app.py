from flask import Flask, render_template, request, Response, redirect
import numpy as np
import cv2
import threading
import pyttsx3
from ultralytics import YOLO

app = Flask(__name__)
# mail = Mail(app)

MODEL_FILE="static/models/yolov8n.pt"
COCO_FILE="static/models/coco.txt"
# // if you have second camera you can set first parameter as 1
video = cv2.VideoCapture('')


@app.route('/')
def index():
     return render_template('index.html')


VIDEO_EXTENSIONS = ['mp4']
PHOTO_EXTENSIONS = ['png', 'jpeg', 'jpg']


def fextension(filename):
    return filename.rsplit('.', 1)[1].lower()


@app.route('/upload', methods=['POST'])
def upload():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_new')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'


@app.route('/upload_gun', methods=['POST'])
def upload_gun():
    global video
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype = fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/video/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/video/' + file.filename)
            return redirect('/video_feed_gun')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/photo/' + file.filename)
            print('photo')
            return render_template('preview_photo.html', file_name=file.filename, type='image/'+exttype)
    return 'Invalid video file'

# def sendmessage(res,sub,body):
#     with app.app_context():
#         msg = Message(
#                     sub,
#                     sender ='shivanwhy999@gmail.com',
#                     recipients = res
#                 )
#         msg.body = body
#         mail.send(msg)


class PrepareImage():
    '''ATTRIBUTES:
    gauss_size : kernel size for the gaussian blur 
                type-> tuple of size 2 with odd and equal 
                       entries > 1
    gauss_deviation : x and y axis standard deviations for 
               gaussian blur 
               type -> list-like of size = 2
    auto_canny : If auto canny is True use median of blurred 
                image to calculate thresholds 
                type-> boolean
    canny_low : the lower threshold of the canny filter 
                type -> int 
    canny_high : the higher threshold of the canny filter 
                type -> int 
    segment_x : the width of segment peak( the triangular 
               segment head). Given as the fraction of the width 
               of the image
               type -> float in (0,1) 0 and 1 exclusive
    segment_y : the height segment peak
                Given as the fraction of the height from the 
                top 
                type -> float in (0,1) 0 and 1 exclusive

    METHODS:
    do_canny : does gaussian blurring and canny thresholding of image 
    segment_image : segments the lane to reduce computation cost 
    get_poly_mask_points : returns the lane area to analyse for curve fitting
    get_binary_image: 
    '''

    def __init__(self,
                 gauss_size=None,
                 gauss_deviation=None,
                 auto_canny=False,
                 canny_low=50,
                 canny_high=175,
                 segment_x=0.5,
                 segment_y=0.5):

        # setting gaussian kernel parameters.
        if (gauss_size is not None):
            if (len(gauss_size) != 2):
                raise Exception("Wrong size for the Gaussian Kernel")
            elif (type(gauss_size) is not tuple):
                raise Exception("Kernel type should be a tuple")
            elif (gauss_size[0] % 2 == 0 or gauss_size[1] % 2 == 0):
                raise Exception("Even entries found in Gaussian Kernel")
        self.gauss_kernel = gauss_size

        if (gauss_deviation is not None):
            if (len(gauss_deviation) != 2):
                raise Exception("Wrong length of gauss deviation")
            else:
                self.gauss_deviation = gauss_deviation

        if (type(auto_canny) is not bool):
            raise TypeError("Incorrect Type mentioned for auto canny")

        # setting canny parameters
        if (auto_canny is False):
            self.auto_canny = False
            if (type(canny_high) is int and type(canny_low) is int):
                self.canny_low = canny_low
                self.canny_high = canny_high
            else:
                raise TypeError(
                    "Incorrect type specified for canny thresholds")
        else:
            self.auto_canny = True

        # setting segment parameters
        if segment_x >= 1 or segment_x <= 0:
            raise Exception("Fraction specified is out of range (0,1)")
        else:
            self.segment_x = segment_x
        if segment_y >= 1 or segment_y <= 0:
            raise Exception("Fraction specified is out of range (0,1)")
        else:
            self.segment_y = segment_y

    def do_canny(self, frame):
        '''PARAMETERS: frame: the frame of the image on which we want to apply the 
                      canny filter 
          RETURNS : a canny filtered frame '''
        # gray the image
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # apply blur
        if (self.gauss_kernel is None):
            self.gauss_kernel = (9, 9)  # using a default kernel size
        if (self.gauss_deviation is None):
            self.gauss_deviation = [3, 3]

        blur = cv2.GaussianBlur(gray, self.gauss_kernel,
                                self.gauss_deviation[0], self.gauss_deviation[1])

        # apply canny filter
        if self.auto_canny is False:
            canny = cv2.Canny(blur, self.canny_low, self.canny_high)
        else:
            # Auto canny trumps specified parameters
            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            canny = cv2.Canny(blur, lower, upper)

        return canny

    def segment_image(self, frame):
        '''PARAMETERS: frame : the frame of the image on which we want to apply the 
                      segementation filter 
        RETURNS : a segmented canny filtered frame '''
        height = frame.shape[0]
        width = frame.shape[1]
        shift = int(0.08 * width)
        points = np.array([
            [(0, height), (width, height), (int(width*self.segment_x)+shift, int(height*self.segment_y)),
             (int(width*self.segment_x)-shift, int(height*self.segment_y))]
        ])
        # create an image with zero intensity with same dimensions as frame.
        mask = np.zeros_like(frame)

        # filling the frame's triangle with white pixels
        cv2.fillPoly(mask, points, 255)
        # do a bitwise and on the canny filtered black and white image and the
        # segment you just created to get a triangular area for lane detection
        segment = cv2.bitwise_and(frame, mask)

        # boundary lines...
        # cv.line(segment,(0,height),(int(width*self.segment_x)-shift,int(height*self.segment_y)),(250,0,0),1)
        # cv.line(segment,(width,height),(int(width*self.segment_x)+shift,int(height*self.segment_y)),(250,0,0),1)
        # cv.line(segment,(int(width*self.segment_x)+shift,int(height*self.segment_y)),
        #      (int(width*self.segment_x)-shift,int(height*self.segment_y)),(250,0,0),1)

        return segment
    # this needs to be less tilted

    def get_poly_maskpoints(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        shift = int(0.08 * width)
        points = np.array([
            [(2*shift, height), (width-2*shift, height), (int(width*self.segment_x)+2*shift, int(height*self.segment_y)),
             (int(width*self.segment_x)-2*shift, int(height*self.segment_y))]
        ])
        left = (points[0][0], points[0][3])
        right = (points[0][1], points[0][2])
        return (left, right)

    def get_binary_image(self, frame):
        can = self.do_canny(frame)
#         cv.imshow(can)
        seg = self.segment_image(can)
        return seg


def speak(text, rate=1.4, voice_type="female", volume=1):
    def speak_voice():
        engine = pyttsx3.init()
        # speech rate
        engine.setProperty('rate', rate*engine.getProperty('rate'))

        # Voice
        if (voice_type == "female"):
            engine.setProperty('voice', engine.getProperty('voices')[1].id)
        else:
            engine.setProperty('voice', engine.getProperty('voices')[0].id)

        # Volume)
        engine.setProperty('volume', volume)
        engine.say(text)
        engine.runAndWait()

    try:
        start_voice = threading.Thread(target=speak_voice)
        start_voice.start()
    except:
        print("Warning")


class Curve():
    '''PARAMETERS: 
                     window_size -> float (0,1): how wide you want the 
                                    window to be 
      METHODS: // to-do
    '''

    def __init__(self, draw=False):
        self.non_zero = []
        self.prev_left = None
        self.prev_right = None
        self.draw = False

    def get_lane_points(self, frame, left, right, pxx=10, pxy=30):
        '''
        PARAMETERS: frame, left -> points of left boundary, right -> points of right boundary
                    pxx -> pixel size of x 
                    pxy -> pixel size in y
        RETURNS: points : a list of 2 tuples of proposed lane coords'''
        x_start = left[0][0]
        x_end = right[0][0]
        y_start = left[1][1]
        y_end = left[0][1]
        x = np.array([], dtype=np.uint32)
        y = np.array([], dtype=np.uint32)
        for i in range(y_start, y_end, pxy):
            for j in range(x_start, x_end, pxx):
                if ((pxx*pxy)/40 < np.count_nonzero(frame[i:i+pxx, j:j+pxy]) < (pxx*pxy)/15):
                    nz = np.nonzero(frame[i:i+pxx, j:j+pxy])
                    x = np.hstack((x, nz[0]+i))
                    y = np.hstack((y, nz[1]+j))

        return np.transpose((x, y))

    def detect_curve(self, img, x, y, left, right):
        ''' PARAMETRS: Frame, x-> X coordinates of white points , y-> Y coordinates of white points
                       left -> Points of left boundary
                       right -> Points of right boundary

            RETURNS: -> Image with the single curve traced
        '''
        img2 = np.zeros_like(img)

#         y = -y
        a, b, c = np.polyfit(x, y, 2)
        x_start = left[0][0]
        x_end = right[0][0]
        y_start = left[1][1]
        y_end = left[0][1]
        for i in range(min(x), max(x)):
            y_ = int(a*i*i+b*i+c)
            try:
                if (y_ < img2.shape[0] and y_ > 0):
                    img2[i, y_] = 255
            except:
                pass
        return img2

    def curveTrace(self, frame, left, right):
        '''
        PARAMETERS:  frame,left - coordinates of left boundary, right - coordinates of right boundary
        '''
        height, width = frame.shape
        self.non_zero = []
        # splitting the image to two parts
        left_img = frame[:, :width//2]
        right_img = frame[:, width//2+1:]

        # Working on the left curve
        try:
            curr_points = self.get_lane_points(left_img, left, right, 10, 30)
         # what if very less points?
            if (self.prev_left is None):
                self.prev_left = curr_points
                self.non_zero.append(curr_points)
                x, y = np.transpose(curr_points)
            else:
                if (len(curr_points) < int(0.6*len(self.prev_left)) or curr_points is None):
                    x, y = np.transpose(self.prev_left)
                    self.non_zero.append(self.prev_left)
                else:
                    x, y = np.transpose(curr_points)
                    self.prev_left = curr_points
                    self.non_zero.append(curr_points)

            left_curve = self.detect_curve(left_img, x, y, left, right)
        except:
            left_curve = left_img

        # Working on the right curve
        try:
            flipped_right_img = cv2.flip(right_img, 1)
            curr_points = self.get_lane_points(
                flipped_right_img, left, right, 10, 30)

        # what if very less points?
            if (self.prev_right is None):
                self.prev_right = curr_points
                x, y = np.transpose(curr_points)
                self.non_zero.append(curr_points)
            else:
                if (len(curr_points) < int(0.6*len(self.prev_right)) or curr_points is None):  # 30 %
                    x, y = np.transpose(self.prev_right)
                    self.non_zero.append(self.prev_right)
                else:
                    self.prev_right = curr_points
                    x, y = np.transpose(curr_points)
                    self.non_zero.append(curr_points)

            right_curve = self.detect_curve(
                flipped_right_img, x, y, left, right)
            flipped_right_curve = cv2.flip(right_curve, 1)
        except:
            flipped_right_curve = right_img

        img2 = np.hstack((left_curve, flipped_right_curve))
        return img2

    def drawCurve(self, image, curve, color=(255, 255, 0), thickness=3):
        '''
        PARAMETERS:  image: Original image colored
                     curve -> Curve to draw on the image
                     color -> color of the curve
                     thickness -> Thickness of the curve '''
        height, width, col = image.shape
        if (self.draw == True):
            start = curve.shape[0]//3
        else:
            start = curve.shape[0]
        for i in range(start, curve.shape[0]):
            for j in range(curve.shape[1]):
                if (curve[i, j] != 0):
                    for x in range(thickness):
                        try:
                            image[i, j+x] = color
                        except:
                            pass
        return image


class Predictions():
    '''Provides predictions for a given binary frame where 
       the noise in the image has been removed.
       PARAMETERS: basis: string -> "mean" or "median" 
                           how do you provide the output 
                           for the lane that you acquired
                   threshold: float(0,1) : how closely you 
                           want the lane to be detected relative 
                           to center of image '''

    def __init__(self, basis="mean",
                 threshold=0.1):

        if (basis not in ["mean", "median"]):
            raise ValueError("Basis should be either mean or median")
        self.basis = basis

        if (threshold <= 0 or threshold >= 1):
            raise ValueError("Invalid range for threshold")
        self.threshold = threshold

    def get_lane_middle(self, img, X):
        '''RETURNS: middle x co-ordinate based on the 
                    basis defined in class parameters '''
        if (self.basis == "mean"):
            try:
                mid = int(np.mean(X))
            except:
                mid = img.shape[1]//2
        else:
            try:
                mid = int(np.median(X))
            except:
                mid = img.shape[1]//2
        return mid

    def shifted_lane(self, frame, deviation):
        '''Generates outputs for where to shift 
        given the deviation of the lane center 
        with the image center orientation 

        RETURNS: frame with shift outputs '''
        height, width = frame.shape[0], frame.shape[1]
        shift_left = ["Lane present on left", "Shift left"]
        shift_right = ["Lane present on right", "Shift right"]
        if (deviation < 0):
            # means person on the right and lane on the left
            # need to shift left
            cv2.putText(frame, shift_left[0],
                        (40, 40), 5, 1.1, (100, 10, 255), 2)
            cv2.putText(frame, shift_left[1],
                        (40, 70), 5, 1.1, (100, 10, 255), 2)

            speak(shift_left)
        else:
            # person needs to shift right
            cv2.putText(frame, shift_right[0],
                        (40, 40), 5, 1.1, (100, 255, 10), 2)
            cv2.putText(frame, shift_right[1],
                        (40, 70), 5, 1.1, (100, 255, 10), 2)

            speak(shift_right)
        return frame

    def get_outputs(self, frame, points):
        '''Generates predictions for walking 
           on a lane 
           PARAMETERS: frame : original frame on which we draw
                             predicted outputs. This already has the 
                             lanes drawn on it 
                       points : list of 2-tuples : the list 
                              which contains the points of the lane 
                              which is drawn on the image 
           RETURNS : a frame with the relevant outputs 
           '''

        height, width = frame.shape[0], frame.shape[1]
        # get the center of frame
        center_x = width//2
        # get the distribution of points on
        # left and right of image center
        left_x, right_x = 0, 0
        X = []
        for i in points:
            for k in i:
                x = k
                if (x < center_x):
                    left_x += 1
                else:
                    right_x += 1
                X.append(k)
        # get the lane middle and draw
        try:
            lane_mid = self.get_lane_middle(frame, X)
        except:
            lane_mid = center_x
        cv2.line(frame, (lane_mid, height-1),
                 (lane_mid, height - width//10), (0, 0, 0), 2)
        # calculate shift
        shift_allowed = int(self.threshold*width)
        # calculate deviations and put on image
        deviation = lane_mid - center_x
        deviation_text = "Deviation: " + \
            str(np.round((deviation * 100/width), 3)) + "%"
        cv2.putText(frame, deviation_text, (int(lane_mid-60),
                    int(height-width//(9.5))), 1, 1.3, (250, 20, 250), 2)
        # speak(deviation_text)

        if (abs(deviation) >= shift_allowed):
            # large deviation : give shift outputs only
            frame = self.shifted_lane(frame, deviation)
            return frame
        else:
            # if deviation lesser then that means either correct path
            # or a turn is approaching : text put at the center of the
            # frame

            total_points = left_x + right_x
            correct = ["Good Lane Maintainance", " Continue straight"]
            left_turn = ["Left turn is approaching",
                         "Please start turning left"]
            right_turn = ["Right turn is approaching",
                          "Please start turning right"]
            # if relative change in percentage of points is < 10% then
            # going fine
            try:
                left_perc = left_x*100/(total_points)
                right_perc = right_x*100/(total_points)
            except:
                left_perc = 50
                right_perc = 50
            if (abs(left_perc - right_perc) < 25):
                cv2.putText(frame, correct[0],
                            (40, 40), 5, 1.1, (100, 255, 10), 2)
                cv2.putText(frame, correct[1],
                            (40, 70), 5, 1.1, (100, 255, 10), 2)

                speak(correct)
            else:
                if (left_perc > right_perc):  # more than 25% relative change
                    # means a approximately a right turn is approaching
                    cv2.putText(
                        frame, right_turn[0], (40, 40), 5, 1.1, (100, 10, 255), 2)
                    cv2.putText(
                        frame, right_turn[1], (40, 70), 5, 1.1, (100, 10, 255), 2)

                    speak(right_turn)
                else:
                    cv2.putText(
                        frame, left_turn[0], (40, 40), 5, 1.1, (100, 10, 255), 2)
                    cv2.putText(
                        frame, left_turn[1], (40, 70), 5, 1.1, (100, 10, 255), 2)

                    speak(left_turn)
            # return the frame with the outputs
            # to-do : output with sound
            return frame


def gen_new(video):
    # Update the path to your video file
    # cap = cv2.VideoCapture(
    # r'C:\Users\aryan\Documents\DIP Proj\Code\WhatsApp Video 2024-06-04 at 22.43.02_fa4a6b88.mp4')
    i = 0
    ImagePreprocessor = PrepareImage(
        (11, 11), (2, 0), False, 50, 170, 0.5, 0.37)
    CurveMaker = Curve(draw=True)
    Predict = Predictions(basis='median', threshold=0.3)

    #yolo model prediction code

    my_file = open(COCO_FILE, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()

    # load a pretrained YOLOv8n model
    model = YOLO(MODEL_FILE, "v8") 

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (500, 350))

    while True:
        ret, frame = video.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        
        if ret:
            # frame = cv2.resize(frame, (500, 350))
            # display_image(frame, "Original Frame")
            # Write the original frame to the output video
            # out.write(frame)
            image = ImagePreprocessor.get_binary_image(frame)
            if image is None or image.size == 0:
                print("Binary image is empty")
                continue
            # display_image(image, "Binary Image")

            # Convert binary image to 3-channel to save it as video frame
            # binary_image_color = cv2.cvtColor(
                #   image, cv2.COLOR_GRAY2BGR)
            # Write the binary image to the output video
            # out.write(binary_image_color)

            left, right = ImagePreprocessor.get_poly_maskpoints(image)
            curve = CurveMaker.curveTrace(image, left, right)
            if curve is None or curve.size == 0:
                print("Curve image is empty")
                continue
            # display_image(curve, "Curve Image")

            # Convert curve image to 3-channel to save it as video frame
            # curve_color = cv2.cvtColor(curve, cv2.COLOR_GRAY2BGR)
            # Write the curve image to the output video
            # out.write(curve_color)

            curve_with_text = CurveMaker.drawCurve(
                frame, curve)
            points = np.argwhere(curve == 255)
            final = Predict.get_outputs(curve_with_text, points)
            # display_image(final, "Final Output")
            # Write the final output to the output video
            # out.write(final)

            ret, jpeg = cv2.imencode('.jpg', final)
            final = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n\r\n')
            i += 1

        #YOLO part here
        frame = cv2.resize(frame, (720, 480))


        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()
        #print(DP)
        no_faces=0
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),(255,255,255),3,)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame,class_list[int(clsID)],(int(bb[0]), int(bb[1]) - 10),font,0.5,(255, 255, 255),1,)
                if class_list[int(clsID)] == "person":
                    person_detect = ["A person is approaching",
                         "Please be cautious"]
                    speak(person_detect)

        
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed_new')
def video_feed_new():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
