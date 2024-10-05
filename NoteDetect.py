# import the open cv image detection libary we need
import cv2

# import numpy to handle and use the data that opencv functions return
import numpy as np

# import networktables to output to RoboRIO
from networktables import NetworkTables

# initialize IP address (address might be wrong)
NetworkTables.initialize(server="127.0.0.1")
sd = NetworkTables.getTable("SmartDashboard")

# tell opencv what camera to use for video capture - opening the camera
capture = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)

# loop that runs and constantly scans from the camera to detect notes
while True:
    # read the current frame from the camera
    ret, frame = capture.read()

    # convert the frame to gray and then back to bgr
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # a bunch of math stuff that Thomas did
    diff = cv2.subtract(frame, gray)
    bitwise = cv2.bitwise_and(gray, frame)
    absdiff = cv2.absdiff(frame, gray)
    absdiff2 = cv2.add(absdiff, diff)
    absdiff2 = cv2.add(absdiff2, diff)
    absdiff2 = cv2.add(absdiff2, diff)
    subdiff = cv2.bitwise_not(absdiff2)
    subdiff = cv2.absdiff(subdiff, absdiff2)
    subdiff = cv2.bitwise_not(subdiff)
    absdiff2 = cv2.subtract(absdiff2, subdiff)
    subdiff2 = cv2.bitwise_and(subdiff, diff)
    bitwise = cv2.subtract(absdiff2, subdiff2)
    #mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = diff
    th = 1
    imask =  mask>th
    
    canvas = np.zeros_like(absdiff2, np.uint8)
    canvas[imask] = frame[imask]

    # blur the frame
    blurred_frame = cv2.GaussianBlur(bitwise, (7, 7), cv2.BORDER_DEFAULT)

    # remove all blue from the image
    absdiff2[:, :, 0] = np.zeros([frame.shape[0], frame.shape[1]])
    
    # convert to grayscale
    absdiff2_gray = cv2.cvtColor(absdiff2, cv2.COLOR_BGR2GRAY)

    retval, thresh = cv2.threshold(absdiff2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    median_filtered = cv2.medianBlur(thresh, 5)

    contours, heirarchy = cv2.findContours(median_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        contour_list = []
        for contour in contours:
            if cv2.contourArea(contour) > 300:
                contour_list.append(contour)

        for note in contour_list:
            approx = cv2.approxPolyDP(note, 0.009 * cv2.arcLength(note, True), True)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

            n = approx.ravel()
            i = 0

            for j in n:
                if (i % 2 == 0):
                    x = n[i] - 320
                    y = n[i + 1] - 240

                    print(f"x: {x}, y: {y}")
                    print("Note Detected")
                    
                    #put value functions here
                    sd.putValue("note_x", int(x))
                    sd.putValue("note_y", int(y))
                    sd.putBoolean("noteDetected", True)
            
    if str(contours) == "()":
        sd.putBoolean("noteDetected", False)
        print("No Note")

    """
    NEXT STEPS

    get the largest contour (assumed to be note)
    filter out red more because it is detecting red things
    get ONLY the x and y of the largest contour
    add x and y to network tables
    make sure network tables data names are the same on this end and robot's end
    """

    # show the final frame that does the  best job at detecting notes
    cv2.imshow("final output", frame)

    # if we click the "q" key, break from the loop
    # stop running the code and stop detecting notes
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# close the windows that open cv opened to show our note detection output
cv2.destroyAllWindows()
