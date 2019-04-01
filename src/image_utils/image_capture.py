import cv2
import os



class ImageCapture:

    def __init__(self):
        pass

    def capture_image(self):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        #this line will place the webcam view on the desktop if uncommented
        #os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')  # To make window active

        print('press SCAPE to take a picture')

        while True:
            ret, frame = cam.read()
            cv2.imshow("image", frame)
            if not ret:
                break

            k = cv2.waitKey(1 )
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                frame = None
                break
            elif k%256 == 32:
                # SPACE pressed#
                break

        cam.release()
        cv2.destroyAllWindows()
        return frame
