import cv2 as cv

def displayVideo():
    
    cap = cv.VideoCapture('drive.mp4')

    while(cap.isOpened()):

        # Read every video frame with cap.read()
        ret, frame = cap.read()
        if frame is None:
            break

        # Display video frame
        cv.imshow('frame', frame)

        # Wait for q key from user to exit video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    displayVideo()