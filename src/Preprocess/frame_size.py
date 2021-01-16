import cv2

#uses opencv to split the video into 1495 frames as required for stream


vidcap = cv2.VideoCapture('video.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    
    hasFrames,image = vidcap.read()
    if hasFrames:

        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1/25 #//it will capture image 25fps
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
print(count-1)