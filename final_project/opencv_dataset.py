def create_dataset(video_file):
    '''
    Input : location of the video input (video_file)
    Output : Creates folders(sorted images) inside your directory depending upon the classes and masks
    '''
    cap = cv2.VideoCapture(video_file)

    count = 0
    #  VIDEO PROPERTIES
    print("Frame Width : ")
    print(cap.get(3))  # Frame Width
    print("Frame Height :")
    print(cap.get(4))  # Frame Height
    fps = (cap.get(cv2.CAP_PROP_FPS))
    print("FPS :", fps)
    # create a mask from the extacted frame.
    fullmask = cv2.createBackgroundSubtractorMOG2()
    while (cap.isOpened):
        ret, frame = cap.read()
        fgmask = fullmask.apply(frame)
        if fgmask is None:
            break
        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 600: # removes smaller moving objects.
                continue

            filename_frame = "/Users/ubuntu/files/frames/frame_%d.jpg" % count
            # resize image
            resized_frame = cv2.resize(frame, (128,128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename_frame, resized_frame)

            filename_mask = "/Users/ubuntu/files/fgmask/fgmask_%d.jpg" % count
            # resize image
            resized_fgmask = cv2.resize(fgmask, (128,128), interpolation=cv2.INTER_AREA)
            # writing the image to a file.
            cv2.imwrite(filename_mask, resized_fgmask)
            count = count + 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
