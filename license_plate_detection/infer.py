



# read video by index
video = cv.VideoCapture(videos[1])
ret, frame = video.read()

# get video dims
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('./outputs/uk_dash_np_2.avi', fourcc, 20.0, size)

# read frames
ret = True

while ret:
    ret, frame = video.read()

    if ret:
        # detect & track objects
        results = np_model.track(frame, persist=True)

        # plot results
        composed = results[0].plot()

        # save video
        out.write(composed)

out.release()
video.release()