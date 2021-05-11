from tkinter import *
from tkinter import filedialog
import cv2
import os
import shutil
from shutil import copy
from PIL import Image, ImageTk
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from categories import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
video_stop_processed = True
video_stop_live = True
seconds = -1
age_cat = ['0-2',
 '4-6',
 '8-12',
 '15-20',
 '25-32',
 '38-43',
 '48-53',
 '60-100']

gender_cat = ['Female', 'Male', 'Infant']

age_total = dict(zip(age_cat, [0] * len(age_cat)))
gender_total = dict(zip(gender_cat, [0] * len(gender_cat)))
total = 0
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model1 = tf.saved_model.load(os.path.join("models","faster_rcnn"))
category_index = label_map_util.create_category_index_from_labelmap(
    os.path.join("models", "label_map.pbtxt"),
    use_display_name=True)
detect_fn = model1.signatures['serving_default']

model2 = tf.keras.models.load_model(os.path.join('models','facenet'))


# Custom Methods
# Create new folder "face_folder" to store uploaded images
def start_up():
    curr_directory = os.getcwd()
    if os.path.isdir('face_folder'):
        shutil.rmtree('face_folder')
    os.mkdir(os.path.join(curr_directory, 'face_folder'))


# Upload image to face_folder
def upload_clicked():
    global video_stop_processed
    global video_stop_live
    video_stop_processed = True
    video_stop_live = True
    file = filedialog.askopenfilename(filetypes=(("files", "*.jpg"),
                                                 ("files", "*.png"),
                                                 ("files", "*.mp4")))
    curr_directory = os.getcwd() + '/face_folder'
    copy(file, curr_directory)
    fileList.insert(END, os.path.basename(file))


# Process selected file to display image and name
def process_selected():
    global img
    global feed
    global age_cat
    global gender_cat
    global video_stop_processed
    global video_stop_live
    global cap
    global seconds
    index = int(fileList.curselection()[0])
    file = fileList.get(fileList.curselection())
    if index == 0:
        video_stop_live = True
        video_stop_processed = False
        cap = cv2.VideoCapture(0)
        process_live_stream()
    elif file.endswith('.mp4'):
        video_stop_live = True
        video_stop_processed = False
        path = os.path.join(os.getcwd(), 'face_folder', file)
        cap = cv2.VideoCapture(path)
        seconds = 0
        process_live_stream()
    else:
        video_stop_processed = True
        video_stop_live = True        
        fileName.config(text=file)    
        path = os.path.join(os.getcwd(), 'face_folder', file)
        image_np = np.array(Image.open(path))
        detections, image = process_image(image_np)
        bounding_boxes = detections['detection_boxes']
        image = Image.fromarray(image)
        image.thumbnail((480, 280), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        feed.configure(image=img)
        feed.image = img
        image = Image.open(path)
        width, height = image.size
        facelist = []
        for i, img_box in enumerate(bounding_boxes):
            (ymin, xmin, ymax, xmax) = (img_box[0] * height, img_box[1] * width, img_box[2] * height, img_box[3] * width)
            crop_img =  np.asarray(image.crop((max(int(xmin)-3,0), max(int(ymin)-3,0), min(int(xmax)+3,width), min(int(ymax)+3,height))).resize((160,160))) / 255.0
            facelist.append(crop_img)
        if facelist:
            face_np = np.array(facelist)
            prediction = model2.predict(face_np)
            age_pred = [age_cat[x] for x in np.argmax(prediction[0],axis=-1)]
            gender_pred = [gender_cat[x] for x in np.argmax(prediction[1],axis=-1)]
            update_live_feed(age_pred, gender_pred)
            update_total_feed(age_pred, gender_pred)
        else:
            update_live_feed([], [])


    # need to implement video
    # need to also run it through the model, then save the output to a local directory
    

# Delete selected file
def delete_selected():
    global video_stop_processed
    global video_stop_live
    video_stop_processed = True
    video_stop_live = True
    file = fileList.get(fileList.curselection())
    index = fileList.get(0, END).index(file)
    if index > 0:
        fileList.delete(index)
        os.remove(os.path.join('face_folder',  file))

# function to show file when clicked
def show_image(event):
    global feed
    global img
    global video_stop_processed
    global video_stop_live
    global cap
    global seconds
    w = event.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    if index == 0:
        video_stop_processed = True
        video_stop_live = False
        seconds = -1
        cap = cv2.VideoCapture(0)
        video_stream()
    elif value.endswith('.mp4'):
        video_stop_processed = True
        video_stop_live = False
        path = os.path.join('face_folder', value)
        cap = cv2.VideoCapture(path)
        seconds = 0
        video_stream()   
    else:
        video_stop_processed = True
        video_stop_live = True
        fileName.config(text=value)
        path = os.path.join('face_folder', value)
        image  = Image.open(path)
        image.thumbnail((480, 280), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        feed.configure(image=img)
        feed.image = img


# function that shows live video
def video_stream():
    global video_stop_live
    global feed
    global cap
    global seconds
    # Capture frame-by-frame
    ret, frame = cap.read()
    if seconds > -1:
        seconds +=1
        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    if ret:
        # Our operations on the frame come here
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        image = Image.fromarray(cv2image)
        image.thumbnail((480, 280), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image=image)
    if video_stop_live or not ret: 
        feed.configure(image='')
        feed.image = ''
    else: 
        feed.configure(image=img)
        feed.image = img
        feed.after(100, video_stream)
# function that processes live video
def process_live_stream():
    global video_stop_processed
    global feed
    global img
    global seconds
    global cap
    # Capture frame-by-frame
    ret, frame = cap.read()
    if seconds > -1:
        seconds +=1
        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    # Our operations on the frame come here
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections, image = process_image(cv2image)
        bounding_boxes = detections['detection_boxes']
        image = Image.fromarray(image)
        image.thumbnail((480, 280), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        feed.configure(image=img)
        feed.image = img  
        image = Image.fromarray(cv2image)
        width, height = image.size
        facelist = []
        if (seconds % 2 == 0) or (seconds == -1):
            for i, img_box in enumerate(bounding_boxes):
                (ymin, xmin, ymax, xmax) = (img_box[0] * height, img_box[1] * width, img_box[2] * height, img_box[3] * width)
                crop_img =  np.asarray(image.crop((max(int(xmin)-3,0), max(int(ymin)-3,0), min(int(xmax)+3,width), min(int(ymax)+3,height))).resize((160,160))) / 255.0
                facelist.append(crop_img)
            if facelist:
                face_np = np.array(facelist)
                prediction = model2.predict(face_np)
                age_pred = [age_cat[x] for x in np.argmax(prediction[0],axis=-1)]
                gender_pred = [gender_cat[x] for x in np.argmax(prediction[1],axis=-1)]
                update_live_feed(age_pred, gender_pred)
                update_total_feed(age_pred, gender_pred)
            else:
                update_live_feed([], [])
    if video_stop_processed or not ret: 
        feed.configure(image='')
        feed.image = ''
    else:
        if seconds > -1:      
            feed.after(100, process_live_stream)
        else:
            feed.after(1000, process_live_stream)

# function that predict bounding boxes, age and gender classificatoin



# function that updates the age pie chart
def create_age_chart(age_pred):
    global pieAge
    global pieimg
    pieAge.delete('all')
    age_dict =  {key: value/len(age_pred) for key, value in Counter(age_pred).items()}
    fig = plt.figure(figsize = [6,6])
    labels = age_dict.keys()
    sizes = age_dict.values()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    image.thumbnail((240, 240), Image.ANTIALIAS)
    pieimg = ImageTk.PhotoImage(image)
    pieAge.pack_forget()
    pieAge = Label(GUI, image=pieimg)
    pieAge.image = pieimg
# function that updates the gender pie chart
# function to export to csv
# function for suggested ads
# function for updating live feed
def update_live_feed(age_pred, gender_pred):
    global liveFeed
    liveFeed.delete('1.0', END)
    liveFeed.config(state=NORMAL)
    liveFeed.insert(END, f"Faces detected: {len(age_pred)}" + "\n")
    for age, gender in zip (age_pred, gender_pred):
        liveFeed.insert(END, "\n" + f"{gender}: {age}")
    
# function for updating total feed
def update_total_feed(age_pred, gender_pred):
    global totalFeed
    global age_total
    global gender_total
    global total
    totalFeed.delete('1.0', END)
    totalFeed.config(state=NORMAL)
    total += len(age_pred)
    for age, gender in zip (age_pred, gender_pred):
        age_total[age] += 1
        gender_total[gender] += 1
    totalFeed.insert(END, f"Total people: {total}" + "\n")
    for key, value in gender_total.items():
        if value != 0: totalFeed.insert(END, "\n" + f"{key}: {value}") 
    totalFeed.insert(END, "\n")
    for key, value in age_total.items():
        if value != 0: totalFeed.insert(END, "\n" + f"Age {key}: {value}") 

# function for classification box
def process_image(image_np):
    global detect_fn
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections_index = min(len(detections['detection_boxes']), sum(detections['detection_scores'] > 0.7))
    detections['detection_boxes']  = detections['detection_boxes'][:detections_index]
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    return detections, image_np_with_detections


# Create a window
start_up()
GUI = Tk()
GUI.title("Gender and Age Classification at Major Events for Valuable Demographic Trends and Statistics")
width, height = GUI.winfo_screenwidth(), GUI.winfo_screenheight()
GUI.geometry('%dx%d+0+0' % (width,height))
GUI.resizable(True, True)

# Image/Video feed
fileName = Label(text="Current File", width=61, font=("Arial Bold", 10))
blankFeed = Label(width=69, height=20)


feed = Label(GUI, image='')
feed.grid(column=0, row=1, columnspan=4, rowspan=4)
feed.image = ''
# List of files
fileLabel = Label(text="File List", width=20, font=("Arial Bold", 10))
fileList = Listbox(GUI, height=16, width=27)
fileScrollbar = Scrollbar(GUI)
# Upload/Delete/Export file classifications
upload = Button(text="Upload", command=upload_clicked)
process = Button(text="Process", command=process_selected)
delete = Button(text="Delete", command=delete_selected)
saveCSV = Button(text="Export as CSV")
# Selected file classification
fileResultsLabel = Label(text="Classifications", width=20, font=("Arial Bold", 10))
fileResults = Text(GUI, height=10, width=23)
fileResultsScrollbar = Scrollbar(GUI)
# Current classification results
liveFeedLabel = Label(text="Live Feed", width=30, font=("Arial Bold", 10))
liveFeed = Text(GUI, height=10, width=32)
liveFeedScrollbar = Scrollbar(GUI)
totalFeedLabel = Label(text="Total Feed", width=30, font=("Arial Bold", 10))
totalFeed = Text(GUI, height=10, width=32)
totalFeedScrollbar = Scrollbar(GUI)
# Pie Chart of total results
ageChartLabel = Label(text="Age Pie Chart", width=30, font=("Arial Bold", 10))
genderChartLabel = Label(text="Gender Pie Chart", width=30, font=("Arial Bold", 10))
pieAge = Canvas(GUI, height=240, width=240)
# pieAge.image = img
pieGender = Canvas(GUI, height=240, width=240)
# Ad suggestion
adLabel = Label(text="Ad Suggestions", width=20, font=("Arial Bold", 10))
ad = Text(GUI, height=19, width=23)


# Add components
# Image/Video feed
fileName.grid(column=0, row=0, columnspan=4)
blankFeed.grid(column=0, row=1, columnspan=4, rowspan=4)
# List of files
fileLabel.grid(column=4, row=0, columnspan=4)
fileList.grid(column=4, row=1, columnspan=4, sticky='e')
fileScrollbar.grid(column=7, row=1, sticky='nsw')
fileList.config(yscrollcommand=fileScrollbar.set)
fileScrollbar.config(command=fileList.yview)
# Upload/Delete/Export file classifications
upload.grid(column=4, row=3)
process.grid(column=5, row=3)
delete.grid(column=6, row=3)
saveCSV.grid(column=4, row=4, columnspan=3)
# Selected file classification
fileResultsLabel.grid(column=4, row=5, columnspan=4)
fileResults.grid(column=4, row=6, columnspan=4, sticky='e')
fileResultsScrollbar.grid(column=7, row=6, sticky='nsw')
fileResults.config(yscrollcommand=fileResultsScrollbar.set)
fileResultsScrollbar.config(command=fileResults.yview)
# Present classification results
liveFeedLabel.grid(column=0, row=5, columnspan=2)
liveFeed.grid(column=0, row=6, sticky='e')
liveFeedScrollbar.grid(column=1, row=6, sticky='nsw')
liveFeed.config(yscrollcommand=liveFeedScrollbar.set)
liveFeedScrollbar.config(command=liveFeed.yview)
totalFeedLabel.grid(column=2, row=5, columnspan=2)
totalFeed.grid(column=2, row=6, sticky='e')
totalFeedScrollbar.grid(column=3, row=6, sticky='nsw')
totalFeed.config(yscrollcommand=totalFeedScrollbar.set)
totalFeedScrollbar.config(command=totalFeed.yview)
# Pie Chart of total results
ageChartLabel.grid(column=0, row=7, columnspan=2)
genderChartLabel.grid(column=2, row=7, columnspan=2)
pieAge.grid(column=0, row=8, columnspan=2)
pieGender.grid(column=2, row=8, columnspan=2)
# Ad suggestion
adLabel.grid(column=4, row=7, columnspan=4)
ad.grid(column=4, row=8, columnspan=4, rowspan=2)


# Changing color of components
GUI.config(background='#606060')
fileName.config(bg='#E0E0E0')
blankFeed.config(background='#f0f0f0')
fileLabel.config(background='#E0E0E0')
fileList.config(background='white')
upload.config(background='#909090')
process.config(background='#909090')
delete.config(background='#909090')
saveCSV.config(background='#909090')
fileResults.config(background='white')
liveFeedLabel.config(background='#E0E0E0')
liveFeed.config(background='white')
totalFeedLabel.config(background='#E0E0E0')
totalFeed.config(background='white')
ageChartLabel.config(background='#E0E0E0')
genderChartLabel.config(background='#E0E0E0')
pieAge.config(background='white')
pieGender.config(background='white')
adLabel.config(background='#E0E0E0')
ad.config(background='white')


# DEBUG
liveFeed.insert(END, "Faces detected: 8" + "\n")
liveFeed.insert(END, "\n" + "Male: 25-32")
liveFeed.insert(END, "\n" + "Female: 25-32")
liveFeed.insert(END, "\n" + "Male: 38-43")
liveFeed.insert(END, "\n" + "Male: 25-32")
liveFeed.insert(END, "\n" + "Female: 25-32")
liveFeed.insert(END, "\n" + "Female: 38-43")
liveFeed.insert(END, "\n" + "Male: 25-32")
liveFeed.insert(END, "\n" + "Female: 25-32")

totalFeed.insert(END, "Total people: 24" + "\n")
totalFeed.insert(END, "\n" + "Male: 16")
totalFeed.insert(END, "\n" + "Female: 8" + "\n")
totalFeed.insert(END, "\n" + "Age 15-20: 4")
totalFeed.insert(END, "\n" + "Age 25-32: 14")
totalFeed.insert(END, "\n" + "Age 38-43: 6")

# data = [['15-20', 0.17, 'yellow'], ['25-32', 0.58, 'green'], ['38-43', 0.25, 'purple']]
# p = 0
# for i in data:
#     angle = i[1] * 360
#     pieAge.create_arc(220, 220, 20, 20, start=p, extent=angle, fill=i[2])
#     p += angle

# data = [['Male', 0.67, 'blue'], ['Female', 0.33, 'red']]
# p = 0
# for i in data:
#     angle = i[1] * 360
#     pieGender.create_arc(220, 220, 20, 20, start=p, extent=angle, fill=i[2])
#     p += angle

ad.insert(END, "Suggested Ads:" + "\n\n")
ad.insert(END, "Sports" + "\n")
ad.insert(END, "Stationary" + "\n")
ad.insert(END, "Computers" + "\n")

fileResults.insert(END, "Classifications" + "\n\n")
fileResults.insert(END, "Female: 25-32" + "\n")
fileResults.insert(END, "Male: 25-32" + "\n")
fileResults.insert(END, "Female: 25-32" + "\n")


fileList.insert(END, 'Live Streaming')
# event when clicking filenames
fileList.bind("<<ListboxSelect>>", show_image)

# Mainloop
GUI.mainloop()
