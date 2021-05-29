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
import pandas as pd
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

df = pd.read_csv('adtable.csv')
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


def reset_play():
    global video_stop_processed
    global video_stop_live
    global feed
    video_stop_processed = True
    video_stop_live = True
    


# Custom Methods
# Create new folder "face_folder" to store uploaded images
def start_up():
    curr_directory = os.getcwd()
    if os.path.isdir('face_folder'):
        shutil.rmtree('face_folder')
    os.mkdir(os.path.join(curr_directory, 'face_folder'))


# Upload image to face_folder
def upload_clicked():
    reset_play()
    file = filedialog.askopenfilename(filetypes=(("files", "*.jpg"),
                                                 ("files", "*.png"),
                                                 ("files", "*.mp4")))
    curr_directory = os.getcwd() + '/face_folder'
    copy(file, curr_directory)
    fileList.insert(END, os.path.basename(file))

def live_selected():
    global video_stop_processed
    global video_stop_live 
    global cap
    global seconds
    video_stop_processed = True
    if not video_stop_live:
        reset_play()
        feed.configure(image='')
        feed.image = ''
    else:        
        video_stop_live = False
        seconds = -1
        cap = cv2.VideoCapture(0)
        video_stream()

def reset_selected():
    global age_total
    global gender_total
    global total
    reset_play()
    age_total = dict(zip(age_cat, [0] * len(age_cat)))
    gender_total = dict(zip(gender_cat, [0] * len(gender_cat)))
    total = 0
    liveFeed.delete('1.0', END)
    totalFeed.delete('1.0', END)
    ad.delete('1.0', END)
    pieAge.configure(image='')
    pieAge.image = ''
    pieGender.configure(image='')
    pieGender.image = ''

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
    
    if not video_stop_processed:
        video_stop_processed = True
        return
    if not video_stop_live and seconds == -1:
        video_stop_live = True
        video_stop_processed = False
        cap = cv2.VideoCapture(0)
        print('run live')
        process_live_stream()
        return
    index = int(fileList.curselection()[0])
    file = fileList.get(fileList.curselection())
    if file.endswith('.mp4'):
        print('run video')
        video_stop_live = True
        video_stop_processed = False
        path = os.path.join(os.getcwd(), 'face_folder', file)
        cap = cv2.VideoCapture(path)
        seconds = 0
        process_live_stream()
    else:
        video_stop_processed = True
        video_stop_live = True        
        # fileName.config(text=file)    
        path = os.path.join(os.getcwd(), 'face_folder', file)
        image_np = np.array(Image.open(path))
        detections, image = process_image(image_np)
        bounding_boxes = detections['detection_boxes']
        image = Image.fromarray(image)
        feed.update()
        image.thumbnail((int(feed.winfo_width()), int(feed.winfo_height())), Image.ANTIALIAS)
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
            create_age_chart(age_pred)
            create_gender_chart(gender_pred)
            update_ad(age_pred, gender_pred)
        else:
            update_live_feed([], [])
            pieAge.configure(image='')
            pieAge.image = ''
            pieGender.configure(image='')
            pieGender.image = ''
            ad.delete('1.0', END)


    # need to implement video
    # need to also run it through the model, then save the output to a local directory
    

# Delete selected file
def delete_selected():
    reset_play()
    file = fileList.get(fileList.curselection())
    index = fileList.get(0, END).index(file)
    fileList.delete(index)
    os.remove(os.path.join('face_folder',  file))
    feed.configure(image='')
    feed.image = ''

# function to show file when clicked
def show_image(event):
    global feed
    global img
    global cap
    global seconds
    global video_stop_live
    reset_play()
    w = event.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    if value.endswith('.mp4'):
        video_stop_live = False
        path = os.path.join('face_folder', value)
        cap = cv2.VideoCapture(path)
        seconds = 0
        video_stream()   
    else:
        # fileName.config(text=value)
        path = os.path.join('face_folder', value)
        image  = Image.open(path)
        feed.update()
        image.thumbnail((int(feed.winfo_width()), int(feed.winfo_height())), Image.ANTIALIAS)
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
        feed.update()
        image.thumbnail((int(feed.winfo_width()), int(feed.winfo_height())), Image.ANTIALIAS)
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
        feed.update()
        image.thumbnail((int(feed.winfo_width()), int(feed.winfo_height())), Image.ANTIALIAS)
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
                create_age_chart(age_pred)
                create_gender_chart(gender_pred)
                update_ad(age_pred, gender_pred)
            else:
                update_live_feed([], [])
                pieAge.configure(image='')
                pieAge.image = ''
                pieGender.configure(image='')
                pieGender.image = ''
                ad.delete('1.0', END)
                
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
    age_dict =  {key: value/len(age_pred) for key, value in Counter(age_pred).items()}
    fig = plt.figure(figsize = [6,6])
    labels = age_dict.keys()
    sizes = age_dict.values()
    plt.pie(sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)
    fig.canvas.draw()
    pieAge.update()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    image.thumbnail((int(pieAge.winfo_width()), int(pieAge.winfo_height())), Image.ANTIALIAS)
    pieimg = ImageTk.PhotoImage(image)
    pieAge.configure(image=pieimg)
    pieAge.image = pieimg
    plt.close(fig)

def create_gender_chart(gender_pred):
    global pieGender
    gender_dict =  {key: value/len(gender_pred) for key, value in Counter(gender_pred).items()}
    fig = plt.figure(figsize = [6,6])
    labels = gender_dict.keys()
    sizes = gender_dict.values()
    color_dict = {'Female':'r', 'Male':'b', 'Infant':'g'}
    colors = [ color_dict[label] for label in labels] 
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, colors = colors,startangle=90)
    fig.canvas.draw()
    pieGender.update()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    image.thumbnail((int(pieGender.winfo_width()), int(pieGender.winfo_height())), Image.ANTIALIAS)
    pieimg = ImageTk.PhotoImage(image)
    pieGender.configure(image=pieimg)
    pieGender.image = pieimg
    plt.close(fig)
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

def update_ad(age_pred, gender_pred):
    global df
    age_dict = Counter(age_pred)
    age = max(age_dict, key=age_dict.get)
    gender_dict = Counter(gender_pred)
    gender = max(gender_dict, key=gender_dict.get)
    ad.delete('1.0', END)
    ad.config(state=NORMAL)
    for i, row in df[(df.Age== age) & (df.Gender ==gender)].iterrows():
        ad.insert(END, f"{row.Interests}" + "\n")

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
root = Tk()
root.title("Gender and Age Classification Model")
width, height = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry('%dx%d+0+0' % (width,height))
root.resizable(True, True)
root.config(background='#606060')
root.iconbitmap(os.path.join('icons', 'uts.ico'))

# Layout of window (column, weight)
root.columnconfigure(0, weight=2)
root.columnconfigure(1, weight=2)
root.columnconfigure(2, weight=3)
root.columnconfigure(3, weight=3)
root.columnconfigure(4, weight=2)
root.columnconfigure(5, weight=2)
root.rowconfigure(0, weight=8)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)


# Screen frame
frame_screen = Frame(root)
frame_screen.grid(row = 0, column = 0, columnspan=4, sticky='nswe')
frame_screen.rowconfigure(0, weight=0)
frame_screen.rowconfigure(1, weight=1)
frame_screen.columnconfigure(0, weight=1)

# # Image/Video feed
fileName = Label(frame_screen,text="",font=("Arial Bold", 12))
fileName.grid(column=0, row=0, sticky='nswe')
feed = Label(frame_screen, image='', padx=1, pady=1)
frame_screen.grid_propagate(0)
feed.grid(column=0,row=1, sticky='nswe')
feed.image = ''

# Filelist frame
frame_filelist = Frame(root)
frame_filelist.grid(row = 0, column =4, sticky='nswe')
frame_filelist.rowconfigure(0, weight=0)
frame_filelist.rowconfigure(1, weight=1)
frame_filelist.columnconfigure(0, weight=1)
frame_filelist.grid_propagate(0)

# List of files
fileLabel = Label(frame_filelist, text="File List", font=("Arial Bold", 12))
fileLabel.grid(column=0, row=0)
fileList = Listbox(frame_filelist, height = 0, width=5, font=("Arial Bold", 12))
fileList.grid(column=0, row=1, sticky='nswe')
fileScrollbar = Scrollbar(frame_filelist)
fileScrollbar.grid(column=1, row=1, sticky='nse')

# Livefeed frame
frame_livefeed = Frame(root)
frame_livefeed.grid(row = 1, column =0,rowspan=2,sticky='nswe')
frame_livefeed.rowconfigure(0, weight=0)
frame_livefeed.rowconfigure(1, weight=1)
frame_livefeed.columnconfigure(0, weight=1)

# Livefeed
liveFeedLabel = Label(frame_livefeed, text="Live Feed", font=("Arial Bold", 12))
liveFeedLabel.grid(column=0, row=0, sticky='nswe')
liveFeed = Text(frame_livefeed, height= 0, width=0, font=("Helvetica", 15))
liveFeed.grid(column=0, row=1, sticky='nswe')
liveFeedScrollbar = Scrollbar(frame_livefeed)
liveFeedScrollbar.grid(column=1, row=1, sticky='nse')


# totalfeed frame
frame_totalfeed = Frame(root)
frame_totalfeed.grid(row = 1, column =1, rowspan=2, sticky='nswe')
frame_totalfeed.rowconfigure(0, weight=0)
frame_totalfeed.rowconfigure(1, weight=1)
frame_totalfeed.columnconfigure(0, weight=1)

# totalfeed
totalFeedLabel = Label(frame_totalfeed, text="Total Feed", font=("Arial Bold", 12))
totalFeedLabel.grid(column=0, row=0, sticky='nswe')
totalFeed = Text(frame_totalfeed, height= 0, width=0, font=("Helvetica", 15))
totalFeed.grid(column=0, row=1, sticky='nswe')
totalFeedScrollbar = Scrollbar(frame_totalfeed)
totalFeedScrollbar.grid(column=1, row=1, sticky='nsw')

# Age Pie Chart frame
frame_agepiechart = Frame(root)
frame_agepiechart.grid(row = 1, column =2, rowspan= 2,sticky='nswe')
frame_agepiechart.rowconfigure(0, weight=0)
frame_agepiechart.rowconfigure(1, weight=1)
frame_agepiechart.columnconfigure(0, weight=1)
frame_agepiechart.grid_propagate(0)

# Age Pie Chart 
ageChartLabel = Label(frame_agepiechart, text="Age Pie Chart", font=("Arial Bold", 12))
ageChartLabel.grid(column=0, row=0, sticky='nswe')
pieAge = Label(frame_agepiechart, image='', anchor= 'n')
pieAge.grid(column=0, row=1, sticky='nswe')
pieAge.image = ''

# gender Pie Chart frame
frame_genderpiechart = Frame(root)
frame_genderpiechart.grid(row = 1, column =3, rowspan = 2, sticky='nswe')
frame_genderpiechart.rowconfigure(0, weight=0)
frame_genderpiechart.rowconfigure(1, weight=1)
frame_genderpiechart.columnconfigure(0, weight=1)
frame_genderpiechart.grid_propagate(0)

# Gender Pie Chart 
genderChartLabel = Label(frame_genderpiechart, text="Gender Pie Chart", font=("Arial Bold", 12))
genderChartLabel.grid(column=0, row=0, sticky='nswe')
pieGender = Label(frame_genderpiechart, image='', anchor= 'n')
pieGender.grid(column=0, row=1, sticky='nswe')
pieGender.image = ''


# Ad list Frame
frame_adlist = Frame(root)
frame_adlist.grid(row = 0, column =5, sticky='nswe')
frame_adlist.rowconfigure(0, weight=0)
frame_adlist.rowconfigure(1, weight=1)
frame_adlist.columnconfigure(0, weight=1)
frame_adlist.grid_propagate(0)
# List of ads
adLabel = Label(frame_adlist, text="Ad suggestion", font=("Arial Bold", 12))
adLabel.grid(column=0, row=0)
ad = Text(frame_adlist, height=5, width=0, font=("Helvetica", 15))
ad.grid(column=0, row=1, sticky='nswe')



# Buttons Frame
frame_buttons = Frame(root)
frame_buttons.grid(row = 1, column =4, columnspan=2, sticky='nswe')
frame_buttons.rowconfigure(0, weight=1)
frame_buttons.rowconfigure(1, weight=1)
frame_buttons.rowconfigure(2, weight=1)
frame_buttons.columnconfigure(0, weight=1) 
frame_buttons.columnconfigure(1, weight=1) 

# button icons
button_size = int(20)
upload_icon = PhotoImage(file = os.path.join('icons', 'upload.png')).subsample(12, 12)
delete_icon = PhotoImage(file = os.path.join('icons', 'delete.png')).subsample(12, 12)
process_icon = PhotoImage(file =os.path.join('icons', 'process.png')).subsample(12, 12)
live_icon = PhotoImage(file = os.path.join('icons', 'live.png')).subsample(12, 12)
reset_icon = PhotoImage(file =os.path.join('icons', 'reset.png')).subsample(12, 12)
exit_icon = PhotoImage(file = os.path.join('icons', 'exit.png')).subsample(12, 12)

# add buttons
upload = Button(frame_buttons, image = upload_icon,text="     Upload", compound="left", command=upload_clicked, font=("Arial Bold", 10))
upload.grid(column=0, row=0,sticky='nswe')
delete = Button(frame_buttons, image = delete_icon, text="     Delete", compound="left", command=delete_selected, font=("Arial Bold", 10))
delete.grid(column=1, row=0,sticky='nswe')
process = Button(frame_buttons, image=process_icon, text="     Process", compound="left", command=process_selected, font=("Arial Bold", 10))
process.grid(column=0, row=1,sticky='nswe')
live = Button(frame_buttons, image=live_icon,text="     Live", compound="left", command=live_selected, font=("Arial Bold", 10))
live.grid(column=1, row=1,sticky='nswe')
reset = Button(frame_buttons, image=reset_icon,text="     Reset", compound="left", command=reset_selected, font=("Arial Bold", 10))
reset.grid(column=0, row=2,sticky='nswe')
exit = Button(frame_buttons, image=exit_icon,text="     Exit", compound="left", command= root.destroy, font=("Arial Bold", 10))
exit.grid(column=1, row=2,sticky='nswe')

# about  Frame
about_blank = Label(root, text="Group 40 - Binary Numbers\nGender and Age Classification\nfor Demographic Trends and Statistics\nWalden Ip\nJohnathan Pham\nYunseok Oh",font=("Arial", 15), justify = 'left', anchor='nw')
about_blank.grid(row = 2, column =4, columnspan = 2, sticky='nswe')

# fileList.insert(END, 'Live Streaming')
# # event when clicking filenames
fileList.bind("<<ListboxSelect>>", show_image)

# Mainloop
root.mainloop()
