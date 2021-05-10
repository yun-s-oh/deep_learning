from tkinter import *
from tkinter import filedialog
import os
import shutil
from shutil import copy
from PIL import Image, ImageTk


# Custom Methods
# Create new folder "face_folder" to store uploaded images
def start_up():
    curr_directory = os.getcwd()
    if os.path.isdir('face_folder'):
        shutil.rmtree('face_folder')
    os.mkdir(os.path.join(curr_directory, 'face_folder'))


# Upload image to face_folder
def upload_clicked():
    file = filedialog.askopenfilename(filetypes=(("files", "*.jpg"),
                                                 ("files", "*.png"),
                                                 ("files", "*.mp4")))
    curr_directory = os.getcwd() + '/face_folder'
    copy(file, curr_directory)
    fileList.insert(END, os.path.basename(file))


# Process selected file to display image and name
def process_selected():
    file = fileList.get(fileList.curselection())
    fileName.config(text=file)

    global img, feed
    path = os.getcwd() + '\\face_folder\\' + file
    image = Image.open(path)
    image.thumbnail((480, 280), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    feed = Label(GUI, image=img)
    feed.grid(column=0, row=1, columnspan=4, rowspan=4)

    # need to implement video
    # need to also run it through the model, then save the output to a local directory


# Delete selected file
def delete_selected():
    file = fileList.get(fileList.curselection())
    index = fileList.get(0, END).index(file)
    fileList.delete(index)
    os.remove(os.getcwd() + '\\face_folder\\' + file)


# function that updates the age pie chart
# function that updates the gender pie chart
# function to export to csv
# function for suggested ads
# function for updating live feed
# function for updating total feed
# function for classification box


# Create a window
start_up()
GUI = Tk()
GUI.title("Gender and Age Classification at Major Events for Valuable Demographic Trends and Statistics")
GUI.geometry('661x758')
GUI.resizable(False, False)

# Image/Video feed
fileName = Label(text="Current File", width=61, font=("Arial Bold", 10))
blankFeed = Label(width=69, height=20)
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

data = [['15-20', 0.17, 'yellow'], ['25-32', 0.58, 'green'], ['38-43', 0.25, 'purple']]
p = 0
for i in data:
    angle = i[1] * 360
    pieAge.create_arc(220, 220, 20, 20, start=p, extent=angle, fill=i[2])
    p += angle

data = [['Male', 0.67, 'blue'], ['Female', 0.33, 'red']]
p = 0
for i in data:
    angle = i[1] * 360
    pieGender.create_arc(220, 220, 20, 20, start=p, extent=angle, fill=i[2])
    p += angle

ad.insert(END, "Suggested Ads:" + "\n\n")
ad.insert(END, "Sports" + "\n")
ad.insert(END, "Stationary" + "\n")
ad.insert(END, "Computers" + "\n")

fileResults.insert(END, "Classifications" + "\n\n")
fileResults.insert(END, "Female: 25-32" + "\n")
fileResults.insert(END, "Male: 25-32" + "\n")
fileResults.insert(END, "Female: 25-32" + "\n")


# Mainloop
GUI.mainloop()
