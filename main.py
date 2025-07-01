import tkinter as tk
import customtkinter
import time
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import mysql.connector
from CustomTkinterMessagebox import CTkMessagebox



#References:
# Various youtubers but this is the main https://www.youtube.com/watch?v=oDaZrqJ2zoo&list=PLiWNvnK7PSPGPDmrdo3jhi_7hvkGrkFlN&index=2 /https://www.youtube.com/watch?v=mkAx_81Pcww&list=PLyDH8KT4GrNd7KMFIgHuxpcklUJyfk_ov
# Tkinter/CustomTkitner = Various youtuber but this guy is the best => https://www.youtube.com/watch?v=Y01r643ckfI&list=PLfZw_tZWahjxJl81b1S-vYQwHs_9ZT77f
# optimization/running the app better = GPT suggestions/ Yung sa threading




mydb = mysql.connector.connect(
    host="localhost", user="root", password="", database="dsp_proj"
)


#---------------------------- Main functions----------------------------

# Starts Camera and Face Recognition
def startCamera():
    def img_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
        face_coords = []
        for x, y, w, h in features:
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray[y : y + h, x : x + w])
            
            mycursor = mydb.cursor()
            mycursor.execute(
                "SELECT student_name FROM student_record WHERE id=" + str(id)
            )  # id yung indicator kung ano name
            idRecord = mycursor.fetchone()
            if idRecord:  # Chinecheck kung mayroon nga na record sa db pag wala unknown.
                idRecord = "".join(idRecord)
                confPercentage = int(100 * (1 - pred / 300))
                if confPercentage > 80:
                    cv.putText(
                        img,
                        idRecord,
                        (x, y - 2),
                        cv.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        0,
                        cv.LINE_AA,
                    )
                else:
                    cv.putText(
                        img,
                        "UNKNOWN",
                        (x, y - 2),
                        cv.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        1,
                        cv.LINE_AA,
                    )
            else:
                cv.putText(
                    img,
                    "UNKNOWN",
                    (x, y - 2),
                    cv.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )
            face_coords = [x, y, w, h]
        return face_coords

    def face_recognize(img, clf, faceFront):
        face_coords = img_boundary(img, faceFront, 1.1, 15, (0, 255, 0), "Face", clf) #Parameters: img, classifier, scaleFactor, minNeighbors, color, text, clf
        return img

    faceFront = cv.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )  # Import classifier para sa frontal face detection

    clf = cv.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")  # Ito yung custom classifier na tinrain.

    cap = cv.VideoCapture(
        0
    )  # "0" is by default if may external camera use -1, 1, 2, ...

    while cap.isOpened():
        ret, img = cap.read()
        img = face_recognize(img, clf, faceFront)
        cv.imshow("Face Recognition", img)
        if cv.waitKey(10) & 0xFF == ord(
            "x"
        ):  # Press x to  exit da goddamn window if needed
            break
    cap.release()
    cv.destroyAllWindows()
# End start camera

# Start Register faces
def collectImage():
    if (
        not nameInput.get() or not idInput.get() or not  courseInput.get()
    ):  # Pag blank yung text input lalabas ng warning notif
        CTkMessagebox.messagebox(title='Alert!', text='Please complete the form!', sound='on', button_text='bruh',size='220x150') #literal na popup notif lang
    else:
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * from student_record")
        myresult = mycursor.fetchall()
        id = 1  # Starts from 1 yung id na ma c-collect iba ito sa auto increment na nasa db. Ito yung para ma identify kung kaninong mukha yung naka display
        # ini-increment bawat insertion ng name, id, course
        for x in myresult:
            id += 1
        sql = "insert into student_record(id, student_name, student_id, student_course) values(%s, %s, %s, %s)"  # bind params lang
        val = (
            id,
            nameInput.get(),
            idInput.get(),
            courseInput.get(),
        )  # Tapos yugn ininsert na value sa nameInput.get(),  courseInput.get(),  courseInput.get() ma s-save sa database.
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv.cvtColor(
                img, cv.COLOR_BGR2GRAY
            )  # Convert images/frames to black and white
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if faces is ():
                return None
            for x, y, w, h in faces:
                cropped_face = img[y : y + h, x : x + w]
            return cropped_face

        cap = cv.VideoCapture(0)
        img_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv.resize(face_cropped(frame), (200, 200))
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                file_path = (
                    "dataset/user." + str(id) + "." + str(img_id) + ".jpg"
                )  # File directory lang
                cv.putText(
                    face,
                    str(img_id),
                    (50, 50),
                    cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )  # Label lang para ma indicate kung ilang frames na habang nag c-collect ng images
                cv.imwrite(file_path, face)
                cv.imshow("Register Face", face)

            if (
                cv.waitKey(1) & 0xFF == ord("x") or int(img_id) == 100
            ):  # press 'x' or wait ng 100 frames para ma complete ang pag collect ng images
                break
        mycursor.close()
        cap.release()
        cv.destroyAllWindows()
        CTkMessagebox.messagebox(title='Success!', text='Collection Complete!', sound='on', button_text='OK',size='220x150') #literal na popup notif lang

        # End Register Faces
def trainData():
    dataset_dir = os.path.join(os.getcwd(), "dataset") #Para di na hardcoded yung file directory ng folder (thank you gpt)
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert("L")
        imageNP = np.array(img, "uint8")
        id = int(
            os.path.split(image)[1].split(".")[1]
        )  # Yung pangalan kasi ng image ay for example "user.1.1" need lang kunin yung precise name
        faces.append(imageNP)
        ids.append(id)

    ids = np.array(ids)

    clf = cv.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")  # S-save yung trained data into .xml file
    CTkMessagebox.messagebox(title='Notification', text='Training data complete!', sound='on', button_text='OK',size='220x150')
#---------------------------- End of Main Functions----------------------------


#---------------------------- Optimization stuffs----------------------------
def threadingCollect():
    threading.Thread(target=collectImage).start()
def threadingCamera():
    threading.Thread(target=startCamera).start()
#--------------------------end of optimziaiton stuff-------------------------

# Start of GUI 
root = customtkinter.CTk()
root.title("Face Recognition:DSP Project")
root.geometry("800x500")
root.resizable(False, False)
# Tabs
tabs = customtkinter.CTkTabview(root,
    width=400,
    height=200                            
                                )
tabs.pack(pady=10)
instructionTab =  tabs.add("Instructions")
registerTab = tabs.add("Register")
trainTab =  tabs.add("Train")
cameraTab =  tabs.add("Camera")




#Instructions
instructions = customtkinter.CTkTextbox(instructionTab, width=600, wrap="char")
instructLabel = customtkinter.CTkLabel(instructionTab, text="INSTRUCTIONS", font=('Helvetica', 20))
inst = """
    1) Enter Name, Student ID, and Course
    2) Click Register, the app will proceed to collect face data for 100 frames. 
    If the user wants to prematurely stop the collection of data click "x" on the keyboard.
    3) If done go to "Train" tab. (Note: Training is required if a new student is registered.)
    4) Go to "Camera" tab to start the face recognition.
"""
instructions.insert(customtkinter.END, inst)
instructLabel.pack()
instructions.pack()

# Enter student info stuff
f1label= customtkinter.CTkLabel(registerTab, text="Register Student",font=('Helvetica', 20) )
f1label.pack()

nameLabel = customtkinter.CTkLabel(registerTab, text="Name", font=('Helvetica', 18))
nameLabel.pack(pady=5)
nameInput = customtkinter.CTkEntry(registerTab, placeholder_text="ex. Juan Dela Cruz")
nameInput.pack(pady=5)

idLabel = customtkinter.CTkLabel(registerTab, text="Student ID", font=('Helvetica', 18))
idLabel.pack(pady=5)
idInput = customtkinter.CTkEntry(registerTab, placeholder_text="ex. 2021123456")
idInput.pack(pady=5)

courseLabel = customtkinter.CTkLabel(registerTab, text="Course", font=('Helvetica', 18))
courseLabel.pack(pady=5)
courseInput = customtkinter.CTkEntry(registerTab, placeholder_text="ex. BS CpE4A")
courseInput.pack(pady=5)

registerData = customtkinter.CTkButton(registerTab, text="Register Face", width=100, height=40, command=threadingCollect)
registerData.pack(pady=20)



# opens the Camera
# cameraStart= customtkinter.CTkLabel(cameraTab, text="")
# cameraStart.pack() 

regFace = customtkinter.CTkButton(cameraTab, text="Start Camera", width=200, command =threadingCamera)
regFace.pack(pady=60, anchor="center")

# train collected face data 
trainFace = customtkinter.CTkButton(trainTab, text="Train Datas", width=200, command =trainData)
trainFace.pack(pady=60, anchor="center")

# Exit the app
exit = customtkinter.CTkButton(
    root,
    text="Exit",
    fg_color="red",
    hover_color="#b30000",
    width=90,
    command=root.destroy,
)
exit.pack(pady=10, anchor="center")



root.mainloop()
