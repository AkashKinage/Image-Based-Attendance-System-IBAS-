from __future__ import division
import re
from django.shortcuts import render, HttpResponse, redirect
from .models import Student, Attendance, Student_Attendance, Student_Attendance_Webcam
import os
import cv2
import numpy as np
from PIL import Image
import pickle

def index(request):
    return render(request, "index.html")

def register_student(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        roll_no = int(request.POST.get('roll_no'))
        department = request.POST.get('department')
        year = request.POST.get('year')
        division = request.POST.get('division')

        # print(first_name)
        # print(last_name)
        # print(roll_no)
        # print(department)
        # print(year)
        # print(division)

        full_name = first_name + " " + last_name
        
        student = Student(
                full_name=full_name,
                first_name=first_name,
                last_name = last_name,
                roll_no = roll_no,
                department = department,
                year = year,
                division = division
            )

        student.save()

        student_dir_path = r'C:\Users\Akash\Desktop\Development\Computer Vision\Image based Attendance System\Students'
        student_path = student_dir_path+"\\"+full_name
        if os.path.exists(student_path):
            print('Folder already exists')
        else:
            os.mkdir(student_path)

        # while True:
        #     department_path = student_dir_path+"\\"+department
        #     year_path = department_path+"\\"+year
        #     division_path = year_path+"\\"+division
        #     student_path = division_path+"\\"+full_name
        #     if os.path.exists(department_path):
        #         if os.path.exists(year_path):
        #             if os.path.exists(division_path):
        #                 if os.path.exists(student_path):
        #                     break
        #                 else:
        #                     os.mkdir(student_path)
        #             else:
        #                 os.mkdir(division_path)
        #         else:
        #             os.mkdir(year_path)
        #     else:
        #         os.mkdir(department_path)

        return render(request, 'index.html')
    return render(request, 'register_student.html')

def search_student(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')

        full_name = first_name + " " + last_name

        student = Student.objects.filter(full_name=full_name).exists()
        # print(student)

        if student == True:
            request.session['first_name'] = first_name
            request.session['last_name'] = last_name
            params = {"available": True}
            return render(request, "register_faces.html", params)
            
    return render(request, "register_faces.html")

def register_faces(request):
    first_name = request.session['first_name']
    last_name = request.session['last_name'] 
    full_name = first_name + " " + last_name
    student_dir = r'C:\Users\Akash\Desktop\Development\Computer Vision\Image based Attendance System\Students'
    path = student_dir+'\\'+full_name
    camera = cv2.VideoCapture(0)
    i = 0
    while i < 100:
        return_value, image = camera.read()
        if os.path.exists(path):
            cv2.imwrite(path+'//'+first_name+str(i)+'.jpg',image)
            # cv2.imwrite('opencv'+str(i)+'.png', image)
        i += 1
    del(camera)

    train_model(request)

    return render(request, 'index.html')

def train_model(request):
    BASE_DIR = r'C:\Users\Akash\Desktop\Development\Computer Vision\Image based Attendance System'
    img_dir = os.path.join(BASE_DIR, 'Students')

    current_id = 0
    y_labels = []
    x_train = []
    label_ids = {}

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg'):
                path = os.path.join(root, file)
                # label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                # label = os.path.basename(root).replace(" ", "-").lower()
                label = os.path.basename(root)
                # print(label, path)

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                # print(label_ids)

                # Converting the image into grayscale
                pil_image = Image.open(path).convert("L")

                size = (550,550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                # Converting the grayscale into numbers (numpy array)
                image_array = np.array(final_image, "uint8")
                # print(image_array)

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")

    return render(request, 'index.html')

def display_faces(request):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    # attendance_image = request.session['attendance_image']
    attendance_image = Attendance.objects.latest('attendance_image').attendance_image.url
    base_dir = r'C:\Users\Akash\Desktop\Development\Computer Vision\Image based Attendance System'
    new_attendance_image = attendance_image.replace('/',os.sep)

    path = base_dir+new_attendance_image

    print('path of the attendance image:', path)

    labels = {}
    try:
        with open("labels.pickle", "rb") as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

            image = cv2.imread(path)
            # print(image)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)
            for (x,y,w,h) in faces:
            # print(x,y,w,h)
                roi_gray = gray[y:y+h,x:x+w]
                roi_color = image[y:y+h,x:x+w]
                
                id_,conf = recognizer.predict(roi_gray)
                if conf>=45 and conf<=85:
                    # print(id_)
                    # print(labels[id_])
                    student_obj = Student.objects.get(full_name=labels[id_])
                    full_name = student_obj.full_name
                    year = student_obj.year
                    division = student_obj.division 
                    roll_no = student_obj.roll_no
                    subject = request.session['subject']
                
                    student_attendance = Student_Attendance(
                        full_name = full_name,
                        year = year,
                        division = division,
                        subject = subject,
                        date = request.session['date'],
                        roll_no = roll_no
                    )

                    student_attendance.save()
                # img_item = 'my-image.png'
                # cv2.imwrite(img_item, roi_gray)
                    # return
                    return

    except:
        return render(request, "upload_attendance.html")

def display_webcam(request):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = {}
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)
        for (x,y,w,h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            
            id_,conf = recognizer.predict(roi_gray)
            if conf>=45 and conf<=85:
                # print(id_)
                # print(labels[id_])

                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                stroke = 2
                color = (255,255,255) #BGR
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

                try:
                    student_obj = Student.objects.get(full_name=labels[id_])
                    full_name = student_obj.full_name
                    year = student_obj.year
                    division = student_obj.division 
                    roll_no = student_obj.roll_no
                    subject = request.session['subject']
                    date = request.session['date']

                    # try:
                    #     sa = Student_Attendance.objects.filter(full_name=labels[id_], year = year, division = division, subject = subject, roll_no = roll_no, date = date).exists()
                
                    # except Student_Attendance.DoesNotExist:
                    #     student_attendance = Student_Attendance(
                    #         full_name = full_name,
                    #         year = year,
                    #         division = division,
                    #         subject = subject,
                    #         date = date,
                    #         roll_no = roll_no
                    #     )

                    #     student_attendance.save()

                    student_attendance_obj = Student_Attendance.objects.filter(full_name=labels[id_], year = year, division = division, subject = subject, roll_no = roll_no, date = date).exists()
                    if student_attendance_obj == False:
                        student_attendance = Student_Attendance(
                            full_name = full_name,
                            year = year,
                            division = division,
                            subject = subject,
                            date = date,
                            roll_no = roll_no
                        )

                        student_attendance.save()

                except Student.DoesNotExist:
                    student_obj = None

            img_item = 'my-image.png'
            cv2.imwrite(img_item, roi_gray)

            # Drawing rectangle on frame
            color = (255, 0, 0) #BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)


        cv2.imshow('frame1',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_attendance(request):
    # if request.method == 'POST' and request.FILES['attendance_image']: 
    if 'image' in request.POST and request.FILES['attendance_image']:
        year = request.POST.get('year')
        division = request.POST.get('division')
        subject = request.POST.get('subject')
        date = request.POST.get('date')
        attendance_image = request.FILES['attendance_image']


        attendance = Attendance(
            year=year, 
            division=division, 
            subject=subject, 
            date=date,
            attendance_image=attendance_image
            )

        attendance.save()

        request.session['subject'] = subject
        request.session['date'] = date

        # train_model()

        display_faces(request)
    
    elif 'webcam' in request.POST:
        year = request.POST.get('year')
        division = request.POST.get('division')
        subject = request.POST.get('subject')
        date = request.POST.get('date')

        attendance = Attendance(
            year=year, 
            division=division, 
            subject=subject, 
            date=date,
            )

        attendance.save()

        request.session['subject'] = subject
        request.session['date'] = date

        display_webcam(request)

    return render(request, "upload_attendance.html")

def view_attendance(request):
    if request.method == 'POST':
        year = request.POST.get('year')
        division = request.POST.get('division')
        date = request.POST.get('date')
        subject = request.POST.get('subject')

        students_present = Student_Attendance.objects.filter(
            year=year,
            subject=subject,
            division=division,
            date=date
            )

        # print('**********************************************')
        # print(year)
        # print(division)
        # print(date)
        # print(subject)
        # print('**********************************************')

        # if students_present is not None:
        #     print(students_present)
        # else:
        #     print('No record')

        params = {'students_present': students_present}
        return render(request, "display_attendance.html",params)
        
    return render(request, "display_attendance.html")