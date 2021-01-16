import sys
import argparse
from yolo import YOLO, detect_video
from timeit import default_timer as timer
from PIL import Image
from PIL import ImageDraw 
import re
from multiprocessing import Process, Queue
import matplotlib as plt
import glob
import cv2 as cv
import csv
import tensorflow as tf
import numpy as np
from time import sleep
from io import BytesIO 
import xlsxwriter 

output = []
query1_time = []
query2_time =[]
query3_time = []
q1_TP = 0
q2_TP = 0
q3_TP = 0

#method to read input frames and detect objects in stage 1
def detect_img(q, ):
    #create a YOLO object to detect inputted image file
    Y = YOLO()
    
    #path to image frames
    images = glob.glob("Frames/*.jpg")
    frame=1
    #set inital frame to 1
    
    #start timer for throughput of stage 1
    TP1_start = timer()
    
    
    #iterate through each frame in order of image number, key function used to sort
    for image in sorted(images, key=keyfunc):
        frame_time = 0
        start_frame = timer()
        info =[]
        
        #open image as pillow object
        img = Image.open(image)
        
        #detect img and return, detected frame, rectangle dimensions for objects and count of cars
        r_image,out_rectangle,time, car_count = Y.detect_image(img)
        
        #append time Query1 
        global query1_time
        query1_time.append(time)
        
        info.append(frame)
        info.append(car_count)
        
        #send image and info if no cars present
        if car_count ==0:
            info.append(time)
            q.put(r_image)
            q.put(info)
            
            
                
        else:
            #if cars present add  time and rectangle dimensions of shape of object
            info.append(out_rectangle)
            info.append(time)
            
            q.put(r_image)
            q.put(info)
        #increment frame
        
        if frame ==1495:
            print(type(query1_time))
            print(len(query1_time))
            with open('query1_time.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(query1_time)
            
            
            
        frame+=1
        
    
    
    
    #when frames completed stage 1 output throughput 
    q1_TP_end = timer()
    global q1_TP
    q1_TP = q1_TP_end - TP1_start
    
    print("Throughput Query 1 = 1495/{} = ".format(q1_TP), 1495/q1_TP)
    




def classify_img(q,q2):
    #load the transfer learning model to classify car type
    model = tf.keras.models.load_model('Final_Model.h5')

    #start throughput timer
    TP2_start = timer()
    
    while True:
        info = []
        
        #recieve items from the queue
        im = q.get()
        global query2_time
        #assign image if image recieved
        if (type(im)!= list):
            image = im
            
        
        else:
            frame_time =0
            #initalise frame time
            
            frame_time_start = timer()
            
            #if no objects add frame and info to queue2
            if len(im) == 3:
                frame,car_count, time = im
                info.append(frame)
                info.append(car_count)
                
                
                frame_time_end = timer()
                frame_time = frame_time_end - frame_time_start +time
                query2_time.append(frame_time)
                info.append(frame_time)
                
                print("No Car:", info)
                q2.put(image)
                q2.put(info)  
                
                

                #if all frames finish print throughput for stage 2
                if frame == 1495:
                    print(type(query2_time))
                    print(len(query2_time))
                    with open('query2_time.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(query2_time)
            
            
                    break
            
            else:
                #check how many cars in frame and assign rectangle coordinates, cropping the image to size
                frame,car_count,out_rectangle,time  = im
                if car_count == 1:
                    left,top,right,bottom = out_rectangle
                else:
                    left,top,right,bottom,left1,top1,right1,bottom1=out_rectangle
                    cropped2 = image.crop((left1,top1,right1,bottom1))
                
                    #convert image to np array to input into mobilenet and resize
                    image2 = load_image(cropped2)
                    
                    #call the method to predict image based on model, return max prediction
                    predictions2 = model.predict(image2).tolist()
                    best2 = predictions[0].index(max(predictions[0]))
                    if best2 == 0:
                        car_type2= "Hatchback"
                    else:
                        car_type2 = "Sedan"
                
                
                #crop image, convert to nparray and resize
                #predict using model and return prediction
            
                cropped = image.crop((left,top,right,bottom))

            
                image1 = load_image(cropped)
                predictions = model.predict(image1).tolist()
                best = predictions[0].index(max(predictions[0]))
            
                if best == 0:
                    car_type= "Hatchback"
                else:
                    car_type = "Sedan"
                
                #add frame time to list for query 2
                
                frame_time_end = timer()
                frame_time = frame_time_end - frame_time_start +time
                query2_time.append(frame_time)
                
                
                #add info to list for queue2 
                info.append(frame)
                info.append(car_count)
                info.append(out_rectangle)
            
                info.append(car_type)
            
                if car_count ==2:
                    info.append(car_type2)
                info.append(frame_time)
                q2.put(image)
                q2.put(info)  
                
                
                
               
    q2_TP_end = timer()
    global q2_TP
    q2_TP = q2_TP_end - TP2_start
    
    global q1_TP
    
    q2_TP = q2_TP + q1_TP
    print("Throughput Query 2 = 1495/{} = ".format(q2_TP), 1495/q2_TP)           
            
        

    

def colour_car(q2,):
    #start timer throughput 3
    TP3_start = timer()

    while True:
        info = []
        #get item from queue 2
        im = q2.get()

        frame_time3 =0
        
        global query3_time
        global output
        start_frame_time3 = timer()
        
        #check type of item, if not list assign image
        if (type(im)!= list):
            image = im
        
        else:
            #if no cars present append info to output
            if len(im)==3:
                frame,car_count, time = im
                info.append(frame)
                info.append(car_count)
                print(info)
                output.append(info)
                
                end_frame_time3 = timer()
                frame_time3 =end_frame_time3 - start_frame_time3 +time
            
                query3_time.append(frame_time3)
                
                #draw labels on image stating no car and frame number
                draw = ImageDraw.Draw(image)
                draw.text((3, 0),"Car Count = {}".format(car_count),(255,255,255))
                draw.text((3,270),"Frame {}".format(frame),(255,255,255))
                image.save('out_frames/image'+str(frame)+'.jpg')
                
                #if end of frames get throughput and sort output files
                if frame ==1495:
                    q3_TP_end = timer()
                    global q3_TP
                    q3_TP = q3_TP_end - TP3_start
                    global q2_TP
                    q3_TP = q3_TP +q2_TP
                    print("Throughput Query 3 = 1495/{} = ".format(q3_TP), 1495/q3_TP)
                    
                    print(type(query3_time))
                    print(len(query3_time))
                    with open('query3_time.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(query3_time)
                   
                    output = sorted(output, key=lambda x: x[0])
                    
                    #create excel output of results
                    create_out(output)
                    break
                    
            #if one car in frame       
            if len(im) ==5:
                frame,car_count,out_rectangle,car_type,time  = im
                
                #convert image object after cropping to opencv numpy array
                left,top,right,bottom = out_rectangle
                cropped = image.crop((left,top,right,bottom))

                colour_image = cv.cvtColor(np.array(cropped), cv.COLOR_RGB2BGR)
                
                
                #predict colour of first - car_colour1
                car_colour1 = ColourClassifier(colour_image)
            
                #append all info to list
                info.append(frame)
                info.append(car_count)
                info.append(car_type)
                info.append(car_colour1)
                
                #output info to list
                output.append(info)
                
                #return query time for frame
                end_frame_time3 = timer()
                frame_time3 =end_frame_time3 - start_frame_time3 +time
            
                query3_time.append(frame_time3)
                
                
                #draw label in image and save image based on car colour, type and count
                
                draw = ImageDraw.Draw(image)
                draw.text((3, 0),"Car Count = {}".format(car_count),(255,255,255))
                draw.text((3,270),"Frame {}".format(frame),(255,255,255))
                draw.text((70, 255),"Car Type 1 = {}".format(car_type),(255,255,255))
                draw.text((70, 270),"Car Colour 1 = {}".format(car_colour1),(255,255,255))
                image.save('out_frames/image'+str(frame)+'.jpg')
                
            elif len(im)>5:
                #crop image and convert from PIL Image object to opencv numpy array
                frame,car_count,out_rectangle,car_type, car_type2,time = im
                left,top,right,bottom,left1,top1,right1,bottom1=out_rectangle
                cropped2 = image.crop((left1,top1,right1,bottom1))
                image2 = cv.cvtColor(np.array(cropped2), cv.COLOR_RGB2BGR)
                cropped = image.crop((left,top,right,bottom))
                image1 = cv.cvtColor(np.array(cropped), cv.COLOR_RGB2BGR)
            
            
            
            
                car_colour1 = ColourClassifier(image1)
                car_colour2 = ColourClassifier(image2)
                #predict 1st colour car_colour1
                #predict colour 2nd crop - car_colour2
                
                
                #apend info to list
                info.append(frame)
                info.append(car_count)
                info.append(car_type)
                info.append(car_colour1)
                info.append(car_type2)
                info.append(car_colour2)
                
                #save output info in list
                output.append(info)
            
                #append time to query for frame
                end_frame_time3 = timer()
                frame_time3 =end_frame_time3 - start_frame_time3 +time
            
                query3_time.append(frame_time3)
                
                #label frame and save image, includes cars type, colour, count
                draw = ImageDraw.Draw(image)
                draw.text((3, 0),"Car Count = {}".format(car_count),(255,255,255))
                draw.text((3,270),"Frame {}".format(frame),(255,255,255))
                draw.text((70, 255),"Car Type 1 = {}".format(car_type),(255,255,255))
                draw.text((70, 270),"Car Colour 1 = {}".format(car_colour1),(255,255,255))
                draw.text((200, 255),"Car Type 2 = {}".format(car_type2),(255,255,255))
                draw.text((200, 270),"Car Colour 2 = {}".format(car_colour2),(255,255,255))
                image.save('out_frames/image'+str(frame)+'.jpg')
            
        
   
    print("Q3 Exit")
      
      
      

#sort each image in folder based in ascending order of number in title
def keyfunc(name):
    nondigits = re.compile("\D")
    return int(nondigits.sub("",name))
    
#preprocess Pillow Image object to np array and resize to input into mobilenet model
def load_image(img_file, target_size = (224,224)):
    img_file = img_file.resize(target_size)
    X = np.zeros((1,*target_size, 3))
    X[0, ] = np.asarray(img_file)
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X
         
#takes inputted image object and outputs colour classification
def ColourClassifier(input_image): 
    def number_of_mask_255_pixels(example_mask):
        count_255 = 0
        count_NOT255 = 0
        for element in example_mask:
            for i in range(len(element)):
                if element[i] == 255:
                    count_255 += 1
            else:
                count_NOT255 += 1
        return count_255

    #Image resizing
    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(input_image.shape[1] * scale_percent / 100)
    height = int(input_image.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)

    # resize image
    resized_image = cv.resize(input_image, dsize)

    hsv = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)
    #set hsv ranges for each colour 
    colour_ranges = [
        ([0,0,0], [179,255,50]), #Black
        ([0,0,209], [179, 18, 255]), #White
        ([0,0,48], [179, 12, 165]), #Silver
        ([164, 120, 51], [179, 255, 114]), #Red
        ([80, 50, 40], [140, 255, 160]) #Blue
    ]

    list_of_colours = ["black", "white", "silver", "red", "blue"]
    list_of_255_pixels = []
    # Check all colour ranges
    for (i, j) in colour_ranges:
        lower_range = np.array(i)
        upper_range = np.array(j)
        mask = cv.inRange(hsv, lower_range, upper_range)
        num_255pixels = number_of_mask_255_pixels(mask)
        list_of_255_pixels.append(num_255pixels)
        
    #find the colour with most pixels in their ranges and output that as classification
    colour_index = list_of_255_pixels.index(max(list_of_255_pixels))
    detected_colour = list_of_colours[colour_index]
    return detected_colour      


#create excel file output the same as the ground truth file used for f1-score comparisons
def create_out(output):
    
    workbook = xlsxwriter.Workbook('predictions.xlsx')
    worksheet = workbook.add_worksheet() 
    
    #add the column names
    worksheet.write('A1', 'Frame No.') 
    worksheet.write('B1', 'Sedan Black') 
    worksheet.write('C1', 'Sedan Silver') 
    worksheet.write('D1', 'Sedan Red') 
    worksheet.write('E1', 'Sedan White') 
    worksheet.write('F1', 'Sedan Blue') 
    worksheet.write('G1', 'Hatchback Black') 
    worksheet.write('H1', 'Hatchback Silver')
    worksheet.write('I1', 'Hatchback Red') 
    worksheet.write('J1', 'Hatchback White') 
    worksheet.write('K1', 'Hatchback Blue') 
    worksheet.write('L1', 'Total')     
    row = 2
    col = 0
    #iterate through each list in output and write to a row in file
    #columns: Frame number, sedan colours(black-blue), hatchback colours (black -blue), car count 
    for frame in output:
        
        out = []
        number = frame[0]
        count = frame[1]
        sedan = {"black":0,"silver":0,"red":0,"white":0,"blue":0}
        hatchback = {"black":0,"silver":0,"red":0,"white":0,"blue":0}
        
        if len(frame)>2:
            car_type1 = frame[2]
            car_colour1 = frame[3]
            
            if car_type1 == "sedan":
                sedan[car_colour1] += 1
            else:
                hatchback[car_colour1] +=1
        
            if len(frame)==6:
                car_type2 = frame[4]
                car_colour2=frame[5]
                if car_type2 == "sedan":
                    sedan[car_colour2] += 1
                else:
                    hatchback[car_colour2] +=1
        out.append(number)
        for key, value in sedan.items():
            out.append(value)
        for key2, value2 in hatchback.items():
            out.append(value2)
        out.append(count)

        for i in out:
            worksheet.write(row, col, i)
            col+=1
        row+=1
        col=0
        
       
        
    workbook.close() 
        




if __name__ == '__main__':
    #create two queues between each stages 1-2 and 2-3 respectively
    q = Queue(maxsize = 1495)
    q2 = Queue(maxsize = 1495)
    #producer starts method for yolo model
    producer = Process(target=detect_img, args=(q,))
    producer.start()
    print("Producer starting...")
    
    #start the two consumers for classifying car type and colour 
    
    consumer = Process(target=classify_img, args=(q, q2))
    consumer.start()
    print("Consumer 1 starting...")
    
    
    consumer2 = Process(target=colour_car, args=(q2,))
    consumer2.start()
    print("consumer 2 starting...")
    

        

