#!/usr/bin/env python
# coding: utf-8

# In[373]:


os.getcwd()


# In[448]:


# Extract color and other norphological traits from seed images
def Process_wheat_seeds_images(Input_folder, Input_format, working_dir):
    

    import os
    
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt    
    import pandas as pd
    
    import random as rng
    rng.seed(12345)
    
    import time
    # Start the timer
    start_batch_time = time.time()
       
    os.chdir(working_dir)
    # Make output directory is it does not already exist
    if not os.path.exists("WheatSeeds_Out"):
        os.mkdir("WheatSeeds_Out")
        
        
    # inner function 
    def Process_images(img_name):
        
        img = cv2.imread(img_name)
        
        # crop image -- change based on your needs
#         y = 0
#         x = 600
#         h = 3000
#         w = 3100
        
#         img = img[y:y+h, x:x+w]
        
        y = 280  # cuts out left vertical
        x = 100 # cuts out up horizontal
        h = 4000  # cuts out right vertical
        w = 2700 # cuts out bottom horizontal

        #img2 = img[y:y+h, x:x+w]

        img = img[x:w, y:h]
        
        # get image name
        image_name = img_name.split('\\')[-1] 
        image_name = image_name[:-4]

        # Make gray image
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(bw, cv2.DIST_L1, 5)

        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
        #_, dist = cv2.threshold(dist, 0.1*dist.max(), 255,0)

        # Erode and dilate the dist image
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #kernel1 = np.ones((5,5), dtype=np.uint8)
        dist = cv2.erode(dist, kernel1)
        dist = cv2.dilate(dist, kernel2)
        dist_8u = dist.astype('uint8')

        # Find total markers
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create the marker image for the watershed algorithm
        markers = np.zeros(dist.shape, dtype=np.int32)

        # Draw the foreground markers
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)

        # Draw the background marker
        cv2.circle(markers, (5,5), 3, (255,255,255), -1)

        # Perform the watershed algorithm
        cv2.watershed(img, markers)

        #mark = np.zeros(markers.shape, dtype=np.uint8)
        mark = markers.astype('uint8')
        mark = cv2.bitwise_not(mark)
        # uncomment this if you want to see how the mark
        # image looks like at that point
        #cv2.imshow('Markers_v2', mark)

        # Generate random colors
        colors = []
        for contour in contours:
        #     if cv2.contourArea(contour) < 100:
        #         continue

            colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))

        # Create the result image
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

        # Fill labeled objects with random colors
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i,j]
                if index > 0 and index <= len(contours):
                    dst[i,j,:] = colors[index-1]

#         # save the image_name final image
#         working_dir = os.getcwd()
#         filename = image_name + "processed" + '\*' + Input_format
        
#         # join with output folder name made in the outer function and the filename
#         path = os.path.join(working_dir, "WheatSeeds_Out", filename)         
#         cv2.imwrite(path, dst)

        # Visualize final image
        # imS = cv2.resize(dst, (1000, 700)) 
        # cv2.imshow('Final Result', imS)
        # cv2.waitKey(0) 

        ### Morphological measurements

        Names_of_images = []
        Red_mean = []
        Green_mean = []
        Blue_mean = []
        Seed_number = []
        Seed_area = []
        Seed_length = []
        Seed_width = []
        Seed_diameter = []
        Bounding_area = [] # for filtering later on


        # Loop through each contour in the image
        for contour in contours:

            # create a mask representing the current contour
            maskImage = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
            cv2.drawContours(maskImage, [contour],-1,1,-1)

            # get mean RGB value of img inside the mask
            meanColor = np.array(cv2.mean(img, mask=maskImage)).astype(np.uint8)

            # store each color in a separate variable
            R = meanColor[0]
            G = meanColor[1]
            B = meanColor[2]
            Red_mean.append(R)
            Green_mean.append(G)
            Blue_mean.append(B)

            # get area
            area = cv2.contourArea(contour)
            Seed_area.append(area)

            # get seed number 
            Seed_number.append(len(Seed_area))

            # get length and width
            (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
            Seed_length.append(MA)
            Seed_width.append(ma)

            # get seed diameter
            equi_diameter = np.sqrt(4*area/np.pi)
            Seed_diameter.append(equi_diameter)
            
            # get seed bounding box area
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            rect_area = w*h
            Bounding_area.append(rect_area)

            # get image name
            Names_of_images.append(image_name)

            # Put the above morphologies in a dataframe
            seeds_per_image = pd.DataFrame(list(zip(Names_of_images, Seed_number, Seed_area, Bounding_area,
                                                    Seed_length, Seed_width, Seed_diameter, Red_mean, 
                                                    Green_mean, Blue_mean)), columns = ['Image_Name', 'Seed',
                                                                                        'Area', 'Bounding_Area',
                                                                                        'Length', 'Width', 
                                                                                        'Diameter', 'Mean_Red', 
                                                                                        'Mean_Green', 'Mean_Blue'])
        # Return dataframe for current image and labelled objects in image
        return seeds_per_image, dst
    
    # create empty dataframe to house final dataset
    processed_file = pd.DataFrame()
    
    images = []
    total_count = len(os.listdir(Input_folder))
    
    # loop through each file in the folder
    for image_path in os.listdir(Input_folder):
        
        # set current time
        start_current_time = time.time()
        
        # to count current image
        images.append(image_path)
        current_count = len(images)
        
        # get image path        
        input_path = os.path.join(Input_folder, image_path)
        
        # some verbose stuff
        print("Processing image", current_count, "of", total_count, "...")
        
        # put file into function defined above
        processed, marked_image = Process_images(input_path)

        # Append each processed data into empty dataframe
        processed_file = processed_file.append(processed)
        
        # save the image_name final image
        #working_dir = os.getcwd()
        filename = image_path
        
        # join with output folder name made in the outer function and the filename
        path = os.path.join(working_dir, "WheatSeeds_Out", filename)         
        cv2.imwrite(path, marked_image)
        
        # more veerbose stuff
        print("Image", image_path.split('\\')[-1], "processed in", time.time() - start_current_time, "secs.")


    #return processed    
    # export procesed data
    processed_file.to_csv(r'WheatSeeds_Out\wheat_seeds_results.csv', header = True, index = False)
    
    # Output proces time for entire batch

    print("Batch completed in", time.time() - start_batch_time, "secs.")


# In[449]:


working_dir = 'C:\\Users\\eadey\\OneDrive\\Desktop\\UMN\\Thesis PhD\\Image Analysis\\Session 1 F5 Scab pics\\seed_ML\\checks'
Input_folder = os.path.join(working_dir, "test")
Input_format = '.tif'


# In[450]:


Process_wheat_seeds_images(Input_folder, Input_format, working_dir)


# In[ ]:





# In[422]:


img = cv2.imread('C:\\Users\\eadey\\OneDrive\\Desktop\\UMN\\Thesis PhD\\Image Analysis\\Session 1 F5 Scab pics\\seed_ML\\checks\\test\\0_1.tif')


# In[442]:


y = 280  # cuts out left vertical
x =100 # cuts out up horizontal
h = 4000  # cuts out right vertical
w = 2700 # cuts out bottom horizontal

#img2 = img[y:y+h, x:x+w]

img2 = img[x:w, y:h]


# In[443]:


imS = cv2.resize(img2, (1000, 700)) 
cv2.imshow('Final Result', imS)
cv2.waitKey(0) 


# In[381]:


imgplot = plt.imshow(img)
plt.show()


# In[ ]:




