import cv2
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt

def crop_images(img_name):        
    img = cv2.imread(img_name)    
    y = 150  # cuts out left vertical
    x = 100 # cuts out up horizontal
    h = 4220  # cuts out right vertical
    w = 2700 # cuts out bottom horizontal

    img = img[x:w, y:h]
    return img

def get_cropped(Input_folder, Output_folder):

    images = []
    cropped = []
    total_count = len(os.listdir(Input_folder))    
    # loop through each file in the folder
    for image_path in os.listdir(Input_folder):        
        
        # get image path        
        input_path = os.path.join(Input_folder, image_path)
             
        # put file into function defined above
        crop_image = crop_images(input_path)
    
        filename = image_path
            
        # join with output folder name made in the outer function and the filename
        path = os.path.join(os.getcwd(), Output_folder, filename)         
        cv2.imwrite(path, crop_image)
     

    print(f"Cropped images saved to {Output_folder}")   


Input_folder = os.path.join(os.getcwd(), "images")
Output_folder = 'Cropped'

get_cropped(Input_folder, Output_folder)
