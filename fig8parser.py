import json
import os
import glob
import pandas as pd
from PIL import Image
import ast
'''
Parse csv file from fig8 and make terraclear style .json for images.
'''
#make output directories
#os.makedirs(os.path.join(os.getcwd(), 'processed_jsons'), exist_ok=True)

path_to_imgs = os.path.join(os.getcwd(),'data/imgs')
path_to_csv = os.path.join(os.getcwd(),'data/csv')

# List of all images in the imgs folder with certain extensions
imgs = [os.path.join(path_to_imgs,i) for i in os.listdir(path_to_imgs) if i.endswith(".jpg") or i.endswith(".png") or i.endswith(".tiff")]

print("Found", len(imgs), "images.")

# Get all the csv's in the directory
csvs = [os.path.join(path_to_csv,f) for f in os.listdir(path_to_csv) if f.endswith(".csv")]

combined_csv = pd.concat([pd.read_csv(csv) for csv in csvs])

# From the csv, need to get:
    # boxes
        # confidence
        # height
        # label
        # left
        # top 
        # width
    # image-height
    # image-width
    # image_name (path/to/image)

# Loop over the images in the directory

labelsDict = {
    "Rock": "0",
    "Motor": "1" 
}

num_of_txts = 0
for img in imgs:
    for img_path, annotation in zip(combined_csv['original_url'].tolist(),combined_csv['annotation'].tolist()):

        # Create absolute path
        img_path = os.path.join(os.getcwd(),"data/imgs",img_path)
        if (img == img_path):
            txt_path = img_path.split(".")[0] + ".txt"

            # open image for dimension
            img_decoded = Image.open(img)
            img_width, img_height = img_decoded.size
            
            # Convert the string to a dictionary
            annotation_dict = ast.literal_eval(annotation)

            if (len(annotation_dict) == 0):
                # Skip images with no labels
                continue
            else:

                img_contents = []

                for i in range(0, len(annotation_dict)):

                    # Retrieve contents of the dict 
                    name = annotation_dict[i]['class']
                    name_id,  = name.items()
                    class_id = labelsDict[name_id[0]]

                    coordinates = annotation_dict[i]['coordinates']
                    # xmins, xmaxs, ymins, ymaxs
                    
                    x = coordinates['x'] / img_width
                    y = coordinates['y'] / img_height
                    w = coordinates['w'] / img_width
                    h = coordinates['h'] / img_height
                    
                    ##normalize 
                    xmin = x 
                    ymin = y 
                    xmax = x + w
                    ymax = y + h 

                    #x_c = x + (w/2)
                    #y_c = y + (h/2)
                    #w = w
                    #h = h 

                    #img_contents.append((class_id, xmin, xmax,ymin,ymax))
                    #img_contents.append((class_id, x_c, y_c, w, h))
                    img_contents.append((class_id, xmin, xmax, ymin,ymax))

                # write out the results
                with open(txt_path, 'w') as txt_file:
                    for line in img_contents:
                        txt_file.write('{} {} {} {} {}\n'.format(*line))

            num_of_txts += 1

print("Generated",num_of_txts, "txt files.")
                        


                    


            