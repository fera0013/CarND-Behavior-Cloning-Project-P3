import numpy as np
import pandas
import os
import matplotlib.image as mpimg
import scipy

def gen(image_names, labels, batch_size, new_shape = None):
    start = 0
    end = start + batch_size
    n = len(image_names)
    while True:     
        X_batch = np.array([mpimg.imread(os.path.join('data',image_path)) 
                   for image_path in image_names[start:end]])
        X_batch = np.array(X_batch)
        y_batch = np.array(labels[start:end])
        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size
        yield (X_batch, y_batch)


def get_steering_angle_data(OFFSET):
    print("Offset: ",OFFSET)
    df = pandas.read_csv(os.path.join('data','driving_log.csv'),
                         sep = None,
                         skipinitialspace=True)
    right_images = df.loc[:,['right','steering']].rename(columns = {'right':'image'})
    right_images['steering'] -= OFFSET
    left_images = df.loc[:,['left','steering']].rename(columns = {'left':'image'})
    left_images['steering'] += OFFSET
    center_images = df.loc[:,['center','steering']].rename(columns = {'center':'image'})
    data = pandas.concat([right_images,left_images,center_images])
    return data