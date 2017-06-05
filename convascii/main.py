import keras, os
from PIL import Image
import numpy as np
from model import get_image, make_features, make_model

img_file = os.path.join("resources", "img.jpg")
font_file = os.path.join("resources", "font.ttf")

print("Making ascii art out of {} with font {} \n".format(img_file, font_file))
print("Making Font Features...")
vocab, features = make_features(font_file)
print("Making ConvNet Model...")
model = make_model(features)
print("Converting Image...")
img = get_image(img_file)[None, ...]
print(img.shape)
correlated_pixels = np.argmax(model.predict(img), axis=-1).squeeze()
print(correlated_pixels)
open('output.txt', 'w').write(str(correlated_pixels))
'''
for row in correlated_pixels:
    line = '\n'
    for index in row:
        line += vocab[index]
    print(line)
'''
