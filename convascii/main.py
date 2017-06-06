import keras, os
from PIL import Image
import numpy as np
from model import get_image, make_features, make_model, log

img_file = os.path.join("resources", "img.jpg")
font_file = os.path.join("resources", "font.ttf")

def main():
    log(clear=True)
    print("Making ascii art out of {} with font {} \n".format(img_file, font_file))
    print("Making Font Features...")
    vocab, features = make_features(font_file)
    print("Making ConvNet Model...")
    model = make_model(features)
    print("Converting Image...")
    img = get_image(img_file)[None, ...]
    #log(img.tolist(), "Image: ")
    output = model.predict(img)
    correlated_pixels = np.argmax(output, axis=-1).squeeze()
    for row in range(0, correlated_pixels.shape[0], 2):
            line = ''
            for col in range(0, correlated_pixels.shape[1], 1):
                line += vocab[correlated_pixels[row, col]]
            print(line)
    #log(output.tolist(), "Model Output: ")
if __name__ == "__main__": main()
