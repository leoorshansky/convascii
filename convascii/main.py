import keras, os, argparse
from PIL import Image
import numpy as np
from model import get_image, make_features, make_model, log

parser = argparse.ArgumentParser(description="Generate ASCII art using deep convolutional neural networks.")
parser.add_argument('fname', metavar='img_file', help='File name of the image to be ASCII\'d')
parser.add_argument('--layer-name', nargs='?',default='block3_conv3', help='The name of the VGG16 layer to process input through')
parser.add_argument('--pooling', action='store_true', help='Whether or not to pool after the last layer')
args = parser.parse_args()

img_file = args.fname
font_file = os.path.join("resources", "font.ttf")
layer_name = args.layer_name
pooling = args.pooling

def main():
    log(clear=True)
    print("Making ascii art out of {} with font {} \n".format(img_file, font_file))
    print("Making Font Features...")
    vocab, features = make_features(font_file,invert=False, layer_name=layer_name)
    print("Making ConvNet Model...")
    model = make_model(features, layer_name, pooling)
    print("Converting Image...")
    img = get_image(img_file)[None, ...]
    output = model.predict(img)
    correlated_pixels = np.argmax(output, axis=-1).squeeze()
    for row in range(0, correlated_pixels.shape[0], 2):
            line = ''
            for col in range(0, correlated_pixels.shape[1], 1):
                line += vocab[correlated_pixels[row, col]]
            print(line)
if __name__ == "__main__": main()
