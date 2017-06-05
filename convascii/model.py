import string
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Layer
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

def make_model(features, layer_name="block2_conv1"):
    vgg = VGG16(include_top=False)
    layer = vgg.get_layer(layer_name)
    x = layer.output
    num_chars, char_w, char_h, char_filters = features.shape
    filters = features.transpose((1, 2, 3, 0)).astype(np.float32)
    print(filters)
    filters = filters[::1, ::1, :, :] / np.sqrt(np.sum(np.square(filters), axis=(0, 1), keepdims=True))
    print(filters)
    x = BatchNormalization()(x)
    specialized_layer = Conv2D(num_chars, (char_w, char_h))
    x = specialized_layer(x)
    biases = np.zeros((num_chars, ))
    specialized_layer.set_weights([filters, biases])
    x = MaxPool2D()(x)
    model = Model(inputs=vgg.input, outputs=x)
    return model

def make_features(font_path, invert=False, layer_name="block2_conv1"):
    fill, background = 'white', 'black'
    if invert:
        background, fill = fill, background
    vocab = string.ascii_uppercase + string.digits + " \\[]!@#$%^&*()_+=-:;.~|"
    font = ImageFont.truetype(font_path, size=80)
    char_widths = [font.getsize(char)[0] for char in vocab]
    char_heights = [font.getsize(char)[1] for char in vocab]
    max_w = max(char_widths)
    max_h = max(char_heights)
    vocab_imgs = []
    img = Image.new('L', (max_w, max_h), background)
    draw = ImageDraw.Draw(img)
    for char in tqdm(vocab, 'Generating Charset'):
        draw.text((0,0), char, fill=fill, font=font)
        vocab_imgs.append(img_to_array(img.copy()))
        draw.rectangle((0, 0, max_w, max_h), fill=background)
    vocab_imgs = np.stack(vocab_imgs)
    vgg = VGG16(include_top=False, input_shape=(max_h, max_w, 3))
    layer = vgg.get_layer(layer_name)
    model = Model(inputs=vgg.input, outputs=layer.output)
    vocab_imgs = preprocess_input(np.repeat(vocab_imgs, 3, -1))
    return vocab, model.predict(vocab_imgs)

def get_image(fname):

    return img_to_array(Image.open(fname).resize((224, 224)).convert('L').convert('RGB'))
