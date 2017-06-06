import string, os
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Layer
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageDraw, ImageFont

vocab = string.ascii_uppercase + string.digits + " \\[]!@#$%^&*()_+=-:;.~|"

def make_model(features, layer_name="block2_conv1"):
    vgg = VGG16(include_top=False)
    layer = vgg.get_layer(layer_name)
    x = layer.output
    num_chars, char_w, char_h, char_filters = features.shape
    filters = features.transpose((1, 2, 3, 0)).astype(int)
    filters = filters / np.sqrt(np.sum(np.square(filters), axis=(0, 1), keepdims=True))
    x = BatchNormalization()(x)
    #log_op = lambda inp: K.tf.py_func(log, [inp], K.tf.float32)
    #log_op = lambda inp: K.tf.Print(inp, [inp])
    #x = Lambda(log_op)(x)
    specialized_layer = Conv2D(num_chars, (char_w, char_h))
    x = specialized_layer(x)
    biases = np.zeros((num_chars, ))
    specialized_layer.set_weights([filters, biases])
    #x = MaxPool2D()(x)
    model = Model(inputs=vgg.input, outputs=x)
    return model

def save_imgs(filters):
    filters = filters.transpose((3, 0, 1, 2))
    bad_chars = {"\\":"backslash", " ":"space", "*":"asterisk", ":":"colon", ";":"semicolon", ".":"period", "~":"tilda", "|":"pipe"}
    char_index = 0
    for filter_set in tqdm(filters, 'Saving Images'):
        charname = vocab[char_index]
        if charname in bad_chars:
            charname = bad_chars[charname]
        folder = os.path.join("resources", "filters", charname)
        if not os.path.exists(folder):
            os.mkdir(folder)
        filter_index = 0
        for filter_ in np.rollaxis(filter_set, 2):
            img = array_to_img(filter_[..., None])
            img.save(os.path.join(folder, str(filter_index) + ".jpg"))
            filter_index += 1
        char_index += 1

def make_features(font_path, invert=False, layer_name="block2_conv1"):
    fill, background = 'white', 'black'
    if invert:
        background, fill = fill, background
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
    vgg = VGG16(include_top=False, input_shape=(max_h, max_w, 3)) #input shape is reversed
    layer = vgg.get_layer(layer_name)
    model = Model(inputs=vgg.input, outputs=layer.output)
    vocab_imgs = preprocess_input(np.repeat(vocab_imgs, 3, -1))
    return vocab, model.predict(vocab_imgs)

def get_image(fname):
    return img_to_array(Image.open(fname).resize((224, 224)).convert('L').convert('RGB'))

def log(obj='', msg="", clear=False):
    w = 'w' if clear else 'a'
    n = '' if clear else '\n'
    with open("log.txt", w) as openfile:
        openfile.write(msg + n)
        openfile.write(str(obj) + n * 2)
    return obj
