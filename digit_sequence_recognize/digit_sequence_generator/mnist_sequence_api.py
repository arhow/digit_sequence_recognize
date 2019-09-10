from __future__ import print_function
from digit_sequence_recognize.digit_sequence_generator.mnist_sequence import MNIST_SEQUENCE
import numpy as np
from PIL import Image


class MNIST_SEQUENCE_API(object):

    def __init__(self, path, name_img, name_lbl):

        # 'data', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte'
        self.sequence_object = MNIST_SEQUENCE(path, name_img, name_lbl)

    def generate_mnist_sequence(self, digits, spacing_range, image_width, image_height):
        img_data = (self.sequence_object.generate_image_sequence(digits, spacing_range[0], spacing_range[1], image_width, image_height) * 255.0).astype(np.uint8)
        return img_data

    def generate_data(self, num_samples, seq_len, spacing_range=(0,0), total_width=-1, image_height=28):
        inputs = []
        labels = []
        if total_width <= 0:
            total_width = (image_height+int(spacing_range[1]/2)) * seq_len
        for i in range(num_samples):
            seq_values = np.random.randint(0, 10, seq_len)
            seq = self.generate_mnist_sequence(seq_values, spacing_range, total_width, image_height)
            inputs.append(seq)
            labels.append(seq_values)
        print("MNIST sequence image dataset of size " + str(num_samples) +
              " has been generated.")
        return np.array(inputs), np.array(labels)

    def save_image(self, img_data, sequence):
        sequence_image = Image.fromarray(img_data)
        img_name = "-".join(list(map(str, sequence)))
        sequence_image.save(img_name + ".png")
        print("Image for the sequence " + img_name +
              " is generated and saved as " + img_name + ".png.")


