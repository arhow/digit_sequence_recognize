import optparse
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from imgaug import augmenters as iaa
from digit_sequence_recognize.utilities.array_saver import save_array, load_array
from digit_sequence_recognize.train_model.model import Model

def get_command_line_arguments():
    parser = optparse.OptionParser()

    parser.set_defaults(save_path='data')
    parser.set_defaults(train_inputs='train_inputs')
    parser.set_defaults(train_labels='train_labels')
    parser.set_defaults(validation_inputs='validation_inputs')
    parser.set_defaults(validation_labels='validation_labels')
    parser.set_defaults(test_inputs='test_inputs')  #
    parser.set_defaults(test_labels='test_labels')  #
    parser.set_defaults(image_width=140)  #
    parser.set_defaults(image_height=28)  #
    parser.set_defaults(seq_len=5)  #
    parser.set_defaults(epochs=10)  #
    parser.set_defaults(batch_size=50)  #
    parser.set_defaults(fit_verbose=1)  #
    parser.set_defaults(augment_times=5)  #-1 means no augment

    parser.add_option('--save_path', dest='save_path')
    parser.add_option('--train_inputs', dest='train_inputs')
    parser.add_option('--train_labels', dest='train_labels')
    parser.add_option('--validation_inputs', dest='validation_inputs')
    parser.add_option('--validation_labels', dest='validation_labels')
    parser.add_option('--test_inputs', dest='test_inputs')
    parser.add_option('--test_labels', dest='test_labels')
    parser.add_option('--image_width', dest='image_width')
    parser.add_option('--image_height', dest='image_height')
    parser.add_option('--seq_len', dest='seq_len')
    parser.add_option('--epochs', dest='epochs')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--fit_verbose', dest='fit_verbose')
    parser.add_option('--augment_times', dest='augment_times')



    (options, args) = parser.parse_args()

    return options


def run(save_path, train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels, image_width, image_height, seq_len, epochs, batch_size, fit_verbose, augment_times):
    '''

    :param save_path:
    :param train_inputs:
    :param train_labels:
    :param validation_inputs:
    :param validation_labels:
    :param test_inputs:
    :param test_labels:
    :param image_width:
    :param image_height:
    :param seq_len:
    :param epochs:
    :param batch_size:
    :param fit_verbose:
    :param augment_times:
    :return:
    '''


    train_inputs = load_array(f'{save_path}/{train_inputs}')
    train_labels = load_array(f'{save_path}/{train_labels}')
    validation_inputs = load_array(f'{save_path}/{validation_inputs}')
    validation_labels = load_array(f'{save_path}/{validation_labels}')
    test_inputs = load_array(f'{save_path}/{test_inputs}')
    test_labels = load_array(f'{save_path}/{test_labels}')

    seq = iaa.Sequential([
        # iaa.Affine(rotate=(-5, 5)),
        iaa.AdditiveGaussianNoise(scale=(30, 90)),
        iaa.Crop(percent=(0, 0.05))
    ], random_order=True)

    if augment_times > 0:
        auged_train_inputs = []
        auged_train_labels = []
        train_inputs_shape = train_inputs.shape
        train_labels_shape = train_labels.shape
        for i in range(train_inputs.shape[0]):
            image = train_inputs[i]
            auged_train_inputs.append([seq.augment_image(image) for _ in range(augment_times)])
            auged_train_labels.append([train_labels[i] for _ in range(augment_times)])
        auged_train_inputs = np.array(auged_train_inputs)
        auged_train_labels = np.array(auged_train_labels)
        train_inputs = auged_train_inputs.reshape(-1, train_inputs_shape[1], train_inputs_shape[2])
        train_labels = auged_train_labels.reshape(-1, train_labels_shape[1])


    sklb = LabelBinarizer()
    sklb.fit(np.unique(train_labels.reshape(-1,1)))

    train_labels_shape = train_labels.shape
    train = {'examples':train_inputs, 'labels':sklb.transform(train_labels.reshape(-1,1)).reshape(train_labels_shape[0],train_labels_shape[1], sklb.classes_.shape[0])}
    validation_labels_shape = validation_labels.shape
    validation = {'examples': validation_inputs, 'labels': sklb.transform(validation_labels.reshape(-1,1)).reshape(validation_labels_shape[0], validation_labels_shape[1], sklb.classes_.shape[0])}
    test_labels_shape = test_labels.shape
    test = {'examples': test_inputs, 'labels': sklb.transform(test_labels.reshape(-1,1)).reshape(test_labels_shape[0], test_labels_shape[1], sklb.classes_.shape[0])}

    m = Model(save_path, image_height, image_width, seq_len)
    his = m.fit(train, validation, test, epochs=epochs, batch_size=batch_size, verbose=fit_verbose)
    pd.DataFrame(his).to_csv(f'{save_path}/fit_history.csv', index=False)
    return


def main():
    start_time_ = time.time()
    cla = get_command_line_arguments()
    param = {
        'save_path': cla.save_path,
        'train_inputs':cla.train_inputs,
        'train_labels':cla.train_labels,
        'validation_inputs':cla.validation_inputs,
        'validation_labels':cla.validation_labels,
        'test_inputs':cla.test_inputs,
        'test_labels':cla.test_labels,
        'image_width':int(cla.image_width),
        'image_height':int(cla.image_height),
        'seq_len':int(cla.seq_len),
        'epochs':int(cla.epochs),
        'batch_size':int(cla.batch_size),
        'fit_verbose':int(cla.fit_verbose),
        'augment_times':int(cla.augment_times),
    }
    try:
        print(param)
        run(**param)
    except Exception as e:
        print(e.__str__())
    finally:
        print('cost time {}'.format(time.time() - start_time_))
    return


if __name__ == "__main__":
    main()