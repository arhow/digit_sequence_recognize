import optparse
import time
from digit_sequence_recognize.digit_sequence_generator.mnist_sequence_api import MNIST_SEQUENCE_API
from digit_sequence_recognize.utilities.array_saver import load_array, save_array

def get_command_line_arguments():
    parser = optparse.OptionParser()

    parser.set_defaults(save_path='data')
    parser.set_defaults(name_img='t10k-images.idx3-ubyte')
    parser.set_defaults(name_lbl='t10k-labels.idx1-ubyte')
    parser.set_defaults(samples=100)
    parser.set_defaults(seq_len=5) # if seq_len=-1 seq_len is not fixed
    parser.set_defaults(min_seq_len=2)  #
    parser.set_defaults(max_seq_len=10)  #
    parser.set_defaults(image_width=-1)  # if total_width=-1 , total_width will be setted with image_height*seq_len
    parser.set_defaults(image_height=28)  #
    parser.set_defaults(min_spacing=0)  #
    parser.set_defaults(max_spacing=0)  #
    parser.set_defaults(graph_input_array_file_name='inputs.bc')  #
    parser.set_defaults(graph_labels_array_file_name='labels.bc')  #
    parser.set_defaults(is_generate_all_digit_sequence_image=0)  #

    parser.add_option('--save_path', dest='save_path')
    parser.add_option('--name_img', dest='name_img')
    parser.add_option('--name_lbl', dest='name_lbl')
    parser.add_option('--samples', dest='samples')
    parser.add_option('--seq_len', dest='seq_len')
    parser.add_option('--min_seq_len', dest='min_seq_len')
    parser.add_option('--max_seq_len', dest='max_seq_len')
    parser.add_option('--image_width', dest='image_width')
    parser.add_option('--image_height', dest='image_height')
    parser.add_option('--min_spacing', dest='min_spacing')
    parser.add_option('--max_spacing', dest='max_spacing')
    parser.add_option('--graph_input_array_file_name', dest='graph_input_array_file_name')
    parser.add_option('--graph_labels_array_file_name', dest='graph_labels_array_file_name')
    parser.add_option('--is_generate_all_digit_sequence_image', dest='is_generate_all_digit_sequence_image')


    (options, args) = parser.parse_args()

    return options


def run(save_path, name_img, name_lbl, samples, seq_len, min_spacing, max_spacing,
        image_width, image_height, graph_input_array_file_name, graph_labels_array_file_name,
        is_generate_all_digit_sequence_image):
    '''

    :param save_path:
    :param name_img:
    :param name_lbl:
    :param samples:
    :param seq_len:
    :param min_spacing:
    :param max_spacing:
    :param image_width:
    :param image_height:
    :param graph_input_array_file_name:
    :param graph_labels_array_file_name:
    :param is_generate_all_digit_sequence_image:
    :return:
    '''

    api_object = MNIST_SEQUENCE_API(save_path, name_img, name_lbl)
    inputs, labels = api_object.generate_data(samples, seq_len, spacing_range=(min_spacing, max_spacing),
                                              total_width=image_width, image_height=image_height)
    save_array(inputs, f'{save_path}/{graph_input_array_file_name}')
    save_array(labels, f'{save_path}/{graph_labels_array_file_name}')
    if is_generate_all_digit_sequence_image:
        api_object.save_image(inputs.reshape(-1,inputs.shape[-1]), range(10))
    return


def main():
    start_time_ = time.time()
    cla = get_command_line_arguments()
    param = {
        'save_path': cla.save_path,
        'name_img':cla.name_img,
        'name_lbl':cla.name_lbl,
        'samples':int(cla.samples),
        'seq_len':int(cla.seq_len),
        'min_spacing':int(cla.min_spacing),
        'max_spacing':int(cla.max_spacing),
        'image_width':int(cla.image_width),
        'image_height':int(cla.image_height),
        'graph_input_array_file_name':cla.graph_input_array_file_name,
        'graph_labels_array_file_name':cla.graph_labels_array_file_name,
        'is_generate_all_digit_sequence_image':int(cla.is_generate_all_digit_sequence_image),
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