## About

`digit_sequence_recognize` is digit sequence recognization task sample


## Installation

```
git clone https://github.com/arhow/digit_sequence_recognize.git
cd digit_sequence_recognize
python setup.py install
```

## Use Example

```
-- create train dataset 
python3 generate_digit_sequence_dataset.py --graph_input_array_file_name train_inputs --graph_labels_array_file_name train_labels --samples 2000

-- create validation dataset
python3 generate_digit_sequence_dataset.py --graph_input_array_file_name validation_inputs --graph_labels_array_file_name validation_labels --samples 400

--create test dataset
python3 generate_digit_sequence_dataset.py --graph_input_array_file_name test_inputs --graph_labels_array_file_name test_labels --samples 2000

--train model
python3 train_model.py --epochs 500

--check train history
vi fit_history.csv
```

## Parameters

```
--generate_digit_sequence_dataset.py

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

--train_model.py

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
    parser.set_defaults(augment_times=5)  #
```
