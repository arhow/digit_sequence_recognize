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
