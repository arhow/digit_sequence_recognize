import os

import numpy as np
import tensorflow as tf


from digit_sequence_recognize.train_model.nn import SequenceReshapedConvolutionBatchnorm


class Model(object):

    def __init__(self, path, image_height, image_width, seq_len):

        self.path = path
        self.image_height = image_height
        self.image_width = image_width
        self.seq_len = seq_len
        # self.nn = SequenceReshapedConvolutionBatchnorm(image_height, image_width, seq_len)
        return

    def fit(self, train, validation, test, epochs=400, batch_size=50, verbose=0):

        his = []
        with tf.Graph().as_default():
            # Wiring
            nn = SequenceReshapedConvolutionBatchnorm(self.image_height, self.image_width, self.seq_len)

            inputs_placeholder, labels_placeholder, keep_prob_placeholder, is_training_placeholder = nn.input_placeholders()
            logits = nn.inference(inputs_placeholder, keep_prob_placeholder, is_training_placeholder)
            loss = nn.loss(logits, labels_placeholder)
            training = nn.training(loss, 0.0001)
            evaluation = nn.evaluation(logits, labels_placeholder)

            # Initialization
            session = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            session.run(init)

            # visualize graph
            writer = tf.summary.FileWriter(f"{self.path}/visualizations/" + nn.get_name())
            writer.add_graph(session.graph)

            # Summaries
            merged_summary = tf.summary.merge_all()

            # Saver to save checkpoints
            saver = tf.train.Saver(max_to_keep=4)

            # Training
            steps_per_epoch = len(train['examples']) // batch_size
            number_of_examples = steps_per_epoch * batch_size
            loss_value_epoch = 0
            for step in range(epochs*steps_per_epoch + 1):
                batch = self.get_batch(train, inputs_placeholder, labels_placeholder, keep_prob_placeholder, 0.85, is_training_placeholder, True, batch_size)
                loss_value, summary, _ = session.run([loss, merged_summary, training], feed_dict=batch)
                loss_value_epoch += loss_value
                writer.add_summary(summary, step)

                if step % steps_per_epoch == 0:
                    # Save checkpoint
                    try:
                        os.makedirs(os.path.join(f"{self.path}/checkpoints", nn.get_name()))
                    except:
                        pass
                    saver.save(session, os.path.join(f"{self.path}/checkpoints", nn.get_name(), nn.get_name()), global_step=step)

                    val_loss = self.evaluate_loss(validation, session, loss, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
                             is_training_placeholder, "validation", writer, step, batch_size)
                    tst_precision = self.evaluate_precision(test, session, evaluation, inputs_placeholder, labels_placeholder, keep_prob_placeholder,
                             is_training_placeholder, "test", writer, step, batch_size)

                    his.append({'epoch':step // steps_per_epoch, 'loss':loss_value_epoch/number_of_examples, 'val_loss':val_loss, 'tst_precision':tst_precision})
                    if verbose > 0:
                        print('epoch', step // steps_per_epoch, 'loss', loss_value_epoch/number_of_examples, 'val_loss', val_loss, 'tst_precision', tst_precision)
                    loss_value_epoch = 0
        return his


    def evaluate_precision(self, dataset, session, operation, inputs_placeholder, labels_placeholder, keep_prob_placeholder, is_training_placeholder, name, summary_writer, learning_step, batch_size, summary_tag_prefix='Accuracy'):

        steps_per_epoch = len(dataset['examples']) // batch_size
        number_of_examples = steps_per_epoch * batch_size

        correct_num = 0
        for step in range(steps_per_epoch):
            batch = self.get_batch(dataset, inputs_placeholder, labels_placeholder, keep_prob_placeholder, 1, is_training_placeholder, False, batch_size)
            corrects_in_batch, corrects_vector, predictions = session.run(operation, feed_dict=batch)
            correct_num += corrects_in_batch

        score = correct_num / number_of_examples
        summary = tf.Summary()
        summary.value.add(tag=f'{summary_tag_prefix}_{name}', simple_value=score)
        summary_writer.add_summary(summary, learning_step)
        return score

    def evaluate_loss(self, dataset, session, operation, inputs_placeholder, labels_placeholder, keep_prob_placeholder, is_training_placeholder, name, summary_writer, learning_step, batch_size, summary_tag_prefix='Loss'):

        steps_per_epoch = len(dataset['examples']) // batch_size
        number_of_examples = steps_per_epoch * batch_size

        loss_epoch = 0
        for step in range(steps_per_epoch):
            batch = self.get_batch(dataset, inputs_placeholder, labels_placeholder, keep_prob_placeholder, 1, is_training_placeholder, False, batch_size)
            loss_batch_i = session.run([operation], feed_dict=batch)
            loss_epoch += loss_batch_i[0]

        loss_example = loss_epoch / number_of_examples
        summary = tf.Summary()
        summary.value.add(tag=f'{summary_tag_prefix}_{name}', simple_value=loss_example)
        summary_writer.add_summary(summary, learning_step)
        return loss_example


    def get_batch(self, dataset, inputs_placeholder, labels_placeholder, keep_prob_placeholder, keep_prob_val, is_training_placeholder, is_traininig, batch_size):
        if "position" not in dataset:
            dataset["position"] = 0
        position = dataset["position"]
        steps_per_epoch = len(dataset['examples']) // batch_size
        inputs = dataset['examples'][batch_size * position: (batch_size * position) + batch_size]
        labels = dataset['labels'][batch_size * position: (batch_size * position) + batch_size]
        position += 1
        if position == steps_per_epoch:
            position = 0
        dataset["position"] = position
        return {inputs_placeholder: inputs, labels_placeholder: labels, keep_prob_placeholder: keep_prob_val, is_training_placeholder: is_traininig}