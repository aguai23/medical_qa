import tensorflow as tf
import numpy as np
from data_processor import DataProcessor
from knowledge_tree import KnowledgeTree

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "graph_path", None,
  "The graph path file, containing all the triples constructing knowledge graph"
)

flags.DEFINE_string(
  "data_path", None,
  "the labeled questions path"
)

flags.DEFINE_string(
  "output_dir", None,
  "the output directory where the model checkpints will be written"
)

flags.DEFINE_integer(
  "max_sequence", 20,
  "the maximum length of one sentence"
)

flags.DEFINE_integer(
  "max_entity", 3,
  "max entity length"
)

flags.DEFINE_integer(
  "train_batch_size", 32,
  "batch size of training")


def build_model(features, labels, hidden_size=256, fc_size=100, num_labels=2):
  sentence_embedding = features[0]
  entity_embedding = features[1]
  lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
  _, sentence_state = tf.nn.dynamic_rnn(lstm_cell, sentence_embedding, dtype=tf.float32)
  _, entity_state = tf.nn.dynamic_rnn(lstm_cell, entity_embedding, dtype=tf.float32)

  # fully connected
  sentence_feature = tf.layers.dense(sentence_state, fc_size, activation=tf.nn.relu)
  entity_feature = tf.layers.dense(entity_state, fc_size, activation=tf.nn.relu)
  combine = tf.multiply(sentence_feature, entity_feature)
  logits = tf.layers.dense(combine, num_labels, activation=None)

  loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
  train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
  return loss, train_op


def model_fn_builder(hidden_size, fc_size, num_labels):

  def model_fn(features, labels, mode, params):
    sentence_embedding = features["sentence_embedding"]
    entity_embedding = features["entity_embedding"]

    with tf.variable_scope("model"):
      with tf.variable_scope("rnn"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=False)
        _, sentence_state = tf.nn.dynamic_rnn(lstm_cell, sentence_embedding, dtype=tf.float32)
        _, entity_state = tf.nn.dynamic_rnn(lstm_cell, entity_embedding, dtype=tf.float32)

      with tf.variable_scope("dense"):
        # fully connected
        with tf.variable_scope("question"):
          sentence_feature = tf.layers.dense(sentence_state, fc_size, activation=tf.nn.relu)
        with tf.variable_scope("entity"):
          entity_feature = tf.layers.dense(entity_state, fc_size, activation=tf.nn.relu)

      combine = tf.multiply(sentence_feature, entity_feature)
      logits = tf.layers.dense(combine, num_labels, activation=None)
      print(labels.shape)
      print(logits.shape)
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss, global_step=global_step)
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
      )
    return output_spec

  return model_fn


def input_fn_builder(question_feature, entity_feature, training_label):

  def input_fn(params):
    num_samples = len(question_feature)

    dataset = tf.data.Dataset.from_tensor_slices(({
      "sentence_embedding":
        tf.constant(question_feature, shape=[num_samples, FLAGS.max_sequence, 100], dtype=tf.float32),
      "entity_embedding":
        tf.constant(entity_feature, shape=[num_samples, FLAGS.max_entity, 100], dtype=tf.float32)
    }, tf.constant(training_label, shape=[num_samples, 2], dtype=tf.float32)))

    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=FLAGS.train_batch_size)
    return dataset

  return input_fn


def main(_):
  knowledge_tree = KnowledgeTree(FLAGS.graph_path)
  data_processor = DataProcessor(FLAGS.data_path, knowledge_tree, FLAGS.max_sequence, FLAGS.max_entity)
  question_feature, entity_feature, labels = data_processor.get_training_samples()

  input_fn = input_fn_builder(question_feature, entity_feature, labels)
  model_fn = model_fn_builder(
    hidden_size=256,
    fc_size=100,
    num_labels=2
  )

  estimator = tf.estimator.Estimator(
    model_dir=FLAGS.output_dir,
    model_fn=model_fn
  )

  estimator.train(input_fn=input_fn, max_steps=600)


if __name__ == "__main__":
  flags.mark_flag_as_required("graph_path")
  flags.mark_flag_as_required("data_path")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
