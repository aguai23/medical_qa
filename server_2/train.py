import tensorflow as tf
import numpy as np
from data_processor import DataProcessor
from knowledge_tree import KnowledgeTree
from sklearn.metrics import accuracy_score
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "graph_path", "./data/graph",
  "The graph path file, containing all the triples constructing knowledge graph"
)

flags.DEFINE_string(
  "data_path", None,
  "the labeled questions path"
)

flags.DEFINE_string(
  "output_dir", "./data/model/",
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
  "batch size of training"
)

flags.DEFINE_integer(
  "train_epoch", 60,
  "training epochs"
)

flags.DEFINE_integer(
  "word_embedding", 100,
  "word2vec dimensions"
)


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
    if len(sentence_embedding.shape) < 3:
      sentence_embedding = sentence_embedding[np.newaxis, :]
      entity_embedding = entity_embedding[np.newaxis, :]
    # print(sentence_embedding.shape)
    # print(entity_embedding.shape)
    # print(labels)
    with tf.variable_scope("model"):
      with tf.variable_scope("rnn"):
        sentence_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=False, name="sentence_cell")
        entity_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=False, name="entity_cell")
        _, sentence_state = tf.nn.dynamic_rnn(sentence_cell, sentence_embedding, dtype=tf.float32)
        _, entity_state = tf.nn.dynamic_rnn(entity_cell, entity_embedding, dtype=tf.float32)

      with tf.variable_scope("dense"):
        # fully connected
        with tf.variable_scope("question"):
          sentence_feature = tf.layers.dense(sentence_state, fc_size, activation=tf.nn.leaky_relu)
        with tf.variable_scope("entity"):
          entity_feature = tf.layers.dense(entity_state, fc_size, activation=tf.nn.leaky_relu)

      combine = tf.multiply(sentence_feature, entity_feature)
      logits = tf.layers.dense(combine, num_labels, activation=None)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope("train"):
        print(labels.shape)
        print(logits.shape)
        predictions = tf.nn.sigmoid(logits)
        description_predictions = tf.to_float(predictions[:, 0] > 0.5)
        continue_predictions = tf.to_float(predictions[:, 1] > 0.5)
        description_accuracy = (tf.reduce_sum(tf.multiply(labels[:,0], description_predictions)) +
                               tf.reduce_sum(tf.multiply(1 - labels[:,0], 1 - description_predictions)))/\
                               tf.to_float(tf.size(description_predictions))
        continue_accuracy = (tf.reduce_sum(tf.multiply(labels[:,1], continue_predictions)) +
                               tf.reduce_sum(tf.multiply(1 - labels[:,1], 1 - continue_predictions)))/\
                               tf.to_float(tf.size(continue_predictions))
        loss_weight = tf.multiply(labels, 3) + 1
        loss = tf.reduce_mean(
          tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), loss_weight))

        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)

        logging_hook = tf.train.LoggingTensorHook(
          {"loss": loss,
           "description": description_accuracy,
           "continue": continue_accuracy},
          every_n_iter=10
        )
        output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook]
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
      logits = tf.nn.sigmoid(logits)
      description_predictions = tf.to_float(logits[:, 0] > 0.5)
      continue_predictions = tf.to_float(logits[:, 1] > 0.5)
      description_accuracy = tf.metrics.accuracy(labels[:, 0], description_predictions)
      continue_accuracy = tf.metrics.accuracy(labels[:, 1], continue_predictions)
      eval_metric = {
        "description_accuracy": description_accuracy,
        "continue_accuracy": continue_accuracy,
        "eval_loss": tf.metrics.mean(loss)
      }
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric,

      )
    else:
      predictions = {
        "logits": tf.nn.sigmoid(logits),
        "sentence_feature": sentence_feature,
      }
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions
      )
    return output_spec

  return model_fn


def input_fn_builder(question_feature, entity_feature, training_label, predict=False):
  def input_fn(params):
    num_samples = len(question_feature)
    # print(num_samples)
    # print(question_feature.shape)
    dataset = tf.data.Dataset.from_tensor_slices(({
                                                    "sentence_embedding":
                                                      tf.constant(question_feature,
                                                                  shape=[num_samples, FLAGS.max_sequence, 100],
                                                                  dtype=tf.float32),
                                                    "entity_embedding":
                                                      tf.constant(entity_feature,
                                                                  shape=[num_samples, FLAGS.max_entity, 100],
                                                                  dtype=tf.float32)
                                                  }, tf.constant(training_label, shape=[num_samples, 2],
                                                                 dtype=tf.float32)))

    if not predict:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=100)
      dataset = dataset.batch(batch_size=FLAGS.train_batch_size)
    return dataset

  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  knowledge_tree = KnowledgeTree(FLAGS.graph_path)
  data_processor = DataProcessor(FLAGS.data_path, knowledge_tree, FLAGS.max_sequence, FLAGS.max_entity)
  question_feature, entity_feature, labels = data_processor.get_training_samples()

  train_numbers = len(question_feature)
  training_steps = int(train_numbers / FLAGS.train_batch_size * FLAGS.train_epoch)

  input_fn = input_fn_builder(question_feature, entity_feature, labels)

  valid_question, valid_entity, valid_label = data_processor.get_valid_samples()
  valid_numbers = len(valid_question)
  valid_steps = int(valid_numbers / FLAGS.train_batch_size)
  evaluate_fn = input_fn_builder(valid_question, valid_entity, valid_label)

  model_fn = model_fn_builder(
    hidden_size=256,
    fc_size=100,
    num_labels=2
  )

  config = tf.estimator.RunConfig(
    save_checkpoints_steps=300,
    log_step_count_steps=10,
    save_summary_steps=10,
    keep_checkpoint_max=10
  )
  estimator = tf.estimator.Estimator(
    model_dir=FLAGS.output_dir,
    model_fn=model_fn,
    config=config
  )

  train_spec = tf.estimator.TrainSpec(
    input_fn=input_fn,
    max_steps=training_steps
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn=evaluate_fn,
    steps=valid_steps,
    throttle_secs=10
  )

  tf.estimator.train_and_evaluate(
    estimator,
    train_spec,
    eval_spec
  )


if __name__ == "__main__":
  flags.mark_flag_as_required("graph_path")
  flags.mark_flag_as_required("data_path")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
