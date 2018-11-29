#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_cors import CORS
import json
import modeling
import tokenization
import tensorflow as tf
from scipy import spatial
app = Flask(__name__)
CORS(app, supports_credentials=True)


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def parse_question(question):
  """parse question into an example"""
  examples = []
  unique_id = 0
  question = question.strip()
  text_a = question
  examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=None))
  return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
      "unique_ids":
        tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
      "input_ids":
        tf.constant(
          all_input_ids, shape=[num_examples, seq_length],
          dtype=tf.int32),
      "input_mask":
        tf.constant(
          all_input_mask,
          shape=[num_examples, seq_length],
          dtype=tf.int32),
      "input_type_ids":
        tf.constant(
          all_input_type_ids,
          shape=[num_examples, seq_length],
          dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=input_type_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map, initialized_variable_names
     ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
      "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
        "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
      InputFeatures(
        unique_id=example.unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids))
  return features


# init model
tf.logging.set_verbosity(tf.logging.INFO)

layer_indexes = [-1]

bert_config = modeling.BertConfig.from_json_file("../model/bert_config.json")

tokenizer = tokenization.FullTokenizer(
  vocab_file="../model/vocab.txt", do_lower_case=True)

model_fn = model_fn_builder(
  bert_config=bert_config,
  init_checkpoint="../model/bert_model.ckpt",
  layer_indexes=layer_indexes,
  use_tpu=False,
  use_one_hot_embeddings=False)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
  master=None,
  tpu_config=tf.contrib.tpu.TPUConfig(
    num_shards=8,
    per_host_input_for_training=is_per_host))

estimator = tf.contrib.tpu.TPUEstimator(
  use_tpu=False,
  config=run_config,
  model_fn=model_fn,
  predict_batch_size=128)

# load question embedding and corresponding answer
print("loading question embedding ...")
question_embedding = []
with open("../data/question_embedding.txt", "r") as f:
  lines = f.readlines()
  for line in lines:
    embedding = json.loads(line)
    question_embedding.append(embedding["features"][0]["layers"][0]["values"])

print("loading questions ...")
questions = []
with open("../data/question.txt", "r") as f:
  lines = f.readlines()
  for line in lines:
    if len(line) > 0:
      questions.append(line)

print("loading answers ...")
answer = []
with open("../data/answer.txt", "r") as f:
  lines = f.readlines()
  for line in lines:
    if len(line) > 0:
      answer.append(line)
print(len(answer))
print(len(question_embedding))
assert len(answer) == len(question_embedding) == len(questions)


@app.route('/')
def welcome():
  return render_template("index.html")


@app.route("/handle_question", methods=['POST'])
def handle_question():
  question = request.get_json()['input_utterance']
  if len(question) == 0:
    return ""
  examples = parse_question(question)
  features = convert_examples_to_features(
    examples=examples, seq_length=128, tokenizer=tokenizer)
  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature
  input_fn = input_fn_builder(features=features, seq_length=128)
  feature_value = None
  for result in estimator.predict(input_fn, yield_single_examples=True):
    layer_output = result["layer_output_0"]
    feature_value = [round(float(x), 6) for x in layer_output[0:1].flat]

  min_distance = 1
  correct_answer = None
  similar_question = None
  for i in range(len(question_embedding)):
    distance = spatial.distance.cosine(feature_value, question_embedding[i])
    if distance < min_distance:
      min_distance = distance
      correct_answer = answer[i]
      similar_question = questions[i]
  print(min_distance)
  print(similar_question)
  print(correct_answer)

  return correct_answer.decode('utf-8')
