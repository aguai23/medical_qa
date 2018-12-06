#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf
from train import model_fn_builder, input_fn_builder
from knowledge_tree import KnowledgeTree
from train import FLAGS
import jieba
import gensim
import numpy as np
app = Flask(__name__)
CORS(app, supports_credentials=True)

knowledge_tree = KnowledgeTree(FLAGS.graph_path)

model_fn = model_fn_builder(
    hidden_size=256,
    fc_size=100,
    num_labels=2
)

estimator = tf.estimator.Estimator(
    model_dir=FLAGS.output_dir,
    model_fn=model_fn
)

entity_embeddings = {}
# build entity embeddings dict
node_list = [knowledge_tree.get_root()]
model = gensim.models.Word2Vec.load("./data/word2vec")
default_embedding = np.zeros(FLAGS.word_embedding)
while len(node_list) > 0:
  node = node_list.pop(0)
  entity = node.value
  entity_tokens = list(jieba.cut(entity))
  embedding = np.zeros((FLAGS.max_entity, FLAGS.word_embedding))
  for i in range(len(entity_tokens)):
    if i == FLAGS.max_entity:
      break
    if entity_tokens[i] in model.wv:
      embedding[i] = model.wv[entity_tokens[i]]
    else:
      embedding[i] = default_embedding
  entity_embeddings[entity] = embedding
  node_list += node.children


def build_question_embedding(question):
  question_tokens = list(jieba.cut(question))
  embedding = np.zeros((FLAGS.max_sequence, FLAGS.word_embedding))
  for i in range(len(question_tokens)):
    if i == FLAGS.max_sequence:
      break
    if question_tokens[i] in model.wv:
      embedding[i] = model.wv[question_tokens[i]]
    else:
      embedding[i] = default_embedding
  return embedding


tf.logging.set_verbosity(tf.logging.INFO)


@app.route("/handle_question", methods=['POST'])
def handle_question():
  question = request.get_json()['input_utterance']
  print(question)
  if len(question) == 0:
    return ""
  queue = [knowledge_tree.get_root()]
  result = []
  question_embedding = build_question_embedding(question)
  # print(question_embedding)
  label_dummy = np.zeros(2)
  while len(queue) > 0:
    node = queue.pop(0)
    entity_embedding = entity_embeddings[node.value]
    # print(entity_embedding)
    input_fn = input_fn_builder(question_embedding[np.newaxis, :], entity_embedding[np.newaxis, :],
                                label_dummy[np.newaxis, :], predict=True)
    predictions = estimator.predict(input_fn, checkpoint_path="./data/model/model.ckpt-1200")
    # print(list(predictions))
    logits = list(predictions)[0]["logits"]
    description_logit = logits[0]
    continue_logit = logits[1]
    print(node.value)
    print("description logit " + str(description_logit))
    print("continue logit " + str(continue_logit))

    if description_logit > 0.5 and node.connection is not None:
      result.append((node.connection, node.value))
    if continue_logit > 0.5:
      queue += node.children
  return str(result) + "\n"
