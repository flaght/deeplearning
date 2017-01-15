# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for word2vec_optimized module."""

import os

import tensorflow as tf

from tensorflow.models.embedding import word2vec_optimized

FLAGS = tf.app.flags.FLAGS

# FLAGS = flags.FLAGS

class Word2VecTest(tf.test.TestCase):

    def setUp(self):
        # FLAGS.train_data = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/embedding/text8"
        # FLAGS.eval_data = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/embedding/questions-words.txt"
        # FLAGS.save_path = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/embedding/tmp"
        #
        FLAGS.train_data = os.path.join(self.get_temp_dir() + "text8")
        FLAGS.eval_data = os.path.join(self.get_temp_dir() + "questions-words.txt")
        FLAGS.save_path = self.get_temp_dir()
        # with open(FLAGS.train_data, "w") as f:
        #   f.write(
        #       """alice was beginning to get very tired of sitting by her sister on
        #       the bank, and of having nothing to do: once or twice she had peeped
        #       into the book her sister was reading, but it had no pictures or
        #       conversations in it, 'and what is the use of a book,' thought alice
        #       'without pictures or conversations?' So she was considering in her own
        #       mind (as well as she could, for the hot day made her feel very sleepy
        #       and stupid), whether the pleasure of making a daisy-chain would be
        #       worth the trouble of getting up and picking the daisies, when suddenly
        #       a White rabbit with pink eyes ran close by her.\n""")
        #   with open(FLAGS.eval_data, "w") as f:
        #     f.write("alice she rabbit once\n")

    def testWord2VecOptimized(self):
        FLAGS.batch_size = 5
        FLAGS.num_neg_samples = 10
        FLAGS.epochs_to_train = 1
        FLAGS.min_count = 1
        word2vec_optimized.main([])


if __name__ == "__main__":
    tf.test.main()
