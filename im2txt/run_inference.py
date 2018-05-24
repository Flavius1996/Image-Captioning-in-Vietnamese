# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json
import codecs

############ Google Translate API ################
#from googletrans import Translator
#translator = Translator()


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("output_file", "", "Output json predict result.")


tf.logging.set_verbosity(tf.logging.INFO)

JSONDATA = []

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    print("Patten: ", file_pattern)
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching",
                  len(filenames))

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)
    
    count = 1
    
    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      basename = os.path.basename(filename)
      dirname = os.path.dirname(filename)
      
      captions = generator.beam_search(sess, image)
      
      ################### PRINT LOG #######################
      AllCaptions = []
      print("[%d/%d] Captions for image %s:" % (count, len(filenames), basename))
      count += 1
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        string = "\t{0}) {1} (p={2})".format(i, sentence, math.exp(caption.logprob))
        print(string)
        AllCaptions.append(string)

      # with open(dirname + '/' + os.path.splitext(basename)[0] + '.txt', 'w') as fout:
      #   fout.write('\n'.join(AllCaptions)) 

      if FLAGS.output_file != "":
        ############## GET JSON RESULT FILE ###########################
        assert len(captions) == 1
        imgid = os.path.splitext(basename)[0].split('_')[-1]
  
        sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
        sentence = " ".join(sentence)
          
        ############## Translate to VN use for EN model ################
        #sentence = translator.translate(sentence, dest='vi').text
        
        ann_dict = {}
        ann_dict["image_id"] = int(imgid)
        ann_dict["caption"] = sentence
        JSONDATA.append(ann_dict)
        
    if FLAGS.output_file != "":
      with codecs.open(FLAGS.output_file, 'w', encoding='utf-8') as jFile:
        json.dump(JSONDATA, jFile, ensure_ascii=False)
      print('#' * 70)
      print("Saved result at file: {}".format(FLAGS.output_file))


if __name__ == "__main__":
  import sys
  reload(sys)
  sys.setdefaultencoding('utf8') 
  
  tf.app.run()
