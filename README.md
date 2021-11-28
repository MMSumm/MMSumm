
### Image Data Generation

---

To generate the rcnn features and labels used by OSCAR, we use [this repo](https://github.com/airsplay/py-bottom-up-attention)

Each image file requires a features file, a labels file, and a vgg-embeddings file.
The former two can be generated using the previously mentioned
[repo](https://github.com/airsplay/py-bottom-up-attention), while the last can be generated using
the file `get_vgg_embeddings.py`.

To generate the TSV files for image data, use `get_idx_files.py`.

### Sentence 

---

For the sentence simplification task, run `simplify.py` on the MSMO dataset. REQUIRED BEFORE
OSCAR SCORING, otherwise the embeddings would not be very meaningful.

### Model training/inference

---

To train the model
1. Go to MMSumm/src/
2. Run train.sh and set image data and train data respectively in train.sh

For validation
1. Go to MMSumm/src/
2. Run val.sh and set image data and train data respectively in val.sh

For testing
1. Go to MMSumm/src/
2. Run test.sh and set image data, train data and model path respectively in test.sh
