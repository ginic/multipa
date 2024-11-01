# multipa
MultIPA is yet another automatic speech transcription model into phonetic IPA.
The idea is that, if we train a multilingual speech-to-IPA model with enough amount of good phoneme representations, the model's output will be approximated to phonetic transcriptions.
Please check out the [Paper](https://arxiv.org/abs/2308.03917) for details.

## Available training languages
At this moment, we have the following languages incorporated available in the training data:
- Finnish
- Hungarian
- Japanese
- Maltese
- Modern Greek
- Polish
- Tamil

We aim to include more languages to take into account linguistic diversity.

## Installation
Use a virtual environment, such as anaconda, with python 3.9. It's also a good idea to upgrade pip before you start: `pip install --upgrade pip`.
Install pytorch dependencies first: `pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0`
Then install remaining requirements: `pip install .`. You can run `pip install -e .[dev,test]` for the developer set up, after which the best way to check that the installation worked is to run unit tests with `pytest`. You will also need to download the unidic dictionary with `python -m unidic download` (in the same python environment as you installation).

For convenience, scripts for setup and installation are provided in `scripts/setup.sh` (local or personal computer) and `scripts/setup_slurm.sh` (Slurm compute cluster).


## How to run
⚠️ *Warning*: This section is out of data.
*TODO*: Update processing and training instructions with new subcommands for each dataset

First, run `pip install .` to install this package and required libraries.

You need to convert the transcription in the CommonVoice dataset into IPA before training a model.
To do so, run `src/multipa/preprocess.py`; for example,
```
multipa-preprocess \
       -l ja pl mt hu fi el ta \
       --num_proc 48
```

Then, run `src/multipa/main.py` to train a model.
For example:
```
multipa-train \
        -l ja pl mt hu fi el ta \
        -tr 1000 1000 1000 1000 1000 1000 1000 \
        -te 200 200 200 200 200 200 200 \
        -qf False False False False False False False \
        -a True \
        -s "japlmthufielta-nq-ns" \
        --no_space \
        -v vocab.json \
        -e 10
```
for training with 7 languages, 1000 training samples and 200 validation samples for each, where audio samples with bad quality are not filtered out, additional data from Forvo are included, the suffix for the output model folder name is `japlmthufielta-nq-ns`, orthographic spaces are removed, the name of the vocab file is `vocab.json`, and the number of epochs is set to 10.

### Evaluation
To evaluate an existing models' performance on the test split of the Buckeye corpus, you can list models from Hugging Face with the `--hf_models` arg or models saved in local files with the `--local_models` flag. If you used the `--no_space` flag during training, you should use it during evaluation as well: 
```
multipa-evaluate --hf_models ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000 \ 
 --local_models data/buckeye_model/wav2vec2-large-xlsr-buckeye-ipa_test \
 --eval_out data/test_eval/buckeye_eval.csv \
 --verbose_results_dir data/test_eval/detailed_results --no_space --data_dir <path_to_preprocessed_buckeye_folder>
 ```

## Model
You can run the model (trained on 1k samples for each language, 9h in total) [here](https://huggingface.co/ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns).

## Notes
- If you are using AFS, `preprocess.py` might cause `OS Error: File too large` due to reaching the limit of the number of files that a directory can accommodate.
- Additional data from Forvo themselves are not uploaded in this repository.
- The full list of IPA symbols was obtained from the [Panphon](https://github.com/dmort27/panphon) library.

## Citation
Chihiro Taguchi, Yusuke Sakai, Parisa Haghani, David Chiang. "Universal Automatic Phonetic Transcription into the International Phonetic Alphabet". INTERSPEECH 2023.\
For the time being, you may cite our arXiv paper:
```
@misc{taguchi2023universal,
      title={Universal Automatic Phonetic Transcription into the International Phonetic Alphabet}, 
      author={Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
      year={2023},
      eprint={2308.03917},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
Feel free to raise issues if you find any bugs.
Also, feel free to contact me `ctaguchi at nd.edu` for collaboration.

# CMU Dict License
The [Carnegie Mellon Pronouncing Dictionary](https://github.com/Alexir/CMUdict/tree/master) distributed with this code in `src/multipa/resources/cmudict-0.7b-ipa.txt` is covered by the following BSD 3-Clause License. This license does not apply to other code in this repository.
```
Copyright (c) 2015, Alexander Rudnicky
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of dictTools nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```