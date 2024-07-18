# Rasa: Building Expressive Speech Synthesis Systems for Indian Languages in Low-resource Settings

> 🎉 Accepted at INTERSPEECH 2024

We release Rasa, the first multilingual expressive TTS dataset for any Indian language, which contains 10 hours of neutral speech and 1-3 hours of expressive speech for each of the 6 Ekman emotions covering 3 languages: Assamese, Bengali, \& Tamil. Our ablation studies reveal that just 1 hour of neutral and 30 minutes of expressive data can yield a Fair system as indicated by MUSHRA scores. Increasing neutral data to 10 hours, with minimal expressive data, significantly enhances expressiveness. This offers a practical recipe for resource-constrained languages, prioritizing easily obtainable neutral data alongside smaller amounts of expressive data. We show the importance of syllabically balanced data and pooling emotions to enhance expressiveness. We also highlight challenges in generating specific emotions, e.g., fear and surprise.

**TL;DR:** We open-source Expressive Text-To-Speech dataset and models for 3 Indian languages: *Assamese, Bengali, and, Tamil.*.


**Authors:** Praveen S V*, Ashwin Sankar*, Giri Raju, Mitesh M. Khapra



### Downloads:
Model checkpoints can be downloaded at https://ai4bharat.iitm.ac.in/rasa/v1/

| Language | Links                                                                                            |
|----------|--------------------------------------------------------------------------------------------------|
| Assamese | [(3.8 GB)](https://indic-tts-public.objectstore.e2enetworks.net/data/expressive_assamese_tts_dataset.tar.gz) |
| Bengali  | [(4.0 GB)](https://indic-tts-public.objectstore.e2enetworks.net/data/expressive_bengali_tts_dataset.tar.gz)  |
| Tamil    | [(9.1 GB)](https://indic-tts-public.objectstore.e2enetworks.net/data/expressive_tamil_tts_dataset.zip) |


## Setup:


### Environment Setup

```
git clone https://github.com/ai4bharat/rasa.git
cd rasa
conda env create -f environment.yml

cd Trainer
pip3 install -e .[all]
cd ..

cd TTS
pip3 install -e .[all]
cd ..
```

### Data Setup

The data should be formatted similar to LJSpeech. You may find the metadata along with the audio here - https://ai4bharat.iitm.ac.in/rasa/v1/.


### Training:
1. Set the configuration with [main.py](./main.py), [vocoder.py](./vocoder.py), [configs](./configs) and [run.sh](./run.sh). Make sure to update the CUDA_VISIBLE_DEVICES in all these files.
2. Train by executing `sh configs/train.sh`

### Inference

```
    output_dir="/path/to/where/you/want/the/output/saved"
    base_path="/path/to/base/directory/of/fastpitch"
    vocoder_path="/path/to/base/directory/of/hifigan"

    mkdir -p ${output_dir}

    python3 -m TTS.bin.synthesize --text "/path/to/metadata_test.csv" \
        --model_path ${base_path}/best_model.pth \
        --config_path ${base_path}/config.json \
        --vocoder_path ${vocoder_path}/best_model.pth \
        --vocoder_config_path ${vocoder_path}/config.json \
        --out_path ${output_dir} \
        --use_cuda t \
        --use_emotion t \
```

Code Reference: 
1. [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
2. [https://github.com/coqui-ai/Trainer](https://github.com/coqui-ai/Trainer)
