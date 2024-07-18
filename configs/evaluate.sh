output_dir="evaluation_samples/Tamil/1hrNeutral/SEmotion/1hr/Happy"
wavs_path="data/rasa/wavs-22k"
base_path="checkpoints/rasa/tamil/happy/1hr/"
vocoder_path="checkpoints/vocoders/tamil/"

mkdir -p ${output_dir}

python3 -m TTS.bin.synthesize --text "datasets/rasa/tamil/splits/disgust/metadata_test.csv" \
    --model_path ${base_path}/best_model.pth \
    --config_path ${base_path}/config.json \
    --vocoder_path ${vocoder_path}/best_model.pth \
    --vocoder_config_path ${vocoder_path}/config.json \
    --out_path ${output_dir} \
    --use_cuda t \
    --use_emotion t \
&& \
python3 scripts/evaluate_mcd.py \
    ${output_dir} \
    ${wavs_path} \
&& \
python3 scripts/evaluate_f0.py \
    ${output_dir} \
    ${wavs_path}