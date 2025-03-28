stage=0

if [ $stage -le 1 ]; then
    python src/utils/prepare_mmau.py \
    --input_file data/MMAU/mmau-test-mini.json \
    --wav_dir data/MMAU/test-mini-audios \
    --out_file data/MMAU/mmau-mini.data
fi