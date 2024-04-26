# Riva build and deploy commands
## Model specification
### Type
Conformer CTC Large fr_FR
### Tokenizer
size of vocabulary: 128
### Download
RIVA Conformer ASR French trainable v2.1
[speechottext_fr_fr_conformer.tlt](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_fr_fr_conformer/files)
### Finetuning framework
TAO

## Run docker container for building and deployment
```bash
docker run --gpus all -it --rm -v ~/riva_quickstart_arm64_v2.13.1/model_repository/artifacts/:/servicemaker-dev -v ~/riva_quickstart_arm64_v2.13.1/model_repository/rmir/:/rmir -v ~/riva_quickstart_arm64_v2.13.1/model_repository/:/data/ --entrypoint="/bin/bash" nvcr.io/nvidia/riva/riva-speech:2.13.1-servicemaker-l4t-aarch64
```


## Riva build
### Build online, streaming model in docker container
```bash
riva-build speech_recognition -f \
  /rmir/tao_v128_online.rmir:tlt_encode \
  /servicemaker-dev/finetuned_entire_plus_model_epoch_0.riva:tlt_encode \
  --name=finetuned-conformer-fr-FR-asr \
  --return_separate_utterances=False \
  --featurizer.use_utterance_norm_params=False \
  --featurizer.precalc_norm_time_steps=0 \
  --featurizer.precalc_norm_params=False \
  --ms_per_timestep=40 \
  --endpointing.start_history=200 \
  --nn.fp16_needs_obey_precision_pass \
  --endpointing.residue_blanks_at_start=-2 \
  --chunk_size=0.16 \
  --left_padding_size=1.92 \
  --right_padding_size=1.92 \
  --decoder_type=flashlight \
  --decoding_language_model_binary=/data/models/conformer-fr-FR-asr-streaming-ctc-decoder-cpu-streaming/1/fr-FR_default_2.1.bin \
  --decoding_vocab=/data/models/conformer-fr-FR-asr-streaming-ctc-decoder-cpu-streaming/1/fr-FR_default_2.1_dict_vocab.txt \
  --flashlight_decoder.lm_weight=0.7 \
  --flashlight_decoder.word_insertion_score=0.75 \
  --flashlight_decoder.beam_threshold=20. \
  --language_code=fr-FR \
  --wfst_tokenizer_model=/data/models/conformer-fr-FR-asr-streaming/1/tokenize_and_classify.far \
  --wfst_verbalizer_model=/data/models/conformer-fr-FR-asr-streaming/1/verbalize.far \
  --flashlight_decoder.max_execution_batch_size=1 \
  --endpointing.min_batch_size=1 \
  --flashlight_decoder.min_batch_size=1 \
  --max_batch_size=1 \
  --endpointing.max_batch_size=1 \
  --flashlight_decoder.max_batch_size=1 \
  --featurizer.opt_batch_size=1 \
  --endpointing.opt_batch_size=1 \
  --flashlight_decoder.opt_batch_size=1
```
  
 if failed, try add  flag --nn.use_trt_fp32

### Build offline model in docker container
```bash
riva-build speech_recognition \
  /rmir/tao_v128_offline.rmir:tlt_encode \
  /servicemaker-dev/finetuned_entire_plus_model_epoch_0.riva:tlt_encode \
  --offline \
  --name=finetuned-conformer-fr-FR-asr-offline \
  --return_separate_utterances=True \
  --featurizer.use_utterance_norm_params=False \
  --featurizer.precalc_norm_time_steps=0 \
  --featurizer.precalc_norm_params=False \
  --ms_per_timestep=40 \
  --endpointing.start_history=200 \
  --nn.fp16_needs_obey_precision_pass \
  --endpointing.residue_blanks_at_start=-2 \
  --chunk_size=4.8 \
  --left_padding_size=1.6 \
  --right_padding_size=1.6 \
  --max_batch_size=16 \
  --featurizer.max_batch_size=512 \
  --featurizer.max_execution_batch_size=512 \
  --decoder_type=flashlight \
  --decoding_language_model_binary=/data/models/conformer-fr-FR-asr-streaming-ctc-decoder-cpu-streaming/1/fr-FR_default_2.1.bin \
  --decoding_vocab=/data/models/conformer-fr-FR-asr-streaming-ctc-decoder-cpu-streaming/1/fr-FR_default_2.1_dict_vocab.txt \
  --flashlight_decoder.lm_weight=0.7 \
  --flashlight_decoder.word_insertion_score=0.75 \
  --flashlight_decoder.beam_threshold=20. \
  --language_code=fr-FR \
  --wfst_tokenizer_model=/data/models/conformer-fr-FR-asr-streaming/1/tokenize_and_classify.far \
  --wfst_verbalizer_model=/data/models/conformer-fr-FR-asr-streaming/1/verbalize.far
```

## Riva deploy
### Manual deployment in docker container  
```bash
riva-deploy -f /rmir/tao_v128_online.rmir:tlt_encode /data/models
riva-deploy -f /rmir/tao_v128_offline.rmir:tlt_encode /data/models
```


### Automatic deployment in riva quickstart folder
```bash
bash riva_init.sh
```

## Riva start
```bash
bash riva_start.sh
```

## Riva client
### Use microphone for online ASR under /home/wnm/python-clients/ folder
```bash
python3 examples/transcribe_mic.py --language-code fr-FR
```

## Other notes
### delete all folders for online model
```bash
sudo rm -r finetuned-conformer-fr-FR-asr finetuned-conformer-fr-FR-asr-endpointing-streaming finetuned-conformer-fr-FR-asr-ctc-decoder-cpu-streaming finetuned-conformer-fr-FR-asr-feature-extractor-streaming riva-trt-finetuned-conformer-fr-FR-asr-am-streaming
```

### delete all folders for offline model
```bash
sudo rm -r riva-trt-finetuned-conformer-fr-FR-asr-offline-am-streaming-offline finetuned-conformer-fr-FR-asr-offline-feature-extractor-streaming-offline finetuned-conformer-fr-FR-asr-offline-endpointing-streaming-offline finetuned-conformer-fr-FR-asr-offline-ctc-decoder-cpu-streaming-offline finetuned-conformer-fr-FR-asr-offline
```

