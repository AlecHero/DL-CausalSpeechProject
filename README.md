# Self-supervised non-causal to causal speech enhancement

Todo:

- [x] gNeptuneLogger (https://app.neptune.ai/o/apros7/org/CausalSpeech/runs/table?viewId=standard-view)
- [x] Implemented ConvTasNet.py
- [ ] Create custom torch dataloader for causal models
- [ ] Create custom torch dataloader for non-causal models
- [x] Create loss/eval scripts
- [ ] Create teacher training script
- [ ] Train teacher
- [ ] Make dataset with predictions from teacher (overfit?)
- [ ] Create student training script
- [ ] Train student
- [ ] Evaluate student

Missing in training script:
- [ ] Making it causal, non-causal
- [ ] Train with batch_size > 1
- [ ] Perform validation and log
- [ ] Work with more than 1 second of audio


Next:
- [ ] Try end-to-end?
- [ ] Try with discriminator model included?

