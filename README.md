# Self-supervised non-causal to causal speech enhancement

Todo:
- [x] Causal 1dConv and 1dTransposeConv layer (hent kode fra torchaudio) (bertram, alex)
- [x] Find ud af hvordan man kører med bsub (Lucas)
- [x] Compile torch forward + backward (logger skal ud) (Lucas)
- [x] Træne teacher (Lucas)
- [x] Prep script for student train
- [ ] Cleanup folder

- [ ] How to best transfer learn from teacher to student
    - [ ] Make a loss for intermediate layer output values (bertram, alex)
- [ ] Figure out accuracy measure. They use PESQ, MCD and PSS in the paper, what should we use? What are the Flops used?

Notes:
- It seems to be better to fully train teacher, then train student: Much more stabel flow for the student to follow. On this note, also address the best alpha, or if there should even be an alpha when the teacher is fully trained when starting student training.


Next:
- [ ] Try end-to-end?
- [ ] Try with discriminator model included?

**Expected results:**
We are going to compare the following approaches:
- Training causal student directly
- Training causal student from teacher with normal loss
- Training causal student from teacher with loss on intermediate layer
- Training causal student and teacher end-to-end with normal loss
- Training causal student and teacher end-to-end with loss on intermediate layer
- Training with discriminators?


