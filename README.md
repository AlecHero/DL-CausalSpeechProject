# Self-supervised non-causal to causal speech enhancement

Todo:
- [ ] Causal 1dConv and 1dTransposeConv layer (hent kode fra torchaudio) (bertram, alex)
- [ ] Find ud af hvordan man kører med bsub (Lucas)
- [ ] Compile torch forward + backward (logger skal ud) (Lucas)
- [ ] Træne teacher (Lucas)
- [ ] Prep script for student train
- [ ] Cleanup folder

- [ ] How to best transfer learn from teacher to student
    - [ ] Make a loss for intermediate layer output values (bertram, alex)

Notes:
- It seems to be better to fully train teacher, then train student: Much more stabel flow for the student to follow. On this note, also address the best alpha, or if there should even be an alpha when the teacher is fully trained when starting student training.


Next:
- [ ] Try end-to-end?
- [ ] Try with discriminator model included?

