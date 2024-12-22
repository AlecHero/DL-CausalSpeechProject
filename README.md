# Self-supervised non-causal to causal speech enhancement

## Demo:
See our demo notebook [here](demo.ipynb)

## Reproduce:

To reproduce our results:
1. Pull this repo
2. If you wish to try the overfitting run, you can do so without setting up the EARS-WHAM dataset and using our sample data.
3. Go the the `config` folder and change the "neptune_api" to your api. You can also leave it blank and let `test_run` be true, which will run the training without logging to neptune.

**Overfitting run:**
```bash
python train.py --config config/overfitting.yaml
```

**Full training run:**
This requires that you have set up the full EARS-WHAM dataset by following this [guide](https://github.com/sp-uhh/ears_benchmark). We decided to resample the data to 16kHz, as this is the sampling rate of the data we used for training. You can do this and see our code for resampling in our Dataloader folder.

```bash
python train.py --config config/student_from_teacher.yaml
```