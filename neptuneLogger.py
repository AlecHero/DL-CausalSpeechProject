import neptune

# https://app.neptune.ai/o/apros7/org/CausalSpeech/runs/table?viewId=standard-view

class NeptuneLogger():
    def __init__(self):
        self.run = neptune.init_run(
            project="apros7/CausalSpeech",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNDg5ZTA0Mi0yNGZlLTRjN2EtODI5Yy0xODI2ZDNiMDY0NGUifQ==",
        )  # your credentials

    def log_metadata(self, metadata):
        #{"learning_rate": 0.001, "optimizer": "Adam"} 
        self.run["parameters"] = metadata

    # def log_loss(self, loss):
    #     self.run["train/loss"].append(loss)

    # def log_f1_score(self, f1_score):
    #     self.run["eval/f1_score"] = f1_score

    def log_metric(self, metric_name, metric_value):
        # if metric_name not in self.run: self.run[metric_name] = []
        self.run[metric_name].append(metric_value)

    def log_teacher_train_soundfile(self, file_path, speaker, idx):
        self.run[f"train/teacher_predictions_speaker{speaker}_index{idx}.wav"].upload(file_path)

    def log_train_soundfile(self, file_path, speaker, idx):
        self.run[f"train/predictions_speaker{speaker}_index{idx}.wav"].upload(file_path)

    def log_val_soundfile(self, file_path, speaker, idx):
        self.run[f"val/predictions_speaker{speaker}_index{idx}.wav"].upload(file_path)

    def stop(self):
        self.run.stop()

if __name__ == "__main__":
    logger = NeptuneLogger()
    logger.run["metric"].append(1.0)
    logger.run["metric"].append(0.5)
    logger.run.stop()