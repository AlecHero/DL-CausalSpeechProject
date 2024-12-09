import torch

model_names = ["student_only_labels", "student_only_teacher", "student_partly_teacher", "e2e_student_from_teacher"]

def save_results(n_datapoints, save_path=None):
    from load_models import get_model_predictions_and_data
    
    datapoints = n_datapoints
    results = get_model_predictions_and_data(
        mock = False,
        save_memory = True,
        datapoints = datapoints,
        deterministic = False
    )
    if save_path: torch.save({ "results": results }, save_path)


def sisnr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def compute_loss(self, ests, refs):
    from itertools import permutations
    
    def sisnr_loss(permute):
        # for one permute
        return sum(
            [self.sisnr(ests[s], refs[t])
                for s, t in enumerate(permute)]) / len(permute)

    # P x N
    N = ests[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(len(ests)))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    return -torch.sum(max_perutt) / N


def compute_metric(results, metric=sisnr):
    baseline_metrics, prediction_metrics = [], []
    
    for (predictions, inputs, outputs) in results:
        baseline_scores = metric(inputs, outputs)
        
        prediction_scores = torch.empty(len(predictions))
        for i in range(len(predictions)):
            prediction_scores[i] = metric(predictions[i][0], outputs)
        
        baseline_metrics.append(baseline_scores)
        prediction_metrics.append(prediction_scores)
    
    return torch.as_tensor(baseline_metrics), torch.stack(prediction_metrics)

def conf(data):
    import numpy as np
    from scipy.stats import t
    data = np.asarray(data)
    n = data.shape[0]
    mean = data.mean(0)
    margin = t.ppf((1 + 0.95) / 2, n-1) * data.std(0, ddof=1) / np.sqrt(n)
    return mean, margin

def print_conf(baseline_metrics, prediction_metrics):
    
    for model_idx in range(num_models-1, -1, -1):
        mean, margin = conf(prediction_metrics[:, model_idx])
        print("{:<25} : {:.2f} ± {:.2f}".format(model_names[model_idx], round(mean, 2), round(margin, 2)))
    print()

    mean, margin = conf(baseline_metrics)
    print("{:<25} : {:.2f} ± {:.2f}".format("baseline", round(mean, 2), round(margin, 2)))

if __name__ == "__main__":
    save_path = "/zhome/f8/2/187151/DL-CausalSpeechProject/results.pt"
    save_results(n_datapoints=632, save_path=save_path)