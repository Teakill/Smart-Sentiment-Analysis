import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_label(y_true: np.array) -> np.array:
    """Convert from one-hot vector into multi-labels vector

    Args:
        y_true (np.array): the one-hot vector of ground truth labels, shape [1,30]

    Returns:
        np.array: ground truth, multi-labels vector, shape [1,6]
    """
    labels = np.array([0, 0, 0, 0, 0, 0])
    for i in range(6):
        for j in range(5):
            if y_true[5*i+j] == 1:
                labels[i] = j + 1
    return labels

def get_prediction(outputs:np.array, threshold=0.5)->np.array:
    """get prediction from logits \\
    Convert from logits into multi-labels vector

    Args:
        outputs (np.array): outputs logits from model, shape: [1, 30]
        threshold (float, optional): sigmoid threshold. Defaults to 0.5.

    Returns:
        np.array: predictions, multi-labels vector, shape: [1,6]
    """
    outputs = sigmoid(outputs)
    result = np.array([0, 0, 0, 0, 0, 0])
    for i in range(6):
        best_score = -999
        index = -1
        for j in range(5):
            if outputs[5*i+j] > best_score:
                best_score = outputs[5*i+j]
                index = j
        if best_score >= threshold:
            result[i] = index + 1
    return result

def sigmoid(x:np.array) -> np.array:
    """sigmoid function

    Args:
        x (np.array): x

    Returns:
        np.array: sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def get_r2_score(y_true: np.array, y_pred: np.array) -> float:
    """Calculate r2 score \\
    R2 = 1 - RSS/K \\
    RSS = residual sum of squares = sigma [(y_hat - y) ^ 2] \\
    K = total sum of squares of max distance = n * (max_sentiment - min_sentiment) ^ 2 \\

    Args:
        y_true (np.array): ground truth
        y_pred (np.array): predictions

    Returns:
        float: r2 score
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, y_true has shape {y_true.shape} while y_pred has shape {y_pred.shape}"
    y_true_ = y_true
    y_pred_ = y_pred
    max_sentiment = 5
    min_sentiment = 1
    rss = 0
    n = 0
    for i in range(len(y_true_)):
        if y_true_[i] * y_pred_[i] > 0:
            rss += (y_true_[i] - y_pred_[i]) ** 2
            n += 1
    try:
        return 1 - rss / (n * (max_sentiment - min_sentiment) ** 2)
    except:
        return 1  # competition rules


def get_precision_recall_f1_score(labels: np.array, predictions: np.array) -> tuple:
    """Calculate precision, recall and f1 score \\
    [0] -> 0 and [1-5] -> 1

    Args:
        labels (np.array): grounth truth
        predictions (np.array): predictions

    Returns:
        tuple: (precision, recall, f1_score)
    """
    labels_ = labels > 0
    predictions_ = predictions > 0
    return (
    precision_score(labels_, predictions_), recall_score(labels_, predictions_), f1_score(labels_, predictions_))


def report_score(labels: np.array, predictions: np.array) -> dict:
    """Get the score: precision, recall, f1 score, r2 score and competition score \\
    Score of 1 aspect

    Args:
        labels (np.array): ground truth, shape: [num_examples, 1]
        predictions (np.array): predictions, shape: [num_examples, 1]

    Returns:
        dict: a score dictionary {precision, recall, f1_score, r2_score, competition_score} of 1 aspect
    """
    score = {}
    score["precision"], score["recall"], score["f1_score"] = get_precision_recall_f1_score(labels, predictions)
    score["r2_score"] = get_r2_score(labels, predictions)
    score["final_score"] = score["f1_score"] * score["r2_score"]

    return score


def convert_to_one_hot(data: np.array) -> np.array:
    """
    Convert data from shape [n_samples, 6] to [n_samples, 30] in a one-hot format.
    """
    one_hot = np.zeros((data.shape[0], 30))

    for i in range(data.shape[0]):
        for j in range(6):
            value = data[i][j]
            if value != 0:
                one_hot[i][j * 5 + (value - 1)] = 1  # Convert to one-hot representation

    return one_hot

if __name__ == '__main__':
    pred = [[1, 0, 3, 4, 2, 0], [0, 2, 2, 5, 1, 0]]
    gt = [[1, 1, 3, 5, 3, 1], [0, 0, 3, 4, 0, 0]]
    # Convert pred and gt to the required one-hot format
    pred_one_hot = convert_to_one_hot(np.array(pred))
    gt_one_hot = convert_to_one_hot(np.array(gt))

    # Get scores using the provided function
    scores = get_score(gt_one_hot, pred_one_hot)

    # Print the results
    for aspect, values in scores.items():
        print(f"\nAspect: {aspect}")
        for metric, value in values.items():
            print(f"{metric}: {value:.4f}")


def get_score_modified(y_true: np.array, y_pred: np.array) -> dict:
    """Get score for format [n_examples, 6]

    Args:
        y_true (np.array): shape: [n_examples, 6]
        y_pred (np.array): shape: [n_examples, 6]

    Returns:
        dict: a score dictionary of each aspect
    """
    score = {}
    aspects = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    for i in range(6):
        score[aspects[i]] = report_score(y_true[:, i], y_pred[:, i])

    return score
