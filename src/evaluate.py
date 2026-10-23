import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    return (tp, fn, fp, tn)

def report(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred)
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    baseline = max(tp + fn, fp + tn) / total

    print("\nConfusion matrix (test set)")
    print("                 predicted dog   predicted not dog")
    print(f"actual dog       {tp:>13}   {fn:>17}")
    print(f"actual not dog   {fp:>13}   {tn:>17}")
    print(f"accuracy:  {accuracy:.1%}")
    print(f"precision: {precision:.1%}  (of the images called dog, how many were)")
    print(f"recall:    {recall:.1%}  (of the real dogs, how many were found)")
    print(f"baseline:  {baseline:.1%}  (always predicting the more common class)")