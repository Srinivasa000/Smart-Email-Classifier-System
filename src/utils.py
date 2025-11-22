import joblib
from sklearn.metrics import classification_report, accuracy_score


def save_obj(obj, path):
	joblib.dump(obj, path)


def load_obj(path):
	return joblib.load(path)


def print_report(y_true, y_pred, target_name=''):
	print(f"\n--- Report for {target_name} ---")
	print(classification_report(y_true, y_pred))
	print("Accuracy:", accuracy_score(y_true, y_pred))