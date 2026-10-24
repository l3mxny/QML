"""
Digit recognizability and class coverage of (generated) low-resolution MNIST images.

A small classifier is fitted on the labelled real training images (preprocess them with
mnist_processing.py --save_labels). It reports
  * the held-out accuracy on real images, i.e., how well the ten digits can be told apart at all at this resolution,
  * for generated images, the predicted class histogram, i.e., whether all ten digits are generated (mode coverage).

Generated images are either loaded from a .npy file (e.g., samples saved during training or by sample_on_ibm.py) or
sampled in simulation from a trained model.

Run from apps/logistics, e.g.
    python ../../run_scripts/digit_coverage.py --data_set_name=mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area
    python ../../run_scripts/digit_coverage.py --data_set_name=mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area \
        --model_name=continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_1234abcd --n_samples=1000
"""

import json
import warnings

import numpy as np
from fire import Fire
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from train_image_qgan import _load_data

N_CLASSES = 10


def fit_classifier(data_set_name, test_size=0.2, seed=0):
    """Fit a digit classifier on the real images. Returns the classifier, its held-out accuracy and class shares."""
    data, data_set_name = _load_data(data_set_name)
    labels = np.load(f"./training_data/{data_set_name}_labels.npy").astype(int)
    assert len(labels) == len(data), f"{len(labels)} labels for {len(data)} images"

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed,
                                                        stratify=labels)
    classifier = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=200, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(x_train, y_train)
    real_class_shares = np.bincount(labels, minlength=N_CLASSES) / len(labels)
    return classifier, classifier.score(x_test, y_test), real_class_shares


def class_coverage(classifier, samples, real_class_shares):
    """Statistics of the predicted digit classes of generated samples."""
    predictions = classifier.predict(samples)
    class_shares = np.bincount(predictions, minlength=N_CLASSES) / len(predictions)
    return dict(
        class_shares=class_shares.round(4).tolist(),
        # a digit counts as covered if it is generated at least half as often as it occurs in the real data
        n_classes_covered=int(np.sum(class_shares >= 0.5 * real_class_shares)),
        # total variation distance between generated and real class distributions (0: identical, 1: disjoint)
        class_tv_distance=float(0.5 * np.abs(class_shares - real_class_shares).sum()),
        mean_classifier_confidence=float(classifier.predict_proba(samples).max(axis=1).mean()),
    )


def _sample_from_model(model_name, n_samples, epoch=None):
    from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler

    model = ContinuousQGANModelHandler()
    if epoch is None:
        epoch = model.last_checkpoint_iteration(model_name)
    model.reload(model_name=model_name, epoch=epoch)
    return model.sample(n_samples), epoch


def main(data_set_name, samples_path=None, model_name=None, epoch=None, n_samples=1000, seed=0, out_path=None):
    """
    Args:
        data_set_name (str): Name of the real data set in ./training_data (with a matching *_labels.npy file).
        samples_path (str, optional): Path of generated images (.npy, shape (n_samples, n_pixels)).
        model_name (str, optional): Trained model in ./experiments to sample from instead (in simulation).
        epoch (int, optional): Checkpoint iteration of the model. Defaults to the latest checkpoint.
        n_samples (int, optional): Number of images to sample from the model. Defaults to 1000.
        seed (int, optional): Seed of the train/test split and classifier. Defaults to 0.
        out_path (str, optional): Path of a JSON file to store the results. Defaults to None.
    """
    classifier, real_accuracy, real_class_shares = fit_classifier(data_set_name, seed=seed)
    result = dict(data_set_name=data_set_name, real_test_accuracy=real_accuracy,
                  real_class_shares=real_class_shares.round(4).tolist())

    samples = None
    if samples_path is not None:
        samples = np.load(samples_path)
        result["samples_path"] = samples_path
    elif model_name is not None:
        samples, epoch = _sample_from_model(model_name, n_samples, epoch=epoch)
        result.update(model_name=model_name, epoch=epoch)

    if samples is not None:
        samples = np.asarray(samples).reshape(len(samples), -1)
        result.update(n_samples=len(samples), **class_coverage(classifier, samples, real_class_shares))

    print(json.dumps(result, indent=2))
    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    Fire(main)
