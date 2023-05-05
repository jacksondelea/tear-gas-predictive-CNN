import glob
import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import csv

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

import glob
from skimage.io import imread
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')
    parser.add_argument(
        '--frames-folder',
        default='path/to/deduplicated_frames',
        help='Path to the folder containing deduplicated frames from YouTube videos.')
    parser.add_argument(
        '--output-csv',
        default='confidence_ratings.csv',
        help='Path to the output CSV file for storing confidence ratings of tear gas canister presence.')

    return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def classify_video_frames(model, frames_folder, output_csv):
    frame_files = os.listdir(frames_folder)

    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Tear_gas_canister_confidence'])

        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)

            # Preprocess the frame
            preprocessed_frame = preprocess_frame(frame_path)

            # Expand dimensions to match the input shape for the model
            input_frame = np.expand_dims(preprocessed_frame, axis=0)

            # Classify the preprocessed frame
            confidence = model.predict(input_frame)[0][0]

            # Output the confidence rating
            print(f"Frame {frame_file}: Tear gas canister presence confidence = {confidence:.2f}")

            # Write the results to the CSV file
            csv_writer.writerow([frame_file, f"{confidence:.2f}"])



def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.task)

    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
    checkpoint_path = "checkpoints" + os.sep + "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + os.sep + timestamp + os.sep

    model.summary()

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["binary_accuracy"])

    if ARGS.evaluate:
        #frames_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data') check directory?
        test(model, datasets.test_data)

        path = ARGS.lime_image
        LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)

    # Preprocess and classify the deduplicated frames
    classify_video_frames(model, ARGS.frames_folder, ARGS.output_csv)

# Make arguments global
ARGS = parse_args()

main()

