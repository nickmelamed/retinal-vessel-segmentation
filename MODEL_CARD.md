# Model card: retinal vessel segmentation U-Net

This card describes the baseline model as of v0.1.0. Sections for later phases
say TBD until their results exist. Every number is copied from
[results/tables](results/tables), which are generated from reported run
`20260928T013730Z-4f5d08` (tag `v0.1.0-rc.1`).

## Model details

A small U-Net in TensorFlow/Keras that takes a preprocessed fundus photograph
and returns, for every pixel, a probability that it belongs to a blood vessel.
A pixel counts as vessel when its probability is at least the fold's
threshold. The settings are in [configs/baseline.yaml](configs/baseline.yaml).
The network has `depth: 4` levels, `base_filters: 32`, batch normalization, and
`dropout: 0.2`, and it is trained with a combined binary cross-entropy and
Dice loss, masked to the field of view (FOV).

Preprocessing takes the green channel, applies CLAHE, scales to [0, 1], and
standardizes each image within its FOV. Training uses random 64 px patches
with flips, 90° rotations, and small brightness and contrast changes. Whole
images are predicted with overlapping sliding windows.

v0.1.0 has no single model. Cross-validation trained five fold models, each
with its own threshold, and every reported number comes from them. A frozen
final model trained on all 20 labeled images is planned, with its checkpoint
hash recorded before any external data is used.

Author: Nick Melamed. Code license: MIT. The data has its own terms.

## Intended use

Research and learning on a small public dataset.

## Out-of-scope uses

Any clinical use, including screening, triage, or any decision about a
patient. The model has been tested only on DRIVE training images, from one
screening program, and has not been tested on any other camera, population,
or dataset.

## Data

DRIVE (Staal et al., 2004): 40 color fundus photographs from a diabetic
retinopathy screening program in the Netherlands, taken with one camera model
and cropped to 565×584 pixels. The vessel labels of the 20 test images are
withheld by the providers, so only the 20 training images are used. Vessels
cover 0.125 of their FOV pixels. The site lists abnormalities for training
images 25, 26, and 32.

DRIVE does not publish patient identifiers; we treat each image as an independent subject, which we cannot verify. <!-- style: ok -->

## Evaluation design

Five-fold cross-validation over the 20 labeled images, split by whole image.
Each fold tests 4 images, uses 2 to stop training and choose the threshold,
and trains on 14. Images 25, 26, and 32 are in three different test folds.
Every image is tested once, and all metrics are computed on these 20
out-of-fold predictions, inside the FOV. The threshold maximizes mean Dice on
the fold's validation images over the grid i/100. The leakage audit returns
zero rows for the run.

## Metrics

Mean over the 20 out-of-fold images, with percentile bootstrap intervals over
images.

| Metric | Mean | 95% CI of mean | Min | Max |
|---|---|---|---|---|
| Dice | 0.794 | 0.759 to 0.818 | 0.510 | 0.845 |
| Sensitivity | 0.783 | 0.731 to 0.821 | 0.392 | 0.873 |
| Specificity | 0.975 | 0.971 to 0.979 | 0.954 | 0.991 |
| Precision | 0.817 | 0.787 to 0.845 | 0.649 | 0.920 |
| AUC-ROC | 0.970 | 0.952 to 0.980 | 0.812 | 0.987 |
| AUC-PR | 0.886 | 0.849 to 0.909 | 0.585 | 0.929 |
| Accuracy | 0.950 | 0.943 to 0.956 | 0.893 | 0.966 |
| Brier score | 0.039 | 0.034 to 0.046 | 0.027 | 0.097 |

Predicting no vessel anywhere would score a mean accuracy of 0.875 on the same
images, so accuracy says little here, and AUC-ROC is flattered by the many
easy background pixels. Sensitivity is 0.564 on thin vessels and 0.951 on
thick ones, with thin meaning a centerline radius of at most 1.414 px in every
fold. These are cross-validation numbers and are not comparable to results on
DRIVE's official test split.

## External validation

TBD. Zero-shot evaluation of the frozen model on STARE and CHASE_DB1 is
planned.

## Uncertainty

TBD. Per-pixel uncertainty from test-time augmentation is planned.

## Calibration

The pooled Brier score over all out-of-fold FOV pixels is 0.039. The
probabilities are more extreme than the observed frequencies. Pixels in the
top bin average 0.981 but are vessel 0.915 of the time, and pixels averaging
0.143 are vessel 0.263 of the time. See the
[reliability diagram](figures/reliability.png) and
[results/tables/calibration.md](results/tables/calibration.md).

## Failure modes

- Thin vessels are missed far more often than thick ones
  ([figures/thin_thick.png](figures/thin_thick.png)).
- Image 34 scores a Dice of 0.510, well below the rest, with many wide vessels
  missed across a large bright area around the optic disc and a false arc
  along the FOV edge ([figures/best_worst.png](figures/best_worst.png)). It is
  not on DRIVE's list of images with abnormalities.
- On image 23, background texture in the lower half is marked as vessel.
- Image 25, one of the images with abnormalities, has the lowest thin-vessel
  sensitivity of all 20, 0.268, and the second-lowest thick-vessel
  sensitivity, 0.875. Three images support a description only.

## Limitations

Twenty labeled images give wide intervals, and one image can move a mean. The
training labels come from one annotator. Annotators were asked to mark pixels they were at least 70% certain were vessel, so thin-vessel edges are uncertain in the labels themselves. <!-- numbers: ok -->
The providers JPEG compressed every image, which can blur the finest vessels.
Rerun variance with the same seed is TBD.

## Ethical considerations

The data is public, no patient identifiers are published, and its images are
not redistributed here. The model comes from one screening program in one country
and would carry that population's and camera's characteristics into any other
setting. Its main error, missing thin vessels, is the kind that can hide
findings, so it should not be presented as an aid to screening or to any
clinical decision.

## Anomaly module

TBD. An LSTM autoencoder that detects simulated narrowings in vessel width
profiles is planned, and will get its own section here.
