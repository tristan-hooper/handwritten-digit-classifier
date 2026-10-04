# Dataset attribution and references

The experiment uses the 1,797-example dataset bundled with scikit-learn's
`load_digits()`. This is the original UCI dataset's test portion. This project
creates its own training/test split within that portion; it does not reproduce
the original UCI training/test experiment.

Dataset citation: Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of
Handwritten Digits* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C50P49

UCI lists the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The sample images and evaluation figures in `reports/` use or derive from this
dataset. These figures were created by this project's evaluation code. The MIT
license for project code does not replace the dataset's license.

References:

- [UCI dataset and license](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [Scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Scikit-learn digit-classification example](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
- [Scikit-learn guidance on preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html)

The scikit-learn example is a learning reference. This project uses logistic
regression with a training-only scaler, a majority-class baseline, stratified
splitting, exported predictions, and verification checks.
