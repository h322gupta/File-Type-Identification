from LangPred import Predictor
from sklearn import metrics
import os
import warnings
from pathlib import Path
from tqdm import tqdm


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def predict(file, model=None):
    # lang in str
    if model is None:
        model = Predictor(lang_json=lang_json)
    myfile = open(file, encoding='utf-8', mode='r').read()
    lang = model.language(myfile)
    return lang


def suffix(file, languages):
    ext = str(file).split(".")[-1]
    for lang, exts in languages.items():
        if ext in exts:
            return lang
    return None


def prediction_and_report(paths, report_path):
    # ==== Prediction ====
    lang_true = []
    lang_pred = []
    model = Predictor(model_dir=model_dir, lang_json=lang_json)
    for file in tqdm(paths[:]):
        try:
            # print(file)
            temp_true = suffix(file, model.languages)
            if temp_true is None:
                continue
            temp_pred = predict(file, model)
            lang_true.append(temp_true)
            lang_pred.append(temp_pred)
            # print("temp_pred:", temp_pred, "temp_true:", temp_true)
        except Exception as e:
            print(f"Error in prediction for {file}: {e}")
            continue

    # ==== Prediction output ====
    print("Predicted on", str(len(lang_pred)), "files. Results are as follows:")
    print(lang_true)
    print(lang_pred)
    result = metrics.confusion_matrix(lang_true, lang_pred)
    print(result)

    report = metrics.classification_report(lang_true, lang_pred)
    print(report)

    with open(report_path, "w") as resultfile:
        resultfile.write("Predicted on " + str(len(lang_pred)) + " files. Results are as follows:\n\n")
        resultfile.write("Confusion Matrix:\n")
        for row in result:
            string = ""
            for column in row:
                string += str(column) + "\t"
            resultfile.write(string + "\n")
        resultfile.write("\nClassification Report\n")
        resultfile.write(report)


# ==== Initialise Paths ====

path =  'working directory/input_dir'#ENTER PATH TO DATA DIR'  # data dir
lang_json = 'working directory/languages.json'#'ENTER PATH TO LANG JSON'  # languages json path
model_dir =  'working directory/models'# model output dir

os.makedirs(model_dir, exist_ok=True)


# ==== Training ====
predictor = Predictor(model_dir=model_dir, lang_json=lang_json)
train_paths, test_paths = predictor.learn(path)


# ==== Testing ====
print("Predicting on test")
prediction_and_report(test_paths, "result_test_final.txt")
print("Predicting on train")
prediction_and_report(train_paths, "result_train21.txt")
