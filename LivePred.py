import pickle
from LangPred import Predictor
from Proccess import extract_from_files , search_files
import os
from tqdm import tqdm
import argparse
import json


path_old =  '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/File type identification - Work sample/FileTypeData' #ENTER PATH TO DATA DIR'  # data dir
lang_json_old = '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/File type identification - Work sample/languages.json'#'ENTER PATH TO LANG JSON'  # languages json path
model_dir_old = '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/File type identification - Work sample/NN_model'  # model output dir

path      = '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/FileTypeData'
lang_json = '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/Work sample/languages.json'#'ENTER PATH TO LANG JSON'  # languages json path
model_dir = '/home/yuv/Downloads/File type identification - Work sample-20220805T123736Z-001/File type identification - Work sample/File type identification - Work sample/Work sample/NN_model'  # model output dir



# path =  'working directory/ input_dir '#ENTER PATH TO DATA DIR'  # data dir
# lang_json = 'working directory/languages.json' #'ENTER PATH TO LANG JSON'  # languages json path
# model_dir = 'working directory/NN_model'   # model output dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Available models are : KNN , RF , XGB , NN ", type=str)

    argv = parser.parse_args()
    classifier = argv.model_name
    test_dir = path

    with open(lang_json) as f:
        languages = json.load(f)

    lang = ['cpp', 'groovy', 'java', 'javascript', 'json', 'python', 'xml', 'yml']

    extensions = [ext for exts in languages.values() for ext in exts]
    files = search_files(test_dir, extensions)
    files = files[22:33]
    pred_list = []
    if classifier != 'NN':
        classifier = pickle.load(open('models/{}.pkl'.format(classifier),'rb'))
        file_feature = extract_from_files(files, languages)
        for item in tqdm(file_feature[0]):
            pred = classifier.predict(item.reshape(1,-1))[0]
            
            pred_list.append(pred)
        # pred_list = [list(languages.values())[i][0] for i in pred_list]
        pred_list = [lang[i] for i in pred_list]
            
    else:
        model = Predictor(model_dir=model_dir, lang_json=lang_json)
        for item in tqdm(files):
            myfile = open(item, encoding='utf-8', mode='r').read()
            pred = model.language(myfile)
            pred_list.append(pred)

    print(pred_list)
    