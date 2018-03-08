"""
Get best performances of given model
"""
import glob
import pickle
import json
import argparse
import numpy as np

def get_model_score(model):
    print("Parsing the results of:", model)
    infos = pickle.load(open('%s/infos-best.pkl' % model, 'rb'), encoding='\"ISO-8859-1')
    val = infos["val_result_history"]
    print("Val:", val)
    Bleus = [v['bleu'] for v in list(val.values())]
    B4 = max(Bleus)
    print("Bleu:", B4)
    score = "<td>%.2f</td>"
    good_score = "<td><b>%.2f</b></td>"
    b4 = good_score % B4 if B4 > 24 else score % B4
    return b4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Load results history:
    table = "<tr>"
    ftable = "\n</tr>\n"
    out = ""
    models = glob.glob('save/*')
    print('Found :', models)
    for model in models:
        b = get_model_score(model)
        out += table + b + ftable
    print(out)
