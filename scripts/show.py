import sys
import os.path as osp
import socket
import glob
import operator
import pickle
import argparse
from collections import OrderedDict
from math import exp
from prettytable import PrettyTable
from html import escape
from parse import *

FIELDS = ["Corpus", 'Loss', 'Reward', 'Vsub', 'Beam', 'Bleu', 'Perplexity', 'best/last']
PAPER_FIELDS = ['Loss', 'Reward', 'Vsub', 'Beam', 'Bleu', 'Perplexity']
PAPER_FIELDS_SELECT = PAPER_FIELDS


def is_required(model, fltr, exclude):
    for fl in fltr:
        if fl not in model:
            return 0
    for exc in exclude:
        if exc in model:
            return 0
    return 1


def correct(word):
    """
    Printable names for key options
    """
    if word == "show_tell":
        return 'Show \\& Tell'
    elif word == 'top_down':
        return "Top-down"
    elif word == "resnet50":
        return "ResNet-50"
    elif word == "resnet152":
        return "ResNet-152"
    elif "cnn" in word:
        return "RNN + CNN"
    else:
        return word


def get_latex(ptab, **kwargs):
    """
    Print prettytable into latex table
    """
    options = ptab._get_options(kwargs)
    lines = []
    rows = ptab._get_rows(options)
    formatted_rows = ptab._format_rows(rows, options)
    aligns = []
    fields = []
    for field in ptab._field_names:
        if options["fields"] and field in options["fields"]:
            aligns.append(ptab._align[field])
            fields.append(field)
    lines = ['|' + '|'.join(['%s' % a for a in aligns]) + '|']
    lines.append('\midrule')
    lines.append(' & '.join(fields) + '\\\\')
    lines.append('\midrule')
    for row in formatted_rows:
        line = []
        for field, datum in zip(ptab._field_names, row):
            if field in fields:
                line.append(correct(datum))
        lines.append(' & '.join(line) + '\\\\')
    lines.append('\midrule')
    return lines


def get_perf(res):
    formatted_res = OrderedDict()
    for k in ['Bleu']:
        if k in res:
            formatted_res[k] = float(res[k])
        else:
            formatted_res[k] = 0
    out = list(formatted_res.values())
    out.append(float(exp(res['ml_loss'])))
    return out


def get_results(model, split='val', verbose=False):
    model_dir = model
    if split == "val":
        compiled = []
        # Read training results:
        if osp.exists('%s/infos.pkl' % model_dir):
            infos = pickle.load(open('%s/infos.pkl' % model_dir, 'rb'))
            Res = infos['val_result_history']
            LL = infos['loss_history']
            iters = list(Res)
            bleus = [Res[it]['bleu'] for it in iters]
            best_iter = iters[bleus.index(max(bleus))]
            last_iter = max(iters)
            out = {}
            out['best/last'] = '%dk / %dk' % (best_iter/1000, last_iter/1000)
            out['Bleu'] = Res[best_iter]['bleu']
            out['ml_loss'] = LL[best_iter]
            params = vars(infos['opt'])
            params['beam_size'] = 3
            compiled = [[params, out]]
        else:
            if verbose:
                print('infos not found in %s' % model_dir)

    elif split == "test":
        # Read post-results
        results = sorted(glob.glob('%s/evaluations/test/*.res' % model_dir))
        params = {}
        compiled = []
        for res in results:
            out = pickle.load(open(res, 'rb'))
            params = out['params']
            del out['params']
            out['best/last'] = "--"
            compiled.append([params, out])
    else:
        raise ValueError('Unknown split %s' % split)

    return compiled


def crawl_results_paper(fltr=[], exclude=[], split="test", verbose=False, beam=-1):
    models = glob.glob('save/*')
    models = [model for model in models if is_required(model, fltr, exclude)]
    tab = PrettyTable()
    tab.field_names = PAPER_FIELDS
    for model in models:
        outputs = get_results(model, split, verbose)
        if len(outputs):
            # print('len(out):', len(outputs))
            if verbose:
                print(model.split('/')[-1])
            for params, res in outputs:
                # params, res = outputs[0]
                data, loss, reward, vsub = parse_name_short(params, verbose)
                perf = get_perf(res)
                if beam != -1:
                    if params['beam_size'] == beam:
                        row = [loss, reward, vsub,
                               params["beam_size"]]
                        row += perf
                        tab.add_row(row)
                else:
                    row = [loss, reward, vsub,
                           params["beam_size"]]
                    row += perf
                    tab.add_row(row)
    return tab


def crawl_results(fltr='', exclude=None, split="val", dir="", save_pkl=False, verbose=False):
    models = glob.glob('save/%s*' % dir)
    models = [model for model in models if is_required(model, fltr, exclude)]
    recap = {}
    tab = PrettyTable()
    tab.field_names = FIELDS
    dump = []
    for model in models:
        outputs = get_results(model, split, verbose)
        if len(outputs):
            if verbose:
                print(model.split('/')[-1])
            # _, loss_version = parse_name_clean(outputs[0][0])
            if save_pkl:
                dump.append(outputs)
            for (p, res) in outputs:
                corpus, loss, reward, vsub = parse_name_short(p, verbose)
                bleu = float(res['Bleu'])
                try:
                    recap[p['alpha']] = bleu
                except:
                    recap[p['alpha_sent']] = bleu
                    recap[p['alpha_word']] = bleu
                try:
                    perpl = float(exp(res['ml_loss']))
                except:
                    perpl = 1.
                row = [corpus, loss, reward, vsub,
                       p['beam_size'],
                       bleu, perpl, res['best/last']]
                tab.add_row(row)
    return tab, dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', '-f', nargs="+", help='keyword to include')
    parser.add_argument('--exclude', '-e', nargs="+", help='keyword to exculde')
    # parser.add_argument('--tex', '-t', action='store_true', help="save results into latex table")
    parser.add_argument('--paper', '-p', action='store_true', help="run paper mode")
    parser.add_argument('--html', action='store_true', help="save results into html")
    parser.add_argument('--pkl', action='store_true', help="save results into pkl")
    parser.add_argument('--split', type=str, default="val", help="split on which to report")
    parser.add_argument('--dir', type=str, default="", help="directory of results")
    parser.add_argument('--sort', type=str, default="Bleu", help="criteria by which to order the terminal printed table")
    parser.add_argument('--verbose', '-v', action="store_true", help="script verbosity")
    parser.add_argument('--beam', '-b', type=int, default=5, help="beam reported")
    args = parser.parse_args()

    split = args.split
    verbose = args.verbose
    fltr = args.filter
    if not fltr:
        fltr = []
    exc = args.exclude
    if not exc:
        exc = []
    if args.verbose:
        print(vars(args))
    fltr_concat = "_".join(fltr)
    if not fltr_concat:
        fltr_concat = ''
    else:
        fltr_concat = '_' + fltr_concat
    if args.paper:
        print('Setting up split=test')
        split = 'test'
        filename = "results/%s%s_%s" % (split, fltr_concat, socket.gethostname())
        if not args.beam == -1:
            filename += '_bw%d' % args.beam
        tab = crawl_results_paper(fltr, exc, split, verbose, args.beam)
        print(tab.get_string(sortby=args.sort, reversesort=True, fields=PAPER_FIELDS_SELECT))
        print('saving latex table in %s.tex' % filename)
        with open(filename+'.tex', 'w') as f:
            tex = get_latex(tab, sortby="Loss",
                            reversesort=False, fields=PAPER_FIELDS_SELECT)
            f.write("\n".join(tex))
    else:
        tab, dump = crawl_results(fltr, exc, split, args.dir,
                                  args.pkl, verbose)
        print(tab.get_string(sortby='Bleu', reversesort=True))
        filename = "results/%s%s_%s" % (split, fltr_concat, socket.gethostname())
        if args.pkl:
            pickle.dump(dump, open(filename+".res", 'wb'))
        if args.html:
            with open(filename+'.html', 'w') as f:
                ss = tab.get_html_string(sortby="Bleu", reversesort=True)
                f.write(ss)

