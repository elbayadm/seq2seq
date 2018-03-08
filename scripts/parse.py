def get_wl(params):
    """
    Word loss settings
    """
    if 'alpha_word' in params:
        alpha = params['alpha_word']
    else:
        alpha = params['alpha']
    if 'cooc' in params.get('similarity_matrix', ''):
        G = "Coocurences"
    elif 'train_coco' in params.get('similarity_matrix', ''):
        G = "Glove-Coco"
    else:
        G = "Glove-Wiki"
    if params.get('rare_tfidf', 0):
        G += ' xIDF(%.1f)' % params.get('rare_tfidf')
    if params.get('word_add_entropy', 0):
        G += ' +H'
    if params.get('exact_dkl', 0):
        G += ' +ExDKL'
    modelparams = ' Word, Sim=%s, $\\tau=%.2f$, $\\alpha=%.1f$' % (G, params['tau_word'], alpha)
    return modelparams


def parse_name_clean(params):
    modelparams = []
    if not params['batch_size'] == 80:
        modelparams.append("Batch=%d" % params['batch_size'])

    # Get the loss:
    if "stratify_reward" in params:
        loss_version = parse_loss(params)
    else:
        loss_version = parse_loss_old(params)
    if params.get('init_decoder_W', ""):
        wdec = ", $W_{dec}=Glove_{coco, 512}$"
        if params.get('freeze_decoder_W', 0):
            wdec += " frozen"
        loss_version += wdec
    if len(modelparams):
        modelparams = ' '.join(modelparams)
    else:
        modelparams = 'Default'
    return modelparams, loss_version


def parse_loss(params):
    combine_loss = params.get('combine_loss', 0)
    loss_version = params['loss_version'].lower()
    if loss_version == "ml":
        loss_version ='ML'
    elif loss_version == "word":
        loss_version = get_wl(params)
    elif loss_version == "seq":
        reward = params['reward']
        if reward == "tfidf":
            reward = 'TFIDF, n=%d, rare=%d' % (params['ngram_length'],
                                               params['rare_tfidf'])
        elif reward == 'hamming':
            reward = 'Hamming, Vpool=%d' % (params['limited_vocab_sub'])
        elif 'bleu' in reward:
            reward = '%s, mode=%d' %(reward, params.get('refs_mode', 1))

        if not params.get('clip_reward', 1) == 1:
            reward += ', clip@%.1f' % params['clip_reward']
        reward += ', $\\tau=%.2f$' % params['tau_sent']

        if params['stratify_reward']:
            loss_version = 'Stratify r=(%s), $\\alpha=%.1f$' % (reward,
                                                                params['alpha_sent'])
            if params.get('lazy_rnn', 0):
                loss_version += ' (LAZY)'

        else:
            sampler = params['importance_sampler']
            tau_q = params.get('tau_sent_q', params['tau_sent'])
            if sampler == "tfidf":
                sampler = 'TFIDF, n=%d, rare=%d $\\tau=%.2f$' % (params['ngram_length'],
                                                                 params['rare_tfidf'],
                                                                 tau_q)
            elif sampler == 'hamming':
                sampler = 'Hamming, Vpool=%d $\\tau=%.2f$' % (params['limited_vocab_sub'],
                                                              tau_q)
            elif sampler == 'greedy':
                sampler = '$p_\\theta$'

            extra = params.get('lazy_rnn', 0) * " (LAZY)"
            loss_version = 'Importance r=(%s), q=(%s),$\\alpha=%.1f$ %s' % (reward,
                                                                            sampler,
                                                                            params['alpha_sent'],
                                                                            extra)
        if combine_loss:
            loss_version = 'Combining Word \& ' + loss_version
    if params.get('penalize_confidence', 0):
        loss_version += ", Penalize(%.2f)" % params['penalize_confidence']
    return loss_version



def parse_loss_old(params):
    sample_cap = params.get('sample_cap', 0)
    sample_reward = params.get('sample_reward', 0)
    alter_loss = params.get('alter_loss', 0)
    sum_loss = params.get('sum_loss', 0)
    combine_loss = params.get('combine_loss', 0)
    multi = alter_loss + sum_loss + combine_loss
    loss_version = params['loss_version']
    if "alpha" in params:
        alpha = (params['alpha'], params['alpha'])
    else:
        alpha = (params['alpha_sent'], params['alpha_word'])
    tau_sent = params['tau_sent']
    tau_word = params['tau_word']
    rare = params.get('rare_tfidf', 0)
    sub = params.get('sub_idf', 0)
    if 'tfidf' in loss_version:
        loss_version = "TFIDF, n=%d, idf_select=%d, idf_sub=%d" % (params.get('ngram_length', 0), rare, sub)
    elif 'hamming' in loss_version:
        loss_version = "Hamming, Vpool=%d" % params.get('limited_vocab_sub', 1)
    if not multi:
        if loss_version == "word2":
            loss_version = get_wl(params)
        elif sample_cap:
            if loss_version == "dummy":
                loss_version = "constant"
            ver = params.get('sentence_loss_version', 1)
            loss_version = ' SampleP, r=%s V%d, $\\tau=%.2f, \\alpha=%.1f$' % (loss_version, ver, tau_sent, alpha[0])
        elif sample_reward:
            mc = params.get('mc_samples', 1)
            loss_version = ' Stratify r=(%s, $\\tau=%.2f), \\alpha=%.1f$' % (loss_version, tau_sent,  alpha[0])
        else:
            # print('Model: %s - assuming baseline loss' % params['modelparams'])
            # modelparams = " ".join(params['modelparams'].split('_'))
            loss_version = " ML"
        if params.get('penalize_confidence', 0):
            loss_version += " Peanlize: %.2f" % params['penalize_confidence']

    else:
        wl = get_wl(params)
        if alter_loss:
            loss_version = " Alternating losses, %s  w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$, $\\gamma=%.1f$ (mode:%s)" \
                           % (wl, loss_version, tau_sent, alpha[0], params.get('gamma', 0), params.get('alter_mode', 'iter'))
        elif sum_loss:
            loss_version = " Sum losses, %s w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$, $\\gamma=%.1f$" \
                           % (wl, loss_version, tau_sent, alpha[0], params.get('gamma', 0))
        elif combine_loss:
            loss_version = " Combining losses, %s w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$" \
                            % (wl, loss_version, tau_sent, alpha[0])
    return loss_version


def parse_name_short(params, verbose=0):
    """
    Return Loss, Reward, Vsub
    """
    if 'stratify_reward' in params:
        if params['loss_version'] == "ml":
            loss = 'ML'
            reward = ""
            vsub = ""
        elif params['loss_version'] == "word":
            loss = 'tok'
            reward = 'Glove sim'
            if params['rare_tfidf']:
                reward += " xIDF"
            vsub = ''
        elif params['loss_version'] == "seq":
            loss = 'Seq'
            vsub = "Vsub%d" % params['limited_vocab_sub']
            reward = params['reward']
            if params['combine_loss']:
                loss = 'Tok-Seq'
    else:
        if params['loss_version'] == "hamming":
            loss = 'seq'
            reward = 'Hamming'
            vsub = 'Vsub%d' % (params.get('limited_vocab_sub', 1))
        if "baseline" in params['modelname']:
            loss = 'ML'
            reward = ""
            vsub = ""
    if verbose:
        loss = params['modelname']
    return loss, reward, vsub



