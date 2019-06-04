from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = Model()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels


        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        if self.bibiFlag:
            self.builders = [VanillaLSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [VanillaLSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             VanillaLSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidBias = self.model.add_parameters((self.hidden_units))

            self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

            self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = self.model.add_parameters((len(self.irels)))


    def  __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())) # + self.outBias
        else:
            #print((sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value())
            #print("len",len((sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value()))
            #nonlayer=(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value()
            #self.nonlayer.append(nonlayer)
            output = self.outLayer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()) # + self.outBias
            #print("output:")
            #print(output.value())

        return output

    def  __getExpr_self2(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())) # + self.outBias
        else:
            #print((sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value())
            #print("len",len((sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value()))
            nonlayer=(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()).value()
            self.nonlayer.append(nonlayer)
            output = self.outLayer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()) # + self.outBias
            #print("output:")
            #print(output.value())

        return output

    def __evaluate_self(self, sentence, train):
        self.nonlayer=[]
        self.output_avg_single=None
        exprs = [ [self.__getExpr_self2(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
        nonlayer_new=self.nonlayer
        return scores, exprs,nonlayer_new

    def __evaluate(self, sentence, train):
        self.output_avg_single=None
        exprs = [ [self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
        return scores, exprs


    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output.value(), output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)))] if self.external_embedding is not None else None
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                heads = decoder.parse_proj(scores)

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                dump = False

                if self.labelsFlag:
                    for modifier, head in enumerate(heads[1:]):
                        scores, exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence



    def get_data_as_indices(self,conll_path):
        with open(conll_path, 'r') as conllFP:
            Data = list(read_conll(conllFP))

        return Data


    def get_repr_self(self,test_X):
        padding_size = 225
        
        save_output_avg = np.zeros((len(test_X), padding_size*100))
        iii=0
        for iSentence, sentence in enumerate(test_X):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
            for entry in conll_sentence:
                try:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm,
                                                                                    0)))] if self.external_embedding is not None else None
                except KeyError:
				    print("KeyError again xxxxxxx")
                entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            if self.blstmFlag:
                lstm_forward = self.builders[0].initial_state()
                lstm_backward = self.builders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = lstm_forward.output()
                    rentry.lstms[0] = lstm_backward.output()

                if self.bibiFlag:
                    for entry in conll_sentence:
                        entry.vec = concatenate(entry.lstms)

                    blstm_forward = self.bbuilders[0].initial_state()
                    blstm_backward = self.bbuilders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.vec)
                        blstm_backward = blstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = blstm_forward.output()
                        rentry.lstms[0] = blstm_backward.output()

            scores, exprs,nonlayer_new= self.__evaluate_self(conll_sentence, True)
            #nonlayer_new_np=np.array(nonlayer_new)
            #output_avg = np.mean(nonlayer_new_np, axis=0)
            #save_output_avg[iii] = output_avg
            list_len=len(nonlayer_new[0])
            padding_list=[]
            for padding_i in range(padding_size):
                padding_list.append([0]*list_len)
            for padding_i in range(len(nonlayer_new)):
                if padding_i>=padding_size:
                   break
                padding_list[padding_i]=nonlayer_new[padding_i]
            padding_arrry=np.array(padding_list)
            padding_arrry_reshape=padding_arrry.reshape((padding_size*100,))
            save_output_avg[iii]=padding_arrry_reshape
            iii+=1

        return save_output_avg




    def Train(self, conll_path):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            eeloss = 0.0

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = None

                    if self.external_embedding is not None:
                        evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    for modifier, head in enumerate(gold[1:]):
                        rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                        wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                        if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                            lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        #self.trainer.update_epoch()
        rate_decay=0.01
        self.trainer.learning_rate /= (1 + rate_decay)
        print "Loss: ", mloss/iSentence


    def Train_self(self, train_x):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        shuffledData =train_x
        random.shuffle(shuffledData)

        errs = []
        lerrs = []
        eeloss = 0.0

        for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                    posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                    evec = None

                    if self.external_embedding is not None:
                        evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                    entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    for modifier, head in enumerate(gold[1:]):
                        rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                        wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                        if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                            lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        #self.trainer.update_epoch()
        rate_decay=0.01
        self.trainer.learning_rate /= (1 + rate_decay)
        print "Loss: ", mloss/iSentence


    def get_avg_nonlayer(self,data1,data2):
        non_layer_sum_data1=self.get_repr_self(data1)
        non_layer_avg_data1=np.mean(non_layer_sum_data1, axis=0)
        #print("non_layer_avg_data1.shape",non_layer_avg_data1.shape,"non_layer_avg_data1",non_layer_avg_data1)
        non_layer_sum_data2=self.get_repr_self(data2)
        non_layer_avg_data2=np.mean(non_layer_sum_data2, axis=0)
        distance = (np.sqrt(np.sum(np.square(non_layer_avg_data1-non_layer_avg_data2))))
        #print("non_layer_avg_data2.shape",non_layer_avg_data2.shape,"non_layer_avg_data2",non_layer_avg_data2)
        return distance


    def predict_self(self, X_test, y_test, X_test_, y_test_, X_dev):
        self.GAMMA=0.99
        #total_loss1=self.Train_self(X_test)
        distance_f1 = self.get_avg_nonlayer(X_test, X_dev)
        total_loss2=self.Train_self(X_test_)
        distance_f2 = self.get_avg_nonlayer(X_test_, X_dev)
        self.td_loss = - distance_f1 - self.GAMMA * distance_f2
        total_loss1 = total_loss2 #2018-9-20[Liu] total_loss1 is useless, just for keeping the interface 
        return self.td_loss,total_loss1,total_loss2


