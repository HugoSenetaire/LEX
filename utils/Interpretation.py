import torch
import matplotlib.pyplot as plt

from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt

import os
from datetime import datetime
from collections.abc import Iterable
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

import itertools
import json
import re

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state
import string




### TEXT INTERPRETATION :

def id_generator(size=15, random_state=None):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipytho n notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))



class TextExplainMapper():
    """Maps feature ids to words or word-positions"""

    def __init__(self, raw_string, vectorizer, class_names, prediction_output = [1.0,0.]):
        """Initializer.
        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        # self.indexed_string = indexed_string
        self.indexed_string = IndexedString(raw_string)
        self.vectorizer = vectorizer
        self.class_names = class_names
        self.prediction_output = prediction_output


    def explain(self, explaination_score):
        self.local_exp = []
        for i, word in enumerate(self.indexed_string.inverse_vocab) :
            print(word)
            try :
                index_nn = self.vectorizer.vocabulary_[word]
            except(KeyError):
                continue
            score_nn = explaination_score[index_nn]
            self.local_exp.append((i, score_nn))





    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.
        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions
        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.indexed_string.word(x[0]),
                '-'.join(
                    map(str,
                        self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp



    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.
        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label
        ans = self.map_exp_ids(self.local_exp, **kwargs)
        ans = [(x[0], float(x[1])) for x in ans]
        return ans


    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.
        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)

        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]) for x in exp]
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        print(all_occurrences[0])
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret



    def save_to_file(self,
                     file_path,
                     labels=None,
                     predict_proba=True,
                     show_predicted_value=True,
                     **kwargs):
        """Saves html explanation to file. .
        Params:
            file_path: file to save explanations to
        See as_html() for additional parameters.
        """
        file_ = open(file_path, 'w', encoding='utf8')
        file_.write(self.as_html(labels=labels,
                                 predict_proba=predict_proba,
                                 show_predicted_value=show_predicted_value,
                                 **kwargs))
        file_.close()


    def as_html(self,
                labels=None,
                predict_proba=True,
                show_predicted_value=True,
                **kwargs):
        """Returns the explanation as an html page.
        Args:
            labels: desired labels to show explanations for (as barcharts).
                If you ask for a label for which an explanation wasn't
                computed, will throw an exception. If None, will show
                explanations for all available labels. (only used for classification)
            predict_proba: if true, add  barchart with prediction probabilities
                for the top classes. (only used for classification)
            show_predicted_value: if true, add  barchart with expected value
                (only used for regression)
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            code for an html page, including javascript includes.
        """

        def jsonize(x):
            return json.dumps(x, ensure_ascii=False)

        # if labels is None and self.mode == "classification":
        #     labels = self.available_labels()

        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(os.path.join(this_dir,"lime"), 'bundle.js'),
                      encoding="utf8").read()

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        # random_id = id_generator(size=15, random_state=0)
        random_id = 0
        out += u'''
        <div class="lime top_div" id="top_div%s"></div>
        ''' % random_id

        label = 0
        # Encart droite
        predict_proba_js = ''
        if predict_proba and self.prediction_output is not None:
            predict_proba_js = u'''
            var pp_div = top_div.append('div')
                                .classed('lime predict_proba', true);
            var pp_svg = pp_div.append('svg').style('width', '100%%');
            var pp = new lime.PredictProba(pp_svg, %s, %s);
            ''' % (jsonize([str(x) for x in self.class_names]), # Class names output
                   jsonize(list(self.prediction_output.astype(float)))) # Score prediction output

        predict_value_js = ''

        # Encart middle

        exp_js = '''var exp_div;
            var exp = new lime.Explanation(%s);
        ''' % (jsonize([str(x) for x in self.class_names]))



        exp = jsonize(self.as_list()) # Here, should give (word, weights)
        exp_js += u'''
        exp_div = top_div.append('div').classed('lime explanation', true);
        exp.show(%s, %d, exp_div);
        ''' % (exp, label)


        raw_js = '''var raw_div = top_div.append('div');'''

        html_data = self.local_exp

        raw_js += self.visualize_instance_html(
                html_data,
                0,
                'raw_div',
                'exp',
                **kwargs)
        out += u'''
        <script>
        var top_div = d3.select('#top_div%s').classed('lime top_div', true);
        %s
        %s
        %s
        %s
        </script>
        ''' % (random_id, predict_proba_js, predict_value_js, exp_js, raw_js)
        out += u'</body></html>'

        return out


class IndexedString(object):
    """String with various indexes."""

    def __init__(self, raw_string, split_expression=r'\W+', bow=True,
                 mask_string=None):
        """Initializer.
        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.
        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.
        Args:
            words_to_remove: list of ids (ints) to remove
        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]