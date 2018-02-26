"""
:Authors: - Wilker Aziz
"""
from dgm4nlp.recipes import smart_ropen
from dgm4nlp.nlputils import Tokenizer
from dgm4nlp.nlputils import Text
import logging


def prepare_training(x_path,
                     # data pre-processing
                     nb_words=None,
                     shortest_sequence=None,
                     longest_sequence=None,
                     # padding
                     bos_str=None,
                     eos_str=None,
                     # normalisation
                     lowercase=False,
                     name='training') -> [Tokenizer, Text]:
    """
    Construct vocabularies/tokenizers and memory-map the training data.

    :param x_path:
    :param y_path:
    :param nb_words:
    :param shortest_sequence:
    :param longest_sequence:
    :param bos_str:
    :param eos_str:
    :param name:
    :return:
    """
    # Prepare vocabularies
    logging.info('Fitting vocabulary')
    tk = Tokenizer(nb_words=nb_words, bos_str=bos_str, eos_str=eos_str, lowercase=lowercase)
    tk.fit_one(smart_ropen(x_path))
    logging.info('  vocab-size=%d', tk.vocab_size())

    # Prepare training corpus
    logging.info('Memory mapping training data')
    training = Text(x_path,
                    tokenizer=tk,
                    shortest=shortest_sequence,
                    longest=longest_sequence,
                    trim=True,
                    mask_dtype='float32',
                    name=name)
    # in case the longest sequence was shorter than we thought
    longest_sequence = training.longest_sequence()
    logging.info(' training-samples=%d longest=%s tokens=%s', training.nb_samples(),
                 longest_sequence, training.nb_tokens())

    return tk, training


def prepare_validation(tk: Tokenizer,
                       x_path,
                       # data pre-processing
                       shortest_sequence=None,
                       longest_sequence=None,
                       name='validation') -> Text:
    """
    Construct vocabularies/tokenizers and memory-map the training data.

    :param tk: trained Tokenizer
    :param x_path:
    :param shortest_sequence:
    :param longest_sequence:
    :param name:
    :return:
    """
    # Prepare training corpus
    logging.info('Memory mapping validation data')
    val = Text(x_path,
               tokenizer=tk,
               shortest=shortest_sequence,
               longest=longest_sequence,
               trim=True,
               mask_dtype='float32',
               name=name)
    # in case the longest sequence was shorter than we thought
    longest_sequence = val.longest_sequence()
    logging.info(' validation-samples=%d longest=%s tokens=%s', val.nb_samples(),
                 longest_sequence, val.nb_tokens())

    return val


def prepare_test(tk: Tokenizer, x_path, name='test') -> Text:
    """
    Construct vocabularies/tokenizers and memory-map the training data.

    :param tk: trained Tokenizer
    :param x_path:
    :param name:
    :return:
    """
    # Prepare training corpus
    logging.info('Memory mapping test data')
    test = Text(x_path,
                tokenizer=tk,
                shortest=None,
                longest=None,
                trim=True,
                mask_dtype='float32',
                name=name)
    # in case the longest sequence was shorter than we thought
    longest_sequence = test.longest_sequence()
    logging.info(' test-samples=%d longest=%s tokens=%s', test.nb_samples(),
                 longest_sequence, test.nb_tokens())

    return test


