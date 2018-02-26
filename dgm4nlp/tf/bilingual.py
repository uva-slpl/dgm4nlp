"""
:Authors: - Wilker Aziz
"""
from dgm4nlp.recipes import smart_ropen
from dgm4nlp.nlputils import Tokenizer
from dgm4nlp.nlputils import Multitext
from dgm4nlp.nlputils import read_naacl_alignments
from dgm4nlp.charutils import Multitext3D
import logging


def prepare_training(x_path, y_path,
                     # data pre-processing
                     nb_words=[None, None],
                     shortest_sequence=[None, None],
                     longest_sequence=[None, None],
                     # padding
                     bos_str=[None, None],
                     eos_str=[None, None],
                     # normalisation
                     lowercase=False,
                     batch_dtype='int64',
                     mask_dtype='float32',
                     name='training') -> [list, Multitext]:
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
    training_paths = [x_path, y_path]
    # Prepare vocabularies
    logging.info('Fitting vocabularies')
    tks = []
    for i, (path, vs, bos, eos) in enumerate(zip(training_paths, nb_words, bos_str, eos_str)):
        logging.info(' stream=%d', i)
        # tokenizer with a bounded vocabulary
        tks.append(Tokenizer(nb_words=vs, bos_str=bos, eos_str=eos, lowercase=lowercase))
        tks[-1].fit_one(smart_ropen(path))
        logging.info('  vocab-size=%d', tks[-1].vocab_size())

    # Prepare training corpus
    logging.info('Memory mapping training data')
    training = Multitext(training_paths,
                         tokenizers=tks,
                         shortest=shortest_sequence,
                         longest=longest_sequence,
                         trim=[True, True],
                         batch_dtype=batch_dtype,
                         mask_dtype=mask_dtype,
                         name=name)
    # in case the longest sequence was shorter than we thought
    longest_sequence = [training.longest_sequence(0), training.longest_sequence(1)]
    logging.info(' training-samples=%d longest=%s tokens=%s', training.nb_samples(),
                 longest_sequence, [training.nb_tokens(0), training.nb_tokens(1)])

    return tks, training


def prepare_training3d(
        x_path, y_path,
        # data pre-processing
        nb_chars=[None, None],
        longest_word=[None, None],
        shortest_sequence=[None, None],
        longest_sequence=[None, None],
        # padding
        bos_str=[None, None],
        eos_str=[None, None],
        # normalisation
        lowercase=False,
        batch_dtype='int32',
        mask_dtype='bool',
        name='training') -> [list, Multitext3D]:
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
    training_paths = [x_path, y_path]
    # Prepare vocabularies
    logging.info('Fitting (char) vocabularies')
    tks = []
    for i, (path, vs, bos, eos, longword) in enumerate(zip(training_paths, nb_chars, bos_str, eos_str, longest_word)):
        logging.info(' stream=%d', i)
        # tokenizer with a bounded vocabulary
        tks.append(Tokenizer(nb_words=vs, bos_str=bos, eos_str=eos, lowercase=lowercase, mode='chars', longest_token=longword))
        tks[-1].fit_one(smart_ropen(path))
        logging.info('  (char) vocab-size=%d', tks[-1].vocab_size())

    # Prepare training corpus
    logging.info('Memory mapping (char) training data')
    training = Multitext3D(
        training_paths,
        tokenizers=tks,
        shortest=shortest_sequence,
        longest=longest_sequence,
        trim=[True, True],
        batch_dtype=batch_dtype,
        mask_dtype=mask_dtype,
        name=name)
    # in case the longest sequence was shorter than we thought
    longest_sequence = [training.longest_sequence(0), training.longest_sequence(1)]
    deepest_sequence = [training.deepest_sequence(0), training.deepest_sequence(1)]
    logging.info(' training-samples=%d longest=%s deepest=%s tokens=%s', training.nb_samples(),
                 longest_sequence, deepest_sequence, [training.nb_tokens(0), training.nb_tokens(1)])

    return tks, training


def prepare_validation(tks, x_path, y_path,
                       wa_path=None,
                       shortest_sequence=[None, None],
                       longest_sequence=[None, None],
                       reverse_alignments=False,
                       batch_dtype='int64',
                       mask_dtype='float32',
                       name='validation') -> [Multitext, tuple]:
    """
    Memory-map validation data.

    :param tks:
    :param x_path:
    :param y_path:
    :param wa_path:
    :param shortest_sequence:
    :param longest_sequence:
    :param name:
    :return:
    """

    # Prepare validation corpus
    logging.info('Memory mapping validation data')
    validation = Multitext(
        [x_path, y_path],
        tokenizers=tks,
        shortest=shortest_sequence,
        longest=longest_sequence,
        trim=[True, True],
        batch_dtype=batch_dtype,
        mask_dtype=mask_dtype,
        name=name)
    logging.info(' dev-samples=%d', validation.nb_samples())

    if wa_path:  # we have a NAACL file for alignments
        logging.info("Working with gold labels for validation: '%s'", wa_path)
        # reads in sets of gold alignments
        val_wa = read_naacl_alignments(wa_path, reverse=reverse_alignments)
        # discard those associated with sentences that are no longer part of the validation set
        # (for example due to length constraints)
        val_wa = [a_sets for keep, a_sets in zip(validation.iter_selection_flags(),
                                                 val_wa) if keep]
        logging.info(' gold-samples=%d', len(val_wa))
    else:
        val_wa = None
    return validation, val_wa


def prepare_validation3d(
        tks, x_path, y_path,
        wa_path=None,
        shortest_sequence=[None, None],
        longest_sequence=[None, None],
        reverse_alignments=False,
        batch_dtype='int32',
        mask_dtype='bool',
        name='validation') -> [Multitext3D, tuple]:
    """
    Memory-map validation data.

    :param tks:
    :param x_path:
    :param y_path:
    :param wa_path:
    :param shortest_sequence:
    :param longest_sequence:
    :param name:
    :return:
    """

    # Prepare validation corpus
    logging.info('Memory mapping (char) validation data')
    validation = Multitext3D(
        [x_path, y_path],
        tokenizers=tks,
        shortest=shortest_sequence,
        longest=longest_sequence,
        trim=[True, True],
        batch_dtype=batch_dtype,
        mask_dtype=mask_dtype,
        name=name)
    logging.info(' dev-samples=%d', validation.nb_samples())

    if wa_path:  # we have a NAACL file for alignments
        logging.info("Working with gold labels for validation: '%s'", wa_path)
        # reads in sets of gold alignments
        val_wa = read_naacl_alignments(wa_path, reverse=reverse_alignments)
        # discard those associated with sentences that are no longer part of the validation set
        # (for example due to length constraints)
        val_wa = [a_sets for keep, a_sets in zip(validation.iter_selection_flags(),
                                                 val_wa) if keep]
        logging.info(' gold-samples=%d', len(val_wa))
    else:
        val_wa = None
    return validation, val_wa


def prepare_test(tks, x_path, y_path, wa_path=None, reverse_alignments=False, name='test') -> [Multitext, tuple]:
    """
    Memory-map test data.

    :param tks:
    :param x_path:
    :param y_path:
    :param wa_path:
    :param name:
    :return:
    """

    logging.info('Memory mapping test data')
    test = Multitext([x_path, y_path],
                     tokenizers=tks,
                     shortest=None,
                     longest=None,
                     trim=[True, True],
                     mask_dtype='float32',
                     name=name)
    logging.info(' test-samples=%d', test.nb_samples())
    if wa_path:  # we have a NAACL file for alignments
        logging.info("Working with gold labels for test: '%s'", wa_path)
        # reads in sets of gold alignments
        test_wa = read_naacl_alignments(wa_path, reverse=reverse_alignments)
        logging.info(' test-gold-samples=%d', len(test_wa))
    else:
        test_wa = None

    return test, test_wa


def prepare_test3d(tks, x_path, y_path,
                   wa_path=None, reverse_alignments=False,
                   batch_dtype='int32', mask_dtype='bool', name='test') -> [Multitext3D, tuple]:
    """
    Memory-map test data.

    :param tks:
    :param x_path:
    :param y_path:
    :param wa_path:
    :param name:
    :return:
    """

    logging.info('Memory mapping (char) test data')
    test = Multitext3D(
        [x_path, y_path],
        tokenizers=tks,
        shortest=None,
        longest=None,
        trim=[True, True],
        batch_dtype=batch_dtype,
        mask_dtype=mask_dtype,
        name=name)
    logging.info(' test-samples=%d', test.nb_samples())
    if wa_path:  # we have a NAACL file for alignments
        logging.info("Working with gold labels for test: '%s'", wa_path)
        # reads in sets of gold alignments
        test_wa = read_naacl_alignments(wa_path, reverse=reverse_alignments)
        logging.info(' test-gold-samples=%d', len(test_wa))
    else:
        test_wa = None

    return test, test_wa
