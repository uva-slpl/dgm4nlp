"""
Here I adapt Text and Multitext to output 3D tensors of the kind
    [batch_size, max_steps, max_depth]
 where I use
    - the first dimension to index sentences,
    - the second dimension to index tokens,
    - and the third dimension to index characters.


:Authors: - Wilker Aziz
"""
import numpy as np
import os
import tempfile
import itertools
from dgm4nlp.recipes import smart_ropen
import dgm4nlp.nlputils as nlputils


def bound_length(input_paths, tokenizers, shortest, longest):
    """
    Return an np.array which flags whether all parallel segments comply with length constraints
    and count the number of tokens in each stream (considering valid sequences only).

    :param input_paths: paths (list/tuple) to each part of the parallel collection
    :param tokenizers: list/tuple of tokenizers
    :param shortest: shortest valid sequence for each part of the parallel collection
    :param longest: longest valid sequence for each part of the parallel collection
    :return: selection [nb_samples] and counts [nb_streams, nb_samples, 2]
        where counts[:,:,0] counts tokens
         and counts[:, :, 1] counts chars
    """

    # get an iterator for each stream
    nb_streams = len(input_paths)
    iterators = [tokenizers[i].to_sequences_iterator(smart_ropen(input_paths[i])) for i in range(nb_streams)]

    # length checks
    selection = []
    shapes = [[] for _ in range(nb_streams)]
    for seqs in zip(*iterators):  # get a sequence from each iterator
        # check if every sequence complies with its respective length bounds
        if not all(lower <= seq.shape[0] <= upper for lower, upper, seq in zip(shortest, longest, seqs)):
            selection.append(False)  # excluded
        else:
            selection.append(True)  # included
            # increase token count
            for i, seq in enumerate(seqs):
                shapes[i].append(seq.shape)

    return np.array(selection, dtype=bool), np.array(shapes)


def construct_mmap(input_path, output_path, tokenizer, selection, mem_units, dtype):
    """
    Construct memory map for selected sentences in a corpus.

    :param input_path: path to text
    :param output_path: path to memory map file
    :param tokenizer: tokenizer for text
    :param selection: array of binary selectors
    :param mem_units: number of memory units in the selected corpus
    :param dtype: data type for memmap
    :return: np.array with shapes where array[i] is the shape information of the ith sequence
    """

    # construct memory mapped array
    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=mem_units)

    # prepare for populating memmap
    offset = 0
    shapes = []

    # populate memory map
    for sid, seq in enumerate(tokenizer.to_sequences_iterator(smart_ropen(input_path))):
        if not selection[sid]:  # skip sentences that do not comply with length constraints
            continue
        # here we have a valid sequence, thus we memory map it
        mmap[offset:offset + seq.size] = seq.flatten()
        offset += seq.size
        shapes.append(seq.shape)

    del mmap

    return np.array(shapes, dtype='int64')


class Text3D:
    """
    This class is used to represent large text collections as a matrix of integers.

    It uses a pre-trained Tokenizer and it can impose a limit on sentence length.
    It uses memory mapped files for memory efficiency,
     and it provides a generator for batches of a given size. This generator may iterate once through the data
     or indefinitely in an endless cycle.

    TODO: reload memmap when possible (I find this a bit dangerous though since several options affect its structure)

    """

    MASK = 0
    TRIM = 1
    COMPLETE = 2
    DISCARD = 3
    STRATEGY_MAP = {'mask': MASK, 'trim': TRIM, 'complete': COMPLETE, 'discard': DISCARD}

    def __init__(self, input_path, tokenizer: nlputils.Tokenizer,
                 shortest=1,
                 longest=np.inf,
                 trim=False,
                 output_dir=None,
                 tmp_dir=None,
                 batch_dtype='int64',
                 mask_dtype='float32',
                 name='text',
                 _selection=None,
                 _mem_units=None):
        """
        Wrap a corpus for string->integer conversion.

        An object of this class cleans up after itself: randomly generated files created by this class
            are removed on destruction. Note that, if a user specifies output_dir,
            then the the memory map will be deleted.

        :param input_path: path to a file containing the raw text
        :param tokenizer: a Tokenizer to turn text sequences into integer sequences
        :param shortest: the length of the shortest valid sequence (defaults to 1 which is also the minimum)
        :param longest: the length of the longest valid sentence (defaults to inf)
        :param trim: trim batches to the longest sentence in the corpus (defaults to False)
            but longest=np.inf causes trim to be overwritten to True
        :param output_dir: where to store the memory map (defaults to None in which case tmp_dir will be used)
        :param tmp_dir: a temporary directory used in case output_dir is None (defaults to None in which case a
            the system's tmp space will be used)
        :param batch_dtype: data type for batches
        :param mask_dtype: data type for masks
        :param name: name of the corpus (file will end in .dat)
            * if the memory map lives in output_dir then its file name will be '{}.dat'.format(name)
            * if the memory map lives in temp_dir then its file name will be obtained with
                tempfile.mkstemp(prefix=name, suffix='.dat', dir=tmp_dir, text=False)
            in this case, the file will be deleted when the Text object gets destructed
        :param _selection: uses a subset of the data specified through a np.array with a binary selector per sample
        :param _mem_units: total number of memory units in the selection
            selection and mem_units are used when multiple texts are simultaneously constrained for length
            users probably would never need to specify these variables by hand
        """
        assert shortest > 0, '0-length sequences are not such a great idea'
        if longest == np.inf:  # overwrites trim when longest is np.inf
            trim = True

        self._input_path = input_path
        self._tokenizer = tokenizer
        self._batch_dtype = batch_dtype
        self._mask_dtype = mask_dtype
        self._to_remove = {}

        # create a file to store the corpus
        if output_dir is None:
            if tmp_dir:
                tmp_dir = os.path.abspath(tmp_dir)  # make it absolute
                os.makedirs(tmp_dir, exist_ok=True)  # make sure it exists
                _, memmap_path = tempfile.mkstemp(prefix=name, suffix='.dat', dir=tmp_dir, text=False)
            else:
                _, memmap_path = tempfile.mkstemp(prefix=name, dir=tmp_dir, text=False)  # create a random file name
            self._to_remove['memmap'] = memmap_path  # mark for deletion (since this lives in a temporary directory)
        else:  # user chose an output (not temporary) directory
            output_dir = os.path.abspath(output_dir)  # make it absolute
            os.makedirs(output_dir, exist_ok=True)  # make sure it exists
            memmap_path = os.path.join(output_dir, '{}.dat'.format(name))
            # do not schedule deletion (user probably wants to keep folder and/or file)
        self._memmap_path = memmap_path

        if _selection is None or _mem_units is None:
            # bound sequences for length and count number of resulting tokens
            # shapes: [1, nb_sentences, 2]
            _selection, shapes = bound_length([input_path], [tokenizer], [shortest], [longest])
            # here we have single stream
            # [nb_sentences, 2]
            shapes = np.squeeze(shapes, 0)
            _mem_units = np.sum(np.prod(shapes, 1))
        # construct mmap given length constraints (expressed through selection and nb_tokens)
        # [nb_sentences, 2]
        self._shapes = construct_mmap(input_path, memmap_path, tokenizer,
                                             _selection, _mem_units, dtype=batch_dtype)
        self._mem_units = _mem_units
        # total number of tokens
        self._nb_tokens = np.sum(self._shapes[:, 0], 0)
        self._nb_samples = self._shapes.shape[0]
        # longest sequence in corpus (possibly not trimmed)
        self._longest = longest if not trim else np.max(self._shapes[:, 0], 0)
        # in terms of number of chars
        self._deepest = np.max(self._shapes[:, 1], 0)
        self._selection = _selection
        self._name = name

    @property
    def name(self):
        return self._name

    def __del__(self):
        if 'memmap' in self._to_remove:
            try:
                os.unlink(self._to_remove['memmap'])
            except FileNotFoundError:
                pass
        if 'tmp_dir' in self._to_remove:
            try:
                os.rmdir(self._to_remove['tmp_dir'])
            except FileNotFoundError:  # the directory somehow disappeared
                pass
            except OSError:  # probably there's more stuff in the directory
                pass

    def iter_selection_flags(self):
        """Iterate over the selection flags"""
        return iter(self._selection)

    def nb_streams(self):
        """A Text is a single stream"""
        return 1

    def to_str(self, token_id: int, stream=0) -> str:
        return self._tokenizer.to_str(token_id)

    def memmap_path(self, stream=0):
        """Where the memory map is stored"""
        return self._memmap_path

    def nb_samples(self):
        """Total number of sequences in the corpus"""
        return self._nb_samples

    def nb_tokens(self, stream=0):
        """Total number of tokens in the corpus"""
        return self._nb_tokens

    def vocab_size(self, stream=0):
        """Size of the vocabulary (including -PAD-, -UNK-, and other special symbols)"""
        return self._tokenizer.vocab_size()

    def longest_sequence(self, stream=0):
        """Length of the longest sequence in the corpus"""
        return self._longest

    def deepest_sequence(self, stream=0):
        return self._deepest

    def sample_length(self, sid: int, stream=0):
        """Length of the sequence sid (0-based) in stream (0-based)"""
        return self._shapes[sid, 0]

    def sample_depth(self, sid: int, stream=0):
        """Length of the sequence sid (0-based) in stream (0-based)"""
        return self._shapes[sid, 1]

    def lengths(self, stream=0):
        return self._shapes[:, 0]

    def depths(self, stream=0):
        return self._shapes[:, 1]

    def batch_iterator(self, batch_size, endless=False, shorter_batch='mask',
                       dynamic_sequence_length=False,
                       dynamic_sequence_depth=False):
        """
        Iterate over an input stream yielding batches of a certain size.

        :param batch_size: number of samples/sequences in the batch
        :param endless: cycle endlessly over the samples in the corpus (defaults to False)
        :param shorter_batch: strategy to deal with a shorter batch at the end of the corpus
            * 'mask': masks missing sequences in last batch
            * 'trim': truncates the last batch (implies dynamic number of samples per batch)
            * 'complete': loops over to the beginning of the corpus gathering samples to complete the batch
            * 'discard': ditches the last batch
            * anything else will silently get mapped to 'mask'
        :param dynamic_sequence_length: with dynamic sequence length with trim columns as to fit the longest
            sample in the batch (default to False)
        :return: generator of pairs (batch, mask)
        """
        mmap = np.memmap(self._memmap_path, dtype=self._batch_dtype, mode='r')

        nb_total_samples = self.nb_samples()

        endless_iterator = itertools.cycle(enumerate(self._shapes))
        offset = 0
        n_steps = self.longest_sequence()  # by default batch and mask are created with as many columns as necessary
        n_channels = self.deepest_sequence()

        # configure length strategy
        if dynamic_sequence_length:
            trim_length = lambda pair, longest: (pair[0][:, :longest, :], pair[1][:, :longest, :])
        else:
            trim_length = lambda pair, longest: pair

        if dynamic_sequence_depth:
            trim_depth = lambda pair, deepest: (pair[0][:, :, :deepest], pair[1][:, :, :deepest])
        else:
            trim_depth = lambda pair, deepest: pair

        # configure shorter batch strategy
        shorter_batch = Text3D.STRATEGY_MAP.get(shorter_batch, Text3D.MASK)
        if shorter_batch == Text3D.TRIM:
            trim_size = lambda pair, size: (pair[0][:size, :, :], pair[1][:size, :, :])
        else:
            trim_size = lambda pair, size: pair

        pad_id = self._tokenizer.pad_id()

        # Generate batches and masks potentially indefinitely
        generating = True
        while generating:
            batch = np.zeros((batch_size, n_steps, n_channels), dtype=self._batch_dtype)
            mask = np.zeros((batch_size, n_steps, n_channels), dtype=self._mask_dtype)
            valid_batch = True
            samples_in_batch = 0
            longest_in_batch = 0
            deepest_in_batch = 0
            for row in range(batch_size):
                seq_id, shape = next(endless_iterator)  # get the next length
                if seq_id == 0:  # we are back at the beginning of the corpus
                    offset = 0
                length, depth = shape
                size = length * depth
                sample = np.reshape(mmap[offset: offset + size], [length, depth])
                batch[row, :length, :depth] = sample
                valid = sample != pad_id
                mask[row, :length, :depth] = valid
                offset += size
                # update tightest possible shape
                longest_in_batch = max(longest_in_batch, length)
                deepest_in_batch = max(deepest_in_batch, np.max(np.sum(valid, -1)))
                samples_in_batch += 1

                if seq_id + 1 == nb_total_samples:  # we are at the 0-based end of the corpus
                    if row + 1 == batch_size:  # we also happened to complete the batch
                        if not endless:  # this is the last batch because the iterator is not endless
                            generating = False
                        break  # batch is done
                    else:  # here we have to deal with an incomplete batch
                        # first let's check whether we may continue after dealing with this batch
                        if not endless:
                            generating = False
                        # now let's decide on how to deal with the batch

                        # we may complete it with samples taken from the beginning of the corpus
                        if shorter_batch == Text3D.COMPLETE:
                            # thus just go on (next samples will come from the beginning of the corpus)
                            continue

                        # we may stop right here and invalidate the batch
                        if shorter_batch == Text3D.DISCARD:  # we are not yielding this batch
                            # thus we invalidate it
                            valid_batch = False

                        # or we stop right here with the batch we have
                        break  # DISCARD/TRIM/MASK all lead to a break for this batch

            if valid_batch:  # here we yield a batch after attempting trimming rows and columns
                yield trim_size(
                    trim_length(
                        trim_depth((batch, mask), deepest_in_batch),
                        longest_in_batch),
                    samples_in_batch)


class Multitext3D:
    """
    This class wraps a collection of parallel Text objects.

    It extends the functionality of Text by allowing parallel streams
    """

    def __init__(self, input_paths: tuple, tokenizers: tuple,
                 shortest=None,
                 longest=None,
                 trim=None,
                 output_dir=None,
                 tmp_dir=None,
                 batch_dtype='int64',
                 mask_dtype='float32',
                 name='bitext',
                 _selection=None,
                 _mem_units=None):
        """
        Wraps a collection of Text objects, one per stream (check Text's note on cleanup).


        :param input_paths: path to each half of the parallel corpus
        :param tokenizers: a Tokenizer for each half of the parallel corpus
        :param shortest: a pair specifying the length of the shortest valid sequence (defaults to 1 for all streams)
        :param longest: a pair specifying the length of the longest valid sentence (defaults to inf for all streams)
        :param trim: a pair specifying whther to trim batches to the longest sentence in the corpus
            defaults to False for all streams, but if longest is unbounded, trim will be overwritten to True
        :param output_dir: where to store the memory map (defaults to None in which case tmp_dir will be used)
        :param tmp_dir: a temporary directory used in case output_dir is None (defaults to None in which case a
            the system's tmp space will be used)
        :param batch_dtype: data type for batches
        :param mask_dtype: data type for masks
        :param name: name of the corpus (file will end in .dat)
            * if memory maps live in output_dir then each file name will be '{}-{}.dat'.format(name, stream_nb)
            * if memory maps live in temp_dir then each file name will be obtained with
                tempfile.mkstemp(prefix='{}-{}'.format(name, stream_nb), suffix='.dat', dir=tmp_dir, text=False)
            in this case, files will be deleted when the Text objects get destructed
        :param _selection: uses a subset of the data specified through a np.array with a binary selector per sample
            Multitext can figure this out by itself.
        :param _mem_units: total number of memory units in the selection
            _selection and _mem_units are used when multiple texts are simultaneously constrained for length
            users probably would never need to specify these variables by hand
        """
        nb_streams = len(input_paths)

        # normalise some default attributes
        if shortest is None:
            shortest = [1] * nb_streams
        if longest is None:
            longest = [np.inf] * nb_streams
        if trim is None:
            trim = [False] * nb_streams

        assert all(lower > 0 for lower in shortest), '0-length sequences are not such a great idea'
        assert len(input_paths) == len(tokenizers) == len(shortest) == len(longest) == len(trim) == nb_streams, \
            'Be consistent wrt input/tokenizers/shortest/longest: I expect %d input streams' % nb_streams

        if _selection is None or _mem_units is None:
            # select parallel sentences complying with length constraints
            # selection: [samples]
            # shapes: [stream, samples, 2]
            _selection, _shapes = bound_length(input_paths, tokenizers, shortest, longest)
            # [streams]
            _mem_units = np.sum(np.prod(_shapes, -1), -1)

        corpora = []  # type: list[Text3D]
        for i in range(nb_streams):
            corpora.append(Text3D(input_path=input_paths[i],
                                  tokenizer=tokenizers[i],
                                  shortest=shortest[i],
                                  longest=longest[i],
                                  trim=trim[i],
                                  output_dir=output_dir,
                                  tmp_dir=tmp_dir,
                                  batch_dtype=batch_dtype,
                                  mask_dtype=mask_dtype,
                                  _selection=_selection,
                                  _mem_units=_mem_units[i],
                                  name='{}-{}'.format(name, i)))

        self._corpora = tuple(corpora)  # type: tuple[Text3D]
        self._nb_samples = _selection.sum()
        self._batch_dtype = batch_dtype
        self._mask_dtype = mask_dtype
        self._selection = _selection
        self._name = name

    @property
    def name(self):
        return self._name

    def iter_selection_flags(self):
        """Iterate over the selection flags"""
        return iter(self._selection)

    def nb_streams(self):
        return len(self._corpora)

    def to_str(self, token_id: int, stream: int) -> str:
        return self._corpora[stream].to_str(token_id)

    def memmap_path(self, stream):
        return self._corpora[stream].memmap_path()

    def nb_tokens(self, stream):
        """Total number of tokens in the corpus"""
        return self._corpora[stream].nb_tokens()

    def vocab_size(self, stream):
        """Size of the vocabulary (including -PAD-, -UNK-, and other special symbols)"""
        return self._corpora[stream].vocab_size()

    def nb_samples(self):
        """Total number of sequences in the corpus"""
        return self._nb_samples

    def longest_sequence(self, stream: int):
        """Length of the longest sequence in the corpus"""
        return self._corpora[stream].longest_sequence()

    def sample_length(self, sid: int, stream: int):
        """Length of the sequence sid (0-based) in stream (0-based)"""
        return self._corpora[stream].length(sid)

    def lengths(self, stream: int):
        return self._corpora[stream].lengths()

    def deepest_sequence(self, stream=0):
        return self._corpora[stream].deepest_sequence()

    def sample_depth(self, sid: int, stream=0):
        """Length of the sequence sid (0-based) in stream (0-based)"""
        return self._corpora[stream].sample_depth(sid)

    def depths(self, stream=0):
        return self._corpora[stream].depths()

    def batch_iterator(self, batch_size, endless=False, shorter_batch='mask',
                       dynamic_sequence_length=False, dynamic_sequence_depth=False):
        """
        Iterate over an input stream yielding batches of a certain size.

        :param batch_size: number of samples/sequences in the batch
        :param endless: cycle endlessly over the samples in the corpus (defaults to False)
        :param shorter_batch: strategy to deal with a shorter batch at the end of the corpus
            * 'mask': masks missing sequences in last batch
            * 'trim': truncates the last batch (implies dynamic number of samples per batch)
            * 'complete': loops over to the beginning of the corpus gathering samples to complete the batch
            * 'discard': ditches the last batch
            * anything else will silently get mapped to 'mask'
        :param dynamic_sequence_length: with dynamic sequence length with trim columns as to fit the longest
            sample in the batch (default to False)
        :return: generator of pairs (batch, mask), one pair per stream
        """

        iterators = [corpus.batch_iterator(batch_size, endless, shorter_batch,
                                           dynamic_sequence_length, dynamic_sequence_depth)
                     for corpus in self._corpora]

        while True:  # because this is a generator, we leave the loop with a StopIteration exception
            yield [next(iterator) for iterator in iterators]


def test():
    path1 = '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.en'
    path2 = '/home/wferrei1/github/dgm4nlp/data/en-fr/trial.en-fr.fr'

    print('Fitting tokenizer')
    tok1 = nlputils.Tokenizer(nb_words=None, bos_str='-NULL-', eos_str=None, mode='chars', longest_token=10)
    tok1.fit_one(smart_ropen(path1))
    print(tok1.vocab_size())
    tok2 = nlputils.Tokenizer(nb_words=None, bos_str=None, eos_str=None, mode='chars', longest_token=12)
    tok2.fit_one(smart_ropen(path2))
    print(tok2.vocab_size())

    print('Memory mapping data')
    text = Multitext3D(
        [path1, path2],
        [tok1, tok2],
        shortest=[2, 2],
        longest=[20, 20],
        trim=[True, True],
        batch_dtype='int32',
        mask_dtype='bool',
    )

    print(text.nb_samples(), text.longest_sequence(0), text.deepest_sequence(0))
    print(text.nb_samples(), text.longest_sequence(1), text.deepest_sequence(1))
    import sys
    for i, (xm, ym) in enumerate(text.batch_iterator(1, dynamic_sequence_length=True, dynamic_sequence_depth=True)):
        x, m1 = xm
        x, m1 = x[0], m1[0]
        y, m2 = ym
        y, m2 = y[0], m2[0]
        print(x)
        print(m1)
        print(y)
        print(m2)
        print()

        tok1.write_as_text([x], sys.stdout)
        tok2.write_as_text([y], sys.stdout)
        if i == 1:
            break


if __name__ == '__main__':
    test()
