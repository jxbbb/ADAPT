# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import json
import os
from io import open

import six
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .file_utils import cached_path

import logging
from src.utils.comm import is_main_process
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
if not is_main_process():
    logger.disabled = True

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = 'model.ckpt'


try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """
        def __init__(self, *args, **kwargs):
            super(Identity, self).__init__()

        def forward(self, input):
            return input


if not six.PY2:
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = ''.join(docstr) + fn.__doc__
            return fn
        return docstring_decorator
else:
    # Not possible to update class docstrings on python2
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator


class PretrainedConfig(object):
    """ Base class for all configuration classes.
        Handle a few common parameters and methods for loading/downloading/saving configurations.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a PretrainedConfig from a pre-trained model configuration.

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load from cache
                    or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **return_unused_kwargs**: (`optional`) bool:
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes:
                ie the part of kwargs which has not been used to update `config` and is otherwise ignored.
            **kwargs**: (`optional`) dict:
                Dictionary of key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used
                to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples::

            >>> config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            >>> config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            >>> config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            >>> config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            >>> assert config.output_attention == True
            >>> config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
            >>>                                                    foo=False, return_unused_kwargs=True)
            >>> assert config.output_attention == True
            >>> assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            return None
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        # add img_layer_norm_eps, use_img_layernorm
        if "img_layer_norm_eps" in kwargs:
            setattr(config, "img_layer_norm_eps", kwargs["img_layer_norm_eps"])
            to_remove.append("img_layer_norm_eps")
        if "use_img_layernorm" in kwargs:
            setattr(config, "use_img_layernorm", kwargs["use_img_layernorm"])
            to_remove.append("use_img_layernorm")
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class PreTrainedModel(nn.Module):
    """ Base class for all models. Handle loading/storing model config and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = PretrainedConfig
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ""
    input_embeddings = None

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self.init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
            Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: does nothing and just returns a pointer to the input tokens Embedding Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embedding Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, 'tie_weights'):
            self.tie_weights()

        return model_embeds

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.
            Args:
                heads_to_prune: dict of {layer_num (int): list of heads to prune in this layer (list of int)}
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are desactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache
                    or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a tensorflow index checkpoint `file` (e.g. `./tf_model/model.ckpt.index`).
                    In this case, ``from_tf`` should be set to True and a configuration object should be
                    provided as `config` argument. This loading option is slower than converting the TensorFlow
                    checkpoint in a PyTorch model using the provided conversion scripts and loading
                    the PyTorch model afterwards.
            **model_args**: (`optional`) Sequence:
                All remaning positional arguments will be passed to the underlying model's __init__ function
            **config**: an optional configuration for the model to use instead of an automatically loaded configuation.
                Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with a `shortcut name` of a pre-trained model), or
                - the model was saved using the `save_pretrained(save_directory)` (loaded by suppling the save directory).
            **state_dict**: an optional state dictionnary for the model to use instead of a state dictionary loaded
                from saved weights file.
                This option can be used if you want to create a model from a pretrained configuraton but load your own weights.
                In this case though, you should check if using `save_pretrained(dir)` and `from_pretrained(save_directory)` is not
                a simpler option.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **output_loading_info**: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            **kwargs**: (`optional`) dict:
                Dictionary of key, values to update the configuration object after loading.
                Can be used to override selected configuration parameters. E.g. ``output_attention=True``.

               - If a configuration is provided with `config`, **kwargs will be directly passed
                 to the underlying model's __init__ method.
               - If a configuration is not provided, **kwargs will be first passed to the pretrained
                 model configuration class loading function (`PretrainedConfig.from_pretrained`).
                 Each key of **kwargs that corresponds to a configuration attribute
                 will be used to override said attribute with the supplied **kwargs value.
                 Remaining keys that do not correspond to any configuration attribute will
                 be passed to the underlying model's __init__ function.

        Examples::

            >>> model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            >>> model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            >>> model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            >>> assert model.config.output_attention == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_model_archive_map.keys()),
                        archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) == 2 and "size mismatch for cls.seq_relationship.weight" in error_msgs[0]:
            logger.info('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        elif len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling
        and beam-search.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.

            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)  # do greedy decoding without beam search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, do_sample=True, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[0][i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, num_beams=3)  # generate sequences using greedy beam search decoding (3 beams)
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences using using greedy search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, "`bos_token_id` should be a positive integer."
        assert isinstance(pad_token_id, int) and pad_token_id >= 0, "`pad_token_id` should be a positive integer."
        assert isinstance(eos_token_ids, (list, tuple)) and (
            e >= 0 for e in eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        if num_return_sequences != 1:
            for i in range(len(output)):
                output[i] = output[i].view(batch_size, num_return_sequences, -1)
        return output

    def _decode_step(self, input_ids, past):
        model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        token_len = outputs[0].shape[1]
        if self.od_labels_len == 0:
            next_token_idx = token_len - 1
        else:
            if token_len == 2:
                assert self._do_output_past(outputs)
                next_token_idx = 1
            else:
                next_token_idx = token_len - self.od_labels_len - 1

        next_token_logits = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
        assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

        # if model has past, then set the past variable to speed up decoding
        if self._do_output_past(outputs):
            past = outputs[1]
        return next_token_logits, past

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        if torch._C._get_tracing_state():
            cur_unfinished = torch.ones(1, dtype=input_ids)
        else:
            cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert outputs[0].shape[1] == token_len
            next_token_logits = outputs[0][:, next_token_idx, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            #for t in input_ids:
                #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        num_keep_best = self.num_keep_best
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        TOPN_PER_BEAM = 2

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert outputs[0].shape[1] == token_len
            scores = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
            assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [TOPN_PER_BEAM] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=TOPN_PER_BEAM)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, TOPN_PER_BEAM)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, TOPN_PER_BEAM).to(next_words.device)
                next_words = next_words.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 1st position
                    reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]

        return decoded, logprobs


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """
    def __init__(self, config):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """
    def __init__(self, config):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert start_states is not None or start_positions is not None, "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz) # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions) # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1) # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, config):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz) # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2) # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz) # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :] # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


class SQuADHead(nn.Module):
    r""" A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~pytorch_transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """
    def __init__(self, config):
        super(SQuADHead, self).__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(self, hidden_states, start_positions=None, end_positions=None,
                cls_index=None, is_impossible=None, p_mask=None):
        outputs = ()

        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1) # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(start_log_probs, self.start_n_top, dim=-1) # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz) # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1) # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1) # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.end_n_top, dim=1) # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs


class SequenceSummary(nn.Module):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'token_ids' => supply a Tensor of classification token indices (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
        if config.summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, token_ids=None):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            token_ids: [optional] index of the classification token if summary_type == 'token_ids',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'token_ids' and token_ids is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'token_ids':
            if token_ids is None:
                token_ids = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
            else:
                token_ids = token_ids.unsqueeze(-1).unsqueeze(-1)
                token_ids = token_ids.expand((-1,) * (token_ids.dim()-1) + (hidden_states.size(-1),))
            # shape of token_ids: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, token_ids).squeeze(-2) # shape (bsz, XX, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer, index, dim=None):
    """ Prune a Conv1D or nn.Linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError("Can't prune layer of class {}".format(layer.__class__))
