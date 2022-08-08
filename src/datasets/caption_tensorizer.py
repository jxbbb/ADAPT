import torch
import random
import os.path as op
from src.utils.logger import LOGGER
import re, html

FLAIR_TAG = {
    "noun": ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "WP", "WP$"],
    "verb": ["VB", "VBD", "VBG", "VBP", "VBZ"],
    "adjective": ["JJ", "JJR", "JJS"],
    "adverb": ["RB","RBR", "RBS", "WRB"],
    "number": ["CD"]}


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            attn_mask_type='seq2seq', is_train=True, mask_b=False,
            text_mask_type='random', tag_to_mask=None,
            mask_tag_prob=0.8, random_mask_prob=0.5):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            attn_mask_type: attention mask type, support seq2seq/bidirectional/cap_s2s/cap_bidir.
            mask_b: whether to mask text_b or not during training.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self.attn_mask_type = attn_mask_type
        self.text_mask_type = text_mask_type
        self.mask_b = mask_b
        self.tag_to_mask = None
        self.mask_tag_prob = 0
        self.random_mask_prob = 1
        if is_train:
            assert attn_mask_type in ('seq2seq', 'bidirectional', 'cap_s2s', 'cap_bidir', 'learn_vid_att')
            assert text_mask_type in ('random', 'bert_attn', 'pos_tag', 'attn_on_the_fly')
            if self.text_mask_type == 'pos_tag':
                self.tag_to_mask = tag_to_mask
                self.included_tags = set()
                for type in self.tag_to_mask:
                    self.included_tags.update(set(FLAIR_TAG[type]))
                self.mask_tag_prob = mask_tag_prob
            if self.text_mask_type != "random":
                self.random_mask_prob = random_mask_prob
        else:
            assert attn_mask_type in ('seq2seq', 'learn_vid_att')
        
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))
    
    def get_pos_tag_mask_idx(self, seq_a_len, text_meta):
        
        ''' The rest   
        ADD	Email
        AFX	Affix
        CC	Coordinating conjunction
        DT	Determiner
        EX	Existential there
        FW	Foreign word
        HYPH	Hyphen
        IN	Preposition or subordinating conjunction
        LS	List item marker
        MD	Modal
        NFP	Superfluous punctuation
        PDT	Predeterminer
        POS	Possessive ending
        RP	Particle
        SYM	Symbol
        TO	to
        UH	Interjection
        WDT	Wh-determiner
        XX
        '''
        # process loaded pos_tags
        pos_tags =  text_meta["bert_pos_tag"] 
        if len(pos_tags) > seq_a_len - 2:
            pos_tags = pos_tags[:seq_a_len-2]
        pos_tags = [None] + pos_tags + [None]
        padding_len = seq_a_len - len(pos_tags)
        pos_tags += [None] * padding_len
        allow_masked_ids = set()
        for bert_idx, tag in enumerate(pos_tags):
            if tag is None:
                continue
            if bert_idx >= seq_a_len:
                break
            if tag not in self.included_tags:
                continue
            allow_masked_ids.add(bert_idx)
        return pos_tags, allow_masked_ids
    
    def get_bert_attn_mask_idx(self, seq_a_len, text_meta, num_masked):
        # process loaded bert attention weights (assuming max_len = 50)
        attn_weights =  text_meta["bert_attn"] 
        if len(attn_weights) > seq_a_len:
            attn_weights = attn_weights[:seq_a_len]
        elif len(attn_weights) < seq_a_len:
            # pad with zeros
            padding_len = seq_a_len - len(attn_weights)
            attn_weights = [0.0] * padding_len
        mask_idx = torch.multinomial(torch.tensor(attn_weights), num_masked).tolist()
        return mask_idx
    
    def get_attn_masks(self, seq_a_len, seq_len):
        # image features
        img_len = self.max_img_seq_len

        max_len = self.max_seq_len + self.max_img_seq_len
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        if self.is_train and self.attn_mask_type == 'bidirectional':
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            attention_mask[c_start : c_end] = 1 # for text_a
            attention_mask[l_start : l_end] = 1 # for text_b if any
            attention_mask[r_start : r_end] = 1 # for image
        elif self.is_train and self.attn_mask_type in ('cap_s2s', 'cap_bidir'):
            # caption is a single modality, and without attention on others
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # no attention between [CLS] and caption
            attention_mask[0, 0] = 1
            if self.attn_mask_type == 'cap_s2s':
                attention_mask[c_start + 1 : c_end, c_start + 1 : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len - 1, 0 : seq_a_len - 1]
                )
            else:
                attention_mask[c_start + 1 : c_end, c_start + 1: c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            # cross attention for L-R, R-L
            attention_mask[l_start : l_end, r_start : r_end] = 1
            attention_mask[r_start : r_end, l_start : l_end] = 1
            # cross attention between [CLS] and L/R
            attention_mask[0, l_start : l_end] = 1
            attention_mask[l_start : l_end, 0] = 1
            attention_mask[0, r_start : r_end] = 1
            attention_mask[r_start : r_end, 0] = 1
        elif self.attn_mask_type in ('learn_vid_att'):
            # prepare attention mask:
            # note that there is no attention from caption to image
            # because otherwise it will violate the triangle attention 
            # for caption as caption will have full attention on image. 
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # triangle mask for caption to caption
            attention_mask[c_start : c_end, c_start : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len, 0 : seq_a_len]
            )
            # full attention for C-L, C-R
            attention_mask[c_start : c_end, l_start : l_end] = 1
            attention_mask[c_start : c_end, r_start : r_end] = 1 
            # full attention for video tokens:
            attention_mask[l_start : r_end, l_start : r_end] = 1
        else:
            # prepare attention mask:
            # note that there is no attention from caption to image
            # because otherwise it will violate the triangle attention 
            # for caption as caption will have full attention on image. 
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # triangle mask for caption to caption
            attention_mask[c_start : c_end, c_start : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len, 0 : seq_a_len]
            )
            # full attention for L-L, R-R
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            # full attention for C-L, C-R
            attention_mask[c_start : c_end, l_start : l_end] = 1
            attention_mask[c_start : c_end, r_start : r_end] = 1
            # full attention for L-R:
            attention_mask[l_start : l_end, r_start : r_end] = 1
            attention_mask[r_start : r_end, l_start : l_end] = 1
        return attention_mask
    
    def get_text_mask_idx(self, seq_a_len, seq_len, text_meta=None):
        # randomly mask words for prediction, ignore [CLS], [PAD]
        # it is important to mask [SEP] for image captioning as it means [EOS].

        # 1. get the number of masked tokens
        if self.mask_b:
            # can mask both text_a and text_b
            num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
        else:
            # only mask text_a
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
        num_masked = int(num_masked)

        # 2. get the masking candidates
        if self.mask_b:
            # text b always random masking
            text_b_candidate = list(range(self.max_seq_a_len, seq_len))
        else:
            text_b_candidate = []
        if self.text_mask_type == 'pos_tag' and random.random() > self.random_mask_prob:
            full_candidate = set(list(range(1, seq_a_len)))
            pos_tags, pos_tag_candidate = self.get_pos_tag_mask_idx(
                text_meta=text_meta,
                seq_a_len=seq_a_len)

            left_over_candidate = list(
                full_candidate.difference(pos_tag_candidate)) + text_b_candidate
            pos_tag_candidate = list(pos_tag_candidate)
            num_pos_tag_masked = min(
                max(1, int(num_masked*self.mask_tag_prob)), len(pos_tag_candidate))
            random.shuffle(pos_tag_candidate)
            masked_idx = pos_tag_candidate[:num_pos_tag_masked]
            
            num_left_overs = num_masked - num_pos_tag_masked
            if num_left_overs > 0:
                random.shuffle(left_over_candidate)
                other_masked_idx = left_over_candidate[:num_left_overs]
                masked_idx += other_masked_idx
        elif self.text_mask_type == 'bert_attn' and random.random() > self.random_mask_prob:
            masked_idx = self.get_bert_attn_mask_idx(seq_a_len, text_meta, num_masked)
        else:
            # random
            candidate_masked_idx = list(range(1, seq_a_len))
            candidate_masked_idx += text_b_candidate
            random.shuffle(candidate_masked_idx)
            masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        return masked_idx
    
    def mask_text_inputs(self, tokens, seq_a_len, seq_len, text_meta=None):
        if self.is_train:
            if self.text_mask_type == "attn_on_the_fly" and random.random() > self.random_mask_prob and len(tokens)> 2:
                # self.text_mask_type == "attn_on_the_fly"
                masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
                masked_pos[1: seq_a_len] += 1
                masked_pos[0] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]
                mlm_targets = [-1] * self.max_masked_tokens
            else:
                masked_idx = self.get_text_mask_idx(seq_a_len, seq_len, text_meta)
                try:
                    masked_token = [tokens[i] for i in masked_idx]
                except Exception as e:
                    overflow_idx = []
                    for i in masked_idx:
                        if i >= len(tokens) or i < 0:
                            overflow_idx.append(i)
                    raise ValueError(f"Error {e}\nOverflow: {overflow_idx} in tokens {tokens}")
                for pos in masked_idx:
                    if random.random() <= 0.8:
                        # 80% chance to be a ['MASK'] token
                        tokens[pos] = self.tokenizer.mask_token
                    elif random.random() <= 0.5:
                        # 10% chance to be a random word ((1-0.8)*0.5)
                        tokens[pos] = self.tokenizer.get_random_token()
                    else:
                        # 10% chance to remain the same (1-0.8-0.1)
                        pass

                masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
                masked_pos[masked_idx] = 1
                
                # get the actual number of masked tokens
                num_masked = len(masked_token)
                mlm_targets = self.tokenizer.convert_tokens_to_ids(masked_token)
                if num_masked < self.max_masked_tokens:
                    mlm_targets = mlm_targets + ([-1] * (self.max_masked_tokens - num_masked))
                assert len(mlm_targets) == self.max_masked_tokens, f"mismatch in len(masked_ids) {len(mlm_targets)} vs. max_masked_tokens {self.max_masked_tokens}"
        elif not self.is_train:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)
            mlm_targets = None
        
        return tokens, masked_pos, mlm_targets
    
    def prepro_raw_txt(self, text):
        # in case there are html special characters
        text = html.unescape(text)
        # FIXME: quick hack for text with emoji, may adopt twitter tokenizer later
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text
    
    def tokenize_text_inputs(
            self, text_a, text_b=None, cls_token_segment_id=0,
            pad_token_segment_id=0, sequence_a_segment_id=0,
            sequence_b_segment_id=1, text_meta=None):
        text_a = self.prepro_raw_txt(text_a)
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
            if self.text_mask_type == "pos_tag":
                assert text_meta is not None and 'bert_pos_tag' in text_meta
                assert len(text_meta['bert_pos_tag']) == len(tokens_a)
            elif self.text_mask_type == "bert_attn":
                assert text_meta is not None and 'bert_attn' in text_meta
                assert (len(text_meta['bert_attn']) == len(tokens_a) + 2 or 
                        len(text_meta['bert_attn']) == self.max_seq_a_len)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            text_b = self.prepro_raw_txt(text_b)
            # pad text_a to keep it in fixed length for better inference.
            # we do not use pos tag for text_b
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        return tokens, segment_ids, seq_a_len, seq_len

    def tensorize_example_e2e(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, text_meta=None):
        # tokenize the texts
        tokens, segment_ids, seq_a_len, seq_len = self.tokenize_text_inputs(
            text_a, text_b, cls_token_segment_id, pad_token_segment_id,
            sequence_a_segment_id, sequence_b_segment_id, text_meta)
        
        # masking the tokens
        tokens_after_masking, masked_pos, mlm_targets = self.mask_text_inputs(
            tokens, seq_a_len, seq_len, text_meta)

        # pad on the right for image captioning
        seq_padding_len = self.max_seq_len - seq_len
        tokens = tokens_after_masking + ([self.tokenizer.pad_token] * seq_padding_len)
        segment_ids += ([pad_token_segment_id] * seq_padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = self.get_attn_masks(seq_a_len, seq_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            mlm_targets = torch.tensor(mlm_targets, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, mlm_targets)
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos


def build_tensorizer(args, tokenizer, is_train=True):
    if hasattr(args, 'mask_od_labels'):
        mask_b = args.mask_od_labels
    else:
        mask_b = False
    if is_train:
        if  args.text_mask_type == "pos_tag":
            # if op.exists(args.tagger_model_path):
            #     tagger = SequenceTagger.load(args.tagger_model_path)
            # else:
            #     LOGGER.info(f'{args.tagger_model_path} does not exists, download on the fly...')
            #     tagger = SequenceTagger.load('pos-fast')
            tag_to_mask = set(args.tag_to_mask)
        # elif args.text_mask_type == "bert_attn":
        #     bert = 
        else:
            tagger = None
            tag_to_mask = None
        return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length,
            max_seq_a_length=args.max_seq_a_length,
            mask_prob=args.mask_prob,
            max_masked_tokens=args.max_masked_tokens,
            attn_mask_type=args.attn_mask_type,
            is_train=True,
            mask_b=mask_b,
            text_mask_type=args.text_mask_type,
            mask_tag_prob=args.mask_tag_prob,
            tag_to_mask=tag_to_mask,
            random_mask_prob=args.random_mask_prob,
            # tagger=tagger,
        )
    return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length if args.add_od_labels else args.max_gen_length,
            max_seq_a_length=args.max_gen_length,
            is_train=False,
            attn_mask_type=args.attn_mask_type,
    )

