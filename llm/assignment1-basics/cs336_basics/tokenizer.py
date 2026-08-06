import json
import os
from collections import defaultdict
from typing import BinaryIO,Iterable

import numpy as np
import regex as re
from typing import List

from cs336_basics.train_tokenizer_bpe import bytes_to_unicode


class BPETokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None=None):
        self.vocab=vocab
        self.id_to_byte=vocab
        self.bytes_to_id={v:k for k,v in vocab.items()}
        self.merges={pair:i for i,pair in enumerate(merges)}
        self.special_tokens=special_tokens or []
        if self.special_tokens:
            sorted_special=sorted(self.special_tokens,key=len,reverse=True)
            special_pattern="|".join(re.escape(t) for t in sorted_special)
            self.special_regex=re.compile(special_pattern)
        else:
            self.special_regex=None

        self.gpt2_pat=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self,text:str)->list[int]:
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)
        tokens=[]
        last_pos=0
        for match in self.special_regex.finditer(text):
            pre_text=text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            special_tok=match.group()
            tokens.append(self.bytes_to_id[special_tok.encode("utf-8")])
            last_pos=match.end()
        remaining_text=text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens

    def decode(self,ids:list[int]):
        byte_segments=[self.id_to_byte[i] for i in ids]
        full_bytes=b"".join(byte_segments)
        return full_bytes.decode("utf-8",errors="replace")

    def encode_iterable(self,iterable:Iterable[str])->Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    def _encode_text_segment(self,text:str)->list[int]:
        ids=[]
        pre_tokens=self.gpt2_pat.findall(text)
        for p_tok in pre_tokens:
            byte_parts=[bytes([b]) for b in p_tok.encode("utf-8")]
            while len(byte_parts)>=2:
                best_pair=None
                min_rank=float('inf')
                for i in range(len(byte_parts)-1):
                    pair=(byte_parts[i],byte_parts[i+1])
                    if pair in self.merges:
                        rank=self.merges[pair]
                        if rank<min_rank:
                            min_rank=rank
                            best_pair=pair
                if best_pair is None:
                    break

                new_byte_parts=[]
                i=0
                while i<len(byte_parts):
                    if i<len(byte_parts)-1 and (byte_parts[i],byte_parts[i+1])==best_pair:
                        new_byte_parts.append(best_pair[0]+best_pair[1])
                        i+=2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i+=1
                byte_parts=new_byte_parts

            for byte_part in byte_parts:
                ids.append(self.bytes_to_id[byte_part])
        return ids

def load_trained_tokenizer(vocab_path:str,merges_path:str,special_tokens:List[str]):
    byte_encoder=bytes_to_unicode()
    byte_decoder={v:k for k,v in byte_encoder.items()}
    with open(vocab_path,"r",encoding="utf-8") as f:
        vocab_raw=json.load(f)
        vocab={
            int(k):bytes([byte_decoder[c] for c in v])
            for k,v in vocab_raw.items()
        }
    merges=[]
    with open(merges_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip('\n')
            if not line: continue
            parts=line.split(' ')
            if len(parts)==2:
                p1=bytes([byte_decoder[c] for c in parts[0]])
                p2=bytes([byte_decoder[c] for c in parts[1]])
                merges.append((p1,p2))
    return BPETokenizer(vocab,merges,special_tokens)

def process_corpus(input_txt:str,output_bin:str,tokenizer:BPETokenizer,chunk_size_mb:int=50):
    def file_chunk_generator(file_path,size):
        with open(file_path,"r",encoding="utf-8") as f:
            while True:
                chunk=f.read(size)
                if not chunk:
                    break
                yield chunk
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"找不到文件{input_txt}")
    chunk_size=1024*1024*chunk_size_mb
    if os.path.exists(output_bin):
        os.remove(output_bin)
    chunks=file_chunk_generator(input_txt,chunk_size)
    token_stream=tokenizer.encode_iterable(chunks)
    total_token=0
    write_batch_size=1_000_000
    token_buffer=[]
    with open(output_bin,"ab") as f_out:
        for token_id in token_stream:
            token_buffer.append(token_id)
            if len(token_buffer)>=write_batch_size:
                np_ids=np.array(token_buffer,dtype=np.uint16)
                f_out.write(np_ids.tobytes())
                total_token+=len(token_buffer)
                token_buffer=[]
        if token_buffer:
            np_ids=np.array(token_buffer,dtype=np.uint16)
            f_out.write(np_ids.tobytes())
            total_token+=len(token_buffer)
    print(f"处理完成，总Token:{total_token}")



if __name__ == '__main__':
    input_file="../data/TinyStoriesV2-GPT4-valid.txt"
    output_file="../data/TinyStoriesV2-GPT4-valid.bin"
    vocab_json = "./tokenizer/vocab.json"
    merges_txt = "./tokenizer/merges.txt"
    special_tokens = ["<|endoftext|>"]
    tokenizer = load_trained_tokenizer(vocab_json, merges_txt, special_tokens)
    process_corpus(input_file, output_file, tokenizer)