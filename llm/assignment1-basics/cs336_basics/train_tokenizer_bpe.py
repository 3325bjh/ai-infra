import json
import os
from collections import defaultdict
from typing import BinaryIO,Iterable
import regex as re
from charset_normalizer.cd import Counter



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab={i:bytes([i]) for i in range(256)}
    num_merges=vocab_size-256-len(special_tokens)
    raw_counts=Counter()


    #预分词
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if special_tokens:
                special_regsx="|".join(re.escape(t) for t in special_tokens)
                parts=re.split(f"({special_regsx})",chunk)
                chunk = "".join(p for p in parts if p not in special_tokens)

            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, chunk):
                word=match.group()
                raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))]+=1


    words_list=[]
    counts_list=[]
    for word_tuple,freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    states=defaultdict(int)
    indices=defaultdict(set)


    for idx,word in enumerate(words_list):
        freq=counts_list[idx]
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            states[pair]+=freq
            indices[pair].add(idx)

    merges=[]

    for _ in range(num_merges):
        if not states:
            break

        best_pair=max(states.items(),key=lambda x:(x[1],x[0]))[0]

        if states[best_pair]<=0:
            break
        merges.append(best_pair)
        new_token=best_pair[0]+best_pair[1]
        relevant_indices=indices[best_pair]
        for idx in relevant_indices:
            word=words_list[idx]
            freq=counts_list[idx]
            i=0
            while i<len(word)-1:
                if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                    if i>0:
                        prev_pair=(word[i-1],word[i])
                        states[prev_pair]-=freq
                        if states[prev_pair]==0:
                            del states[prev_pair]
                    if i<len(word)-2:
                        next_pair=(word[i+1],word[i+2])
                        states[next_pair]-=freq
                        if states[next_pair]==0:
                            del states[next_pair]

                    word[i]=new_token
                    del word[i+1]

                    if i>0:
                        new_prev=(word[i-1],word[i])
                        states[new_prev]+=freq
                        indices[new_prev].add(idx)
                    if i<len(word)-1:
                        new_next=(word[i],word[i+1])
                        states[new_next]+=freq
                        indices[new_next].add(idx)
                else:
                    i+=1

            if best_pair in states: del states[best_pair]
            if best_pair in indices: del indices[best_pair]

    for pair in merges:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]

    for s_tok in special_tokens:
        s_bytes=s_tok.encode("utf-8")
        vocab[len(vocab)]=s_bytes

    return vocab,merges

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs=bs[:]
    n=0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n+=1
    cs=[chr(n) for n in cs]
    return dict(zip(bs,cs))

def save_tokenizer_files(vocab,merges,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    byte_encoder=bytes_to_unicode()
    json_vocab={
        k:"".join(byte_encoder[b] for b in v)
        for k,v in vocab.items()
    }
    with open(os.path.join(out_dir,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump(json_vocab,f,indent=4)

    with open(os.path.join(out_dir,"merges.txt"),"w",encoding="utf-8") as f:
        for p1,p2 in merges:
            s1="".join(byte_encoder[b] for b in p1)
            s2="".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")





if __name__ == '__main__':
    vocab,merges=run_train_bpe("../data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
    save_tokenizer_files(vocab,merges,"tokenizer")



