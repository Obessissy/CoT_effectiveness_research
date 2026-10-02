import pathlib, sentencepiece as spm

p = pathlib.Path("v1_sp_bpe_8k_seed1234/tokenizer.model")
sp = spm.SentencePieceProcessor(model_proto=p.read_bytes())   # 用 bytes 绕开中文路径 bug

sp.get_piece_size()        # 8192      词表大小
sp.id_to_piece(22)         # '<Z_017>' ID → piece
sp.piece_to_id('<Z_017>')  # 22        piece → ID
print(sp.get_score(5218))         # -4925.0   piece 的分数
#sp.id_to_piece(99999)      # '<unk>'   越界返回 unk_id(0)
print(sp.piece_to_id('1/3'))