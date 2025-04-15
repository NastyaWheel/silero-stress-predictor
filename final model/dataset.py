import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


vowels = 'аеёиоуыэюя'
VOWELS = set(vowels) 

class StressDataset(Dataset):
    def __init__(self, csv_path_or_df, char2idx, is_test=False):
        """
        csv_path_or_df: путь к .csv или готовый pandas DataFrame
        char2idx: вручную созданный словарь {символ: индекс}
        is_test: если True, то нет метки stress
        """
        if isinstance(csv_path_or_df, str):
            self.data = pd.read_csv(csv_path_or_df)
        elif isinstance(csv_path_or_df, pd.DataFrame):
            self.data = csv_path_or_df.copy()
        else:
            raise ValueError("csv_path_or_df должен быть либо строкой (путь к файлу), либо DataFrame")

        self.is_test = is_test
        self.char2idx = char2idx
        
    def __len__(self):
        return len(self.data)

    def encode_word(self, word):
        """
        Преобразует слово в список индексов, маску гласных,
        нормированные позиции символов и гласных.
        """
        word_encoded = []
        vowels_mask = []
        norm_char_pos = []   # нормированная позиция символа в слове
        norm_vowel_pos = []    # нормированная позиция гласной среди гласных (0 для согласных)

        word_len = len(word)
        num_vowels = sum(1 for char in word if char in VOWELS)

        current_vowel_idx = 0

        for i, char in enumerate(word):
            idx = self.char2idx.get(char, self.char2idx.get('<unk>'))
            word_encoded.append(idx)

            # маска гласных
            is_vowel = 1 if char in VOWELS else 0
            vowels_mask.append(is_vowel)

            # нормированная позиция символа
            norm_char = i / (word_len - 1) if word_len > 1 else 0.0
            norm_char_pos.append(norm_char)

            # нормированная позиция гласной
            if is_vowel and num_vowels > 1:
                norm_vowel = current_vowel_idx / (num_vowels - 1)
                current_vowel_idx += 1
            elif is_vowel and num_vowels == 1:
                norm_vowel = 0.0
                current_vowel_idx += 1
            else:
                norm_vowel = -1.0

            norm_vowel_pos.append(norm_vowel)

        return (
            torch.tensor(word_encoded, dtype=torch.long),      
            torch.tensor(vowels_mask, dtype=torch.float),      
            torch.tensor(norm_char_pos, dtype=torch.float),    
            torch.tensor(norm_vowel_pos, dtype=torch.float)    
        )

    def __getitem__(self, idx):
        """
        Возвращает один элемент из датасета по индексу idx датасета.
        """
        row = self.data.iloc[idx]
        word_tensor, vowel_mask, norm_char_pos, norm_vowel_pos = self.encode_word(row['word'])

        if self.is_test:
            item_id = row['id'] if 'id' in row else idx
            return (word_tensor, vowel_mask, norm_char_pos, norm_vowel_pos), item_id

        stress_label = torch.tensor(row['stress'], dtype=torch.long)

        return (word_tensor, vowel_mask, norm_char_pos, norm_vowel_pos), stress_label
    

def collate_fn(batch):
    word_tensors = []
    vowel_masks = []
    norm_char_pos_list = []
    norm_vowel_pos_list = []
    lengths = []
    targets_or_ids = []

    target = batch[0][1]
    is_test = not torch.is_tensor(target)

    for (word_tensor, vowel_mask, norm_char_pos, norm_vowel_pos), target_or_id in batch:
        word_tensors.append(word_tensor)
        vowel_masks.append(vowel_mask)
        norm_char_pos_list.append(norm_char_pos)
        norm_vowel_pos_list.append(norm_vowel_pos)
        lengths.append(len(word_tensor))
        targets_or_ids.append(target_or_id)

    padded_word_tensors = pad_sequence(word_tensors, batch_first=True, padding_value=0)
    padded_vowel_masks = pad_sequence(vowel_masks, batch_first=True, padding_value=0.0)
    padded_norm_char_pos = pad_sequence(norm_char_pos_list, batch_first=True, padding_value=0.0)
    padded_norm_vowel_pos = pad_sequence(norm_vowel_pos_list, batch_first=True, padding_value=-1.0)

    lengths = torch.tensor(lengths, dtype=torch.long)

    if is_test:
        batch_ids = [int(i) for i in targets_or_ids]
        return (padded_word_tensors, padded_vowel_masks, lengths, padded_norm_char_pos, padded_norm_vowel_pos), batch_ids
    else:
        stress_labels = torch.stack(targets_or_ids)
        return (padded_word_tensors, padded_vowel_masks, lengths, padded_norm_char_pos, padded_norm_vowel_pos), stress_labels