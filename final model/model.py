import torch
import torch.nn as nn

class StressLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_features_dim = embedding_dim + 3

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=self.input_features_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, inputs):
        padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos = inputs

        x_embed = self.embedding(padded_word_tensors)

        vowel_mask_expanded = padded_vowel_masks.unsqueeze(-1)
        norm_char_pos_expanded = norm_char_pos.unsqueeze(-1)
        norm_vowel_pos_expanded = norm_vowel_pos.unsqueeze(-1)

        x = torch.cat([x_embed, vowel_mask_expanded, norm_char_pos_expanded, norm_vowel_pos_expanded], dim=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)

        logits_list = []
        for i in range(output.size(0)):
            mask = padded_vowel_masks[i].bool()
            output_vowels_only = output[i][mask]

            logits_vowels = self.fc(output_vowels_only).squeeze(-1)
            logits_list.append(logits_vowels)

        return logits_list