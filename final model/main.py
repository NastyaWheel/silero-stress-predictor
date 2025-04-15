import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from dataset import StressDataset, collate_fn
from model import StressLSTM
from train import Trainer


TRAIN_CSV = '/kaggle/input/silero-stress-predictor/train.csv'
TEST_CSV = '/kaggle/input/silero-stress-predictor/test.csv'
CHECKPOINT_PATH = 'best_model.pt'
PLOT_PATH = 'training_plot.png'
SUBMISSION_PATH = 'submission.csv'
VOCAB_SAVE_PATH = 'char2idx.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vowels = 'аеёиоуыэюя'
consonants = 'бвгджзйклмнпрстфхцчшщ'
special_chars = 'ьъ-'

alphabet = list(vowels + consonants + special_chars)
char2idx = {char: idx + 1 for idx, char in enumerate(sorted(alphabet))}
char2idx['<unk>'] = len(char2idx) + 1


def inference_and_save(trainer, test_loader, submission_path='/kaggle/working/submission.csv'):
    trainer.model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for (padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos), batch_ids in tqdm(test_loader, desc='Inferencing'):

            padded_word_tensors = padded_word_tensors.to(trainer.device)
            padded_vowel_masks = padded_vowel_masks.to(trainer.device)
            lengths = lengths.to(trainer.device)
            norm_char_pos = norm_char_pos.to(trainer.device)
            norm_vowel_pos = norm_vowel_pos.to(trainer.device)

            logits_list = trainer.model((
                padded_word_tensors, 
                padded_vowel_masks, 
                lengths, 
                norm_char_pos, 
                norm_vowel_pos
            ))

            batch_size = len(logits_list)

            for i in range(batch_size):
                logits_vowels = logits_list[i]
                pred_vowel_index = logits_vowels.argmax().item()
                stress_syllable = pred_vowel_index + 1

                predictions.append(stress_syllable)
                ids.append(batch_ids[i])

    submission_df = pd.DataFrame({
        'id': ids,
        'stress': predictions
    })

    submission_df.sort_values('id', inplace=True)
    submission_df.reset_index(drop=True, inplace=True)

    submission_df.to_csv(submission_path, index=False)
    print(f" Submission saved to {submission_path}")



def main():
    full_df = pd.read_csv(TRAIN_CSV)

    lemmas = full_df['lemma'].unique()
    train_lemmas, val_lemmas = train_test_split(lemmas, test_size=0.2, random_state=1)

    train_split = full_df[full_df['lemma'].isin(train_lemmas)].reset_index(drop=True)
    val_split = full_df[full_df['lemma'].isin(val_lemmas)].reset_index(drop=True)

    train_dataset = StressDataset(train_split, char2idx, is_test=False)
    val_dataset = StressDataset(val_split, char2idx, is_test=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    vocab_size = len(char2idx)
    embedding_dim = 128
    hidden_dim = 256
    dropout = 0.3
    lr=1e-3

    model = StressLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=lr,
        epochs=8,
        patience=3,
        checkpoint_path=CHECKPOINT_PATH,
        plot_path=PLOT_PATH
    )

    trainer.train()


    test_dataset = StressDataset(TEST_CSV, char2idx=char2idx, is_test=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    inference_and_save(trainer, test_loader, submission_path=SUBMISSION_PATH)

if __name__ == '__main__':
    main()
