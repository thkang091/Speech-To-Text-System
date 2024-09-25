import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
from scipy.signal import lfilter, butter
from scipy.fftpack import dct
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import warnings

# Suppress WavFileWarning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

# 1. Preprocessing and Feature Extraction (MFCC) - Same as before

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def frame_signal(signal, frame_size, frame_stride, sample_rate):
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def power_spectrum(frames, NFFT):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    return pow_frames

def filter_banks(pow_frames, sample_rate, nfilt=40, NFFT=512):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

def mfcc(signal, sample_rate, num_ceps=13):
    emphasized_signal = preemphasis(signal)
    frames = frame_signal(emphasized_signal, 0.025, 0.01, sample_rate)
    pow_frames = power_spectrum(frames, 512)
    fb = filter_banks(pow_frames, sample_rate)
    mfcc_features = dct(fb, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    return mfcc_features

# 2. Advanced Acoustic Modeling (HMM)

class HMMModel:
    def __init__(self, n_components=5):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

    def fit(self, X, lengths):
        self.model.fit(X, lengths)

    def score(self, X, lengths):
        return self.model.score(X, lengths)

# 3. Simple Language Model

class LanguageModel:
    def __init__(self):
        self.word_freq = {}
        self.total_words = 0

    def train(self, sentences):
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
                self.total_words += 1

    def score(self, sentence):
        words = sentence.split()
        score = 1.0
        for word in words:
            score *= (self.word_freq.get(word, 0) + 1) / (self.total_words + len(self.word_freq))
        return score

# 4. RNN for End-to-End ASR

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# 5. Improved Speech Recognition System

class ImprovedSpeechRecognitionSystem:
    def __init__(self, hmm_components=5, rnn_hidden_size=64):
        self.hmm_models = {}
        self.language_model = LanguageModel()
        self.rnn_model = None
        self.hmm_components = hmm_components
        self.rnn_hidden_size = rnn_hidden_size
        self.label_to_index = {}
        self.index_to_label = {}

    def prepare_data(self, audio_files, labels):
        X = []
        y = []
        for audio_file, label in zip(audio_files, labels):
            try:
                sample_rate, signal = wavfile.read(audio_file)
                mfcc_features = mfcc(signal, sample_rate)
                X.append(mfcc_features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
        return X, y

    def train_hmm(self, X, y):
        for label in set(y):
            hmm_model = HMMModel(n_components=self.hmm_components)
            label_data = [x for x, l in zip(X, y) if l == label]
            lengths = [len(x) for x in label_data]
            hmm_model.fit(np.vstack(label_data), lengths)
            self.hmm_models[label] = hmm_model

    def train_language_model(self, sentences):
        self.language_model.train(sentences)

   
    def train_rnn(self, X, y, epochs=50, batch_size=1):  # Changed epochs and batch_size
        # Prepare data for RNN
        X_padded = nn.utils.rnn.pad_sequence([torch.FloatTensor(x) for x in X], batch_first=True)
        y_encoded = [self.label_to_index[label] for label in y]
        y_tensor = torch.LongTensor(y_encoded)

        # Create and train RNN model
        input_size = X_padded.shape[2]
        output_size = len(set(y))
        self.rnn_model = SimpleRNN(input_size, self.rnn_hidden_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)

        n_samples = len(X)
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_X = X_padded[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                outputs = self.rnn_model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_samples  # Changed to use n_samples instead of batches
            if epoch % 10 == 0:  # Print loss every 10 epochs
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def train(self, audio_files, labels, sentences):
        X, y = self.prepare_data(audio_files, labels)
        if not X or not y:
            print("No valid data to train on. Please check your audio files and labels.")
            return

        self.train_hmm(X, y)
        self.train_language_model(sentences)

        # Prepare label encoding for RNN
        unique_labels = list(set(y))
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}

        self.train_rnn(X, y)
    def recognize(self, audio_file):
        sample_rate, signal = wavfile.read(audio_file)
        mfcc_features = mfcc(signal, sample_rate)

        # HMM-based recognition
        hmm_scores = {label: model.score(mfcc_features, [len(mfcc_features)]) for label, model in self.hmm_models.items()}
        hmm_prediction = max(hmm_scores, key=hmm_scores.get)

        # RNN-based recognition
        with torch.no_grad():
            rnn_input = torch.FloatTensor(mfcc_features).unsqueeze(0)
            rnn_output = self.rnn_model(rnn_input)
            rnn_prediction = self.index_to_label[rnn_output.argmax().item()]

        # Combine predictions (simple averaging for demonstration)
        combined_scores = {label: (hmm_scores[label] + (1 if label == rnn_prediction else 0)) / 2 for label in hmm_scores}
        combined_prediction = max(combined_scores, key=combined_scores.get)

        # Apply language model (if applicable)
        lm_score = self.language_model.score(combined_prediction)
        final_prediction = combined_prediction if lm_score > 0 else "Unknown"

        return final_prediction

# 6. Evaluation

def evaluate_system(system, test_files, true_labels):
    correct = 0
    total = len(test_files)

    for test_file, true_label in zip(test_files, true_labels):
        predicted_label = system.recognize(test_file)
        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total
    return accuracy

# Main execution

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    train_files = ["./data/harvard.wav", "./data/jackhammer.wav"]  # Added jackhammer.wav to training
    train_labels = ["harvard", "jackhammer"]  # Updated labels
    test_files = ["./data/jackhammer.wav"]
    test_labels = ["jackhammer"]
    sentences = ["harvard sentences are used for voice testing",
                 "jackhammer noise is loud and disruptive",
                 "speech recognition systems process audio input"]

    # Create and train the system
    system = ImprovedSpeechRecognitionSystem()
    system.train(train_files, train_labels, sentences)
    
    # Evaluate the system
    accuracy = evaluate_system(system, test_files, test_labels)
    print(f"Improved system accuracy: {accuracy * 100:.2f}%")