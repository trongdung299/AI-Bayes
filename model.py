import re
import unicodedata
import pickle

class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.vocabulary = []
        self.parameters_spam = {}
        self.parameters_ham = {}
        self.p_spam = 0
        self.p_ham = 0

    def clean_text(self, text):
        # Chuẩn hóa unicode
        text = unicodedata.normalize('NFC', text)
        
        # Loại bỏ các ký tự đặc biệt nhưng giữ lại dấu tiếng Việt
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Tách từ (giữ nguyên dấu tiếng Việt)
        words = text.split()
        
        # Loại bỏ các từ quá ngắn (ít hơn 2 ký tự)
        words = [word for word in words if len(word) > 1]
        
        return words

    def train(self, training_set):
        # Tạo một từ điển để lưu số lần xuất hiện của mỗi từ trong spam và ham
        word_counts_per_sms = {'spam': {}, 'ham': {}}

        # Tổng số lượng tin nhắn spam và ham
        n_spam = 0
        n_ham = 0

        for index, row in training_set.iterrows():
            label = row['Label']
            message = self.clean_text(row['SMS'])
            if label == 'spam':
                n_spam += 1
            else:
                n_ham += 1

            for word in message:
                if word not in word_counts_per_sms[label]:
                    word_counts_per_sms[label][word] = 1
                else:
                    word_counts_per_sms[label][word] += 1

                if word not in self.vocabulary:
                    self.vocabulary.append(word)

        # Tính xác suất của từng từ cho spam và ham
        self.parameters_spam = {}
        self.parameters_ham = {}

        for word in self.vocabulary:
            if word not in word_counts_per_sms['spam']:
                word_counts_per_sms['spam'][word] = 0
            if word not in word_counts_per_sms['ham']:
                word_counts_per_sms['ham'][word] = 0

            p_word_given_spam = (word_counts_per_sms['spam'][word] + self.alpha) / (n_spam + self.alpha * len(self.vocabulary))
            p_word_given_ham = (word_counts_per_sms['ham'][word] + self.alpha) / (n_ham + self.alpha * len(self.vocabulary))

            self.parameters_spam[word] = p_word_given_spam
            self.parameters_ham[word] = p_word_given_ham

        # Xác suất trước
        self.p_spam = n_spam / len(training_set)
        self.p_ham = n_ham / len(training_set)

    def classify(self, message):
        message = self.clean_text(message)
        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham
        
        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message *= self.parameters_spam[word]
            if word in self.parameters_ham:
                p_ham_given_message *= self.parameters_ham[word]
                
        return p_ham_given_message > p_spam_given_message

    def save_model(self, filename):
        """Lưu mô hình vào file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        """Đọc mô hình từ file"""
        with open(filename, 'rb') as f:
            return pickle.load(f) 