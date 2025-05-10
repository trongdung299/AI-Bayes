import pandas as pd
from model import NaiveBayes

def train_model():
    """Huấn luyện và lưu mô hình"""
    try:
        # Read training data with error handling
        training_data = pd.read_csv('train_sms_vi.csv', encoding='utf-8', on_bad_lines='skip')
        
        # Clean data: remove rows with missing values and ensure correct column names
        training_data = training_data.dropna()
        training_data.columns = ['Label', 'SMS']
        
        # Convert labels to lowercase for consistency
        training_data['Label'] = training_data['Label'].str.lower()
        
        # Initialize and train the model
        model = NaiveBayes()
        model.train(training_data)
        
        # Save the trained model
        model.save_model('spam_classifier.pkl')
        print("Đã huấn luyện và lưu mô hình thành công!")
        
    except FileNotFoundError:
        print("Không tìm thấy file train_sms_vi.csv!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    train_model()