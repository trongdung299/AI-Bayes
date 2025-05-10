import pandas as pd
from model import NaiveBayes

def train_model():
    """Huấn luyện và lưu mô hình"""
    try:
        # Đọc dữ liệu huấn luyện với xử lý lỗi
        training_data = pd.read_csv('train_sms_vi.csv', encoding='utf-8', on_bad_lines='skip')
        
        # Làm sạch dữ liệu: xóa các dòng có giá trị thiếu và đảm bảo tên cột đúng
        training_data = training_data.dropna()
        training_data.columns = ['Label', 'SMS']
        
        # Chuyển đổi nhãn thành chữ thường để đồng nhất
        training_data['Label'] = training_data['Label'].str.lower()
        
        # Khởi tạo và huấn luyện mô hình
        model = NaiveBayes()
        model.train(training_data)
        
        # Lưu mô hình đã huấn luyện
        model.save_model('spam_classifier.pkl')
        print("Đã huấn luyện và lưu mô hình thành công!")
        
    except FileNotFoundError:
        print("Không tìm thấy file train_sms_vi.csv!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    train_model()