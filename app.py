import os
import pytesseract
from flask import Flask, request, jsonify
from PIL import Image
import io
import openai
import logging
import cv2
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Đặt đường dẫn đến tệp tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'

# Đặt API key cho OpenAI GPT-4
openai.api_key = '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'

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
learned_data = []

# ================================================================
# Hàm nạp dữ liệu đã học từ tệp văn bản
def load_learned_data():
    try:
        with open('learned_data.txt', 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        logging.warning("No learned data found, initializing with empty data.")
        return []

# Lưu dữ liệu đã học vào tệp văn bản
def save_learned_data(text):
    with open('learned_data.txt', 'a', encoding='utf-8') as file:
        file.write(text + "\n")
    logging.info("Learned data saved successfully.")

# Hàm để reset bộ nhớ học khi file không tồn tại
def reset_learned_data():
    global learned_data
    learned_data = []
    logging.info("Learned data has been reset.")

# Kiểm tra nếu file 'learned_data.txt' không tồn tại, reset bộ nhớ học
if not os.path.exists('learned_data.txt'):
    reset_learned_data()

# Nạp dữ liệu học vào bộ nhớ
learned_data = load_learned_data()

# ================================================================
# Hàm để làm sạch văn bản và chỉ giữ lại các ký tự quan trọng
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s.,!?áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ-]', '', text).strip()
    return text

# Hàm tìm câu trả lời từ dữ liệu đã học
def find_answer_from_learned_data(question):
    question_cleaned = clean_text(question)
    documents = [clean_text(doc) for doc in learned_data]

    if not documents:  # Kiểm tra nếu không có dữ liệu học
        return None

    # Tạo vector TF-IDF từ câu hỏi và các văn bản đã học
    vectorizer = TfidfVectorizer().fit_transform([question_cleaned] + documents)
    vectors = vectorizer.toarray()

    # Tính toán độ tương đồng cosine giữa câu hỏi và các câu đã học
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()

    # Lấy văn bản có độ tương đồng cao nhất
    best_match_index = np.argmax(cosine_similarities)
    
    if cosine_similarities[best_match_index] > 0.3:  # Giảm ngưỡng để chấp nhận nhiều kết quả hơn
        return documents[best_match_index]
    return None

# ================================================================
# Hàm tìm thông tin cụ thể từ câu hỏi
def find_specific_information(question, data):
    keywords = re.findall(r'\b\w+\b', question.lower())  # Tìm tất cả các từ khóa trong câu hỏi
    filtered_data = []

    for d in data:
        if all(keyword in d.lower() for keyword in keywords):
            filtered_data.append(d)

    # Loại bỏ dấu ** trong dữ liệu
    cleaned_data = [re.sub(r'\*\*', '', d) for d in filtered_data]
    
    return cleaned_data

# Hàm query dữ liệu đã học và GPT-4
def query_learned_data(history):
    try:
        question = history[-1]['content']
        
        # Lọc thông tin cụ thể từ dữ liệu đã học dựa trên từ khóa trong câu hỏi
        filtered_data = find_specific_information(question, learned_data)
        
        if filtered_data:
            return "\n".join(filtered_data)  # Trả về dữ liệu cụ thể đã lọc
        else:
            # Khi không có dữ liệu cụ thể, chuyển sang chế độ tư vấn linh hoạt
            messages = [
                {"role": "system", "content": "Bạn là một trợ lý hữu ích chuyên về xe đạp. Hãy tư vấn một cách linh hoạt và hữu ích."},
                {"role": "system", "content": "\n".join(learned_data) },  
                {"role": "user", "content": question}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=4096,  # Giới hạn token tối đa
                temperature=0.7,
                stop=None  
            )
            
            answer = response.choices[0].message['content'].strip()
            
            # Kiểm tra và yêu cầu thêm thông tin nếu câu trả lời bị ngắt quãng
            while len(answer.split()) < 50:
                messages.append({"role": "user", "content": "Hãy tiếp tục."})
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=4096,  # Tiếp tục yêu cầu thêm thông tin
                    temperature=0.7,
                    stop=None
                )
                answer += "\n" + response.choices[0].message['content'].strip()
            
            return answer  # Trả về câu trả lời từ GPT-4
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn GPT-4: {str(e)}")
        return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn."

# ================================================================
# Endpoint để học từ văn bản
@app.route('/learn', methods=['POST'])
def learn():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Không có văn bản được cung cấp"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Văn bản rỗng"}), 400
    
    # Lưu văn bản đã học vào bộ nhớ và tệp văn bản
    learned_data.append(text)
    save_learned_data(text)
    
    return jsonify({"message": "Học văn bản thành công"}), 200

# ================================================================
# Hàm trích xuất văn bản từ hình ảnh và làm sạch dữ liệu
def extract_and_clean_text_from_image(image_content):
    try:
        image = Image.open(io.BytesIO(image_content))
        text = pytesseract.image_to_string(image, lang='eng')
        cleaned_text = clean_text(text)
        return cleaned_text
    except Exception as e:
        logging.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")
        return None

# Phân tích hình ảnh với YOLO
def analyze_image_with_yolo(image_content):
    # Load YOLO model with the correct file paths
    net = cv2.dnn.readNet(r"F:/FPT/python/yolov3.weights", r"F:/FPT/python/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load class names (labels) used by YOLO
    with open(r"F:/FPT/python/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Convert image content to a format YOLO can process
    image = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Store the details
                boxes.append([center_x, center_y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove duplicate boxes
    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    else:
        indexes = []

    detected_labels = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            detected_labels.append(classes[class_ids[i]])

    return detected_labels

# Endpoint để học từ hình ảnh sử dụng YOLO
@app.route('/learn_from_image', methods=['POST'])
def learn_from_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['file']
    try:
        image_content = image_file.read()

        # Trích xuất và làm sạch văn bản từ hình ảnh
        extracted_text = extract_and_clean_text_from_image(image_content)
        
        if not extracted_text:
            return jsonify({"error": "Không thể trích xuất văn bản từ hình ảnh"}), 500

        # Phân tích hình ảnh với YOLO
        labels_from_image = analyze_image_with_yolo(image_content)
        
        # Lưu văn bản và nhãn từ hình ảnh vào dữ liệu học
        learned_data.append(extracted_text)
        save_learned_data(extracted_text)
        
        return jsonify({"message": "Image learned successfully", "extracted_text": extracted_text, "labels": labels_from_image}), 200
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ================================================================
# Endpoint để gửi câu hỏi đến GPT-4 hoặc dữ liệu đã học
@app.route('/ask', methods=['POST'])
def ask():
    history = request.json['history']
    
    # Truy vấn thông tin từ dữ liệu đã học và lịch sử hội thoại
    answer = query_learned_data(history)
    
    return jsonify({"response": answer}), 200

if __name__ == '__main__':
    app.run(debug=True)
