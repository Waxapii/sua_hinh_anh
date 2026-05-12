import os
import io
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Xử lý ảnh 
def process_image_core(img, params):
    params = params or {}
    
    # Đọc và chuẩn bị tham số từ ảnh
    brightness = float(params.get('brightness', 0))
    contrast = float(params.get('contrast', 0))
    gamma = float(params.get('gamma', 0))
    sat = float(params.get('saturation', 0))
    hue = float(params.get('hue', 0))
    vib = float(params.get('vibrance', 0))
    temp = float(params.get('temperature', 0))
    tint = float(params.get('tint', 0))
    r_gain = float(params.get('r_gain', 0)) / 100.0 + 1.0
    g_gain = float(params.get('g_gain', 0)) / 100.0 + 1.0
    b_gain = float(params.get('b_gain', 0)) / 100.0 + 1.0
    r_offset = float(params.get('r_offset', 0))
    g_offset = float(params.get('g_offset', 0))
    b_offset = float(params.get('b_offset', 0))
    sharpen = float(params.get('sharpen', 0))
    vignette = float(params.get('vignette', 0)) 
    grayscale = float(params.get('grayscale', 0))

    img_float = img.astype(np.float32)

    # 1. Ánh sáng: Độ sáng & Tương phản
    alpha = 1.0 + contrast / 100.0
    img_float = img_float * alpha + brightness

    # 2. Ánh sáng: Gamma
    if gamma != 0:
        g = 1.0 + gamma / 100.0
        g = max(0.1, g)
        inv = 1.0 / g
        img_float = 255.0 * np.power(np.clip(img_float / 255.0, 0.0, 1.0), inv)

    # 3. KÊNH RGB (Gain + Offset)
    if any(v != 1.0 for v in [r_gain, g_gain, b_gain]) or any(v != 0 for v in [r_offset, g_offset, b_offset]):
        img_float[:, :, 0] = img_float[:, :, 0] * b_gain + b_offset
        img_float[:, :, 1] = img_float[:, :, 1] * g_gain + g_offset
        img_float[:, :, 2] = img_float[:, :, 2] * r_gain + r_offset

    # 4. Màu sắc: Nhiệt độ màu & Ám màu
    if temp != 0 or tint != 0:
        if temp != 0:
            img_float[:, :, 0] += temp * 0.6  # Blue
            img_float[:, :, 2] -= temp * 0.6  # Red
        if tint != 0:
            img_float[:, :, 1] += tint * 0.6  # Green

    # 5. Màu sắc: Bão hòa, Hue, Vibrance trong không gian HSV
    if sat != 0 or hue != 0 or vib != 0:
        hsv = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        if hue != 0:
            H = (H + hue) % 180
        if vib != 0:
            boost = (1.0 - S / 255.0) * (vib / 100.0)
            S = S + (S * boost)
        if sat != 0:
            S = S * (1.0 + sat / 100.0)
        S = np.clip(S, 0, 255)
        H = np.clip(H, 0, 179)
        hsv = cv2.merge([H, S, V]).astype(np.uint8)
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    # 6. Hiệu ứng: Sắc nét
    if sharpen > 0:
        amount = sharpen / 50.0
        blurred = cv2.GaussianBlur(np.clip(img_float, 0, 255).astype(np.uint8), (0, 0), 3).astype(np.float32)
        img_float = img_float + amount * (img_float - blurred)

    # 7. Hiệu ứng: Viền tối
    if vignette > 0:
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        mask = (kernel_y * kernel_x.T) / (kernel_y * kernel_x.T).max()
        alpha_v = vignette / 100.0
        img_float = img_float * ((1.0 - alpha_v) + mask[:, :, np.newaxis] * alpha_v)

    # 8. Hiệu ứng: Thang xám
    if grayscale > 0:
        alpha_g = grayscale / 100.0
        gray = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        img_float = img_float * (1.0 - alpha_g) + gray3 * alpha_g
# Trả về ảnh đã xử lý, đảm bảo giá trị nằm trong khoảng 0-255 và kiểu dữ liệu uint8
    return np.clip(img_float, 0, 255).astype(np.uint8)

@app.route('/')
def index():
    return render_template('index.html')
# tải ảnh lên
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    file.save(filepath)
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})
# xử lý ảnh và trả về ảnh đã chỉnh sửa
@app.route('/process', methods=['POST'])
# xem trước ảnh đã chỉnh sửa mà không cần tải về    
def process():
    params = request.get_json(silent=True) or {}
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404
    processed_img = process_image_core(img, params)
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})
# tải ảnh đã chỉnh sửa về
@app.route('/download', methods=['POST'])
def download():
    params = request.get_json(silent=True) or {}
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    img = cv2.imread(filepath)
    
    if img is None:
        return jsonify({'error': 'Image not found'}), 404

    # Xử lý ảnh với các thông số hiện tại
    processed_img = process_image_core(img, params)
    
    # Mã hóa ảnh sang định dạng PNG trong bộ nhớ (RAM) thay vì lưu ra ổ cứng
    is_success, buffer = cv2.imencode('.png', processed_img)
    if not is_success:
        return jsonify({'error': 'Failed to encode image'}), 500
        
    # Chuyển đổi buffer thành dạng file nhị phân (BytesIO) để gửi đi
    io_buf = io.BytesIO(buffer)
    
    # Trả thẳng file về cho trình duyệt tải xuống
    return send_file(io_buf, mimetype='image/png', as_attachment=True, download_name='anh_da_chinh_sua.png')
# Chạy ứng dụng Flask
if __name__ == '__main__':
    # Lấy cổng từ biến môi trường của Render, mặc định là 5000 nếu chạy cục bộ
    port = int(os.environ.get("PORT", 5000))
    # Phải đặt host='0.0.0.0' để Render có thể truy cập vào ứng dụng
    app.run(host='0.0.0.0', port=port)