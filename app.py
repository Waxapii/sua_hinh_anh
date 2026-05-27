import os
import io
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# ==============================================================================
# HÀM BỔ TRỢ: CHUYỂN ĐỔI ĐỊNH DẠNG DỮ LIỆU GIỮA FRONTEND VÀ BACKEND
# ==============================================================================

def base64_to_cv2(base64_string):
    """
    NOTE: Giải mã chuỗi Base64 từ trình duyệt gửi lên thành mảng numpy cho OpenCV.
    Tại sao? Vì ảnh truyền qua Internet ở dạng text (Base64), OpenCV không đọc trực tiếp được.
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1] # Tách bỏ phần header 'data:image/png;base64,'
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8) # Đọc dữ liệu thô dạng byte thành mảng 8-bit không dấu
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Giải mã thành ma trận ảnh BGR chuẩn OpenCV
    except Exception:
        return None

def cv2_to_base64(img, ext='.jpg'):
    """
    NOTE: Mã hóa ma trận ảnh OpenCV thành chuỗi Base64 để gửi ngược lại cho trình duyệt hiển thị.
    """
    _, buffer = cv2.imencode(ext, img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    mime = 'image/jpeg' if ext == '.jpg' else 'image/png'
    return f'data:{mime};base64,{img_base64}'

# ==============================================================================
# HÀM XỬ LÝ ẢNH LÕI (CORE IMAGE PROCESSING)
# ==============================================================================

def process_image_core(img, params):
    params = params or {}
    
    # Ép kiểu dữ liệu từ request JavaScript (mặc định dạng chuỗi hoặc số nguyên) sang float 
    # để phục vụ tính toán toán học chính xác cao ở các bước dưới.
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

    # --------------------------------------------------------------------------
    # NOTE QUAN TRỌNG VỀ KIỂU DỮ LIỆU: Tại sao phải dùng np.float32?
    # --------------------------------------------------------------------------
    # - Ảnh gốc đọc bởi OpenCV mặc định ở dạng 'np.uint8' (Số nguyên không dấu 8-bit: từ 0 đến 255).
    # - Nếu giữ nguyên uint8 để tính toán (ví dụ: nhân độ tương phản hoặc cộng độ sáng):
    #   + Lỗi tràn số trên (Overflow): 250 + 10 sẽ bằng 4 (thay vì 260) vì uint8 kịch trần là 255.
    #   + Lỗi tràn số dưới (Underflow): 5 - 10 sẽ bằng 251 (thay vì -5).
    # - Chuyển sang 'np.float32' (Số thực 32-bit) cho phép giá trị vượt qua giới hạn [0, 255], 
    #   giữ được phần thập phân, giúp tính toán mượt mà, không bị mất chi tiết hoặc sai màu sắc.
    img_float = img.astype(np.float32)

    # 1. Ánh sáng: Độ sáng (Brightness) & Tương phản (Contrast)
    # Công thức: New_Pixel = Old_Pixel * Alpha + Brightness
    alpha = 1.0 + contrast / 100.0
    img_float = img_float * alpha + brightness

    # 2. Ánh sáng: Gamma (Chỉnh vùng tối/vùng sáng theo đường cong)
    if gamma != 0:
        g = max(0.1, 1.0 + gamma / 100.0) # Tránh g = 0 gây lỗi chia cho 0
        inv = 1.0 / g
        # Chuẩn hóa ảnh về khoảng [0.0, 1.0], lũy thừa mũ Gauss rồi nhân lại 255
        img_float = 255.0 * np.power(np.clip(img_float / 255.0, 0.0, 1.0), inv)

    # 3. KÊNH RGB (Gain + Offset) - Can thiệp trực tiếp vào 3 kênh màu riêng biệt
    # Lưu ý: Thứ tự các kênh màu trong OpenCV mặc định là BGR (Blue - Green - Red) chứ không phải RGB.
    if any(v != 1.0 for v in [r_gain, g_gain, b_gain]) or any(v != 0 for v in [r_offset, g_offset, b_offset]):
        img_float[:, :, 0] = img_float[:, :, 0] * b_gain + b_offset  # Kênh 0: Blue
        img_float[:, :, 1] = img_float[:, :, 1] * g_gain + g_offset  # Kênh 1: Green
        img_float[:, :, 2] = img_float[:, :, 2] * r_gain + r_offset  # Kênh 2: Red

    # 4. Màu sắc: Nhiệt độ màu (Temperature) & Ám màu (Tint)
    if temp != 0 or tint != 0:
        if temp != 0:
            img_float[:, :, 0] += temp * 0.6  # Tăng Blue (Làm ảnh lạnh hơn) hoặc ngược lại
            img_float[:, :, 2] -= temp * 0.6  # Giảm Red (Làm ảnh bớt ấm) hoặc ngược lại
        if tint != 0:
            img_float[:, :, 1] += tint * 0.6  # Tăng/Giảm sắc xanh lá (Green)

    # 5. Màu sắc: Bão hòa (Saturation), Sắc độ (Hue), Sinh động (Vibrance) trong không gian HSV
    # NOTE không gian màu: Ta phải ép tạm thời về uint8 để hàm `cvtColor` của OpenCV hoạt động chính xác,
    # vì không gian màu HSV yêu cầu các dải giá trị tiêu chuẩn cố định (H: 0-179, S: 0-255, V: 0-255).
    if sat != 0 or hue != 0 or vib != 0:
        hsv = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv) # Tách thành 3 ma trận độc lập
        if hue != 0:
            H = (H + hue) % 180 # Hue dạng vòng tròn từ 0-179 độ nên dùng phép chia lấy dư (%)
        if vib != 0:
            # Vibrance (Độ sinh động): Chỉ tăng bão hòa cho những pixel màu vốn đang nhạt (S thấp)
            boost = (1.0 - S / 255.0) * (vib / 100.0)
            S = S + (S * boost)
        if sat != 0:
            S = S * (1.0 + sat / 100.0) # Saturation: Tăng bão hòa đồng đều toàn bộ ảnh
        
        S = np.clip(S, 0, 255)
        H = np.clip(H, 0, 179)
        hsv = cv2.merge([H, S, V]).astype(np.uint8)
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) # Trả về BGR dạng float32 tiếp tục xử lý

    # 6. Hiệu ứng: Sắc nét (Sharpen)
    # LƯU Ý ĐÃ SỬA: Không ép kiểu về uint8 ở đây như code cũ, thực hiện tính toán làm mờ (Blur) trực tiếp
    # trên ma trận float32 để tránh hiện tượng vỡ ảnh hoặc xuất hiện đốm đen đốm trắng do lệch kiểu dữ liệu.
    if sharpen > 0:
        amount = sharpen / 50.0
        blurred = cv2.GaussianBlur(img_float, (0, 0), 3) # Làm mờ ảnh bằng bộ lọc Gauss
        # Công thức Unsharp Masking: Ảnh sắc nét = Ảnh gốc + Lượng tăng cường * (Ảnh gốc - Ảnh mờ)
        img_float = img_float + amount * (img_float - blurred)

    # 7. Hiệu ứng: Viền tối (Vignette)
    if vignette > 0:
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        mask = (kernel_y * kernel_x.T) / (kernel_y * kernel_x.T).max() # Tạo mặt nạ gradient hình tròn từ tâm sáng ra biên tối
        alpha_v = vignette / 100.0
        img_float = img_float * ((1.0 - alpha_v) + mask[:, :, np.newaxis] * alpha_v)

    # 8. Hiệu ứng: Thang xám (Grayscale)
    if grayscale > 0:
        alpha_g = grayscale / 100.0
        gray = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) # Đưa về ảnh 1 kênh xám
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) # Nhân bản thành 3 kênh xám để đồng bộ với img_float
        img_float = img_float * (1.0 - alpha_g) + gray3 * alpha_g # Hòa trộn giữa ảnh màu và ảnh xám

    # --------------------------------------------------------------------------
    # NOTE TRẢ VỀ: Tại sao phải np.clip và ép ngược về .astype(np.uint8)?
    # --------------------------------------------------------------------------
    # Sau tất cả các phép toán nhân chia cộng trừ trên kiểu float32, giá trị pixel có thể vọt lên 300 hoặc tụt xuống -50.
    # - 'np.clip(..., 0, 255)': Ép tất cả các giá trị nhỏ hơn 0 thành 0, lớn hơn 255 thành 255.
    # - '.astype(np.uint8)': Chuyển đổi ngược từ số thực về số nguyên 8-bit chuẩn. 
    # Đây là định dạng bắt buộc mà tất cả các thư viện hiển thị ảnh (và trình duyệt web) yêu cầu để có thể đọc được file ảnh.
    return np.clip(img_float, 0, 255).astype(np.uint8)

# ==============================================================================
# CÁC ROUTE ĐIỀU HƯỚNG FLASK (WEB API)
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    LƯU Ý ĐÃ SỬA: Code cũ lưu ảnh xuống ổ cứng server (`uploads/current.png`). 
    Code mới đọc ảnh trực tiếp từ bộ nhớ RAM của luồng request đó, chuyển thành Base64 rồi trả ngay về cho Client.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    filestr = file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400
        
    return jsonify({'image': cv2_to_base64(img)})

@app.route('/process', methods=['POST'])
def process():
    """
    LƯU Ý ĐÃ SỬA: Trình duyệt gửi lên JSON gồm { image: "chuỗi base64 gốc", params: { ... } }.
    Nếu `params` không tồn tại, backend sẽ lấy tất cả các trường khác ngoài `image` làm tham số.
    Server giải mã ảnh, xử lý theo tham số rồi trả về Base64. Hoàn toàn phi trạng thái, không lưu file trên ổ cứng.
    """
    data = request.get_json(silent=True) or {}
    base64_str = data.get('image')
    params = data.get('params')
    if params is None:
        params = {k: v for k, v in data.items() if k != 'image'}
    
    if not base64_str:
        return jsonify({'error': 'No image data provided'}), 400
        
    img = base64_to_cv2(base64_str)
    if img is None:
        return jsonify({'error': 'Invalid image data'}), 400
        
    processed_img = process_image_core(img, params)
    return jsonify({'image': cv2_to_base64(processed_img)})

@app.route('/download', methods=['POST'])
def download():
    """
    Nhận JSON gồm { image: "chuỗi base64 gốc", params: { ... } }.
    Nếu `params` không tồn tại, backend sẽ lấy tất cả các trường khác ngoài `image` làm tham số.
    Xử lý ảnh và trả về file PNG tải xuống, không sử dụng ổ cứng.
    """
    data = request.get_json(silent=True) or {}
    base64_str = data.get('image')
    params = data.get('params')
    if params is None:
        params = {k: v for k, v in data.items() if k != 'image'}
    
    if not base64_str:
        return jsonify({'error': 'No image data provided'}), 400
        
    img = base64_to_cv2(base64_str)
    if img is None:
        return jsonify({'error': 'Invalid image data'}), 400

    processed_img = process_image_core(img, params)
    is_success, buffer = cv2.imencode('.png', processed_img) # Mã hóa định dạng PNG chất lượng cao không nén vỡ hình
    if not is_success:
        return jsonify({'error': 'Failed to encode image'}), 500
        
    io_buf = io.BytesIO(buffer) # Tạo file ảo trong bộ nhớ RAM (In-memory stream)
    return send_file(io_buf, mimetype='image/png', as_attachment=True, download_name='anh_da_chinh_sua.png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
