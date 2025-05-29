# Báo Cáo: Nhận Diện Khuôn Mặt Sử Dụng Hàm Mất Mát Triplet Với Mô Hình ResNet18

## Giới Thiệu

Nhận diện khuôn mặt là một lĩnh vực quan trọng trong thị giác máy tính, đóng vai trò thiết yếu trong các ứng dụng như bảo mật, xác minh danh tính, giám sát, và cá nhân hóa trải nghiệm người dùng. Dự án này triển khai một hệ thống nhận diện khuôn mặt tiên tiến, tận dụng hàm mất mát Triplet kết hợp với kiến trúc ResNet18 tùy chỉnh để tạo ra các embedding khuôn mặt chất lượng cao. Hệ thống không chỉ phát hiện khuôn mặt từ các ảnh trong cấu trúc thư mục lồng nhau mà còn huấn luyện một mô hình học sâu để đảm bảo rằng các embedding của cùng một người gần nhau hơn trong không gian đặc trưng so với các embedding của người khác.

Dự án sử dụng tập dữ liệu bao gồm hơn 200.000 ảnh, áp dụng kỹ thuật phát hiện khuôn mặt bằng Haar Cascade và huấn luyện mô hình để tạo ra các embedding 128 chiều nhằm hỗ trợ xác minh danh tính. Sự kết hợp giữa các phương pháp truyền thống và học sâu hiện đại giúp hệ thống đạt hiệu suất cao, đồng thời mở ra tiềm năng ứng dụng trong các kịch bản thực tế.

## Mô Tả Vấn Đề

Mục tiêu chính của dự án là xây dựng một hệ thống nhận diện khuôn mặt toàn diện, đáp ứng các yêu cầu sau:
1. **Phát hiện khuôn mặt**: Xác định và trích xuất các khuôn mặt từ ảnh trong cấu trúc thư mục lồng nhau.
2. **Tạo embedding**: Sử dụng mô hình học sâu để chuyển đổi các khuôn mặt thành các vector đặc trưng 128 chiều.
3. **Huấn luyện mô hình**: Áp dụng hàm mất mát Triplet để đảm bảo khoảng cách giữa các embedding của cùng một người nhỏ hơn so với khoảng cách giữa các embedding của người khác.
4. **Đánh giá hiệu suất**: Đo lường khả năng phân biệt danh tính của mô hình trên tập huấn luyện và kiểm tra, sử dụng các chỉ số như mất mát và độ chính xác.

Hệ thống được thiết kế để xử lý các thách thức như biến đổi ánh sáng, góc nhìn, biểu cảm khuôn mặt, và cấu trúc thư mục phức tạp, đồng thời tối ưu hóa hiệu suất trên tập dữ liệu thực tế.

## I. Phương Pháp

### 1. Phát Hiện Khuôn Mặt
Hệ thống sử dụng bộ phân loại Haar Cascade của OpenCV (`haarcascade_frontalface_default.xml`) để phát hiện khuôn mặt trong ảnh. Quy trình bao gồm:
- **Tiền xử lý**: Chuyển đổi ảnh màu RGB sang thang độ xám để giảm độ phức tạp tính toán và tăng tốc độ xử lý.
- **Phát hiện khuôn mặt**: Áp dụng bộ phân loại Haar Cascade với các tham số:
  - `scaleFactor=1.1`: Điều chỉnh tỷ lệ thu nhỏ để phát hiện khuôn mặt ở các kích thước khác nhau.
  - `minNeighbors=5`: Yêu cầu số lượng láng giềng tối thiểu để xác nhận một vùng là khuôn mặt, giảm thiểu phát hiện sai.
  - `minSize=(30, 30)`: Đặt kích thước tối thiểu của khuôn mặt để lọc bỏ các vùng quá nhỏ.
- **Lưu kết quả**: Các khuôn mặt được cắt và lưu vào thư mục đầu ra `Faces_detected` với cấu trúc thư mục tương ứng với thư mục gốc.

Hàm `detect_faces_in_nested_folders` được thiết kế để xử lý cấu trúc thư mục lồng nhau, đảm bảo khả năng xử lý lỗi (ví dụ: ảnh không đọc được) và trả về một từ điển ghi lại số lượng khuôn mặt phát hiện trong mỗi ảnh.

### 2. Chuẩn Bị Dữ Liệu
Dữ liệu được tổ chức và chuẩn bị thông qua lớp `TripletFaceDataset`, được thiết kế để tạo ra các bộ ba (anchor, positive, negative) cho huấn luyện:
- **Anchor**: Một ảnh của một người bất kỳ.
- **Positive**: Một ảnh khác của cùng người đó.
- **Negative**: Một ảnh của một người khác.
- **Biến đổi dữ liệu**:
  - **Huấn luyện**:
    - Thay đổi kích thước ảnh thành 224x224 để phù hợp với đầu vào của mô hình ResNet18.
    - Lật ngang ngẫu nhiên để tăng cường dữ liệu, mô phỏng các góc nhìn khác nhau.
    - Điều chỉnh màu sắc (độ sáng=0.2, độ tương phản=0.2) để mô phỏng các điều kiện ánh sáng thực tế.
    - Chuyển thành tensor để tương thích với PyTorch.
  - **Kiểm tra**:
    - Thay đổi kích thước thành 224x224.
    - Chuyển thành tensor mà không áp dụng tăng cường dữ liệu để đảm bảo tính nhất quán trong đánh giá.

**Ví dụ về cấu trúc thư mục lồng nhau**:
Tập dữ liệu huấn luyện (`Combined_dataset`) được tổ chức theo cấu trúc lồng nhau, trong đó mỗi thư mục con đại diện cho một cá nhân, và các tệp ảnh được lưu bên trong. Ví dụ:
```
/kaggle/working/combined_dataset/
├── person_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── image_003.jpg
├── person_002/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── image_003.jpg
└── person_003/
    ├── image_001.jpg
    └── image_002.jpg
```
Trong cấu trúc này, thư mục `person_001` chứa các ảnh của một cá nhân (ví dụ: `image_001.jpg`, `image_002.jpg`), và tương tự cho các thư mục `person_002`, `person_003`. Sau khi phát hiện khuôn mặt, thư mục đầu ra `Faces_detected` sẽ có cấu trúc tương tự, nhưng chứa các ảnh khuôn mặt đã cắt, ví dụ:
```
/kaggle/working/Faces_detected/
├── person_001/
│   ├── face_0_result_image_001.jpg
│   ├── face_0_result_image_002.jpg
│   └── face_1_result_image_002.jpg
├── person_002/
│   ├── face_0_result_image_001.jpg
│   └── face_0_result_image_002.jpg
└── person_003/
    ├── face_0_result_image_001.jpg
```
Lớp `TripletFaceDataset` sử dụng cấu trúc này để chọn ngẫu nhiên các ảnh từ cùng một thư mục cho cặp anchor-positive và từ một thư mục khác cho negative, đảm bảo tính đa dạng trong các bộ ba.

Tập dữ liệu huấn luyện được lấy từ `Faces_detected`, trong khi tập kiểm tra sử dụng dữ liệu từ `input/vggface2/val`. Lớp `TripletFaceDataset` tạo ra 10,000 bộ ba, đảm bảo tính ngẫu nhiên và đa dạng trong quá trình huấn luyện.

### 3. Kiến Trúc ResNet18
ResNet18 là một mạng nơ-ron tích chập (CNN) thuộc họ ResNet (Residual Network), được giới thiệu bởi He và cộng sự trong bài báo "Deep Residual Learning for Image Recognition" (2015). Mô hình này được chọn cho dự án nhờ vào hiệu suất cao, độ phức tạp tính toán vừa phải, và khả năng học các đặc trưng phức tạp thông qua cơ chế kết nối tắt (skip connections).

#### Cấu trúc của ResNet18
ResNet18 bao gồm 18 tầng (layers) với các khối cơ bản (`BasicBlock`) sử dụng kết nối tắt để giải quyết vấn đề biến mất gradient (vanishing gradient) trong các mạng sâu:
- **Lớp tích chập đầu tiên**: Một lớp tích chập 7x7 (stride=2) với 64 bộ lọc, tiếp theo là chuẩn hóa hàng loạt (batch normalization) và hàm kích hoạt ReLU, giảm kích thước không gian của ảnh đầu vào.
- **Max Pooling**: Lớp max pooling 3x3 (stride=2) để giảm thêm kích thước không gian, chuẩn bị dữ liệu cho các tầng ResNet.
- **Bốn tầng ResNet**:
  - `layer1`: 2 khối `BasicBlock`, mỗi khối gồm hai lớp tích chập 3x3 với 64 bộ lọc.
  - `layer2`: 2 khối `BasicBlock`, mỗi khối với 128 bộ lọc, sử dụng stride=2 để giảm kích thước.
  - `layer3`: 2 khối `BasicBlock`, mỗi khối với 256 bộ lọc, tiếp tục giảm kích thước.
  - `layer4`: 2 khối `BasicBlock`, mỗi khối với 512 bộ lọc, tạo ra các đặc trưng cấp cao.
- **Average Pooling**: Lớp average pooling thích nghi chuyển đổi đầu ra thành vector 1x1 (512 chiều).
- **Lớp kết nối đầy đủ**: Trong phiên bản tùy chỉnh, lớp này được thay thế để tạo ra embedding 128 chiều, được chuẩn hóa L2 để đảm bảo tính nhất quán trong không gian đặc trưng.

#### Kết nối tắt (Skip Connections)
Mỗi `BasicBlock` trong ResNet18 sử dụng kết nối tắt để cộng đầu vào của khối với đầu ra sau hai lớp tích chập, giúp:
- Giảm thiểu vấn đề biến mất gradient, cho phép huấn luyện các mạng sâu hơn.
- Tăng cường luồng thông tin qua các tầng, cải thiện khả năng học các đặc trưng phức tạp.

#### Lý do chọn ResNet18
ResNet18 được chọn vì:
- **Hiệu quả tính toán**: Với 18 tầng, ResNet18 có độ phức tạp thấp hơn so với các mô hình sâu hơn như ResNet50 hay ResNet101, phù hợp với tài nguyên tính toán.
- **Hiệu suất đã được chứng minh**: ResNet18 đạt hiệu suất cao trên ImageNet, và trọng số được huấn luyện trước (`ResNet18_Weights.IMAGENET1K_V1`) cung cấp điểm khởi đầu tốt để chuyển giao học tập (transfer learning).
- **Tính linh hoạt**: Kiến trúc có thể được tùy chỉnh dễ dàng để phù hợp với nhiệm vụ nhận diện khuôn mặt, chẳng hạn như thay đổi lớp kết nối đầy đủ để tạo embedding 128 chiều.
- **Khả năng học đặc trưng**: Kết nối tắt giúp mô hình học được các đặc trưng phân cấp (hierarchical features), từ các đặc trưng cấp thấp (cạnh, góc) đến cấp cao (đặc điểm khuôn mặt).

Trong dự án, ResNet18 được tùy chỉnh bằng cách thay thế lớp kết nối đầy đủ cuối cùng để tạo ra embedding 128 chiều, và các trọng số được tải trước từ ImageNet được áp dụng cho 120 tham số tương thích, giúp tăng tốc độ hội tụ và cải thiện hiệu suất ban đầu.

### 4. Hàm Mất Mát Triplet
Hàm mất mát Triplet là trung tâm của quá trình huấn luyện, với mục tiêu tối ưu hóa không gian embedding:
$$
\mathcal{L} = \max(d(a, p) - d(a, n) + \text{margin}, 0)
$$

trong đó:
- $d(a, p)$: Khoảng cách Euclidean giữa embedding của anchor và positive.
- $d(a, n)$: Khoảng cách Euclidean giữa embedding của anchor và negative.
- `margin=1.0`: Biên độ để đảm bảo sự tách biệt giữa các cặp positive và negative.

Hàm mất mát này khuyến khích các embedding của cùng một người gần nhau hơn so với các embedding của người khác, tạo ra một không gian đặc trưng mạnh mẽ và phân biệt tốt cho nhận diện khuôn mặt.

### 5. Huấn Luyện và Đánh Giá
Mô hình được huấn luyện trong 100 epoch với các thiết lập sau:
- **Bộ tối ưu hóa**: Adam với tốc độ học $10^{-4}$ và suy giảm trọng số $10^{-5}$ để ổn định gradient và tránh quá khớp.
- **Kích thước lô**: 32, cân bằng giữa hiệu quả tính toán và độ chính xác của gradient.
- **DataLoader**: Sử dụng xáo trộn ngẫu nhiên cho tập huấn luyện và 2 worker để tăng tốc độ tải dữ liệu.

Các chỉ số đánh giá bao gồm:
- **Mất mát**: Trung bình của hàm mất mát Triplet trên mỗi lô, phản ánh mức độ lỗi trong việc sắp xếp các embedding.
- **Độ chính xác**: Tỷ lệ các bộ ba mà khoảng cách anchor-positive nhỏ hơn khoảng cách anchor-negative, được tính bằng hàm `compute_accuracy`.

Sau mỗi epoch, mô hình được đánh giá trên tập kiểm tra để theo dõi khả năng tổng quát hóa. Mô hình cuối cùng được lưu tại `final_triplet_resnet18_ss2.pth` sau khi hoàn thành huấn luyện.

## Chi Tiết Triển Khai

Dự án được triển khai với mã nguồn rõ ràng, được tổ chức để đảm bảo tính mạnh mẽ, dễ bảo trì, và khả năng tái sử dụng:
- **Phát hiện khuôn mặt**: Hàm `detect_faces_in_nested_folders` xử lý lỗi một cách mạnh mẽ (ví dụ: ảnh không đọc được hoặc định dạng không hợp lệ) và tự động tạo các thư mục đầu ra cần thiết.
- **Tập dữ liệu**: Lớp `TripletFaceDataset` tạo ra các bộ ba một cách động, sử dụng cấu trúc thư mục lồng nhau để chọn ngẫu nhiên các ảnh, đảm bảo tính đa dạng và tránh thiên vị.
- **Mô hình**: ResNet18 tùy chỉnh được khởi tạo với 120 tham số từ trọng số ImageNet, giảm thời gian huấn luyện và cải thiện hiệu suất ban đầu.
- **Huấn luyện**: Vòng lặp huấn luyện được tối ưu hóa để xử lý các lô dữ liệu lớn, với các thông báo chi tiết về mất mát và độ chính xác được in ra sau mỗi epoch để theo dõi tiến trình.
- **Lưu trữ**: Mô hình được lưu sau khi hoàn thành 100 epoch, đảm bảo rằng trạng thái tốt nhất được bảo toàn cho các ứng dụng sau này.

Quá trình xử lý dữ liệu và huấn luyện được thực hiện trên môi trường Kaggle, tận dụng tài nguyên GPU để tăng tốc độ tính toán và đảm bảo hiệu quả.

## Kết Quả

Quá trình huấn luyện trong 100 epoch mang lại các kết quả đáng chú ý, minh chứng cho hiệu quả của hệ thống:
- **Mất mát huấn luyện**: Giảm đều đặn từ 0.6090 (epoch 1) xuống 0.2537 (epoch 100), cho thấy mô hình học được các đặc trưng khuôn mặt một cách hiệu quả.
- **Độ chính xác huấn luyện**: Tăng từ 0.7706 lên 0.9474, thể hiện khả năng phân biệt danh tính mạnh mẽ trên tập huấn luyện.
- **Mất mát kiểm tra**: Dao động từ mức thấp nhất 0.5360 (epoch 22) đến cao nhất 0.7430 (epoch 90), kết thúc ở 0.6535, cho thấy sự biến động trong khả năng tổng quát hóa.
- **Độ chính xác kiểm tra**: Đạt đỉnh tại 0.8159 (epoch 22), nhưng giảm nhẹ xuống 0.7462 vào cuối quá trình, phản ánh một số thách thức trong việc áp dụng mô hình trên dữ liệu mới.

Bảng dưới đây tóm tắt các chỉ số tại một số mốc quan trọng:

| Epoch | Mất Mát Huấn Luyện | Độ Chính Xác Huấn Luyện | Mất Mát Kiểm Tra | Độ Chính Xác Kiểm Tra |
|-------|--------------------|-------------------------|------------------|-----------------------|
| 1     | 0.6090            | 0.7706                 | 0.6438           | 0.7609               |
| 10    | 0.3775            | 0.8732                 | 0.6239           | 0.7637               |
| 50    | 0.2900            | 0.9242                 | 0.6494           | 0.7497               |
| 100   | 0.2537            | 0.9474                 | 0.6535           | 0.7462               |

Để minh họa chất lượng tiền xử lý, một ảnh khuôn mặt được cắt đã được hiển thị trong quá trình phát hiện khuôn mặt, sử dụng matplotlib để trực quan hóa kết quả. Điều này xác nhận rằng các khuôn mặt được trích xuất chính xác và phù hợp để đưa vào quá trình huấn luyện.

## Thảo Luận

Hệ thống thể hiện hiệu quả vượt trội trong việc học các đặc trưng trên tập huấn luyện, đạt độ chính xác gần 95% và mất mát huấn luyện rất thấp (0.2537). Tuy nhiên, độ chính xác kiểm tra dao động quanh mức 75-80%, cho thấy một số hạn chế trong khả năng tổng quát hóa. Các nguyên nhân tiềm tàng bao gồm:
- **Độ đa dạng dữ liệu**: Tập dữ liệu huấn luyện (`Faces_detected`) có thể thiếu sự đa dạng về góc nhìn, ánh sáng, biểu cảm, hoặc bối cảnh, dẫn đến hiệu suất kiểm tra không ổn định.
- **Chênh lệch dữ liệu**: Sự khác biệt trong chất lượng ảnh, tiền xử lý, hoặc phân bố dữ liệu giữa tập huấn luyện và kiểm tra (`input/vggface2/val`) có thể ảnh hưởng đến hiệu suất.
- **Siêu tham số**: Biên độ 1.0 trong hàm mất mát Triplet và tốc độ học $10^{-4}$ có thể chưa được tối ưu hóa hoàn toàn cho tập dữ liệu này.
- **Cấu trúc thư mục**: Mặc dù cấu trúc lồng nhau giúp tổ chức dữ liệu hiệu quả, nhưng nếu số lượng ảnh trong mỗi thư mục (cá nhân) không đồng đều, điều này có thể gây ra thiên vị trong việc chọn bộ ba.

Để cải thiện hiệu suất, các hướng tiếp cận sau được đề xuất:
1. **Tăng cường dữ liệu**: Áp dụng thêm các biến đổi như xoay ảnh, cắt ngẫu nhiên, thay đổi độ bão hòa, hoặc thêm nhiễu để tăng tính mạnh mẽ của mô hình trước các biến thể thực tế.
2. **Tối ưu hóa siêu tham số**: Thử nghiệm các giá trị biên độ khác nhau (ví dụ: 0.5, 1.5, 2.0) và điều chỉnh tốc độ học theo lịch trình giảm dần (learning rate scheduling) để cải thiện hội tụ.
3. **Tăng quy mô dữ liệu**: Sử dụng tập dữ liệu lớn hơn, chẳng hạn như toàn bộ tập VGGFace2, hoặc kết hợp nhiều nguồn dữ liệu để tăng tính đa dạng.
4. **Hard triplet mining**: Ưu tiên các bộ ba khó (nơi khoảng cách anchor-positive và anchor-negative gần nhau) để cải thiện khả năng học các đặc trưng phân biệt.
5. **Cân bằng dữ liệu**: Đảm bảo rằng mỗi thư mục trong cấu trúc lồng nhau có số lượng ảnh tương đương, tránh thiên vị khi chọn bộ ba.

## Kết Luận

Dự án đã xây dựng thành công một hệ thống nhận diện khuôn mặt toàn diện, từ phát hiện khuôn mặt bằng Haar Cascade, tổ chức dữ liệu trong cấu trúc thư mục lồng nhau, đến huấn luyện mô hình học sâu sử dụng hàm mất mát Triplet. Việc sử dụng ResNet18 tùy chỉnh, với trọng số được huấn luyện trước từ ImageNet, đã mang lại hiệu quả vượt trội, đạt độ chính xác huấn luyện 94.74% và mất mát huấn luyện 0.2537 sau 100 epoch. Mặc dù độ chính xác kiểm tra dao động quanh 75-80%, hệ thống vẫn thể hiện tiềm năng lớn trong các ứng dụng thực tế như xác minh danh tính, kiểm soát truy cập, hoặc nhận diện tự động.

Mô hình cuối cùng, được lưu tại `final_triplet_resnet18_ss2.pth`, sẵn sàng cho các tác vụ xác minh khuôn mặt và có thể được cải tiến thêm thông qua các kỹ thuật tối ưu hóa đề xuất. Với sự kết hợp giữa kỹ thuật truyền thống (Haar Cascade) và học sâu hiện đại (ResNet18, Triplet Loss), dự án không chỉ đạt được hiệu quả kỹ thuật mà còn mở ra các hướng phát triển cho các hệ thống nhận diện khuôn mặt tiên tiến hơn trong tương lai.

## II. Nhận diện
1. **Xây dựng thư viện nhúng và nhận diện khuôn mặt**: Sử dụng mạng Triplet dựa trên ResNet18 để tạo nhúng (embeddings) cho các ảnh trong thư viện và thực hiện nhận diện khuôn mặt từ ảnh đầu vào hoặc frame camera.
2. **Giao diện người dùng (GUI)**: Sử dụng Tkinter để tạo giao diện cho phép người dùng tải ảnh hoặc sử dụng camera để nhận diện khuôn mặt thời gian thực.

Hệ thống được thiết kế để chạy trên môi trường có GPU (CUDA) và sử dụng các thư viện như PyTorch, OpenCV, face_recognition, PIL và Tkinter.

## 1. Xây Dựng Thư Viện Nhúng và Nhận Diện Khuôn Mặt

### Mục Đích
Xây dựng một thư viện nhúng khuôn mặt từ một thư mục ảnh và cung cấp các hàm để nhận diện khuôn mặt từ ảnh hoặc frame camera, sử dụng mô hình Triplet đã được huấn luyện trước đó.

### Phương Pháp
- **Mô hình TripletNet**:
  - Dựa trên ResNet18, với lớp cuối cùng được thay thế bằng một lớp tuyến tính để tạo nhúng 128 chiều.
  - Chuẩn hóa L2 được áp dụng cho nhúng để đảm bảo tính nhất quán trong không gian nhúng.
  - Hàm `forward` xử lý ba đầu vào (anchor, positive, negative) để sử dụng trong huấn luyện triplet, nhưng chỉ sử dụng `forward_once` để tạo nhúng trong quá trình nhận diện.
- **Xây dựng thư viện nhúng**:
  - Hàm `build_individual_gallery_embeddings` duyệt qua thư mục ảnh (`./input_test`), tạo nhúng cho từng ảnh bằng mô hình TripletNet.
  - Mỗi nhúng được lưu trong một từ điển với khóa là `person_name_img_name` và giá trị là cặp `(person_name, embedding)`.
- **Nhận diện khuôn mặt**:
  - **Phương pháp cơ bản (`identify_image`)**: Tính khoảng cách Euclidean giữa nhúng của ảnh đầu vào và tất cả nhúng trong thư viện, chọn người có khoảng cách nhỏ nhất nếu nhỏ hơn ngưỡng (threshold=0.4).
  - **Phương pháp voting (`identify_image_voting`)**: Lấy top-k (mặc định k=5) ảnh có khoảng cách nhỏ nhất, thực hiện voting để chọn người xuất hiện nhiều nhất, chỉ tính các ảnh có khoảng cách dưới ngưỡng.
  - **Nhận diện từ frame camera (`identify_frame`)**: Sử dụng `face_recognition` để phát hiện khuôn mặt, cắt vùng khuôn mặt, tạo nhúng và so sánh với thư viện nhúng để nhận diện.
- **Đầu vào**:
  - Mô hình đã huấn luyện (`final_triplet_resnet18_ss2.pth`).
  - Thư mục ảnh (`./input_test`) chứa các thư mục con, mỗi thư mục con chứa ảnh của một người.
- **Đầu ra**:
  - Thư viện nhúng (`gallery_embeddings`).
  - Kết quả nhận diện: tên người, khoảng cách nhỏ nhất, và tên ảnh khớp (nếu có).

### Chi Tiết Triển Khai
- **Biến đổi dữ liệu**:
  - Sử dụng `transforms.Compose` để thay đổi kích thước ảnh thành 224x224, chuyển thành tensor, và chuẩn hóa với mean=[0.485, 0.456, 0.406] và std=[0.229, 0.224, 0.225].
- **Xử lý lỗi**: Kiểm tra định dạng ảnh hợp lệ (JPG, JPEG, PNG) và xử lý ngoại lệ khi mở ảnh.
- **Hiệu suất**: Sử dụng GPU (CUDA) để tăng tốc tính toán nhúng.

## 2. Giao Diện Người Dùng (GUI)

### Mục Đích
Cung cấp một giao diện sử dụng Tkinter để người dùng tương tác với hệ thống nhận diện khuôn mặt, hỗ trợ cả tải ảnh và nhận diện thời gian thực qua camera.

### Phương Pháp
- **Giao diện chính**:
  - Sử dụng Tkinter để tạo cửa sổ với kích thước 1200x700, nền màu sáng (`#e6ecf0`).
  - Bao gồm:
    - Tiêu đề "HỆ THỐNG NHẬN DIỆN KHUÔN MẶT".
    - Frame chứa ba nút: "Chọn ảnh để nhận diện", "Bật Camera", "Dừng Camera".
    - Frame hiển thị ảnh hoặc frame camera (kích thước 500x500).
    - Nhãn hiển thị kết quả nhận diện (tên người, khoảng cách, độ chính xác).
- **Chức năng tải ảnh (`upload_and_identify`)**:
  - Cho phép người dùng chọn ảnh (JPG, JPEG, PNG) qua hộp thoại `filedialog`.
  - Hiển thị ảnh thu nhỏ (300x300) trên giao diện.
  - Sử dụng hàm `identify_image` để nhận diện, so sánh tên ảnh với tên người dự đoán để đánh giá tính đúng/sai.
  - Hiển thị kết quả với màu chữ xanh (đúng) hoặc đỏ (sai).
- **Chức năng camera thời gian thực (`start_camera`, `stop_camera`, `identify_frame`, `display_frame`)**:
  - Mở camera (`cv2.VideoCapture(0)`) để nhận frame.
  - Sử dụng `face_recognition` để phát hiện khuôn mặt, cắt vùng khuôn mặt, và nhận diện bằng cách so sánh nhúng với thư viện.
  - Vẽ hình chữ nhật và nhãn (tên người, khoảng cách) lên frame.
  - Cập nhật frame mỗi 50ms trên giao diện Tkinter.
  - Nút "Dừng Camera" dừng camera và xóa frame hiển thị.

### Chi Tiết Triển Khai
- **Thiết kế giao diện**:
  - Nút bấm sử dụng màu sắc hiện đại (xanh dương, xanh ngọc, đỏ nhạt) với hiệu ứng `activebackground`.
  - Nhãn kết quả sử dụng font Helvetica, căn giữa, màu chữ tối (`#2c3e50`).
- **Xử lý frame camera**:
  - Frame OpenCV (BGR) được chuyển sang RGB, xử lý khuôn mặt, và hiển thị bằng `ImageTk.PhotoImage`.
  - Kích thước frame được thay đổi thành 500x500 để phù hợp với giao diện.
- **Ngưỡng nhận diện**: Mặc định 0.7 cho camera, 0.4 cho nhận diện ảnh tĩnh.
- **Xử lý lỗi**: Kiểm tra lỗi mở camera, đọc frame, và xử lý trường hợp không phát hiện khuôn mặt.

## Ghi Chú Triển Khai
- **Môi trường**: Yêu cầu GPU (CUDA) để tính toán nhúng nhanh. Mô hình và thư viện ảnh được tải từ đường dẫn cụ thể (`./final_triplet_resnet18_ss2.pth`, `./input_test`).
- **Phụ thuộc**: PyTorch, torchvision, OpenCV, face_recognition, PIL, Tkinter.
- **Hiệu suất**: Nhận diện thời gian thực phụ thuộc vào tốc độ camera và khả năng xử lý của GPU.
- **Khả năng mở rộng**: Hệ thống có thể mở rộng bằng cách thêm các phương pháp nhận diện khác (như voting) hoặc hỗ trợ nhiều camera.

## Cải Tiến Tiềm Năng
- **Nhận diện khuôn mặt**:
  - Tích hợp phương pháp voting (`identify_image_voting`) vào nhận diện camera để tăng độ chính xác.
  - Thử nghiệm các ngưỡng khác nhau hoặc tự động điều chỉnh ngưỡng dựa trên dữ liệu.
- **Giao diện**:
  - Thêm chức năng lưu kết quả nhận diện (ảnh hoặc video).
  - Hiển thị top-k kết quả thay vì chỉ một kết quả để cung cấp thêm thông tin.
- **Hiệu suất**:
  - Tối ưu hóa xử lý frame camera để giảm độ trễ.
  - Sử dụng mô hình nhẹ hơn (như MobileNet) để tăng tốc độ trên thiết bị yếu.
- **Bảo mật**:
  - Thêm xác thực người dùng hoặc mã hóa thư viện nhúng để bảo vệ dữ liệu.

## 3. Kết quả thực nghiệm 
Hệ thống đã được triển khai và kiểm thử trên máy tính cá nhân với các tình huống thực tế khác nhau. Kết quả thực nghiệm cho thấy:
-	Phát hiện khuôn mặt: Khá tốt
-	Nhận diện khuôn mặt bằng ảnh và camera: Hệ thống nhận diện khá tốt trong trường hợp ảnh rõ nét và điều kiện ánh sáng phù hợp. Tiến hành test 90 ảnh đúng 67 ảnh, tỉ lệ chính xác đạt khoảng 70%
-	Hiệu suất thời gian thực: Hệ thống xử lý và hiển thị kết quả gần như thời gian thực (real-time), đảm bảo trải nghiệm mượt mà khi sử dụng webcam.
Kết luận:
Hệ thống đã đáp ứng tốt các yêu cầu cơ bản về nhận diện khuôn mặt, tuy nhiên hiệu quả có thể bị ảnh hướng với các góc độ khác nhau hoặc điều kiện môi trường. Đây sẽ là những thứ cần cải thiện trong tương lai.

## Kết Luận
Hai đoạn mã cung cấp một hệ thống nhận diện khuôn mặt hoàn chỉnh, kết hợp mạng Triplet dựa trên ResNet18 để tạo nhúng và giao diện Tkinter để tương tác người dùng. Hệ thống hỗ trợ cả nhận diện ảnh tĩnh và thời gian thực qua camera, với khả năng hiển thị trực quan và đánh giá độ chính xác. Đây là một nền tảng mạnh mẽ cho các ứng dụng nhận diện khuôn mặt, với tiềm năng cải tiến để tăng độ chính xác và trải nghiệm người dùng.
