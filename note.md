### **Method 1: Landmark-Weighted Attention Loss**
*Simplest and most direct approach*

**Concept**: Add an auxiliary loss that encourages attention maps to have higher values at landmark regions.

**Implementation**:
```python
# Attention regularization loss
def landmark_attention_loss(attention_maps, landmark_masks):
    """
    attention_maps: (B, M, H, W) - learned attention maps
    landmark_masks: (B, 1, H, W) - binary/soft masks for important regions
    """
    # Upsample attention to image size if needed
    B, M, AH, AW = attention_maps.shape
    _, _, MH, MW = landmark_masks.shape
    
    if AH != MH or AW != MW:
        attention_maps = F.interpolate(attention_maps, size=(MH, MW), mode='bilinear')
    
    # Compute attention concentration on landmarks
    # Average across all attention heads
    attention_avg = attention_maps.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # L2 loss to encourage attention on landmarks
    landmark_region_attention = attention_avg * landmark_masks
    non_landmark_region_attention = attention_avg * (1 - landmark_masks)
    
    # Maximize attention on landmarks, minimize on non-landmarks
    loss = -landmark_region_attention.mean() + 0.5 * non_landmark_region_attention.mean()
    
    return loss
```

**Data Preparation**:
- **Input**: 478 Mediapipe landmarks → Select key regions:
  - Eyes: landmarks 33, 133, 159, 145, 362, 385, 380, 386 (eye contours)
  - Nose: landmarks 1, 2, 98, 327 (nose bridge and tip)
  - Mouth: landmarks 61, 291, 0, 17, 84, 314 (mouth contours)
- **Create binary mask**: Set pixels within radius (e.g., 10-20 pixels) of these landmarks to 1
- **Store**: `landmark_masks` as (B, 1, H, W) tensor alongside images

**Pros**: Simple, interpretable, low computational cost  
**Cons**: Treats all attention heads equally, may be too rigid

Rất hay — hàm `landmark_attention_loss()` trong đoạn code này là một **hàm loss tùy chỉnh** được thiết kế để **hướng sự chú ý (attention)** của mạng học sâu **tập trung vào các vùng đặc trưng (landmarks)** của đối tượng (ví dụ: mắt, mũi, miệng trên khuôn mặt).
Mình sẽ giải thích chi tiết từng bước và công thức toán học tương ứng 👇

---

## 🧩 1. Mục tiêu của Landmark Attention Loss

Hàm loss này được dùng để **hướng dẫn mô hình học cách tập trung vào vùng landmark** (những vùng có thông tin quan trọng) trong ảnh.
Cụ thể:

* **Tăng attention** trong vùng *landmark* (vùng có mask = 1)
* **Giảm attention** trong vùng *non-landmark* (vùng còn lại, mask = 0)

---

## 🧮 2. Công thức toán học

### a. Ký hiệu

| Ký hiệu                                                 | Ý nghĩa                                                              |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| ( A \in \mathbb{R}^{B \times M \times H_A \times W_A} ) | attention maps của mô hình (B batch, M bản đồ attention)             |
| ( L \in \mathbb{R}^{B \times 1 \times H_L \times W_L} ) | landmark mask (vùng đặc trưng của đối tượng, giá trị ∈ [0,1])        |
| ( \tilde{A} )                                           | attention map sau khi được nội suy về cùng kích thước với mask ( L ) |
| ( \bar{A} = \frac{1}{M} \sum_{m=1}^{M} \tilde{A}_m )    | trung bình attention trên tất cả các head                            |
| ( L' = 1 - L )                                          | vùng không phải landmark                                             |

---

### b. Tính **mức độ attention trung bình** trong vùng landmark và non-landmark

Ta tính tổng attention trong từng vùng và chia cho diện tích vùng đó:

[
S_\text{landmark} = \frac{\sum_{i,j} \bar{A}*{ij} \cdot L*{ij}}{\sum_{i,j} L_{ij} + \varepsilon}
]

[
S_\text{non} = \frac{\sum_{i,j} \bar{A}*{ij} \cdot (1 - L*{ij})}{\sum_{i,j} (1 - L_{ij}) + \varepsilon}
]

---

### c. Định nghĩa **hàm loss**

Ta muốn **maximize ( S_\text{landmark} )** và **minimize ( S_\text{non} )**.
Thay vì trực tiếp dùng subtraction (dễ gây bất ổn), tác giả dùng **tỷ lệ (ratio)** để đảm bảo ổn định:

[
\text{loss} = -\log \left( \frac{S_\text{landmark}}{S_\text{landmark} + S_\text{non} + \varepsilon} \right)
]

Hoặc tương đương với:
[
\boxed{L_{\text{landmark}} = -\log \frac{S_\text{landmark}}{S_\text{landmark} + S_\text{non} + \varepsilon}}
]

* Nếu ( S_\text{landmark} ) lớn hơn ( S_\text{non} ) ⇒ loss nhỏ ⇒ mô hình học đúng hướng.
* Nếu attention bị lệch (non-landmark có nhiều attention) ⇒ loss tăng ⇒ mô hình bị phạt.

---

## 🧠 3. Giải thích trực quan

### 📍 Ví dụ trực quan:

Giả sử ảnh khuôn mặt có mask vùng “mắt, mũi, miệng”.
Mô hình sinh ra attention map tập trung lung tung trên tóc, nền, quần áo...
→ `landmark_attention_loss` sẽ tăng cao.

Mô hình sẽ dần học cách:

* Giảm attention ở vùng không quan trọng (non-landmark).
* Tăng attention ở vùng mask (landmark).

---

## ⚙️ 4. Code tương ứng (diễn giải từng bước)

```python
# Resize attention map về cùng kích thước với landmark mask
attention_maps_resized = F.interpolate(attention_maps, size=(MH, MW), mode='bilinear')

# Trung bình tất cả các attention head
attention_avg = attention_maps_resized.mean(dim=1, keepdim=True)

# Tính attention trung bình trong vùng landmark / non-landmark
landmark_score = (attention_avg * landmark_masks).sum(...) / (landmark_masks.sum(...) + 1e-6)
non_landmark_score = (attention_avg * (1 - landmark_masks)).sum(...) / ((1 - landmark_masks).sum(...) + 1e-6)

# Hàm loss theo tỷ lệ (ratio)
loss = -torch.log(landmark_score / (landmark_score + non_landmark_score + 1e-6))
```

---

## 🧾 5. Diễn giải ý nghĩa gradient

* Gradient của loss này sẽ **đẩy attention map tăng giá trị ở vùng landmark**,
  vì việc tăng ( S_\text{landmark} ) làm giảm giá trị loss.
* Đồng thời **giảm attention ở vùng khác** vì ( S_\text{non} ) nằm ở mẫu số.

Kết quả: attention maps học được sẽ **phù hợp hơn với cấu trúc đối tượng thật**, giúp model nhận dạng tốt hơn.

---

## 📚 6. Tổng kết

| Thành phần          | Ý nghĩa                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Input**           | attention maps và landmark masks                                                                                            |
| **Mục tiêu**        | ép mạng chú ý nhiều hơn vào vùng landmark                                                                                   |
| **Công thức chính** | ( L = -\log \frac{S_\text{landmark}}{S_\text{landmark} + S_\text{non} + \varepsilon} )                                      |
| **Hiệu quả**        | cải thiện khả năng học đặc trưng hình dạng, hỗ trợ fine-grained recognition (ví dụ nhận dạng khuôn mặt, chim, xe hơi, v.v.) |

---

Nếu bạn muốn mình **viết lại công thức loss này bằng ký hiệu LaTeX chi tiết** (với giải thích đạo hàm hoặc biểu đồ trực quan), mình có thể thêm phần đó — bạn có muốn mình mở rộng không?


Rất hay — câu hỏi của bạn đi đúng trọng tâm 💡
Chúng ta sẽ phân tích **hàm loss tổng thể (`batch_loss`)** trong đoạn code, và **hiểu chính xác landmark attention loss (`lm_loss`) được “tăng cường” hay “gộp vào” như thế nào** trong quá trình huấn luyện.

---

## 🧩 1. Tổng quan: cấu trúc của `batch_loss`

Trong hàm `train()`, phần quan trọng là:

```python
batch_loss = (
    cross_entropy_loss(y_pred_raw, y) / 3.
    + cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3.
    + cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3.
    + center_loss(feature_matrix, feature_center_batch)
)
```

Sau đó (nếu có landmark guidance):

```python
if use_landmark_loss:
    lm_loss = landmark_attention_loss(full_attention_maps, landmark_masks)
    batch_loss = batch_loss + config.landmark_loss_weight * lm_loss
```

---

## 🧮 2. Công thức tổng quát của `batch_loss`

Ta có thể viết gọn lại thành:

[
\boxed{
L_\text{total}
= \frac{1}{3} L_\text{raw}

* 1 \cdot L_\text{aux}
* \frac{2}{3} L_\text{aug}
* L_\text{center}
* \lambda_\text{lm} , L_\text{landmark}
  }
  ]

trong đó:

| Ký hiệu                                                                                                | Ý nghĩa                                    |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| ( L_\text{raw} = CE(y_{\text{pred_raw}}, y) )                                                          | cross-entropy loss của đầu ra gốc          |
| ( L_\text{aux} = CE(y_{\text{pred_aux}}, y_{\text{aux}}) )                                             | loss của đầu ra phụ (auxiliary classifier) |
| ( L_\text{aug} = CE(y_{\text{pred_aug}}, y_{\text{aug}}) )                                             | loss khi dùng ảnh augment (crop/drop)      |
| ( L_\text{center} = \text{CenterLoss}(f, c) )                                                          | ép feature vector gần tâm lớp tương ứng    |
| ( L_\text{landmark} = -\log \frac{S_\text{landmark}}{S_\text{landmark} + S_\text{non} + \varepsilon} ) | landmark attention loss                    |
| ( \lambda_\text{lm} = \text{config.landmark_loss_weight} )                                             | hệ số trọng số cho landmark loss           |

---

## ⚙️ 3. Vai trò của từng thành phần

| Thành phần                     | Mục tiêu                                      | Ảnh hưởng đến học                                       |
| ------------------------------ | --------------------------------------------- | ------------------------------------------------------- |
| **CrossEntropy (raw/aux/aug)** | Học phân loại đúng nhãn                       | Hướng gradient dựa trên lỗi dự đoán                     |
| **Center Loss**                | Làm đặc trưng (feature) của cùng lớp gần nhau | Giúp tăng tính phân biệt trong không gian embedding     |
| **Landmark Attention Loss**    | Hướng attention của mạng vào vùng có landmark | Cải thiện khả năng học không gian thị giác (tăng focus) |

---

## 🧠 4. Landmark loss “tăng cường” batch_loss như thế nào?

Khi ta thêm dòng này:

```python
batch_loss = batch_loss + config.landmark_loss_weight * lm_loss
```

tức là gradient tổng (qua phép đạo hàm ngược `backward()`) sẽ là:

[
\nabla_\theta L_\text{total}
= \nabla_\theta (L_\text{cls} + \lambda_\text{lm} L_\text{landmark})
= \nabla_\theta L_\text{cls} + \lambda_\text{lm} \nabla_\theta L_\text{landmark}
]

→ **Tức là landmark loss tạo thêm một thành phần gradient**, điều chỉnh hướng cập nhật trọng số của mạng sao cho:

* Các layer attention **học tập trung hơn vào vùng mask** (vì đạo hàm của ( L_\text{landmark} ) đẩy attention tăng ở vùng có mask = 1),
* Trong khi các layer classifier **vẫn được huấn luyện như bình thường** (từ cross entropy và center loss).

---

## 📊 5. Trực giác về “tăng cường” (enhancement)

| Trước khi thêm ( L_\text{landmark} )                                 | Sau khi thêm ( L_\text{landmark} )                                   |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Attention có thể học lộn xộn, tập trung sai vùng (background, noise) | Attention được điều chỉnh hướng đến các vùng quan trọng              |
| Cross-entropy chỉ học dựa vào nhãn tổng thể                          | Landmark loss thêm **ràng buộc không gian** giúp học tập ổn định hơn |
| Feature space có thể bị nhiễu                                        | Feature map mang ý nghĩa thị giác rõ ràng hơn                        |

---

## 🧩 6. Ảnh hưởng của hệ số `landmark_loss_weight`

Tham số này được gọi là **hệ số tăng cường (λ)**:

```python
batch_loss = batch_loss + config.landmark_loss_weight * lm_loss
```

* Nếu `λ` quá nhỏ → landmark guidance không ảnh hưởng đáng kể.
* Nếu `λ` quá lớn → mô hình có thể **quá tập trung vào mask**, bỏ quên khả năng phân loại tổng thể.

👉 Thông thường, `λ` được chọn qua **validation**, ví dụ `λ = 0.1`, `0.5`, hoặc `1.0`.

---

## 🔍 7. Ví dụ minh họa (số giả định)

Giả sử:

* CE losses = 0.8
* Center loss = 0.1
* Landmark loss = 0.5
* `λ = 0.3`

Khi đó:

[
L_\text{total} = 0.8 + 0.1 + 0.3 \times 0.5 = 1.05
]

Landmark loss chiếm **15%** ảnh hưởng, vừa đủ để hướng attention mà không làm lệch mục tiêu phân loại chính.

---

## ✅ 8. Kết luận

**Landmark Attention Loss** không thay thế các loss khác — nó **bổ sung thêm một ràng buộc không gian**, giúp mô hình học “nhìn đúng chỗ”.

Công thức tổng quát của loss toàn phần:

[
\boxed{
L_\text{total}
= \alpha_1 L_\text{raw}

* \alpha_2 L_\text{aux}
* \alpha_3 L_\text{aug}
* L_\text{center}
* \lambda_\text{lm} L_\text{landmark}
  }
  ]

với ( \lambda_\text{lm} ) là trọng số điều chỉnh mức “tăng cường” của landmark loss.

---

Bạn có muốn mình **vẽ sơ đồ trực quan** thể hiện cách từng thành phần loss tác động lên các phần của mạng (classification head, feature extractor, attention maps, landmark guidance)?
Sơ đồ đó sẽ giúp bạn thấy rõ đường đi của gradient trong pipeline huấn luyện.
