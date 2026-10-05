# Bản đồ tĩnh repo RSIAT (G1)

Tài liệu này ghi nhận mã nguồn hiện có trước khi bắt đầu các thay đổi kiến trúc QR-RSIAT/Kaggle. Chỉ đọc mã; không chạy mô hình và không sửa file gốc.

## Phạm vi và trạng thái

- Kế hoạch tham chiếu: [`agents/AGENT_PLAN_v2_KIEN_TRUC_KAGGLE.md`](../agents/AGENT_PLAN_v2_KIEN_TRUC_KAGGLE.md), đặc biệt mục 5, giai đoạn G1.
- Repo có lối vào `main.py`, `trainer.py`, `args.sh`; cấu hình thí nghiệm trong `exps/`; mã mô hình trong `models/`, `network/`, `utils/`; quản lý dữ liệu trong `data/`.
- `docs/REPO_MAP.md` là sản phẩm duy nhất của G1. Chưa triển khai module mới hay sửa mã nguồn RSIAT.

## Cây mã liên quan

| Thành phần | Vai trò đã xác minh |
|---|---|
| [`main.py`](../main.py) | Đọc `--config` (mặc định `./exps/adapter_imageneta.json`), nạp JSON, hợp nhất đối số CLI với cấu hình JSON, gọi `RSIAT_train`. JSON ghi đè giá trị cùng tên trong dict đối số. |
| [`trainer.py`](../trainer.py) | Lặp các seed, tạo log dưới `logs/<model>/<dataset>/<init_cls>/<increment>/`, thiết lập device/seed, tạo `DataManager` và learner; chạy từng task, đánh giá, ghi accuracy curve. |
| [`args.sh`](../args.sh) | Sáu lệnh gọi `main.py` lần lượt cho `cifar224`, `cub`, `imageneta`, `imagenetr`, `omnibench`, `vtab`. |
| [`exps/`](../exps) | Sáu JSON cấu hình, một cấu hình mỗi dataset. Có các trường `device`, `seed`, chia task, optimizer, batch/epoch, learning rate, loss và RAE/CA. |
| [`data/data.py`](../data/data.py) | Dataset wrapper, transform train/test, ánh xạ class order và nạp dữ liệu theo từng dataset. |
| [`data/data_manager.py`](../data/data_manager.py) | Chọn dataset, chia class thành task, remap nhãn, tạo `DummyDataset` trả về `(idx, image, label)`. |
| [`utils/model_factory.py`](../utils/model_factory.py) | Factory hiện chỉ ánh xạ `model_name == "adapter"` sang `models.RSIAT_adapter.Learner`. Đây là điểm đăng ký model hiện hữu. |
| [`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py) | Learner RSIAT: huấn luyện base/incremental task, RS loss, RAE/alignment và gọi các bước tính thống kê, displacement, classifier compacting. |
| [`models/base.py`](../models/base.py) | `BaseLearner`: state task/class, đánh giá, tính prototype/covariance, displacement và classifier retraining từ Gaussian. |
| [`utils/inc_net.py`](../utils/inc_net.py) | Dựng `SimpleVitNet`/backbone và classifier; có helper `load_state_vision_model`. |
| [`network/vision_transformer_adapter.py`](../network/vision_transformer_adapter.py) | Định nghĩa Vision Transformer/adapter và các hàm dựng ViT pretrained adapter. |
| [`network/classifier.py`](../network/classifier.py) | `SimpleContinualLinear`, `CosineLinear` và continual heads. |
| [`utils/loss.py`](../utils/loss.py) | Angular penalty classification loss (CosFace/ArcFace/SphereFace/cross-entropy). |
| [`utils/toolkit.py`](../utils/toolkit.py) | Tiện ích tham số, accuracy, tensor/NumPy, one-hot và autoencoder `AutoencoderSigmoid` dùng như projector RAE. |
| [`requirements.txt`](../requirements.txt) | Dependency versions pin; hiện bao gồm cả `torch==2.8.0+cu126` và `torchvision==0.23.0+cu126`. |
| [`README.md`](../README.md) | Hướng dẫn dữ liệu tổng quát, yêu cầu Ubuntu 20.04/Python 3.10/CUDA 11.8 và cách chạy theo `args.sh`. |

## Luồng chạy và huấn luyện

1. `main()` nạp config và truyền dict đã hợp nhất vào `RSIAT_train` ([`main.py`](../main.py#L6)).
2. `RSIAT_train` lặp `seed`; `_train` tạo logger, gọi `_set_random()` và `_set_device()`, sau đó tạo `DataManager`, `get_model`, và vòng task ([`trainer.py`](../trainer.py#L13), [`trainer.py`](../trainer.py#L25)).
3. `model_factory.get_model()` chọn learner `adapter` ([`utils/model_factory.py`](../utils/model_factory.py#L1)). `Learner` yêu cầu tên backbone có `adapter` và khởi tạo `SimpleVitNet` ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L21)).
4. Mỗi task tăng số class, tạo train loader cho class mới và test loader cho toàn bộ class đã thấy, huấn luyện, xử lý displacement (từ task sau base), lưu classifier state tạm để CA, tính class statistics và tùy config retrain classifier ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L67)).
5. `after_task()` cập nhật số class đã biết và sao chép/freeze network làm model cũ ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L41)).
6. Mỗi epoch có training accuracy và gọi `_compute_accuracy`; sau khi train mỗi task, `eval_task()` gọi `_eval_cnn()`, tổng hợp top-1/top-5/grouped accuracy. Trainer ghi các đường cong top-1/top-5 và trung bình top-1; không thấy lệnh gọi `save_checkpoint()` trong luồng này ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L162), [`models/base.py`](../models/base.py#L177), [`models/base.py`](../models/base.py#L190), [`trainer.py`](../trainer.py#L60)).

## Loss, RAE, prototype/covariance và drift

- Base task: `loss_c` là CosFace trên logits của class mới; `loss_rt` là `lambda_rs * RS_Loss`, với `lambda_rs` warm-up tuyến tính theo epoch. Tổng loss là `loss_c + loss_rt` ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L162), [`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L212)).
- `RS_Loss` chuẩn hóa feature, dựng mask positive/negative theo nhãn, tính cosine matrix và margin loss ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L227)).
- Incremental task: `old_ae` là `AutoencoderSigmoid(768, ae_code_dims)`; `_inc_loss` áp projector lên feature cũ và prototype cũ, tính MSE alignment cùng tổng similarity normalized, rồi nhân `beta`/`gamma` ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L70), [`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L201), [`utils/toolkit.py`](../utils/toolkit.py#L65)).
- Với mỗi class mới, `_compute_class_mean()` lấy feature từ train samples ở transform test/eval, tính mean theo NumPy và covariance mẫu bằng `torch.cov` trên float64 rồi cộng `1e-3 I`. Các class cũ được chép sang mảng thống kê mở rộng. Ở base task còn tính `radius` từ covariance NumPy với hiệu chỉnh `1e-4 I` ([`models/base.py`](../models/base.py#L227)).
- Từ incremental task, learner trích feature của cùng train loader qua model cũ/mới. `displacement()` lấy hiệu feature mới-cũ, tạo trọng số theo khoảng cách bình phương tới old class means với `sigma=4.0`, rồi lấy trung bình trọng số theo class. Khi `ssca` bật, drift cộng vào prototype class cũ ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L104), [`models/base.py`](../models/base.py#L286)).
- Khi CA được bật và `ca_epochs > 0`, `_stage2_compact_classifier()` lấy 256 feature Gaussian mỗi class từ mean/covariance, chỉ train classifier bằng SGD/CrossEntropy trong vòng epoch riêng; gọi đánh giá trong epoch ([`models/base.py`](../models/base.py#L52)).

## Dữ liệu, transforms và cấu hình

- `DataManager` gọi dataset wrapper tương ứng, nhận class order; nếu `shuffle=true` thì permutation được tạo từ seed cấu hình, còn nếu false lấy `class_order` của wrapper. Nhãn train/test sau đó được remap theo order ([`data/data_manager.py`](../data/data_manager.py#L138)).
- Train transform: `RandomResizedCrop(224, scale=(0.05, 1.0))`, horizontal flip, `ToTensor`; eval transform: resize ngắn cạnh lên 256, center crop 224, `ToTensor` ([`data/data.py`](../data/data.py#L14)).
- CIFAR-100 dùng `torchvision.datasets.CIFAR100` với `root="./data/datasets"` và `download=True`. Các dataset ImageFolder dùng `./data/datasets/<dataset>/{train,test}`; dataset được hỗ trợ là ImageNet-R, ImageNet-A, CUB, VTAB và OmniBenchmark ([`data/data.py`](../data/data.py#L52), [`data/data.py`](../data/data.py#L75)).
- Các JSON trong `exps/` có batch size từ 48 tới 128; mọi config hiện đọc được đều đặt `device: ["0"]`, `seed: [1993]`, backbone `pretrained_vit_b16_224_in21k_adapter`, `optimizer: "sgd"`. Các thông số epochs/increment/loss riêng theo dataset được giữ nguyên trong từng JSON.
- README mô tả dataset ở `datasets/<tên>/`, khác với đường dẫn thực tế `./data/datasets/...` trong `data/data.py`. Chưa có lớp registry hoặc biến môi trường root trong mã được đọc.

## Backbone, pretrained weights và classifier

- `get_convnet()` nhận `convnet_type`. Với `_adapter`, cấu hình adapter có `ffn_num` từ JSON và gọi hàm dựng tương ứng; output dimension đặt 768 ([`utils/inc_net.py`](../utils/inc_net.py#L9)).
- `vit_base_patch16_224_in21k_adapter()` tạo kiến trúc adapter, gọi `timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)`, biến đổi trọng số QKV thành Q/K/V và `mlp.fc` sang tên module tương ứng, sau đó nạp với `strict=False`. Tham số có trong `missing_keys` được mở gradient; phần còn lại bị đóng băng ([`network/vision_transformer_adapter.py`](../network/vision_transformer_adapter.py#L344)). Nhánh adapter pretrained không dùng tham số `pretrained` truyền vào để quyết định cờ pretrained; lời gọi hiện đặt `pretrained=True`.
- Nhánh ViT không adapter trong `get_convnet()` cũng gọi `timm.create_model(..., pretrained=True)` ([`utils/inc_net.py`](../utils/inc_net.py#L12)).
- `load_state_vision_model()` là helper riêng đọc checkpoint từ đường dẫn được truyền, nhưng trong các file đã rà không tìm thấy nơi gọi nó ([`utils/inc_net.py`](../utils/inc_net.py#L60)). Cơ chế cache/offline và vị trí chính xác của checkpoint do `timm` lựa chọn chưa được định nghĩa trong repo.
- `SimpleContinualLinear` thêm một head theo task; `after_task()` freeze bản sao network trước task sau. Stage-2 classifier training dùng các thống kê Gaussian, không dùng ảnh đầu vào.

## Thiết bị, workers, ghi log/checkpoint

- `trainer._set_device()` chuyển mỗi phần tử `args["device"]` sang `torch.device("cuda:<id>")`; nhánh CPU kiểm tra `device_type == -1` bên trong vòng lặp qua `device_type`, trong khi JSON hiện cung cấp list chuỗi. `BaseLearner` lấy phần tử đầu làm `_device` và giữ cả list trong `_multiple_gpus` ([`trainer.py`](../trainer.py#L95), [`models/base.py`](../models/base.py#L26)).
- Các chỗ thiết bị cứng đã tìm thấy: `.cuda()` trên batch và label trong `Learner.extract_features()` ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L50)); `torch.device("cuda:...")`/`torch.device("cpu")` trong `_set_device()` ([`trainer.py`](../trainer.py#L95)); `nn.DataParallel` khi huấn luyện/inference và stage-2 cùng các nhánh unwrap/kiểm tra ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L93), [`models/base.py`](../models/base.py#L46), [`models/base.py`](../models/base.py#L68), [`models/base.py`](../models/base.py#L213)). Các `.to(self._device)` dùng device đã tạo ở trainer/base learner: learner lines 72, 97, 124, 172, 205; base learner lines 65, 81, 84, 90, 91, 181, 194, 215, 219. RS loss lấy device từ `features.device` để tạo mask ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L234)). Không thấy `device_ids` hoặc `CUDA_VISIBLE_DEVICES` trong các mã Python đã rà.
- Batch size của train/eval loader lấy từ config, nhưng `num_workers=8` được truyền trực tiếp cho loader trong `RSIAT_adapter.py`; biến module `num_workers=8` không được dùng tại các lời gọi đó. Loader tính thống kê trong `BaseLearner` dùng batch size module `64` và `num_workers=4` ([`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L18), [`models/RSIAT_adapter.py`](../models/RSIAT_adapter.py#L87), [`models/base.py`](../models/base.py#L11), [`models/base.py`](../models/base.py#L246)).
- Log là file + stdout theo cấu trúc `logs/...`; trainer có đoạn lưu checkpoint theo task nhưng đang comment. `BaseLearner.save_checkpoint()` tồn tại và gọi `torch.save`, nhưng không thấy được gọi trong luồng huấn luyện hiện tại ([`trainer.py`](../trainer.py#L28), [`trainer.py`](../trainer.py#L75), [`models/base.py`](../models/base.py#L136)).

## Vấn đề đã làm rõ và quyết định triển khai

Các quyết định sau do người dùng xác nhận sau khi hoàn tất G1. Đây là yêu cầu cho các giai đoạn tiếp theo, chưa có thay đổi mã nguồn nào được thực hiện.

| # | Vấn đề | Quyết định/phương án đã chốt | Điều vẫn cần kiểm tra khi triển khai |
|---|---|---|---|
| 1 | Hợp đồng CPU/device | Dùng `RuntimeContext` làm nguồn device duy nhất; CPU dùng được cho chẩn đoán, còn huấn luyện CPU nếu không được hỗ trợ thì phải dừng sớm và nêu rõ. Không hardcode CUDA. | Các đường `.cuda()` và `DataParallel` đã liệt kê ở trên phải được thay/đưa qua context nhất quán. |
| 2 | Gốc dữ liệu | Dùng `--data_root`/`DATA_ROOT`, fallback đường dẫn RSIAT tương thích, và trên Kaggle dò `/kaggle/input`; tạo symlink tương thích khi xác định được dataset. | Xác nhận tồn tại/cấu trúc train-test trước khi tạo loader; nếu thiếu dữ liệu thì báo chính xác dataset cần gắn. |
| 3 | Weights pretrained | Dùng checkpoint/cache có sẵn trước; thiếu weights thì báo rõ mô hình/đường dẫn cần cung cấp, không âm thầm dùng backbone ngẫu nhiên. | Xác thực checkpoint và khả năng nạp offline; chưa xác minh trước tên file/cache cụ thể của timm. |
| 4 | Class order/ImageFolder | Kiểm tra mapping class train/test, số lượng và thứ tự lớp trước huấn luyện; ghi mapping đã dùng, không tự đoán hoặc âm thầm remap sai protocol. | Đối chiếu class mapping thực tế của các bộ dữ liệu được gắn trên Kaggle. |
| 5 | Checkpoint/resume và thống kê | Tìm checkpoint trong thư mục output của repo/phiên hiện tại trước khi bắt đầu. Nếu tìm thấy checkpoint hợp lệ thì nạp state và tiếp tục; nếu không có checkpoint hợp lệ thì bắt đầu lượt chạy mới. Khi DDP, gộp count/tổng/tích theo lớp bằng `all_reduce(SUM)` trước khi tính mean/covariance. | Xác định schema/checkpoint đầy đủ (model khả huấn luyện, means/covariances, task, cấu hình/RNG); checkpoint không hợp lệ phải có cảnh báo rõ và không được xem như resume thành công. |
| 6 | Đánh giá top-5 | Dùng `k=min(5, số lớp)` khi số lớp nhỏ hơn 5 và thể hiện rõ metric top-k thực tế. | Đảm bảo nhãn/tên metric trong báo cáo không gọi top-k thấp hơn là top-5. |
| 7 | Seed/thứ tự lớp | Seed mặc định là `1993`. Seed thí nghiệm điều khiển nguồn ngẫu nhiên; class order phải tách biệt với seed thí nghiệm để không đổi protocol ngoài ý muốn. | Không ghi đè seed cấu hình khi thiết lập RNG; giữ thứ tự lớp theo protocol được chọn. |
| 8 | Tương thích môi trường | Kaggle dùng PyTorch/CUDA cài sẵn; dependencies bổ sung không cài lại `torch`/`torchvision`. Ghi nhận phiên bản runtime và fail-fast nếu không tương thích. | Đối chiếu phiên bản runtime Kaggle thực tế khi chạy lần đầu; chưa xác minh môi trường baseline ban đầu. |

## Vấn đề mở còn lại

- Cấu trúc và mapping class thực tế của dataset Kaggle chưa được quan sát; chỉ xác minh tĩnh từ code là chưa đủ.
- Chưa có checkpoint để xác nhận schema, vị trí output và tính hợp lệ của resume. Mặc định “không có checkpoint hợp lệ thì chạy mới” đã được chốt, nhưng chi tiết schema sẽ được thiết kế ở giai đoạn checkpoint.
- Chưa xác minh tên file/cache pretrained weights mà phiên Kaggle cung cấp; implementation cần dò và thông báo lỗi cụ thể.
- Chưa xác minh phiên bản Python/PyTorch/CUDA thực tế trên Kaggle hoặc phiên bản dùng cho baseline.

## Ranh giới G1

G1 chỉ tạo bản đồ tĩnh này. Không chạy mô hình, không tái lập baseline, không viết test, và không chỉnh sửa mã/config gốc. Các quyết định ở trên được ghi nhận cho G2–G8; chúng chưa được triển khai hoặc kiểm chứng runtime.
