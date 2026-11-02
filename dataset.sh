mkdir -p ./data/datasets

# 1. ImageNet-A -> ./data/datasets/imagenet-a/{train,test}
curl -L "https://drive.usercontent.google.com/download?id=19l52ua_vvTtttgVRziCZJjal0TPE9f2p&export=download&confirm=t" -o ina.zip
md5sum ina.zip  # Kiểm tra: 7f4aaf0b1532b7a0bd670822c152d3c5
unzip -q ina.zip -d ./data/datasets/
rm ina.zip

# 2. ImageNet-R -> ./data/datasets/imagenet-r/{train,test} (lưu ý: imagenet-r.zip không có folder bọc ngoài)
mkdir -p ./data/datasets/imagenet-r
curl -L "https://drive.usercontent.google.com/download?id=1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR&export=download&confirm=t" -o imagenet-r.zip
md5sum imagenet-r.zip  # Kiểm tra: 730aa3fb9c7d2ec0b6381c6b23040681
unzip -q imagenet-r.zip -d ./data/datasets/imagenet-r/
rm imagenet-r.zip

# 3. CUB-200 -> ./data/datasets/cub/{train,test}
curl -L "https://drive.usercontent.google.com/download?id=1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb&export=download&confirm=t" -o cub.zip
md5sum cub.zip  # Kiểm tra: 85e342a2d5f941e740193602057df7a5
unzip -q cub.zip -d ./data/datasets/
rm cub.zip

# 4. VTAB -> ./data/datasets/vtab/{train,test}
curl -L "https://drive.usercontent.google.com/download?id=1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_&export=download&confirm=t" -o vtab.zip
md5sum vtab.zip  # Kiểm tra: a9c37bde6105a2516c900b0c33a437c1
unzip -q vtab.zip -d ./data/datasets/
rm vtab.zip

# 5. OmniBenchmark -> ./data/datasets/omnibenchmark/{train,test}
curl -L "https://drive.usercontent.google.com/download?id=1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u&export=download&confirm=t" -o omnibenchmark.zip
md5sum omnibenchmark.zip  # Kiểm tra: f1c30808a707f9197c6c811ea52e5cbb
unzip -q omnibenchmark.zip -d ./data/datasets/
rm omnibenchmark.zip