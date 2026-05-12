#!/bin/bash
# PaddleOCR-VL-1.5 + MetaX GPU 环境信息采集脚本

echo "========== 硬件信息 =========="
mx-smi 2>/dev/null || echo "mx-smi 不可用"

echo ""
echo "========== 软件版本 =========="
pip show paddlepaddle 2>/dev/null | grep -E "Name|Version" || echo "paddlepaddle 未安装"
pip show paddle-metax-gpu 2>/dev/null | grep -E "Name|Version" || echo "paddle-metax-gpu 未安装"
pip show paddleocr 2>/dev/null | grep -E "Name|Version" || echo "paddleocr 未安装"
pip show opencv-contrib-python-headless 2>/dev/null | grep -E "Name|Version" || echo "opencv-contrib-python-headless 未安装"
dpkg -l libgl1 2>/dev/null | grep libgl1 || echo "libgl1 未安装"

echo ""
echo "========== MACA 驱动 =========="
for maca_path in /opt/maca /usr/local/maca /opt/maca-3.3.0; do
    if [ -d "$maca_path" ]; then
        echo "MACA path: $maca_path"
        ls "$maca_path/" | head -10
        break
    fi
done
[ -d "/opt/mxdriver" ] && echo "MX Driver path: /opt/mxdriver" && ls /opt/mxdriver/ | head -5
echo "MACA_VERSION: $(cat /opt/maca/Version.txt 2>/dev/null || echo 'N/A')"
echo "MX_DRIVER_VERSION: $(cat /opt/mxdriver/Version.txt 2>/dev/null || echo 'N/A')"

echo ""
echo "========== Python & OS =========="
python3 --version
uname -r
cat /etc/os-release 2>/dev/null | grep -E "PRETTY_NAME|VERSION"

echo ""
echo "========== Paddle 设备检测 =========="
python3 -c "
import paddle
print('Paddle version:', paddle.__version__)
print('Device:', paddle.device.get_device())
print('Device count:', paddle.device.device_count())
try:
    t = paddle.zeros([1], dtype='float32')
    t = t.to('metax_gpu:0')
    print('MetaX GPU 可用: True')
except Exception as e:
    print('MetaX GPU 可用: False,', e)
"

echo ""
echo "========== FastDeploy 版本 =========="
pip show fastdeploy-metax-gpu 2>/dev/null | grep -E "Name|Version" || echo "fastdeploy-metax-gpu 未通过pip安装"
pip show fastdeploy 2>/dev/null | grep -E "Name|Version" || echo "fastdeploy 未通过pip安装"
python3 -c "import fastdeploy; print('FastDeploy version:', fastdeploy.__version__)" 2>/dev/null || echo "无法获取 fastdeploy 版本"
cd /data/FastDeploy && git log --oneline -1 2>/dev/null || echo "非git仓库"
