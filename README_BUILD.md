# ncnn_mobileclip ARM64 Linux 编译指南

## 用途
在 GitHub Actions（Ubuntu x86 runner + QEMU 模拟 ARM64）上编译出可在 Android LXC 容器（Ubuntu 24.04 ARM64）中运行的 MobileCLIP 推理二进制。

## 步骤

### 1. 创建 GitHub 仓库
- 新建一个公开仓库（**必须公开**，否则无法使用免费的 ARM64/QEMU 模拟 runner）
- 仓库名随意，比如 `ncnn-mobileclip-build`

### 2. 上传 workflow
把本目录下的 `.github/workflows/build.yml` 上传到仓库对应路径。
可以直接在 GitHub 网页端创建文件夹和文件，粘贴内容。

### 3. 触发编译
- 上传后会自动触发（因为 workflow 配置了 `on: push`）
- 也可以在仓库的 **Actions** 标签页手动点 **Run workflow**

### 4. 下载产物
编译完成后（约 10~20 分钟），在 Actions 页面点击最新的 Run，拉到最下面下载 `ncnn_mobileclip_arm64.zip`。

### 5. 下载模型
模型文件在 Google Drive（由 futz12/ncnn_mobileclip 作者提供）：
```
https://drive.google.com/file/d/1WFQEwWxUCFhDASbXv7fAlXUHn1BnVGGI/view?usp=drive_link
```
下载后解压，应该得到一个包含 `*.param` 和 `*.bin` 的目录。

### 6. 部署到 u24_test
把以下文件全部传到 u24_test 的 `/root/` 目录：
- `ncnn_mobileclip_full`（可执行文件）
- `bpe_simple_vocab_16e6.txt`
- `vocab.txt`
- 模型文件（`*.param`, `*.bin`）

### 7. 运行
```bash
export LD_LIBRARY_PATH=/root:$LD_LIBRARY_PATH
export MESA_LOADER_DRIVER_OVERRIDE=kgsl
export TURNIP_KMD=kgsl
chmod +x ncnn_mobileclip_full
./ncnn_mobileclip_full test.jpg "a photo of a cat" "a photo of a dog" "a photo of a car"
```

## 注意事项
1. **必须公开仓库**：GitHub 的 ARM64 runner 和 Docker QEMU 在私有仓库上可能受限。公开仓库完全免费。
2. **模型不需要放进仓库**：模型太大，单独下载后手动传到手机。
3. **如果编译报错**：检查 `CMakeLists.txt` 里是否有硬编码的 Windows 路径（`3rdparty/ocvm/x64/vc17/staticlib`），workflow 已经用 `sed` 删掉了。
4. **运行时的环境变量**：必须设置 `MESA_LOADER_DRIVER_OVERRIDE=kgsl` 和 `TURNIP_KMD=kgsl`，否则 ncnn 会尝试走 DRM 路径，找不到 GPU。

## 可选：改用 ncnn 官方预编译
如果不想编译 ncnn，也可以尝试直接下载 ncnn 官方 Release：
`https://github.com/Tencent/ncnn/releases`
找 `ncnn-YYYYMMDD-android-vulkan.zip`（Android 版，但理论上兼容 ARM64 Linux），或者找 Linux 版。
