import os

def write_image_list_to_txt(image_dir: str, output_txt: str, extensions=None):
    """
    读取 image_dir 下的所有图片文件名，按字典序排序，并写入 output_txt。

    Args:
        image_dir (str): 要扫描的图片目录路径。
        output_txt (str): 输出的 txt 文件路径。
        extensions (set[str], optional): 允许的文件后缀集合（含点），默认 {' .jpg','.jpeg','.png','.bmp','.tiff'}。
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 列出目录下符合后缀的文件
    filenames = [
        fn for fn in os.listdir(image_dir)
        if os.path.splitext(fn)[1].lower() in extensions
    ]

    # 按文件名排序
    filenames.sort()

    # 写入 txt
    with open(output_txt, 'w', encoding='utf-8') as f:
        for fn in filenames:
            f.write(f"rainy/{fn}\n")


# 示例用法
if __name__ == "__main__":
    write_image_list_to_txt(
        image_dir="./data/Train/Derain/rainy",
        output_txt="./data_dir/rainy/rainTrain.txt"
    )
