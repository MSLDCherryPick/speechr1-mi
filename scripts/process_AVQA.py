import h5py
import numpy as np
import json
import os
def avqa_audio_feature_generator(file_path, batch_size=1):
    """
    AVQA音频特征数据生成器
    
    参数:
        file_path (str): HDF5文件的路径
        batch_size (int): 批次大小，默认为1
        
    yields:
        dict: 包含一批次音频特征的字典
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 获取数据集大小
            import pdb; pdb.set_trace()
            total_samples = next(iter(f.values())).shape[0]
            
            # 按批次生成数据
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_data = {}
                
                # 读取每个特征的一个批次
                for key in f.keys():
                    batch_data[key] = f[key][start_idx:end_idx]
                
                yield batch_data
                
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        yield None

def process_ava_data(json_path, output_path, audio_dir="/data/jianwei/data/vggsound/audio"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    new_data = []
    success_count = 0
    for item in data:
        audio_path = os.path.join(audio_dir, item["video_name"] + ".wav")
        item["audio_path"] = audio_path
        item["dataset_name"] = "AVQA"
        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            continue
        success_count += 1
        new_data.append(item)
    
    with open(output_path, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    print(f"成功处理 {success_count} 条数据")


# 使用示例
if __name__ == "__main__":
    task = "AVQA"
    if task == "AVQA":
        json_path = "/data/jianwei/experiment/AudioGeneration/thirdparty/r1-aqa/data/AQA/train_qa.json"
        output_path = "/data/jianwei/experiment/AudioGeneration/thirdparty/r1-aqa/data/AQA/train_qa_processed.json"
        process_ava_data(json_path, output_path)


    if task == "H5":
        file_path = "/mnt/conversationhubhot/tmp/jianweiyu/datasets/AVQA/AVQA_extracted_features/AVQA_audio_PANNs_feat.h5"
        
        # 示例1：单条数据迭代
        print("示例1：逐条读取数据")
        for i, features in enumerate(avqa_audio_feature_generator(file_path)):
            if features is not None:
                print(f"样本 {i} 的特征形状：")
                for key, value in features.items():
                    print(f"特征 {key} 的形状：{value.shape}")
            if i >= 2:  # 只显示前3个样本
                break
        
        # 示例2：批次数据迭代
        print("\n示例2：按批次读取数据（batch_size=4）")
        for i, batch_features in enumerate(avqa_audio_feature_generator(file_path, batch_size=4)):
            if batch_features is not None:
                print(f"批次 {i} 的特征形状：")
                for key, value in batch_features.items():
                    print(f"特征 {key} 的形状：{value.shape}")
            if i >= 2:  # 只显示前3个批次
                break
