import os
import subprocess
import time

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    video_path = data['ref_video']
   
    model_choice = data['model_choice']
     # 获取任务ID或使用视频文件名作为ID，用于区分工作区
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    task_id = data.get('task_id', video_name) 

    print(f"[backend.model_trainer] 任务ID: {task_id}, 模型: {model_choice}")

    print("[backend.model_trainer] 模型训练中...")

    if data['model_choice'] == "SyncTalk":
        try:
            # 构建命令
            cmd = [
                "./SyncTalk/run_synctalk.sh", "train",
                "--video_path", data['ref_video'],
                "--gpu", data['gpu_choice'],
                "--epochs", data['epoch']
            ]

            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            # 执行训练命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print("[backend.model_trainer] 训练输出:", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] 错误输出:", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] 训练失败，退出码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return video_path
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return video_path
    elif data['model_choice'] == "ER-NeRF":
        # ToDo : 实现ER-NeRF训练逻辑
        try:
            print("[backend.model_trainer] 开始 ER-NeRF 训练流程...")
            
            # 1. 准备路径和参数
            # 假设 ER-NeRF 项目位于当前目录下的 ER-NeRF 文件夹
            er_nerf_root = "./ER-NeRF" 
            # 使用视频文件名作为 workspace ID (task_id)
            task_id = os.path.splitext(os.path.basename(video_path))[0]
            # ER-NeRF 数据存放的标准路径
            dataset_path = os.path.join("data", task_id)

            # 2. 数据预处理 (提取帧、音频、Landmarks等)
            # 对应命令: python data_utils/process.py <video> --task <id>
            print(f"[backend.model_trainer] [1/2] 正在进行数据预处理: {video_path}")
            process_cmd = [
                "python", os.path.join(er_nerf_root, "data_utils", "process.py"),
                video_path,
                "--task", task_id
            ]
            # 执行预处理
            subprocess.run(process_cmd, check=True)

            # 3. 模型训练
            # 对应命令: python main.py data/<id> --workspace <id> -O --iters <epoch>
            print(f"[backend.model_trainer] [2/2] 开始训练 ER-NeRF 模型...")
            train_cmd = [
                "python", os.path.join(er_nerf_root, "main.py"),
                dataset_path, 
                "--workspace", task_id,
                "-O",  # 开启优化选项
                "--iters", str(data.get('epoch', 5000))
            ]

            # 设置 GPU
            env = os.environ.copy()
            if 'gpu_choice' in data:
                env['CUDA_VISIBLE_DEVICES'] = str(data['gpu_choice'])

            print(f"[backend.model_trainer] 执行训练命令: {' '.join(train_cmd)}")
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            print("[backend.model_trainer] ER-NeRF 训练完成")
            # 训练完成后，checkpoint 会保存在 ER-NeRF/trial_<task_id>/checkpoints/

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] 训练失败，退出码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return video_path
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return video_path
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return video_path


    print("[backend.model_trainer] 训练完成")
    return video_path


#  我发现老师的训练逻辑是单独训练视频生成模型，然后再单独提取语音特征进行克隆，所以在模型训练这里应该不用再提取语音特征


