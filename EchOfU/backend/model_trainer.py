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

        #Todo : 实现ER-NeRF训练逻辑
        try:
            print("[backend.model_trainer] 开始 ER-NeRF 训练流程...")
            
            # 路径配置 (根据实际项目目录结构调整 ER-NeRF 的位置)
            er_nerf_root = "./ER-NeRF"
            # 数据预处理输出路径 data/<task_id>
            dataset_path = os.path.join(er_nerf_root, "data", task_id) 
            
            # 数据预处理 (调用 ER-NeRF/data_utils/process.py)
            # 从视频中提取图像、音频，计算 Landmark，提取 HuBERT/DeepSpeech 特征用于口型同步
           
            print(f"[backend.model_trainer] [1/2] 正在进行数据预处理: {video_path}")
            
            process_script = os.path.join(er_nerf_root, "data_utils", "process.py")
            process_cmd = [
                "python", process_script, 
                video_path,
                "--task", task_id 
            ]
            
            # 注意：ER-NeRF 的 process.py 通常依赖 ffmpeg 和 face-alignment 库
            print(f"[backend.model_trainer] 执行预处理命令: {' '.join(process_cmd)}")
            
            # 使用 check=True 确保如果预处理失败直接抛出异常，不进行训练
            subprocess.run(process_cmd, check=True) # 可以根据需要添加 capture_output=True
            
            print("[backend.model_trainer] 数据预处理完成。")

           
            #  模型训练 (调用 ER-NeRF/main.py)
            # 训练 NeRF 渲染网络
          
            print(f"[backend.model_trainer] [2/2] 开始训练 ER-NeRF 模型...")
            
            train_script = os.path.join(er_nerf_root, "main.py")
            # 构造训练参数
            train_cmd = [
                "python", train_script, 
                dataset_path,                  # 数据集路径
                "--workspace", task_id,        # 工作区名称 (用于保存模型 checkpoint)
                "-O",                          # 开启优化选项 (Head/Torso 分离等默认配置)
                "--iters", str(data['epoch']), # 训练迭代次数
                "--save_latest"                # 保存最新的 checkpoint
            ]

            # 如果前端传递了显卡选择，设置环境变量
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
            
            print("[backend.model_trainer] ER-NeRF 训练日志摘要:")
            # 打印最后几行日志以确认状态
            print('\n'.join(result.stdout.splitlines()[-10:]))
            
            print(f"[backend.model_trainer] ER-NeRF 训练成功完成！模型保存在 workspace/{task_id}")

      

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

