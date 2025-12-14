import subprocess
import os
import json
from datetime import datetime

# 导入OpenVoice相关功能
from .voice_generator import extract_speaker_feature

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    video_path = data['ref_video']
    print(f"输入视频：{video_path}")

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

            extract_trait_from_audio(data)

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

            extract_trait_from_audio(data)


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

def extract_trait_from_audio(data):
    try:

        video_path = data['ref_video']

        print("[backend.model_trainer] 开始OpenVoice语音特征提取...")

        # 从视频中提取音频
        extracted_audio = extract_audio_from_video(video_path)



        # ToDo : 可以让用户自定义语音名字
        speaker_id=data["speaker_id"]
        if speaker_id:
            print()
        else:
            # 生成的说话人ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            speaker_id = f"user_{timestamp}"

        # 提取并保存说话人特征
        success = extract_speaker_feature(speaker_id, extracted_audio)

        if success:
            print(f"[backend.model_trainer] OpenVoice说话人特征提取成功: {speaker_id}")

            # 保存训练信息
            model_info = {
                "model_type": "OpenVoice",
                "speaker_id": speaker_id,
                "reference_video": video_path,
                "extracted_audio": extracted_audio,
                "training_time": datetime.now().isoformat(),
                "status": "completed"
            }

            info_path = f"models/OpenVoice/{speaker_id}_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"[backend.model_trainer] 模型信息已保存: {info_path}")

        else:
            print("[backend.model_trainer] OpenVoice说话人特征提取失败")

    except Exception as e:
        print(f"[backend.model_trainer] OpenVoice训练失败: {e}")

def extract_audio_from_video(video_path):
    """
    从视频中提取音频
    """
    try:
        # 确保输出目录存在
        os.makedirs("static/audios", exist_ok=True)

        # 生成音频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_output = f"static/audios/extracted_{timestamp}.wav"

        # 使用ffmpeg提取音频
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # 禁用视频
            "-acodec", "pcm_s16le",  # 音频编码
            "-ar", "16000",  # 采样率
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            audio_output
        ]

        print(f"[backend.model_trainer] 提取音频命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1分钟超时
        )

        if result.returncode == 0:
            print(f"[backend.model_trainer] 音频提取成功: {audio_output}")
            return audio_output
        else:
            print(f"[backend.model_trainer] 音频提取失败: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("[backend.model_trainer] 音频提取超时")
        return None
    except Exception as e:
        print(f"[backend.model_trainer] 音频提取异常: {e}")
        return None
