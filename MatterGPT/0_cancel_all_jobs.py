import os
import glob
import subprocess
import shutil
import logging
import warnings
import time

warnings.filterwarnings("ignore")
def is_slurm_available():
    """
    检测系统是否安装了 SLURM 作业管理系统。
    返回: True 如果存在 SLURM, 否则 False
    """
    return shutil.which('sinfo') is not None or shutil.which('squeue') is not None
print("\n检测到取消操作。正在终止所有 'pt_main_thread' 和 'python' 进程...")
logging.info("检测到取消操作。尝试终止所有 'pt_main_thread' 和 'python' 的进程。")

use_slurm = is_slurm_available()
if use_slurm:
    os.system('scancel --user=root')
    print("All jobs have been canceled")
else:
    try:
        
        # 定义要查找的进程名称列表
        process_names = ["pt_main_thread", "python"]
        all_pids = []
        for proc_name in process_names:
            try:
                # 使用 pgrep 查找精确匹配的进程名
                pgrep_output = subprocess.check_output(["pgrep", "-f", proc_name], stderr=subprocess.DEVNULL).decode().strip()
                pids = pgrep_output.split('\n') if pgrep_output else []
                pids = [pid for pid in pids if pid.isdigit()]
                all_pids.extend(pids)
            except subprocess.CalledProcessError:
                # 如果没有找到对应的进程，继续
                continue
        
        if not all_pids:
            print("未找到任何名为 'pt_main_thread' 或 'python' 的进程。")
            logging.info("未找到任何名为 'pt_main_thread' 或 'python' 的进程。")
        else:
            unique_pids = list(set(all_pids))  # 移除重复的 PID
            print(f"找到以下 PID: {', '.join(unique_pids)}")
            logging.info(f"找到以下 PID: {', '.join(unique_pids)}")
            
            # 发送 SIGTERM 信号以温和终止进程
            print("正在发送 SIGTERM 信号...")
            logging.info("发送 SIGTERM 信号给进程。")
            try:
                subprocess.run(["kill"] + unique_pids, check=True)
            except subprocess.CalledProcessError as e:
                print(f"发送 SIGTERM 信号失败: {e}")
                logging.error(f"发送 SIGTERM 信号失败: {e}")
            
            # 等待 5 秒以允许进程优雅终止
            time.sleep(5)
            
            # 检查哪些进程仍在运行
            remaining_pids = []
            for proc_name in process_names:
                try:
                    remaining_pgrep = subprocess.check_output(["pgrep", "-f", proc_name], stderr=subprocess.DEVNULL).decode().strip()
                    rem_pids = remaining_pgrep.split('\n') if remaining_pgrep else []
                    rem_pids = [pid for pid in rem_pids if pid.isdigit()]
                    remaining_pids.extend(rem_pids)
                except subprocess.CalledProcessError:
                    continue
            
            remaining_pids = list(set(remaining_pids))  # 移除重复的 PID
            
            if remaining_pids:
                print(f"以下进程未终止，正在发送 SIGKILL 信号: {', '.join(remaining_pids)}")
                logging.info(f"以下进程未终止，发送 SIGKILL 信号: {', '.join(remaining_pids)}")
                try:
                    subprocess.run(["kill", "-9"] + remaining_pids, check=True)
                    print("所有相关进程已被强制终止。")
                    logging.info("所有相关进程已被强制终止。")
                except subprocess.CalledProcessError as e:
                    print(f"发送 SIGKILL 信号失败: {e}")
                    logging.error(f"发送 SIGKILL 信号失败: {e}")
            else:
                print("所有相关进程已成功终止。")
                logging.info("所有相关进程已成功终止。")
    except subprocess.CalledProcessError:
        print("未找到任何名为 'pt_main_thread' 或 'python' 的进程。")
        logging.info("未找到任何名为 'pt_main_thread' 或 'python' 的进程。")
    except Exception as e:
        print(f"终止进程时发生错误: {e}")
        logging.error(f"终止进程时发生错误: {e}")

    print("开始清理任务目录...")
    logging.info("开始清理任务目录。")
    # 清理 job_* 目录
    job_dirs = glob.glob("job_*")
    for job_dir in job_dirs:
        try:
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir)
                print(f"已删除目录: {job_dir}")
                logging.info(f"已删除目录: {job_dir}")
        except FileNotFoundError:
            print(f"目录不存在: {job_dir}")
            logging.warning(f"目录不存在: {job_dir}")
        except Exception as e:
            print(f"无法删除目录 {job_dir}: {e}")
            logging.error(f"无法删除目录 {job_dir}: {e}")

