import os,sys
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import json
from contextlib import contextmanager
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
def process_dataset(csv_file, cif_col, *prop_cols):
    """处理上传的数据集"""
    try:
        df = pd.read_csv(csv_file.name)
        
        # Filter out None values from prop_cols
        valid_prop_cols = [col for col in prop_cols if col is not None]
        
        # 将列名转换为索引
        cif_col_idx = df.columns.get_loc(cif_col)
        prop_col_indices = [df.columns.get_loc(col) for col in valid_prop_cols]
        
        # 构建命令
        cmd = [
            "python", 
            "run.py",
            "--raw_data_path", csv_file.name,
            "--cif_column_index", str(cif_col_idx),
            "--prop_column_index_list"
        ] + [str(idx) for idx in prop_col_indices]
        
        log_output = f"运行命令: {' '.join(cmd)}\n"
        
        dataset_dir = os.path.join(os.getcwd(), "0_dataset")
        
        # 使用Popen来实时获取输出
        process = subprocess.Popen(
            cmd, 
            cwd=dataset_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时读取输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_output += output
                yield log_output, None  # 更新日志但不更新预览
        
        if process.returncode != 0:
            error = process.stderr.read()
            log_output += f"\n错误输出: {error}"
            yield log_output, f"数据集构建失败: {error}"
            return
        
        train_df = pd.read_csv(os.path.join(dataset_dir, "train_data.csv"))
        yield log_output, train_df.head(10).to_html()
    
    except Exception as e:
        yield f"数据集构建过程中发生异常: {str(e)}", None

def train_model(run_name, batch_size, max_epochs, n_embd, n_layer, n_head, lr, n_visible, *prop_cols):
    """训练模型"""
    try:
        # Save n_head and prop_cols to config file
        model_dir = os.path.join(os.getcwd(), "1_train_generate/model")
        config_path = os.path.join(model_dir, f"{run_name}.ini")
        
        # 只收集前 n_visible 个下拉菜单中的值
        n_visible = int(n_visible)
        valid_prop_cols = [col for col in prop_cols[:n_visible] if col is not None]
        
        with open(config_path, 'w') as f:
            json.dump({
                "n_head": n_head,
                "prop_cols": valid_prop_cols
            }, f)

        train_dir = os.path.join(os.getcwd(), "1_train_generate")
        
        cmd = [
            "python",
            "-u",
            "train.py",
            "--run_name", run_name,
            "--batch_size", str(batch_size),
            "--max_epochs", str(max_epochs),
            "--n_embd", str(n_embd),
            "--n_layer", str(n_layer), 
            "--n_head", str(n_head),
            "--learning_rate", str(lr),
            "--prop_column_index_list"
        ]
        
        # 读取训练数据集以获取列索引
        dataset_dir = os.path.join(os.getcwd(), "0_dataset")
        train_df = pd.read_csv(os.path.join(dataset_dir, "train_data.csv"))
        prop_col_indices = [train_df.columns.get_loc(col) for col in valid_prop_cols]
        cmd.extend([str(idx) for idx in prop_col_indices])
        
        log_output = f"运行命令: {' '.join(cmd)}\n"
        
        # 使用Popen来实时获取输出
        process = subprocess.Popen(
            cmd, 
            cwd=train_dir,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # 行缓冲
        )
        
        # 实时读取输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_output += output
                yield log_output, None  # 更新日志但不更新图表
        
        if process.returncode != 0:
            error = process.stderr.read()
            log_output += f"\n错误输出: {error}"
            yield log_output, None
            return
        
        # 训练完成后返回训练曲线图
        history = pd.read_csv(os.path.join(train_dir, "train.log"))
        fig = plt.figure()
        plt.plot(history["Train Loss"], label="Train")
        plt.plot(history["Val Loss"], label="Val")
        plt.legend()
        yield log_output, fig

    except Exception as e:
        yield f"训练过程中发生异常: {str(e)}", None

def generate_structures(model_weight, prop_targets, gen_size, batch_size):
    """生成结构"""
    try:
        model_dir = os.path.join(os.getcwd(), "1_train_generate/model")
        train_dir = os.path.join(os.getcwd(), "1_train_generate")
        # Load n_head from config file
        config_path = os.path.join(model_dir, f"{os.path.splitext(model_weight)[0]}.ini")
        output_csv = os.path.join(train_dir, f"{os.path.splitext(model_weight)[0]}_generated.csv")
        with open(config_path, 'r') as f:
            config = json.load(f)
            n_head = config["n_head"]

        cmd = [
            "python",
            "generate.py",
            "--model_weight", model_weight,
            "--prop_targets", prop_targets,
            "--gen_size", str(gen_size),
            "--batch_size", str(batch_size),
            "--n_head", str(n_head),
            "--output_csv", output_csv
        ]
        
        log_output = f"运行命令: {' '.join(cmd)}\n"
        
        process = subprocess.Popen(
            cmd, 
            cwd=train_dir,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_output += output
                yield log_output, None, None  # Add None for preview and download during processing
        
        if process.returncode != 0:
            error = process.stderr.read()
            log_output += f"\n错误输出: {error}"
            yield log_output, None, None
            return
        
        # 生成完成后返回预览和下载链接
        df = pd.read_csv(output_csv)
        yield log_output, df.head(10).to_html(), output_csv

    except Exception as e:
        yield f"生成过程中发生异常: {str(e)}", None, None

def decode_slices(run_name):
    """解码SLICES字符串"""
    try:
        decode_dir = os.path.join(os.getcwd(), "2_decode")
        train_dir = os.path.join(os.getcwd(), "1_train_generate")
        input_csv = os.path.join(train_dir, f"{run_name}_generated.csv")
        output_csv = os.path.join(decode_dir, f"{run_name}_decoded.csv")
        
        cmd = [
            "python",
            "run.py",
            "--input_csv", input_csv,
            "--output_csv", output_csv
        ]
        
        log_output = f"运行命令: {' '.join(cmd)}\n"
        
        process = subprocess.Popen(
            cmd,
            cwd=decode_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_output += output
                yield log_output, None  # Add None for the file output during processing
        
        if process.returncode != 0:
            error = process.stderr.read()
            log_output += f"\n错误输出: {error}"
            yield log_output, None
            return
        
        # Return both the log and the file path for download
        yield log_output + f"\n解码完成，结果保存在: {output_csv}", output_csv
        
    except Exception as e:
        yield f"解码过程中发生异常: {str(e)}", None

def check_novelty(run_name, structure_json):
    """新颖性检查"""
    try:
        decode_dir = os.path.join(os.getcwd(), "2_decode")
        novelty_dir = os.path.join(os.getcwd(), "3_novelty")
        input_csv = os.path.join(decode_dir, f"{run_name}_decoded.csv")
        output_csv = os.path.join(novelty_dir, f"{run_name}_novelty_checked.csv")
        
        cmd = [
            "python",
            "run.py",
            "--input_csv", input_csv,
            "--output_csv", output_csv,
            "--structure_json_for_novelty_check", structure_json
        ]
        
        log_output = f"运行命令: {' '.join(cmd)}\n"
        
        process = subprocess.Popen(
            cmd,
            cwd=novelty_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_output += output
                yield log_output, None, None  # Add None for result and file during processing
        
        if process.returncode != 0:
            error = process.stderr.read()
            log_output += f"\n错误输出: {error}"
            yield log_output, None, None
            return
        
        # 返回结果统计
        df = pd.read_csv(output_csv)
        novel_count = (df["novelty"] == 1).sum()
        yield log_output, f"发现{novel_count}个新颖结构", output_csv
        
    except Exception as e:
        yield f"新颖性检查过程中发生异常: {str(e)}", None, None

def add_property_dropdown(n_props):
    """添加新的性质下拉框"""
    return gr.Dropdown.update(visible=True)

def remove_prop(n):
    n = int(n)
    if n > 1:  # 保持至少一个性质列
        return {
            prop_dropdowns[n-1]: gr.Dropdown(visible=False),
            n_visible_props: n - 1
        }
    return {n_visible_props: n}

def sync_props_to_train(n, *prop_values):
    """将数据集页面的性质列同步到训练页面"""
    n = int(n)
    updates = {}
    for i in range(5):
        if i < n:
            updates[prop_dropdowns_train[i]] = gr.Dropdown(value=prop_values[i], visible=True)
        else:
            updates[prop_dropdowns_train[i]] = gr.Dropdown(visible=False)
    updates[n_visible_props_train] = n
    return updates

def add_prop_train(n):
    """添加训练页面的性质列"""
    n = int(n)
    if n < len(prop_dropdowns_train):
        updates = []
        for i in range(len(prop_dropdowns_train)):
            if i == n:
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update())
        return updates + [n + 1]
    return [gr.update() for _ in range(len(prop_dropdowns_train))] + [n]

def remove_prop_train(n):
    """删除训练页面的性质列"""
    n = int(n)
    if n > 1:
        updates = []
        for i in range(len(prop_dropdowns_train)):
            if i == (n-1):
                updates.append(gr.update(visible=False))
            else:
                updates.append(gr.update())
        return updates + [n - 1]
    return [gr.update() for _ in range(len(prop_dropdowns_train))] + [n]

def update_dropdowns(csv_file):
    if csv_file is None:
        return {
            preview_output: None,
            cif_col_dropdown: gr.Dropdown(choices=[]),
            **{dropdown: gr.Dropdown(choices=[]) for dropdown in prop_dropdowns}
        }
    
    df = pd.read_csv(csv_file.name)
    preview = df.head(1).to_html()
    columns = df.columns.tolist()
    
    # 自动选择CIF列
    default_cif_col = next((col for col in columns if col in ["CIF", "cif"]), None)
    
    # 初始下拉列表排除CIF列
    available_props = [col for col in columns if col != default_cif_col]
    
    return {
        preview_output: preview,
        cif_col_dropdown: gr.Dropdown(choices=columns, value=default_cif_col),
        **{dropdown: gr.Dropdown(choices=available_props) for dropdown in prop_dropdowns}
    }

def update_available_props(cif_col, *prop_values):
    """更新可用的性质列表，排除已选择的列"""
    if not csv_input.value:
        return [gr.Dropdown() for _ in prop_dropdowns]
    
    df = pd.read_csv(csv_input.value.name)
    all_columns = df.columns.tolist()
    
    # 排除CIF列和已选择的性质
    selected_props = [prop for prop in prop_values if prop]
    available_props = [col for col in all_columns if col not in [cif_col] + selected_props]
    
    updates = []
    for i, current_value in enumerate(prop_values):
        if current_value and current_value not in available_props:
            available_props.append(current_value)
        updates.append(gr.Dropdown(choices=available_props, value=current_value))
    
    return updates

def update_train_dropdowns(csv_file, cif_col, *prop_values):
    try:
        # 直接从训练数据集读取列名
        dataset_dir = os.path.join(os.getcwd(), "0_dataset")
        train_df = pd.read_csv(os.path.join(dataset_dir, "train_data.csv"))
        columns = train_df.columns.tolist()
        
        # 排除 SLICES 列
        available_props = [col for col in columns if col != "SLICES"]
        
        return {dropdown: gr.Dropdown(choices=available_props) for dropdown in prop_dropdowns_train}
    except Exception as e:
        # 如果读取失败（比如文件不存在），返回空选项
        return {dropdown: gr.Dropdown(choices=[]) for dropdown in prop_dropdowns_train}

def load_train_columns():
    """加载训练数据集的列名"""
    try:
        dataset_dir = os.path.join(os.getcwd(), "0_dataset")
        train_df = pd.read_csv(os.path.join(dataset_dir, "train_data.csv"))
        columns = train_df.columns.tolist()
        # 排除 SLICES 列
        available_props = [col for col in columns if col != "SLICES"]
        return available_props
    except Exception as e:
        return []

def load_model_info(model_weight):
    """加载模型配置并生成目标性质填写指南"""
    try:
        model_dir = os.path.join(os.getcwd(), "1_train_generate/model")
        config_path = os.path.join(model_dir, f"{os.path.splitext(model_weight)[0]}.ini")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            prop_cols = config.get("prop_cols", [])
            
        # 生成示例格式
        example = "[[" + ", ".join(f"{col}目标值" for col in prop_cols) + "]]"
        
        # 生成说明文本
        guide = f"请按以下格式填写目标性质值：\n{example}\n\n"
        guide += "如需设计多组目标性质的新材料，请使用如下格式：\n"
        multi_example = "[[" + ", ".join(f"{col}目标值1" for col in prop_cols) + "], "
        multi_example += "[" + ", ".join(f"{col}目标值2" for col in prop_cols) + "]]"
        guide += multi_example
        
        return {
            prop_guide: guide,
            prop_targets: "[[]]",
            n_head_value: config["n_head"]
        }
    except Exception as e:
        return {
            prop_guide: f"加载模型配置失败: {str(e)}",
            prop_targets: "",
            n_head_value: None
        }

def find_latest_generated_file():
    """查找train_dir下最新的*_generated.csv文件，返回其运行名称"""
    train_dir = os.path.join(os.getcwd(), "1_train_generate")
    generated_files = [f for f in os.listdir(train_dir) if f.endswith("_generated.csv")]
    
    if generated_files:
        # 从文件名中提取运行名称（去掉_generated.csv后缀）
        return os.path.splitext(generated_files[0])[0].replace("_generated", "")
    return "test_run"

def update_train_tab_dropdowns(csv_file):
    """当切换到训练标签页时更新性质列下拉菜单"""
    try:
        # 首先尝试从训练数据集读取列名
        dataset_dir = os.path.join(os.getcwd(), "0_dataset")
        train_data_path = os.path.join(dataset_dir, "train_data.csv")
        
        if os.path.exists(train_data_path):
            # 如果训练数据集存在，使用其列名
            df = pd.read_csv(train_data_path)
            columns = df.columns.tolist()
            # 排除 SLICES 列
            available_props = [col for col in columns if col != "SLICES"]
        else:
            # 如果训练数据集不存在，尝试使用数据集页面的数据
            if csv_file:
                df = pd.read_csv(csv_file.name)
                columns = df.columns.tolist()
                # 排除已选择的CIF列
                if cif_col_dropdown.value:
                    available_props = [col for col in columns if col != cif_col_dropdown.value]
                else:
                    available_props = columns
            else:
                available_props = []
        
        return {dropdown: gr.Dropdown(choices=available_props) for dropdown in prop_dropdowns_train}
    
    except Exception as e:
        print(f"Error updating train tab dropdowns: {str(e)}")
        return {dropdown: gr.Dropdown(choices=[]) for dropdown in prop_dropdowns_train}

def update_model_weight(run_name):
    """根据训练标签页的运行名称更新模型权重文件名"""
    return f"{run_name}.pt"

def update_train_and_model(csv_file, run_name):
    """处理顶层标签页切换时的更新"""
    # 获取训练性质列的更新
    dropdown_updates = update_train_tab_dropdowns(csv_file)
    # 获取模型权重的更新
    model_weight_update = update_model_weight(run_name)
    
    # 合并更新结果
    return {
        **dropdown_updates,
        model_weight: model_weight_update
    }

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center;">
            <h1 style="margin-right: 10px; font-size: 36px;">
                Matter<span style="color:#00AAFF;">GPT</span>
            </h1>
        </div>
        """
    )
    with gr.Tab("数据集构建"):
        csv_input = gr.File(label="上传CSV文件")
        preview_output = gr.HTML(label="数据预览")
        
        with gr.Row():
            cif_col_dropdown = gr.Dropdown(
                label="CIF列",
                choices=[],
                interactive=True
            )
        
        prop_cols_container = gr.Column()
        prop_dropdowns = []
        
        # 创建最多5个性质下拉框(初始只显示1个)
        for i in range(5):
            with prop_cols_container:
                with gr.Row():
                    dropdown = gr.Dropdown(
                        label=f"性质列 {i+1}",
                        choices=[],
                        interactive=True,
                        visible=(i==0)  # 只显示第一个
                    )
                    prop_dropdowns.append(dropdown)
        
        with gr.Row():
            add_prop_btn = gr.Button("添加性质列")
            remove_prop_btn = gr.Button("删除性质列")
        process_btn = gr.Button("处理数据集")
        log_output = gr.Textbox(label="处理日志", lines=10)
        result_preview = gr.HTML(label="结果预览")
        
        # 用于跟踪当前显示的性质下拉框数量
        n_visible_props = gr.Number(value=1, visible=False)
        
        def update_dropdowns(csv_file):
            if csv_file is None:
                return {
                    preview_output: None,
                    cif_col_dropdown: gr.Dropdown(choices=[]),
                    **{dropdown: gr.Dropdown(choices=[]) for dropdown in prop_dropdowns}
                }
            
            df = pd.read_csv(csv_file.name)
            preview = df.head(1).to_html()
            columns = df.columns.tolist()
            
            # 自动选择CIF列
            default_cif_col = next((col for col in columns if col in ["CIF", "cif"]), None)
            
            # 初始下拉列表排除CIF列
            available_props = [col for col in columns if col != default_cif_col]
            
            return {
                preview_output: preview,
                cif_col_dropdown: gr.Dropdown(choices=columns, value=default_cif_col),
                **{dropdown: gr.Dropdown(choices=available_props) for dropdown in prop_dropdowns}
            }

        
        def add_prop(n):
            n = int(n)
            if n < len(prop_dropdowns):
                return {
                    prop_dropdowns[n]: gr.Dropdown(visible=True),
                    n_visible_props: n + 1
                }
            return {n_visible_props: n}
        
        csv_input.change(
            update_dropdowns,
            inputs=[csv_input],
            outputs=[preview_output, cif_col_dropdown] + prop_dropdowns
        )
        
        add_prop_btn.click(
            add_prop,
            inputs=[n_visible_props],
            outputs=prop_dropdowns + [n_visible_props]
        )
        
        remove_prop_btn.click(
            remove_prop,
            inputs=[n_visible_props],
            outputs=prop_dropdowns + [n_visible_props]
        )
        
        process_btn.click(
            process_dataset,
            inputs=[csv_input, cif_col_dropdown] + prop_dropdowns,
            outputs=[log_output, result_preview]
        )

    train_generate_tab = gr.Tab("训练与生成")
    with train_generate_tab:
        with gr.Tab("训练", elem_id="train_tab") as train_tab:
            with gr.Row():
                run_name = gr.Textbox(label="运行名称", value="test_run")
                batch_size = gr.Number(label="Batch Size", value=20)
                max_epochs = gr.Number(label="最大Epoch数", value=50)
            with gr.Row():
                n_embd = gr.Number(label="Embedding维度", value=512)
                n_layer = gr.Number(label="层数", value=8)
                n_head = gr.Number(label="注意力头数", value=8)
                lr = gr.Number(label="学习率", value=3.3e-4)
            
            # 添加性质列设置
            prop_cols_container_train = gr.Column()
            prop_dropdowns_train = []
            
            # 获取训练数据的列名
            train_columns = load_train_columns()
            
            # 创建最多5个性质下拉框(初始显示第一个)
            for i in range(5):
                with prop_cols_container_train:
                    with gr.Row():
                        dropdown = gr.Dropdown(
                            label=f"训练性质列 {i+1}",
                            choices=train_columns,  # 直接使用训练数据的列名
                            interactive=True,
                            visible=(i==0)
                        )
                        prop_dropdowns_train.append(dropdown)
            
            with gr.Row():
                add_prop_btn_train = gr.Button("添加训练性质列")
                remove_prop_btn_train = gr.Button("删除训练性质列")
            
            n_visible_props_train = gr.Number(value=1, visible=False)
            
            # 更新训练标签页的性质下拉框
            def update_train_dropdowns(csv_file, cif_col, *prop_values):
                if csv_file is None:
                    return {dropdown: gr.Dropdown(choices=[]) for dropdown in prop_dropdowns_train}
                
                df = pd.read_csv(csv_file.name)
                columns = df.columns.tolist()
                
                # 排除CIF列和数据集页面已选择的性质
                selected_props = [prop for prop in prop_values if prop]
                available_props = [col for col in columns if col not in [cif_col] + selected_props]
                
                return {dropdown: gr.Dropdown(choices=available_props) for dropdown in prop_dropdowns_train}
            
            add_prop_btn_train.click(
                add_prop_train,
                inputs=[n_visible_props_train],
                outputs=prop_dropdowns_train + [n_visible_props_train]
            )
            
            remove_prop_btn_train.click(
                remove_prop_train,
                inputs=[n_visible_props_train],
                outputs=prop_dropdowns_train + [n_visible_props_train]
            )
            
            train_btn = gr.Button("开始训练")
            train_plot = gr.Plot(label="训练曲线")
            train_log = gr.Textbox(label="训练日志", lines=10)  # 新增训练日志输出
            
            train_btn.click(
                train_model,
                inputs=[run_name, batch_size, max_epochs, n_embd, n_layer, n_head, lr, n_visible_props_train] + prop_dropdowns_train,
                outputs=[train_log, train_plot]
            )
            
            # 添加标签页切换事件，包含csv_input作为输入
            train_tab.select(
                update_train_tab_dropdowns,
                inputs=[csv_input],
                outputs=prop_dropdowns_train
            )
        
        with gr.Tab("生成") as generate_tab:
            with gr.Row():
                model_weight = gr.Textbox(label="模型权重文件", value="test_run.pt")
                load_model_btn = gr.Button("加载模型")
            
            prop_guide = gr.Markdown(label="目标性质填写指南")
            n_head_value = gr.Number(visible=False)
            
            with gr.Row():
                prop_targets = gr.Textbox(
                    label="目标性质",
                    value="[[]]",
                    scale=2
                )
                gen_size = gr.Number(
                    label="生成数量",
                    value=50,
                    scale=1
                )
                gen_batch = gr.Number(
                    label="生成批次大小",
                    value=5,
                    scale=1
                )
            
            gen_btn = gr.Button("开始生成")
            gen_log = gr.Textbox(label="生成日志", lines=10)
            with gr.Row():
                gen_preview = gr.HTML(label="生成结果预览")
                gen_download = gr.File(label="下载生成结果", interactive=False)  # Add download widget
            
            # 添加加载模型按钮的事件处理
            load_model_btn.click(
                load_model_info,
                inputs=[model_weight],
                outputs=[prop_guide, prop_targets, n_head_value]
            )
            
            gen_btn.click(
                generate_structures,
                inputs=[model_weight, prop_targets, gen_size, gen_batch],
                outputs=[gen_log, gen_preview, gen_download]
            )

        # 为生成标签页添加切换事件
        generate_tab.select(
            update_model_weight,
            inputs=[run_name],
            outputs=[model_weight]
        )

    with gr.Tab("解码与新颖性"):
        # 初始化时查找已存在的生成文件
        initial_run_name = find_latest_generated_file()
        
        run_name_decode = gr.Textbox(
            label="运行名称",
            value=initial_run_name,
            interactive=True
        )
        
        # 检查默认文件是否存在
        train_dir = os.path.join(os.getcwd(), "1_train_generate")
        data_dir = os.path.join(os.getcwd(), "0_dataset")
        default_file = os.path.join(train_dir, f"{initial_run_name}_generated.csv")
        if not os.path.exists(default_file):
            run_name_decode.value = ""
        
        with gr.Row():
            decode_btn = gr.Button("解码SLICES字符串")
            decode_log = gr.Textbox(label="解码日志", lines=10)
            decode_download = gr.File(label="下载解码结果", interactive=False)  # Add download for decoded results
        
        structure_json = gr.Textbox(
            label="结构JSON文件路径",
            value=os.path.join(data_dir, "cifs_filtered.json")
        )
        
        with gr.Row():
            check_btn = gr.Button("新颖性检查")
            check_log = gr.Textbox(label="检查日志", lines=10)
            check_result = gr.Textbox(label="检查结果")
            novelty_download = gr.File(label="下载新颖性检查结果", interactive=False)  # Add download for novelty results

        # 添加生成完成后的联动
        def update_decode_run_name(model_weight):
            """从模型权重文件名中提取运行名称"""
            return os.path.splitext(model_weight)[0]  # 去掉.pt后缀
        
        # 在生成tab中，生成按钮点击后更新解码tab的运行名称
        gen_btn.click(
            update_decode_run_name,
            inputs=[model_weight],
            outputs=[run_name_decode]
        )
        
        decode_btn.click(
            decode_slices,
            inputs=[run_name_decode],
            outputs=[decode_log, decode_download]
        )
        
        check_btn.click(
            check_novelty,
            inputs=[run_name_decode, structure_json],
            outputs=[check_log, check_result, novelty_download]
        )

    # 在数据集页面添加性质列变化时的联动
    for dropdown in prop_dropdowns:
        dropdown.change(
            sync_props_to_train,
            inputs=[n_visible_props] + prop_dropdowns,
            outputs=prop_dropdowns_train + [n_visible_props_train]
        )

    # 添加CIF列和性质列的联动
    cif_col_dropdown.change(
        update_available_props,
        inputs=[cif_col_dropdown] + prop_dropdowns,
        outputs=prop_dropdowns
    )
    
    # 添加性质列之间的联动
    for dropdown in prop_dropdowns:
        dropdown.change(
            update_available_props,
            inputs=[cif_col_dropdown] + prop_dropdowns,
            outputs=prop_dropdowns
        )

    # 为顶层标签页添加切换事件，同时更新训练性质列和模型权重
    train_generate_tab.select(
        update_train_and_model,
        inputs=[csv_input, run_name],
        outputs=[
            *prop_dropdowns_train,
            model_weight
        ]
    )

print(f"Running on local URL: http://localhost:7860")
with suppress_output():
    demo.launch(server_name="0.0.0.0", server_port=7860)