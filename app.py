import torch
import os
import random
import json
import gradio as gr
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
remote_model_name = "microsoft/Florence-2-base-ft"
model = AutoModelForCausalLM.from_pretrained(remote_model_name, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(remote_model_name, trust_remote_code=True)
checkpoint_dir = "G:/linux/epoch_69"
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("本地权重加载成功。")
mapping_file = r"H:\aUltimatedataset\vqaoracle\datasets\chatgptvqaset\caption.json"
with open(mapping_file, 'r', encoding='utf-8') as f:
    mapping_list = json.load(f)
mapping_dict = {item['answer']: item['label'] for item in mapping_list}
def process_image_caption(image, question="What is the meaning of this oracle bone character?"):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    task_prompt = "<CAPTION>"
    prompt = f"{task_prompt}{question}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<CAPTION>",
        image_size=(image.width, image.height)
    )
    
    generated_ans = parsed_answer.get("<CAPTION>", "").strip()
    mapped_label = mapping_dict.get(generated_ans, generated_ans)
    
    return generated_ans, mapped_label

def process_image_detailed_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    task_prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    
    return parsed_answer.get("<DETAILED_CAPTION>", "").strip()

def process_image_detection(image, text_input="oracle bone character"):
    if image.mode != "RGB":
        image = image.convert("RGB")
    task_prompt = f"<OD>{text_input}"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height)
    )
    detection_results = parsed_answer.get("<OD>", {})
    if detection_results:
        bboxes = detection_results.get('bboxes', [])
        labels = detection_results.get('labels', [])
        result_text = f"检测到 {len(bboxes)} 个目标:\n"
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            result_text += f"目标 {i+1}: {label}, 位置: {bbox}\n"
        return result_text
    else:
        return "未检测到目标"
def process_image_segmentation(image, text_input="oracle bone character"):
    """分割功能"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    task_prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{text_input}"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        image_size=(image.width, image.height)
    )
    segmentation_results = parsed_answer.get("<REFERRING_EXPRESSION_SEGMENTATION>", {})
    if segmentation_results:
        polygons = segmentation_results.get('polygons', [])
        labels = segmentation_results.get('labels', [])
        result_text = f"分割到 {len(polygons)} 个区域:\n"
        for i, (polygon, label) in enumerate(zip(polygons, labels)):
            result_text += f"区域 {i+1}: {label}, 多边形点数: {len(polygon)}\n"
        return result_text
    else:
        return "未找到分割区域"

with gr.Blocks() as demo:
    gr.Markdown("")
    gr.Markdown("支持甲骨文识别、详细字幕生成、目标检测和图像分割功能")

    with gr.Tabs():
        # 甲骨文识别标签页
        with gr.TabItem("甲骨文识别"):
            gr.Markdown("### 上传甲骨文图片，获取识别结果")
            with gr.Row():
                with gr.Column():
                    image_input_caption = gr.Image(type="pil", label="上传甲骨文图像")
                    question_input = gr.Textbox(
                        value="What is the meaning of this oracle bone character?",
                        label="问题"
                    )
                    submit_btn_caption = gr.Button("识别甲骨文")

                with gr.Column():
                    raw_output_caption = gr.Textbox(label="生成答案")
                    mapped_output_caption = gr.Textbox(label="中文解释")

            submit_btn_caption.click(
                fn=process_image_caption,
                inputs=[image_input_caption, question_input],
                outputs=[raw_output_caption, mapped_output_caption]
            )
        with gr.TabItem("详细字幕生成"):
            gr.Markdown("### 生成图像的详细描述")
            with gr.Row():
                with gr.Column():
                    image_input_detailed = gr.Image(type="pil", label="上传图像")
                    submit_btn_detailed = gr.Button("生成详细字幕")

                with gr.Column():
                    detailed_output = gr.Textbox(label="详细字幕", lines=5)

            submit_btn_detailed.click(
                fn=process_image_detailed_caption,
                inputs=[image_input_detailed],
                outputs=[detailed_output]
            )
        with gr.TabItem("目标检测"):
            gr.Markdown("### 检测图像中的指定目标")
            with gr.Row():
                with gr.Column():
                    image_input_detection = gr.Image(type="pil", label="上传图像")
                    detection_text = gr.Textbox(
                        value="oracle bone character",
                        label="检测目标",
                        placeholder="请输入要检测的目标名称"
                    )
                    submit_btn_detection = gr.Button("开始检测")

                with gr.Column():
                    detection_output = gr.Textbox(label="检测结果", lines=8)

            submit_btn_detection.click(
                fn=process_image_detection,
                inputs=[image_input_detection, detection_text],
                outputs=[detection_output]
            )

        # 图像分割标签页
        with gr.TabItem("图像分割"):
            gr.Markdown("### 分割图像中的指定区域")
            with gr.Row():
                with gr.Column():
                    image_input_segmentation = gr.Image(type="pil", label="上传图像")
                    segmentation_text = gr.Textbox(
                        value="oracle bone character",
                        label="分割目标",
                        placeholder="请输入要分割的目标名称"
                    )
                    submit_btn_segmentation = gr.Button("开始分割")

                with gr.Column():
                    segmentation_output = gr.Textbox(label="分割结果", lines=8)

            submit_btn_segmentation.click(
                fn=process_image_segmentation,
                inputs=[image_input_segmentation, segmentation_text],
                outputs=[segmentation_output]
            )

    # 示例图片（仅在甲骨文识别页面显示）
    gr.Examples(
        examples=[
            r"G:\florenceuse\dataset\oracle1.jpg",
            r"G:\florenceuse\dataset\oracle2.png"
        ],
        inputs=image_input_caption,
        outputs=[raw_output_caption, mapped_output_caption],
        fn=process_image_caption,
        cache_examples=True
    )

# 7. 启动界面
if __name__ == "__main__":
    demo.launch(share=True)
