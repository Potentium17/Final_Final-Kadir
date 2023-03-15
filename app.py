from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr
import torch
import os

device = "cuda" if torch.cuda.is_available else "cpu" 
os.system(f"git lts clone https://huggingface.co/WarriorMama777/AbyssOrangeMix /home/user/app/AbyssOrangeMix")

anime_model_list = [
    #"WarriorMama777/BloodOrangeMix",
    "AbyssOrangeMix",
    #"WarriorMama777/ElyOrangeMix",
    #"WarriorMama777/Other",
    #"WarriorMama777/AbyssOrangeMix2",
    #"WarriorMama777/EerieOrangeMix"
]

prompt_list = [
    "a photo of an anime girl."
]

bad_prompt_list = [
    "bad, ugly"
]

image_list = [
    "girl.png"
]

example_text_image = [[
   anime_model_list[0],
   prompt_list[0],
]]

example_image_image = [[
    image_list[0],
    anime_model_list[0],
    prompt_list[0],
    bad_prompt_list[0]
]]


def orangemixs_text_image_generator(
    model_id: str = anime_model_list[0],
    prompt: str = prompt_list[0],
    negative_prompt=bad_prompt_list[0]
    ):

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]

    return image

def orangemixs_image_image_generator(
    image_path: str = image_list[0],
    model_id: str = anime_model_list[0],
    prompt: str = prompt_list[0],
    negative_prompt=bad_prompt_list[0]
    ):

    init_image = Image.open(image_path)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image).images[0]

    return image


app = gr.Blocks()
with app:
    gr.Markdown(
        """
        <p style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </p>
        """
    )  
    with gr.Row():
        with gr.Column():
            with gr.Tab('Text'):
                text_model_id = gr.Dropdown(choices=anime_model_list, value=anime_model_list[0], label='Model Id')
                text_prompt = gr.Textbox(lines=1, value=prompt_list[0], label='Base Prompt')
                bad_text_prompt = gr.Textbox(lines=1, value=bad_prompt_list[0], label='Bad Prompt')
                text_predict = gr.Button(value='Predict')

            with gr.Tab('Image'):
                image_file = gr.Image(type='filepath', label='Image File')
                image_model_id = gr.Dropdown(choices=anime_model_list, value=anime_model_list[1], label='Model Id')
                image_prompt = gr.Textbox(lines=1, value=prompt_list[0], label='Image Prompt')
                bad_image_prompt = gr.Textbox(lines=1, value=bad_prompt_list[0], label='Bad Prompt')
                image_predict = gr.Button(value='Predict')

        with gr.Tab('Output'):
            with gr.Column():
                output_image = gr.Image(label='Output Image')

    text_predict.click(
        fn = orangemixs_text_image_generator,
        inputs = [text_model_id,text_prompt,bad_text_prompt],
        outputs = [output_image]
        )

    image_predict.click(
        fn = orangemixs_image_image_generator,
        inputs = [image_file, image_model_id, image_prompt, bad_image_prompt],
        outputs = [output_image]
        )

    gr.Examples(
            examples=example_text_image, 
            inputs=[text_model_id, text_prompt, bad_text_prompt], 
            outputs = [output_image],
            fn=orangemixs_text_image_generator, 
            cache_examples=True,
            label='Text Example'
        )
app.launch()