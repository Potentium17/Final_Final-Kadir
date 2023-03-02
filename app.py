from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import gradio as gr
import torch

device = "cuda" if torch.cuda.is_available else "cpu" 

anime_orange_model_list = [
    "WarriorMama777/BloodOrangeMix",
    "WarriorMama777/AbyssOrangeMix",
    "WarriorMama777/ElyOrangeMix",
    "WarriorMama777/Other",
    "WarriorMama777/AbyssOrangeMix2",
    "WarriorMama777/EerieOrangeMix"
]

example_prompt_list = [
    "a photo of an anime girl."
]

example_image_list = [
    "girl.png"
]

example_text_image = [
   [anime_orange_model_list[1]],
   [example_prompt_list[0]]
]

example_image_image = [
    [example_image_list[0]],
    [anime_orange_model_list[0]],
    [example_prompt_list[0]]
]

def orangemixs_text_image_generator(
    model_id: str = 'WarriorMama777/AbyssOrangeMix',
    prompt: str = 'a photo of an anime girl.'
    ):

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt=prompt).images[0]

    return image

def orangemixs_image_image_generator(
    image_path: str = example_image_list[0],
    model_id: str = anime_orange_model_list[1],
    prompt: str = 'a photo of an anime girl.'
    ):

    init_image = Image.open(image_path)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt=prompt, image=init_image).images[0]

    return image


app = gr.Blocks()
with app:
    gr.Markdown(
    """
    ![](https://raw.githubusercontent.com/WarriorMama777/imgup/main/img/AOM3/AOM3_G_Top_comp001.webp"image_orangemixs_infographics_01")
    """
    )
    gr.Markdown(
        """
        <h5 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h5>
        """
    )  
    with gr.Row():
        with gr.Column():
            with gr.Tab('Text'):
                text_model_id = gr.Dropdown(choices=anime_orange_model_list, value=anime_orange_model_list[0], label='Model Id')
                text_prompt = gr.Textbox(lines=1, value=example_prompt_list[0], label='Text Prompt')
                text_predict = gr.Button(value='Predict')

            with gr.Tab('Text'):
                image_path = gr.Image(type='filepath', label='Image File')
                image_model_id = gr.Dropdown(choices=anime_orange_model_list, value=anime_orange_model_list[0], label='Model Id')
                image_prompt = gr.Textbox(lines=1, value=example_prompt_list[0], label='Image Prompt')
                image_predict = gr.Button(value='Predict')


        with gr.Column():
            output_image = gr.Image(label='Output Image')

    text_predict.click(
        fn = orangemixs_text_image_generator,
        inputs = [text_model_id,text_prompt],
        outputs = output_image
        )

    image_predict.click(
        fn = orangemixs_text_image_generator,
        inputs = [image_path, image_model_id, image_prompt],
        outputs = output_image
        )

    gr.Examples(
            examples=example_text_image, 
            inputs=[text_model_id,text_prompt], 
            outputs = [output_image],
            fn=orangemixs_text_image_generator, 
            cache_examples=True,
            label='Text Example'
        )
app.launch()