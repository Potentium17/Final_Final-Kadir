from diffusers import StableDiffusionPipeline
import gradio as gr
import torch

text_examples = ["a anime man"]


def orangemixs_image_generator(prompt: str = 'a anime man'):

    device = "cuda" if torch.cuda.is_available else "cpu" 
    pipe = StableDiffusionPipeline.from_pretrained("ArtGAN/AbyssOrangeMix_base", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]  

    return image


app = gr.Blocks()
with app:
    gr.Markdown("# **<h2 align='center'>Prompt-to-Prompt Image Editing with Cross-Attention Control<h2>**")
    gr.Markdown(
        """
        <h5 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a> |
        </h5>
        """
    )  
    with gr.Row():
        with gr.Column():
            base_prompt = gr.Textbox(lines=1, value=text_examples[0], label='Base Prompt')
            predict = gr.Button(value='Predict')

        with gr.Column():
            output_image = gr.Image(label='Output Image')

    predict.click(
        fn = orangemixs_image_generator,
        inputs = [base_prompt],
        outputs = [output_image],
        )

    gr.Examples(
            examples=text_examples[0], 
            inputs=[base_prompt], 
            outputs=[output_image], 
            fn=orangemixs_image_generator, 
            cache_examples=True,
            label='Text Example'
        )
app.launch()