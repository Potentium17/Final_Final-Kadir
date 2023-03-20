from inpaint_zoom.app.zoom_out_app import stable_diffusion_zoom_out_app
from inpaint_zoom.app.zoom_in_app import StableDiffusionZoomIn

import gradio as gr

app = gr.Blocks()
with app:
    gr.HTML(
        """
        <h1 style='text-align: center'>
       Stable Diffusion Infinite Zoom Out and Zoom In
        </h1>
        """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h3>
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab('Zoom In'):
                StableDiffusionZoomIn.app()
            with gr.Tab('Zoom Out'):
                stable_diffusion_zoom_out_app()

app.launch(debug=True)