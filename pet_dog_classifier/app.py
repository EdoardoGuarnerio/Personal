from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

model = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = model.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image(height=192, width=192)
label = gr.Label()
examples:list = ['./dog.png', './cat.png']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)