from transformers import pipeline
import gradio as gr

pipe = pipeline(model="mohamedsaeed823/whisper-small-arbyeg")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Egyptian",
    description="Realtime demo for Egyptian Arabic speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()