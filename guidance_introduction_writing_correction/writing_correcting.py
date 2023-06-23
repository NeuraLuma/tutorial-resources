from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file for our API Keys

import guidance # Guidance
import gradio as gr # Used for our UI

llm = guidance.llms.OpenAI("gpt-4")

correction_program = guidance('''
{{#system~}}
You are a friendly expert in writing. You exceed in Grammar, Spelling, Vocabulary and Style. Assist the user with correcting their writing.
Only correct text that is delimited by <text></text>. When responding corrected text, do not add <text></text>.
Only correct text based on: 
{{~#each correction_criteria}}
- {{this}}
{{~/each}}
{{~/system}}

{{#user~}}
I need assistance with my writing. Help correct any Grammar and Spelling errors and suggest ways to improve my writing such as suggesting a diverse vocabulary.
Here is the text I wrote:
<text>
{{text_input}}
</text>
{{~/user}}

{{#assistant~}}
Sure, I am happy to help you. Here is the corrected text I wrote:
{{~/assistant}}

{{#assistant~}}
{{gen 'answer' temperature=0 max_tokens=500}}
{{~/assistant}}''', llm=llm)

with gr.Blocks() as iface:
    # Options for the checkboxes
    options=['Grammar', 'Spelling', 'Style', 'Vocabulary']

    def handle_submit(input_text, selected_options):
        # Get our result from the chain and append to history:
        result = correction_program(text_input=input_text, correction_criteria=selected_options)

        return result['answer']

    title = gr.components.Markdown("# Writing Correction")
    description = gr.components.Markdown("This Application assists you in correcting your writing. You can use the text input box to write your text and the correction options to select what you want to correct.")

    with gr.Row():
        input_textbox = gr.Textbox(lines=10, label="Text to be corrected")
        output_textbox = gr.Textbox(lines=10, label="Corrected Text", show_copy_button=True)
    
    option_checkboxes = gr.CheckboxGroup(choices=options, label="Correction Options")
    submit_button = gr.Button("Submit")
    submit_button.click(fn=handle_submit, inputs=[input_textbox, option_checkboxes], outputs=output_textbox)
    
# Launch the Gradio app
iface.launch()
