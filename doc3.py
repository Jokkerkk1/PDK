import time
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import FewShotPromptTemplate

import openai
import gradio as gr

os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-OKALqg46qdC65159Z5Wk4pWiGkbFwfiRWRVkvsudwUlikn6X"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4-1106-preview"
)

def load_documents(directory="/home/zhouk23/xinde/xinde"):
    """
    加载books下的文件，进行拆分
    :param directory:
    :return:
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # text_spliter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    text_spliter = CharacterTextSplitter(
        separator = "Q:",
        chunk_size = 0,
        chunk_overlap  = 0,
        is_separator_regex = True,
    )

    split_docs = text_spliter.split_documents(documents)
    # print(split_docs[:2])
    return split_docs

# load_documents


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


# load embedding module
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
# load database
if not os.path.exists('VectorStore'):
    documents = load_documents()
    # documents_Rule = load_prompt_documents()
    db = store_chroma(documents, embeddings)

else:
    db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)

# Create some examples
examples = [
    {
        "query": "NW_2:NW width for NW resistor is 1.2",
        "answer":
            """NW_2 {
                @ NW width for NW resistor is 1.2
                INT NWR < 1.20 ABUT<90 OPPOSITE REGION
                }"""
    }, {
        "query": "NW_15:P+AA minimum enclosure by NW is 0.08, excluding LDMOS area.",
        "answer":
            """NW_15 {
                @ P+AA minimum enclosure by NW is 0.08, excluding LDMOS area.
                ENC (PACT OUTSIDE (INST OR LDBK)) NW < 0.08 ABUT<90 SINGULAR REGION
                }
            """
    }, {
        "query": "RESNWST_8：AA enclosure of NW(the NW interacted with RESNW) EN2 >= 0.3",
        "answer":
            """RESNWST_8 {
                @ AA enclosure of NW(the NW interacted with RESNW) EN2 >= 0.3
                ENC (NW INTERACT RESNW) AA <0.3 ABUT<90 REGION
                }
            """
    }, {
        "query": "SDOP_1：SDOP width is 0.18",
        "answer":
            """SDOP_1 {
                @ SDOP width is 0.18
                INT SDOP < 0.18 ABUT<90 SINGULAR REGION
                }
            """
    }, {
        "query": "Min Overlap of NW and DNW is 0.4",
        "answer":
            """DNW_4 {
                @ Min Overlap of NW and DNW is 0.4
                OUT1 = INT NW DNW < 0.4 ABUT<90 SINGULAR REGION
                OUT1 NOT MARKS 
                }
            """
    }, {
        "query": "It is not allowed that N+AA CUT DNW",
        "answer":
            """DNW_7 {
                @ It is not allowed that N+AA CUT DNW
                OUT1 = NACT CUT DNW
                OUT1 NOT MARKS 
                }
            """
    }, {
        "query": """Width of 45-degree AA. ≥ 0.45um
 	                45-degree AA edge length. ≥ 0.45um """,
        "answer":
            """AA_3_30 {
                @ Width of 45-degree AA. ≥ 0.45um
                @ 45-degree AA edge length. ≥ 0.45um
                X = ANGLE AA_with_dummy >= 44.9 <= 45.1
                OUT1 = EXPAND EDGE (LENGTH X < 0.45) INSIDE BY 0.001
                OUT2 = INT X < 0.45 ABUT<90 REGION 
                (OUT1 OR OUT2) NOT MARKS 
                }
            """
    }, {
        "query": """pace between two AAs inside DG/TG.
 	                DRC doesn’t check LDBK region.	≥	0.15	um """,
        "answer":
            """AA_4b {
                @ Space between two AAs inside DG/TG.
                @ DRC doesn’t check LDBK region.	≥	0.15	um
                X = EXT (AA INSIDE DGTG) < 0.15 ABUT < 90 SINGULAR REGION
                OUT1 = X NOT INSIDE LDBK
                OUT1 NOT MARKS 
                }
            """
    }, {
        "query": "Space between AADMP and (AA or AA_DMY) inside DG/TG. (overlap is not allowed). ≥	0.15um",
        "answer":
            """AADMP_4	{
                @ Space between AADMP and (AA or AA_DMY) inside DG/TG. (overlap is not allowed). ≥	0.15um
                X = AA INSIDE DGTG
                OUT1 = EXT AADMP X < 0.15 ABUT < 90 SINGULAR REGION
                OUT1 NOT MARKS 
                X1 = AA_DMY INSIDE DGTG
                OUT2 = EXT AADMP X1 < 0.15 ABUT < 90 SINGULAR REGION
                OUT2 NOT MARKS 

                OUT3 = AADMP AND ((AA OR AADUM_G) INSIDE DGTG)
                OUT3 NOT MARKS 
                }
            """
    }, {
        "query": "Space between AADMP and NW	≥	0.08	um",
        "answer":
            """AADMP_7	{
                @ Space between AADMP and NW	≥	0.08	um
                OUT1 =  EXT AADMP NW < 0.08 ABUT < 90 SINGULAR REGION
                OUT1 NOT MARKS 
                }
            """
    }
]
# Create a sample template
example_template = """
User: {query}
AI: {answer}
"""
# Create a prompt example using the template above
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
prefix = """
            suppose you are an expert in the SVRF(Standard Verification Rule Format) language. \n
            You need to associate the local knowledge base file with the entered question. \n
            I need you to convert the design into the corresponding SVRF(Standard Verification Rule Format) code\n
            Do not make up answers.\n
        """
suffix = """
User: {query}
AI: """
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)
#qa
QA_CHAIN_PROMPT = PromptTemplate.from_template(""" 
Hello, I need your help to write a programming language for a specific field. Later, I will give you a question, and then you need to return the corresponding code code. 
Here is an example: My question:SRDOP_102：SDOP min. width is 0.18
The answer you should come up with:
SRDOP_102 
@ SDOP min. width is 0.18
INT SDOP < 0.18 ABUT<90 SINGULAR REGION

Among them,SRDOP_102 represents the rule name, and the annotation after @ needs you to repeat my question. The code I gave you is missing curly braces like C code. The curly braces should surround the code except for the rule name. Your subsequent code should follow this format. 
I will make numerical modifications to the question being asked. Here is an example: 
When my question is:SRDOP_102：SDOP min. width is 0.19
What answer should you give:
SRDOP_102 
@ SDOP min. width is 0.19
INT SDOP < 0.19 ABUT<90 SINGULAR REGION

When my question is modified to:SRDOP_102：SDOP min. width is 0.13
Your expected answer is:
SRDOP_102 
@ SDOP min. width is 0.13
INT SDOP < 0.13 ABUT<90 SINGULAR REGION

When my question is modified to:Space between ALL_AA, except INST region >= 0.09um
Your expected answer is: 
AA_S_1 
@ Space between ALL_AA, except INST region >= 0.09um
 err1 = EXT ALL_AA < 0.09 ABUT<90 SINGULAR REGION
 err1 NOT INSIDE INST

When my question is modified to:Core_NW space, when the width <= 0.235um, expect INST region <= 0.365um
Your expected answer is:
NW_S_2_e 
@ Core_NW space, when the width <= 0.235um, expect INST region <= 0.365um
 nw_meet_wid_sides = INT [core_NW] <= 0.235 ABUT<90 OPPOSITE
 nw_meet_sps_sides = EXT [core_NW] <= 0.365 ABUT<90 OPPOSITE
 nw_errs_sps_sides = EXT (core_NW) <= 0.365 ABUT<90 OPPOSITE
 err1 = INT (nw_meet_wid_sides NOT COIN EDGE nw_meet_sps_sides) nw_errs_sps_sides <= 0.235 OPPOSITE REGION
 err1 NOT INSIDE INST

 
When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to these and modify the values:
···
{context}
···
Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
You need to provide the corresponding code for the question:{question}

""")



retriever = db.as_retriever(
    # search_type="similarity_score_threshold",
    # search_kwargs={"score_threshold": 0.77, "k":1}
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
# response = qa.run("Please write the SVRF code related to DNW.W.1")
# print(response)


# def chat(question, history):
#     response = qa.run(question)
#     return response
#
# demo = gr.ChatInterface(chat)
#
# demo.launch(inbrowser=True, share=True)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    """
    :param history:
    :param file:
    :return:
    """
    global qa
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    db = store_chroma(documents, embeddings)
    retriever = db.as_retriever(
        # search_type="similarity_score_threshold",
        # search_kwargs={"score_threshold": 0.77, "k": 1}
    )
    qa.retriever = retriever
    history = history + [((file.name,), None)]
    return history


def bot(history):
    """
    :param history:
    :return:
    """
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "File uploaded successfully!"
    else:
        response = qa({"query": message})['result']
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, None),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", ".txt", ".pdf", ".png"])
        # btn.upload(upload_file, upload_button, file_output)

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
