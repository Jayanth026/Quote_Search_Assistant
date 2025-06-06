import streamlit as st
import pandas as pd
import torch
import ast
from sentence_transformers import SentenceTransformer,util

# Page configuration
st.set_page_config(
    page_title="Quote Search Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load and cache model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model=load_model()

# Load data and compute embeddings
@st.cache_data(show_spinner=False)
def load_data_and_embeddings(csv_path:str):
    df=pd.read_csv(csv_path)

    def build_full_text(row):
        quote=row["quote"]
        author=row["author"]
        raw_tags=row["tags"]
        if isinstance(raw_tags,str):
            try:tags_list=ast.literal_eval(raw_tags)
            except:tags_list=[]
        elif isinstance(raw_tags,list):tags_list=raw_tags
        else:tags_list=[]
        return f"{quote} {author} {' '.join(tags_list)}"

    df["full_text"]=df.apply(build_full_text,axis=1)
    embeddings=model.encode(df["full_text"].tolist(),convert_to_tensor=True)
    return df,embeddings

csv_file_path="C:\\Users\\jayan\\Downloads\\quotes_cleaned (1).csv"
df,corpus_embeddings=load_data_and_embeddings(csv_file_path)

# App title and sidebar
st.title("Quote Search Assistant")
st.markdown("Enter a natural-language query to retrieve similar quotes from the dataset.")

with st.sidebar:
    st.markdown("## Instructions")
    st.markdown(
        "- Type something like **\"quotes about courage by women authors\"**\n"
        "- Adjust **Top K results** using the slider below\n"
        "- Toggle detailed view for formatted output"
    )

# Input
query=st.text_input("Enter your question (e.g., 'quotes about courage by women authors')")
top_k=st.slider("Top K results",min_value=1,max_value=10,value=5,step=1)

# Search and results
if query:
    query_embedding=model.encode(query,convert_to_tensor=True)
    scores=util.cos_sim(query_embedding,corpus_embeddings)[0]
    top_results=torch.topk(scores,k=min(top_k,scores.shape[0]))

    results=[]
    for idx_tensor,score_tensor in zip(top_results.indices,top_results.values):
        idx=idx_tensor.item()
        sim_score=round(score_tensor.item(),4)
        row=df.iloc[idx]
        raw_tags=row["tags"]
        if isinstance(raw_tags,str):
            try:tags_list=ast.literal_eval(raw_tags)
            except:tags_list=[]
        elif isinstance(raw_tags,list):tags_list=raw_tags
        else:tags_list=[]
        results.append({
            "quote":row["quote"],
            "author":row["author"],
            "tags":tags_list,
            "similarity_score":sim_score
        })

    st.subheader("Top K Results (Structured JSON)")
    st.json(results)

    if st.checkbox("Show detailed results with similarity scores"):
        st.subheader("Retrieved Quotes")
        for item in results:
            st.markdown(f"**Quote:** {item['quote']}")
            st.markdown(f"**Author:** {item['author']}")
            st.markdown(f"**Tags:** {', '.join(item['tags']) if item['tags'] else '–'}")
            st.markdown(f"**Similarity:** {item['similarity_score']}")
            st.markdown("---")

# Footer
st.markdown(
    "<br><hr><br>"
    "Built with :heart: using Streamlit and SentenceTransformers",
    unsafe_allow_html=True
)
