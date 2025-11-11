# ==========================================
# 🧠 BERT Semantic Attention Visualizer (Professional Dark Minimal)
# ==========================================
import torch
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import BertTokenizer, BertModel

# ==========================================
# 🔹 Page Setup
# ==========================================
st.set_page_config(page_title="BERT Attention Visualizer", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: #fafafa;}
    .stApp {background-color: #0e1117;}
    h1, h2, h3, h4, h5, h6, p {color: #fafafa !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 BERT Semantic Attention Visualizer")
st.caption("Interactive exploration of token-to-token attention from **BERT-base** — visualize semantic relationships via graph and heatmap.")
st.markdown(
    """
    <div style='font-size:14px; color:#9ca3af;'>
    Built for research and interpretability by <b>Girish G H</b> · 
    <a href='https://girishgh7.github.io/personal-Website/' target='_blank' style='color:#3b82f6;'>Website</a> · 
    <a href='mailto:girishghegde7@gmail.com' style='color:#3b82f6;'>Email</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 🔹 Load Model + Tokenizer
# ==========================================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==========================================
# 🔹 User Input
# ==========================================
sentence = st.text_input("✏️ Enter a sentence:", "Tom ate the apple off the tree")
threshold = st.slider("Attention Strength Threshold", 0.0, 1.0, 0.3, 0.05)

# ==========================================
# 🔹 Visualization
# ==========================================
if st.button("🔍 Visualize Attention"):
    with st.spinner("Analyzing BERT attention..."):
        # Tokenize and run model
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

        # Average across layers & heads
        attn_tensor = torch.stack(attentions).squeeze(1)
        attn_mean = attn_tensor.mean(dim=(0, 1))
        attn = attn_mean.detach().cpu().numpy()

        # Remove [CLS] and [SEP]
        valid_indices = [i for i, t in enumerate(tokens) if t not in ["[CLS]", "[SEP]"]]
        words = [tokens[i] for i in valid_indices]
        attn = attn[valid_indices][:, valid_indices]
        attn = attn / attn.max()

        # ==========================================
        # 🔸 Build Graph
        # ==========================================
        G = nx.DiGraph()
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j and attn[i, j] > threshold:
                    G.add_edge(w1, w2, weight=attn[i, j])

        if not G.edges:
            st.warning("⚠️ No strong attention links found. Try lowering the threshold.")
        else:
            pos = nx.spring_layout(G, seed=42)
            edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
            min_w, max_w = min(edge_weights), max(edge_weights)

            # Edge traces
            edge_traces = []
            viridis = px.colors.sequential.Viridis
            for u, v, data in G.edges(data=True):
                w = data["weight"]
                norm = (w - min_w) / (max_w - min_w + 1e-6)
                color = viridis[int(norm * (len(viridis) - 1))]
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        line=dict(width=1 + 4 * w, color=color),
                        mode="lines",
                        hoverinfo="text",
                        text=[f"<b>{u}</b> → <b>{v}</b><br>Weight: {w:.3f}"],
                    )
                )

            # Node traces
            node_x, node_y = zip(*[pos[w] for w in G.nodes()])
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=24,
                    color="#3b82f6",
                    line=dict(width=2, color="#60a5fa"),
                    opacity=0.95,
                ),
                hoverinfo="text",
            )

            # Graph figure
            graph_fig = go.Figure(data=edge_traces + [node_trace])
            graph_fig.update_layout(
                title=dict(text="BERT Attention Graph", x=0.5, font=dict(size=20)),
                template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                hovermode="closest",
            )

            # ==========================================
            # 🔸 Heatmap
            # ==========================================
            heatmap_fig = px.imshow(
                attn,
                x=words,
                y=words,
                color_continuous_scale="Viridis",
                labels=dict(x="Key Token", y="Query Token", color="Attention Weight"),
                title="Token-to-Token Attention Heatmap",
            )
            heatmap_fig.update_layout(
                template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40),
                coloraxis_colorbar=dict(title="Weight"),
            )

            # ==========================================
            # 🔹 Display Side-by-Side
            # ==========================================
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(graph_fig, use_container_width=True)
            with col2:
                st.plotly_chart(heatmap_fig, use_container_width=True)

            st.success("✅ Visualization complete.")
