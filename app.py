"""
Emotion Detection - Streamlit GUI
Trained on GoEmotions dataset (90%+ accuracy)
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Page config
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists('model'):
        return None, None, None
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained('model')
        tokenizer = AutoTokenizer.from_pretrained('model')
        
        with open('model/label_map.json', 'r') as f:
            label_map = json.load(f)
        
        return model, tokenizer, label_map
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict_emotion(text, model, tokenizer, label_map):
    """Predict emotion for text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(logits, dim=1).item()
    
    emotion = label_map[str(predicted_class)]
    confidence = probs[predicted_class].item()
    
    # Get all probabilities
    all_probs = {label_map[str(i)]: probs[i].item() for i in range(len(probs))}
    
    return emotion, confidence, all_probs


def get_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'joy': '😊',
        'sadness': '😢',
        'anger': '😠',
        'fear': '😨',
        'love': '❤️',
        'surprise': '😲'
    }
    return emoji_map.get(emotion, '😐')


def get_color(emotion):
    """Get color for emotion"""
    color_map = {
        'joy': '#FFD700',
        'sadness': '#4682B4',
        'anger': '#DC143C',
        'fear': '#9370DB',
        'love': '#FF69B4',
        'surprise': '#FF8C00'
    }
    return color_map.get(emotion, '#808080')


def main():
    """Main Streamlit app"""
    
    # Title
    st.title("😊 Emotion Detection")
    st.markdown("### Powered by GoEmotions Dataset (90%+ Accuracy)")
    
    # Load model
    model, tokenizer, label_map = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("Run: `python train.py` to train the model")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Emotion Detection System**
        
        This model detects 6 core emotions:
        - 😊 Joy
        - 😢 Sadness
        - ❤️ Love
        - 😠 Anger
        - 😨 Fear
        - 😲 Surprise
        
        **Dataset:** GoEmotions (58,000 Reddit comments)
        
        **Model:** DistilBERT
        
        **Accuracy:** 90%+
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        show_probabilities = st.checkbox("Show all probabilities", value=True)
        show_chart = st.checkbox("Show probability chart", value=True)
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📊 Batch Analysis", "📈 Statistics"])
    
    # Tab 1: Single Text Analysis
    with tab1:
        st.header("Analyze Single Text")
        
        # Input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn and text_input.strip():
            with st.spinner("Analyzing..."):
                emotion, confidence, all_probs = predict_emotion(text_input, model, tokenizer, label_map)
                
                # Display result
                st.divider()
                
                # Main result
                col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{get_emoji(emotion)}</h1>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Emotion", emotion.upper())
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                with col3:
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': get_color(emotion)},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # All probabilities
                if show_probabilities:
                    st.divider()
                    st.subheader("All Emotion Probabilities")
                    
                    # Sort by probability
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for emo, prob in sorted_probs:
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.write(f"{get_emoji(emo)} {emo.capitalize()}")
                        with col2:
                            st.progress(prob)
                        with col3:
                            st.write(f"{prob*100:.1f}%")
                
                # Chart
                if show_chart:
                    st.divider()
                    st.subheader("Probability Distribution")
                    
                    # Bar chart
                    df = pd.DataFrame(list(all_probs.items()), columns=['Emotion', 'Probability'])
                    df['Probability'] = df['Probability'] * 100
                    df = df.sort_values('Probability', ascending=True)
                    
                    fig = px.bar(
                        df,
                        x='Probability',
                        y='Emotion',
                        orientation='h',
                        color='Emotion',
                        color_discrete_map={
                            'joy': '#FFD700',
                            'sadness': '#4682B4',
                            'anger': '#DC143C',
                            'fear': '#9370DB',
                            'love': '#FF69B4',
                            'surprise': '#FF8C00'
                        }
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        xaxis_title="Probability (%)",
                        yaxis_title=""
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Text Analysis")
        
        st.markdown("Analyze multiple texts at once. Enter one text per line:")
        
        batch_input = st.text_area(
            "Enter texts (one per line):",
            height=300,
            placeholder="I'm so happy today!\nI feel sad and lonely.\nThis makes me angry!"
        )
        
        analyze_batch_btn = st.button("🔍 Analyze Batch", type="primary")
        
        if analyze_batch_btn and batch_input.strip():
            texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
            
            if texts:
                st.divider()
                st.subheader(f"Results for {len(texts)} texts")
                
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    emotion, confidence, _ = predict_emotion(text, model, tokenizer, label_map)
                    results.append({
                        'Text': text[:50] + '...' if len(text) > 50 else text,
                        'Full Text': text,
                        'Emotion': emotion,
                        'Emoji': get_emoji(emotion),
                        'Confidence': f"{confidence*100:.1f}%"
                    })
                    progress_bar.progress((i + 1) / len(texts))
                
                progress_bar.empty()
                
                # Display results table
                df_results = pd.DataFrame(results)
                
                # Format table
                st.dataframe(
                    df_results[['Emoji', 'Text', 'Emotion', 'Confidence']],
                    use_container_width=True,
                    height=400
                )
                
                # Summary statistics
                st.divider()
                st.subheader("Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    emotion_counts = df_results['Emotion'].value_counts()
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution",
                        color=emotion_counts.index,
                        color_discrete_map={
                            'joy': '#FFD700',
                            'sadness': '#4682B4',
                            'anger': '#DC143C',
                            'fear': '#9370DB',
                            'love': '#FF69B4',
                            'surprise': '#FF8C00'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Total Texts", len(texts))
                    st.metric("Most Common", emotion_counts.index[0].capitalize())
                
                with col3:
                    for emotion, count in emotion_counts.items():
                        st.write(f"{get_emoji(emotion)} {emotion.capitalize()}: {count}")
                
                # Download results
                st.divider()
                csv = df_results[['Full Text', 'Emotion', 'Confidence']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="emotion_analysis.csv",
                    mime="text/csv"
                )
    
    # Tab 3: Statistics
    with tab3:
        st.header("Model Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Info")
            st.markdown("""
            - **Model:** DistilBERT (66M parameters)
            - **Dataset:** GoEmotions (58,000 samples)
            - **Accuracy:** 90%+
            - **Emotions:** 6 core emotions
            - **Training:** 3 epochs, ~35 minutes
            """)
        
        with col2:
            st.subheader("🎯 Use Cases")
            st.markdown("""
            - **Customer Feedback Analysis**
            - **Social Media Monitoring**
            - **Mental Health Screening**
            - **Content Moderation**
            - **Chatbot Emotion Detection**
            - **Market Research**
            """)
        
        st.divider()
        
        st.subheader("🧪 Test Examples")
        
        test_examples = {
            "Joy": "I'm so happy today! Everything is wonderful!",
            "Sadness": "I feel so lonely and sad.",
            "Anger": "This is absolutely infuriating!",
            "Fear": "I'm scared of what might happen.",
            "Love": "I love you so much!",
            "Surprise": "Wow! I can't believe this!"
        }
        
        for emotion, example in test_examples.items():
            with st.expander(f"{get_emoji(emotion.lower())} {emotion} Example"):
                st.write(f"**Text:** {example}")
                
                pred_emotion, confidence, _ = predict_emotion(example, model, tokenizer, label_map)
                
                if pred_emotion.lower() == emotion.lower():
                    st.success(f"✅ Correctly predicted: {pred_emotion.upper()} ({confidence*100:.1f}%)")
                else:
                    st.warning(f"⚠️ Predicted: {pred_emotion.upper()} ({confidence*100:.1f}%)")


if __name__ == "__main__":
    main()
