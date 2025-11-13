# evaluation_data.py

EVAL_DATASET = [
    # --- Original Questions ---
    {
        "question": "What are the two main components of the Transformer architecture as described in 'Attention Is All You Need'?",
        "ground_truth_answer": "The Transformer architecture is composed of an encoder and a decoder. The encoder maps an input sequence to a continuous representation, and the decoder generates an output sequence using the encoder's output."
    },
    {
        "question": "What does the acronym BERT stand for?",
        "ground_truth_answer": "BERT stands for Bidirectional Encoder Representations from Transformers."
    },
    {
        "question": "What two unsupervised tasks is BERT pre-trained on?",
        "ground_truth_answer": "BERT is pre-trained on two unsupervised tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP)."
    },
    {
        "question": "How many parameters does the largest GPT-3 model have?",
        "ground_truth_answer": "The largest GPT-3 model has 175 billion parameters."
    },

    # --- New, More Detailed Questions ---

    # From 'Attention Is All You Need'

    {
        "question": "What is the purpose of Multi-Head Attention?",
        "ground_truth_answer": "Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. It projects the queries, keys, and values multiple times with different learned linear projections, allowing each 'head' to focus on different aspects of the input."
    },

    # From 'BERT'
    {
        "question": "What is the function of the special [CLS] token in BERT?",
        "ground_truth_answer": "The [CLS] token is a special symbol added to the beginning of every input sequence. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks, such as sentiment analysis or sentence-pair classification."
    },

    # From 'GPT-3'
    {
        "question": "What does 'few-shot learning' mean in the context of GPT-3?",
        "ground_truth_answer": "In the context of GPT-3, 'few-shot learning' refers to the model's ability to perform a task by being provided with only a few examples in the prompt's context, without any gradient updates or fine-tuning."
    },

]