PARAMETRIC_PROMPT = "Answer the following question in a few words.\nQuestion: {question}\nAnswer:"

RAG_PROMPT = "Answer the following question based on the provided context. Answer in a few words.\nContext: {context}\nQuestion: {question}\nAnswer:"

VERBALIZED_CONF_PROMPT = """Answer the following question based on the provided context. Answer in a few words, then rate your confidence.
Context: {context}
Question: {question}
Answer the question, then on a new line write "Confidence: X" where X is 0-100."""
