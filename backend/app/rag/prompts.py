RAG_SYSTEM_PROMPT = """You are a precise, professional document assistant. Your sole task is to answer the user's question using ONLY the provided excerpts from the uploaded PDF document.

CRITICAL INSTRUCTIONS:
1. Base your answer solely on the context provided below.
2. If the context does not contain sufficient information to answer the question truthfully and completely, state exactly: "I cannot find the answer to this question in the uploaded document."
3. Do not assume, speculate, extrapolate, or utilize external knowledge not present in the context.
4. Keep the answer clear, objective, and concise.
5. Do not include any emojis, informal language, or conversational filler.
6. When answering, you may mention the relevant page number if helpful.
"""

RAG_USER_PROMPT_TEMPLATE = """CONTEXT FROM UPLOADED PDF:
--------------------------------------------------
{context}
--------------------------------------------------

QUESTION:
{question}

ANSWER:"""


def build_rag_prompt(question: str, context_items: list[dict]) -> str:
    """Formats retrieved context items and question into the final LLM prompt."""
    formatted_context_blocks = []
    for item in context_items:
        page_num = item.get("page", "Unknown")
        text = item.get("text", "").strip()
        formatted_context_blocks.append(f"[Source Page {page_num}]:\n{text}")

    context_str = "\n\n".join(formatted_context_blocks)
    prompt = f"{RAG_SYSTEM_PROMPT}\n\n" + RAG_USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question.strip(),
    )
    return prompt
