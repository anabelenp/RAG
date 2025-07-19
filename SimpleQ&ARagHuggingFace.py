# Step 1: Install the necessary packages BEFORE running this code.
# !pip install transformers datasets faiss-cpu

# Step 2: Import required modules from Hugging Face Transformers
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# 🧠 Step 3: Load the RAG tokenizer
# This will help convert your question (text) into a format the model can understand (numbers).
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# 🔍 Step 4: Load the retriever
# The retriever searches for helpful documents from a small sample dataset.
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",      # Pretrained model name
    index_name="exact",              # Type of search method used
    use_dummy_dataset=True           # Use a tiny built-in dataset for testing
)

# 🤖 Step 5: Load the main RAG model
# This model takes the question + documents found, and writes a smart answer.
model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever              # Plug in the retriever we just created
)

# 🧾 Step 6: Ask your question
# You can change the question to anything you like.
question = "Where do bats live?"

# 🧰 Step 7: Prepare the input for the model
# This turns your text question into numbers (tokens) and puts it in PyTorch format.
input_dict = tokenizer.prepare_seq2seq_batch(
    question,
    return_tensors="pt"             # 'pt' means we are using PyTorch
)

# ✍️ Step 8: Generate an answer using the model
# The model will look for relevant documents and use them to write a good answer.
#"input_ids" is a built-in key that comes from how the Hugging Face tokenizer works.
generated = model.generate(input_ids=input_dict["input_ids"])

# 📤 Step 9: Decode the model’s output from tokens back to text
# skip_special_tokens=True removes unnecessary formatting tokens like <pad>, <s>, etc.
answer = tokenizer.batch_decode(generated, skip_special_tokens=True)

# ✅ Step 10: Print the final answer
print("Question:", question)
print("Answer:", answer[0])  # answer is a list with one item, so we print answer[0]
