
# ⚡ DISCOM Chatbot

A smart assistant to answer electricity-related queries using your own custom data.

---

## ✅ How to Run the DISCOM Chatbot (Step-by-Step)

### 🔧 Step 1: Install All Required Libraries

Before running the chatbot, install all the necessary Python libraries:

```bash
pip install -r requirements.txt
```

---

### 🚀 Step 2: Run the Chatbot App

Now it's time to run the chatbot!

#### 🟢 If you're using a fresh setup or virtual environment:

```bash
python -m streamlit run ChatBot_App.py
```

#### 🟢 If you're already in an environment (like Anaconda Prompt):

```bash
streamlit run ChatBot_App.py
```

💬 This will launch the chatbot in your default web browser.

---

## 📚 Optional: Teach New Data to the Chatbot

To add more Q&A pairs and improve chatbot knowledge:

### ✍️ Step 1: Update the CSV File

1. Open the file: `Que and Ans.csv`
2. Add new **questions** in the `Query` column and **answers** in the `Answer` column.
3. Save the file.

📌 Ensure the file is in the same folder as your chatbot files.

---

### 🧠 Step 2: Train the Chatbot on New Data

Run the training script:

```bash
python To_add_data.py
```

Then follow the prompt:

```
📁 Enter path to your Q&A file (CSV/XLSX):
```

➡ Type the file name (e.g., `Que and Ans.csv`) and press Enter.

You'll see:

```
➕ Adding X new Q&A pairs...
✅ Updated data has been added to the vector database.
```

---

## 🎉 You’re All Set!

Your chatbot is now ready with updated knowledge and can answer the new queries as well.

---

Made with ❤️ for DISCOM solutions.
